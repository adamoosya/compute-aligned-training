from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch.utils.data import DataLoader

from cat.data import CausalCollator, encode_math_selection, select_math_examples
from cat.objectives import majority_sft_loss, passn_sft_loss
from cat.testing.tiny import ByteChatTokenizer, TinyCausalLM, tiny_math_records
from cat.training import (SFTObjective, completion_log_probs, evaluate_sft_loss,
                          load_sft_snapshot, save_sft_snapshot, sft_loss, train_sft_epoch)


def small_batch():
    labels = torch.tensor([[-100, -100, 1, 2, 3], [-100, 2, 1, -100, -100]])
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    generator = torch.Generator().manual_seed(6)
    logits = torch.randn(2, 5, 4, generator=generator, dtype=torch.float64, requires_grad=True)
    return logits, labels, mask


def test_causal_shift_sequence_sum_against_manual_scores():
    logits, labels, mask = small_batch()
    scores = completion_log_probs(logits, labels, mask)
    log_softmax = torch.log_softmax(logits, dim=-1)
    expected = []
    for row in range(labels.shape[0]):
        expected.append(sum(log_softmax[row, pos-1, labels[row, pos]]
                            for pos in range(1, labels.shape[1]) if labels[row, pos] != -100))
    torch.testing.assert_close(scores.log_prob, torch.stack(expected))
    assert scores.target_tokens.tolist() == [3, 2]


def test_masked_positions_do_not_contribute_to_logit_gradient():
    logits, labels, mask = small_batch()
    (-completion_log_probs(logits, labels, mask).log_prob.sum()).backward()
    assert torch.equal(logits.grad[0, 0], torch.zeros(4, dtype=torch.float64))
    assert torch.equal(logits.grad[:, -1], torch.zeros(2, 4, dtype=torch.float64))
    assert logits.grad[0, 1].abs().sum() > 0  # predicts the first supervised token
    assert torch.equal(logits.grad[1, 2:], torch.zeros(3, 4, dtype=torch.float64))


@pytest.mark.parametrize('strategy', ['ce', 'passn', 'majority'])
@pytest.mark.parametrize('reduction', ['sequence_mean', 'token_mean'])
def test_sft_objective_reductions(strategy, reduction):
    logits, labels, mask = small_batch()
    objective = SFTObjective(strategy, n=1 if strategy == 'ce' else 4,
                             k=2 if strategy == 'majority' else None, reduction=reduction)
    scores = completion_log_probs(logits, labels, mask)
    losses = (-scores.log_prob if strategy == 'ce' else
              passn_sft_loss(scores.log_prob, 4, reduction='none') if strategy == 'passn' else
              majority_sft_loss(scores.log_prob, 4, 2, reduction='none'))
    denominator = 2 if reduction == 'sequence_mean' else 5
    torch.testing.assert_close(sft_loss(logits, labels, objective, mask), losses.sum()/denominator)


@pytest.mark.parametrize('strategy', ['passn', 'majority'])
def test_sft_autograd_finite_differences(strategy):
    logits, labels, mask = small_batch()
    objective = SFTObjective(strategy, n=4, k=2 if strategy == 'majority' else None)
    assert torch.autograd.gradcheck(lambda x: sft_loss(x, labels, objective, mask), (logits,),
                                    eps=1e-6, atol=2e-6, rtol=1e-4)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_score_precision(dtype):
    logits, labels, mask = small_batch()
    output = completion_log_probs(logits.to(dtype), labels, mask)
    assert output.log_prob.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    assert torch.isfinite(output.log_prob).all()


@pytest.mark.parametrize('error_kind', ['first_label', 'no_target', 'pad_label', 'previous_pad',
                                       'bad_mask', 'bad_vocab', 'bad_shape', 'nan_logits'])
def test_invalid_supervision_rejected(error_kind):
    logits, labels, mask = small_batch()
    if error_kind == 'first_label': labels[0, 0] = 1
    if error_kind == 'no_target': labels[0] = -100
    if error_kind == 'pad_label': labels[1, -1] = 1
    if error_kind == 'previous_pad': mask[0, 1] = 0
    if error_kind == 'bad_mask': mask[0, 1] = 2
    if error_kind == 'bad_vocab': labels[0, -1] = 7
    if error_kind == 'bad_shape': logits = logits[:, :-1]
    if error_kind == 'nan_logits':
        logits = logits.detach().clone()
        logits[0, 1] = float('nan')
    with pytest.raises(ValueError):
        completion_log_probs(logits, labels, mask)


@pytest.mark.parametrize('kwargs', [
    {'strategy': 'oops'}, {'strategy': 'ce', 'n': 2}, {'n': 0}, {'n': True},
    {'strategy': 'majority', 'n': 3}, {'strategy': 'majority', 'n': 3, 'k': 4},
    {'k': 1}, {'reduction': 'wrong'},
])
def test_bad_objectives(kwargs):
    with pytest.raises(ValueError):
        SFTObjective(**kwargs)


def data_loader(batch_size):
    rows = tiny_math_records()
    rows[1]['answer'] = 'a longer test answer'
    rows[-1]['answer'] = 'yet another answer'
    selection = select_math_examples(rows, split='fixture', shuffle=False)
    dataset = encode_math_selection(selection, ByteChatTokenizer(), target='answer')
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=CausalCollator(0))


@pytest.mark.parametrize('strategy', ['ce', 'passn', 'majority'])
@pytest.mark.parametrize('reduction', ['sequence_mean', 'token_mean'])
def test_accumulation_matches_effective_batch_with_unequal_lengths_and_final_tail(strategy, reduction):
    torch.manual_seed(6)
    full = TinyCausalLM().double()
    accumulated = deepcopy(full)
    objective = SFTObjective(strategy, 1 if strategy == 'ce' else 4,
                             2 if strategy == 'majority' else None, reduction)
    opt_full = torch.optim.SGD(full.parameters(), lr=0.001)
    opt_acc = torch.optim.SGD(accumulated.parameters(), lr=0.001)
    steps = []
    class Scheduler:
        def step(self): steps.append(True)
    a = train_sft_epoch(full, data_loader(4), opt_full, objective)
    b = train_sft_epoch(accumulated, data_loader(2), opt_acc, objective,
                         accumulation_steps=2, scheduler=Scheduler())
    for key, value in full.state_dict().items():
        torch.testing.assert_close(value, accumulated.state_dict()[key], rtol=1e-10, atol=1e-12)
    assert a.sequences == b.sequences == 5
    assert a.target_tokens == b.target_tokens
    assert a.optimizer_steps == b.optimizer_steps == 2
    assert len(steps) == 2
    assert a.loss == pytest.approx(b.loss, rel=1e-10)
    assert all(p.grad is None for p in accumulated.parameters())


def test_empty_loaders_and_invalid_accumulation():
    model = TinyCausalLM()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    with pytest.raises(ValueError, match='empty'):
        train_sft_epoch(model, [], opt, SFTObjective())
    with pytest.raises(ValueError, match='empty'):
        evaluate_sft_loss(model, [], SFTObjective())
    with pytest.raises(ValueError, match='accumulation_steps'):
        train_sft_epoch(model, data_loader(2), opt, SFTObjective(), accumulation_steps=0)
    with pytest.raises(ValueError, match='max_grad_norm'):
        train_sft_epoch(model, data_loader(2), opt, SFTObjective(), max_grad_norm=float('nan'))


@pytest.mark.parametrize('mode', [True, False])
def test_evaluation_restores_model_mode_and_keeps_weights(mode):
    model = TinyCausalLM()
    model.train(mode)
    before = deepcopy(model.state_dict())
    stats = evaluate_sft_loss(model, data_loader(2), SFTObjective())
    assert model.training is mode
    assert stats.optimizer_steps == 0
    assert stats.sequences == 5
    for key, value in before.items():
        assert torch.equal(value, model.state_dict()[key])
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize('form', ['tensor', 'mapping', 'attribute'])
def test_model_return_protocols(form):
    class Model(TinyCausalLM):
        def forward(self, **kwargs):
            result = super().forward(**kwargs)
            return result.logits if form == 'tensor' else {'logits': result.logits} if form == 'mapping' else result
    model = Model()
    stats = train_sft_epoch(model, data_loader(2), torch.optim.SGD(model.parameters(), lr=0.001), SFTObjective())
    assert stats.optimizer_steps == 3


def test_snapshot_is_create_only_and_round_trips(tmp_path):
    model = TinyCausalLM()
    path = save_sft_snapshot(tmp_path/'snapshot', model, {'objective': asdict(SFTObjective())})
    restored = TinyCausalLM()
    assert load_sft_snapshot(path, restored)['objective']['strategy'] == 'ce'
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    old = (path/'model_state.pt').read_bytes()
    with pytest.raises(FileExistsError):
        save_sft_snapshot(path, model, {})
    assert (path/'model_state.pt').read_bytes() == old
    with pytest.raises(ValueError):
        save_sft_snapshot(tmp_path/'invalid_metadata', model, {'loss': float('nan')})
    assert not (tmp_path/'invalid_metadata').exists()


def test_offline_smoke_script(tmp_path):
    script = Path(__file__).resolve().parents[1]/'scripts'/'smoke_sft.py'
    output = tmp_path/'run'
    result = subprocess.run([sys.executable, str(script), '--output', str(output)],
                             capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    assert 'SFT smoke test passed' in result.stdout
    report = json.loads((output/'summary.json').read_text())
    assert report['kind'] == 'smoke_only_not_paper_results'
    assert set(report['checks']) == {'ce', 'passn', 'majority'}
    assert all(item['after'] < item['before'] for item in report['checks'].values())
    repeat = subprocess.run([sys.executable, str(script), '--output', str(output)],
                             capture_output=True, text=True, timeout=45)
    assert repeat.returncode != 0
