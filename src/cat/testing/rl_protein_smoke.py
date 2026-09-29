"""Offline optimizer/rollout fixtures; no pretrained models or benchmark downloads."""
from __future__ import annotations
import copy
from contextlib import redirect_stdout
import io
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import torch
from cat.experiments.rl_config import load_rl_config
from cat.experiments.rl import _train_loop
from cat.training.rl import RolloutBatch, cat_weights, CATWeightConfig
from cat.training.rollouts import RLPrompt
from cat.training.checkpoints import save_sft_snapshot, load_sft_snapshot
from cat.testing.tiny import TinyCausalLM
from cat.rewards.protein import protein_reward
from cat.evaluation.protein import expected_max_reward


def _fixture_collector(model, tokenizer, prompts, sampling, *, step, task):
    """Sample a tiny categorical completion followed by a fixed fixture terminator.

    This tests the actual loop with sampled rewards; it is not MATH/ProtGPT2.
    Production rollout collection uses the previously tested chunked HF sampler.
    """
    m = sampling.num_candidates
    with torch.random.fork_rng(), torch.no_grad():
        torch.manual_seed(sampling.seed + step)
        start = torch.zeros((len(prompts) * m, 1), dtype=torch.long)
        probs = model(start).logits[:, -1, 2:6].softmax(-1)
        chosen = torch.multinomial(probs, 1).squeeze(1) + 2
    ids = torch.stack([torch.zeros_like(chosen), chosen, torch.ones_like(chosen)], 1)
    labels = ids.clone(); labels[:, 0] = -100
    texts = {2: "DDDDDDDDDD", 3: "AAAADDDDDD", 4: "AAAAAAADDD", 5: "AAAAAAAAAA"}
    reward_rows, keys, records = [], [], []
    for j, prompt in enumerate(prompts):
        samples = chosen[j * m:(j + 1) * m].tolist()
        if task == "math":
            k = [str(v - 2) for v in samples]
            r = [float(v == prompt.answer) for v in k]
        else:
            k = [None] * m
            r = [protein_reward(texts[v], input_h=prompt.input_h if task == "protein_conditional" else None) for v in samples]
        keys.append(k); reward_rows.append(r)
        records.append({"example_id": prompt.example_id, "fixture": True, "rewards": r,
                        "sampled_token_ids": samples})
    return RolloutBatch(ids, torch.ones_like(ids), labels, torch.tensor(reward_rows),
                         keys if task == "math" else None), records


def run_rl_protein_smoke(output_parent="outputs"):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    parent = Path(output_parent); parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="rl-protein-smoke-", dir=parent)).resolve()
    try:
        configs = [
          "configs/rl/math_passn_baseline.json", "configs/rl/math_passn_n4_raw.json",
          "configs/rl/math_passn_n4_log.json", "configs/rl/math_majority_n4_raw.json",
          "configs/rl/math_majority_n4_log.json", "configs/protein/protein_unconditional_baseline.json",
          "configs/protein/protein_unconditional_bon4.json", "configs/protein/protein_conditional_baseline.json",
          "configs/protein/protein_conditional_bon4.json"]
        torch.manual_seed(13)
        base = TinyCausalLM(vocab_size=6, hidden_size=8)
        summary = []
        for filename in configs:
            c = load_rl_config(filename)
            c = replace(c, model=replace(c.model, device="cpu", precision="fp32", quantization="none"),
                        update=replace(c.update, precision="fp32", forward_batch_size=3),
                        training={**c.training, "max_steps": 3, "prompts_per_step": 4, "learning_rate": .01},
                        sampling=replace(c.sampling, num_candidates=8, chunk_size=4))
            model = copy.deepcopy(base)
            reference = copy.deepcopy(base).requires_grad_(False)
            frozen = {k: v.clone() for k, v in reference.state_dict().items()}
            prompts = [RLPrompt(f"fixture:{i}", f"fixture {i}", answer=str(i % 4), input_h=.2 + .2 * i) for i in range(4)]
            run = root / c.name; run.mkdir()
            with redirect_stdout(io.StringIO()):
                metrics = _train_loop(model, None, prompts, c, run, reference_model=reference,
                                      collector=_fixture_collector)
            if len(metrics) != 3 or not any(not torch.equal(v, base.state_dict()[k]) for k, v in model.state_dict().items()):
                raise AssertionError(f"Parameters did not update: {c.name}")
            for k, v in reference.state_dict().items():
                torch.testing.assert_close(v, frozen[k], rtol=0, atol=0)
            save_sft_snapshot(run / "snapshot", model, {"kind": "smoke_only"})
            restored = copy.deepcopy(base); load_sft_snapshot(run / "snapshot", restored)
            for k, v in model.state_dict().items():
                torch.testing.assert_close(v, restored.state_dict()[k], rtol=0, atol=0)
            summary.append({"config": c.name, "updates": 3, "reference_frozen": True, "reload_match": True})
        r = torch.tensor([[1., 0., 0., 0.], [1., 1., 1., 0.]])
        w, _ = cat_weights(r, torch.full_like(r, -2), CATWeightConfig("majority", 8, normalize=True),
                            answer_keys=[["yes", "a", "b", "c"], ["yes", "yes", "yes", "a"]])
        if torch.allclose(w, torch.ones_like(w)):
            raise AssertionError("Cross-prompt CAT weighting was cancelled")
        if expected_max_reward([-2., 4.], 2) != 2.5:
            raise AssertionError("Protein expected-max check failed")
        (root / "summary.json").write_text(json.dumps({"kind": "smoke_only", "checks": summary}, indent=2) + "\n")
        print("MATH: baseline, raw/log Pass@N and raw/log Majority Vote updates OK")
        print("Proteins: unconditional and conditional baseline/BoN updates OK")
        print("Cross-prompt weights preserved; references frozen; save/reload OK")
        print("Combined RL/protein smoke test passed (CPU fixtures; no model or dataset downloads).")
        print(f"Local smoke outputs: {root}")
        return root
    finally:
        torch.set_num_threads(threads)
