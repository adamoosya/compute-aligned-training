from dataclasses import replace
import sys
from types import SimpleNamespace

import pytest
import torch

from cat.data import select_math_examples
from cat.generation import SamplingConfig, encode_prompt, sample_example
from cat.generation.sampling import chunk_seed
from generation_support import FakeGenerationConfig, FakeModel, FakeTokenizer


@pytest.fixture
def fixture_sampling(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(GenerationConfig=FakeGenerationConfig))
    example = select_math_examples([{"problem": "999 + 888?", "answer": "6", "solution": "secret reference"}],
                                   split="test", shuffle=False).examples[0]
    config = SamplingConfig(num_candidates=5, chunk_size=2, max_new_tokens=4)
    return FakeModel(), FakeTokenizer(), example, config


def test_chunk_count_and_no_reference_leak(fixture_sampling):
    model, tokenizer, example, config = fixture_sampling
    record, details = sample_example(model, tokenizer, example, config)
    assert [x["num_return_sequences"] for x in model.calls] == [2, 2, 1]
    assert len(record.completions) == len(details["attempts"]) == 5
    assert tokenizer.messages == [[{"role": "user", "content": "Problem:\n999 + 888?"}]]
    assert all(len(tokens) == 4 for tokens in tokenizer.decoded)
    assert all("999" not in text and "888" not in text for text in record.completions)
    assert details["prompt_token_ids"] == [1, 4, 5]
    assert all(c["do_sample"] and c["num_beams"] == 1 for c in model.calls)
    assert all(c["top_k"] == 0 and c["temperature"] == 0.8 for c in model.calls)


@pytest.mark.parametrize("behavior", ["normal", "eos", "empty", "fail"])
def test_rng_modes_and_weights_restored(fixture_sampling, behavior):
    model, tokenizer, example, config = fixture_sampling
    model.train()
    model.drop.eval()
    states = [m.training for m in model.modules()]
    params = [p.detach().clone() for p in model.parameters()]
    before = torch.random.get_rng_state().clone()
    model.behavior = behavior
    if behavior == "fail":
        with pytest.raises(RuntimeError):
            sample_example(model, tokenizer, example, config)
    else:
        sample_example(model, tokenizer, example, config)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert [m.training for m in model.modules()] == states
    for a, b in zip(params, model.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert b.grad is None


@pytest.mark.parametrize("same_pad", [True, False])
def test_immediate_eos_remains_an_attempt(fixture_sampling, same_pad):
    model, tokenizer, example, config = fixture_sampling
    if same_pad:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.behavior = "empty"
    record, details = sample_example(model, tokenizer, example, config)
    assert record.completions == ("",) * 5
    assert all(a["token_ids"] == [2] and a["finish_reason"] == "eos" for a in details["attempts"])


@pytest.mark.parametrize("behavior", ["short", "long", "prefix", "count", "float", "bad_after_eos", "token_range"])
def test_bad_generation_aborts(fixture_sampling, behavior):
    model, tokenizer, example, config = fixture_sampling
    model.behavior = behavior
    with pytest.raises((ValueError, TypeError)):
        sample_example(model, tokenizer, example, config)


def test_generation_is_repeatable_and_order_independent(fixture_sampling):
    model, tokenizer, example, config = fixture_sampling
    a, _ = sample_example(model, tokenizer, example, config)
    other = replace(example, example_id="other_id", problem="Different prompt")
    sample_example(model, tokenizer, other, config)
    b, _ = sample_example(model, tokenizer, example, config)
    assert a == b
    c, _ = sample_example(model, tokenizer, example, replace(config, seed=43))
    assert c.completions != a.completions


@pytest.mark.parametrize("field,value", [
    ("num_candidates", 0), ("num_candidates", True), ("chunk_size", 0), ("chunk_size", 99),
    ("max_prompt_tokens", 0), ("max_new_tokens", -1), ("temperature", 0), ("temperature", float("nan")),
    ("temperature", True), ("top_p", 1.1), ("top_p", 0), ("top_p", float("inf")),
    ("top_k", -1), ("top_k", True), ("seed", -1), ("seed", 2**63),
    ("prompt_template", ""), ("prompt_template", "{answer}"), ("prompt_template", "{problem!r}"),
    ("prompt_template", "{problem:>5}"), ("prompt_template", "{problem} {problem}"),
    ("prompt_template", "{problem.__class__}"), ("prompt_template", "{problem"),
])
def test_bad_sampling_settings(field, value):
    with pytest.raises((ValueError, TypeError)):
        replace(SamplingConfig(), **{field: value})


def test_sampling_settings_roundtrip():
    config = SamplingConfig(prompt_template="Question: {problem}\nUse \\boxed{{}}.")
    assert SamplingConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize("ids", [[], [1.0], [True], [-1], "tokens"])
def test_invalid_tokenizer_ids(fixture_sampling, ids):
    _, tokenizer, example, config = fixture_sampling
    tokenizer.ids = ids
    with pytest.raises(ValueError):
        encode_prompt(tokenizer, example.problem, config)


def test_prompt_and_context_limits_before_generation(fixture_sampling):
    model, tokenizer, example, config = fixture_sampling
    with pytest.raises(ValueError, match="not truncated"):
        sample_example(model, tokenizer, example, replace(config, max_prompt_tokens=2))
    model.config.max_position_embeddings = 6
    with pytest.raises(ValueError, match="context"):
        sample_example(model, tokenizer, example, config)
    assert not model.calls


@pytest.mark.parametrize("field,value", [("eos_token_id", None), ("pad_token_id", None), ("eos_token_id", -1),
                                        ("pad_token_id", True), ("eos_token_id", 99)])
def test_invalid_special_tokens(fixture_sampling, field, value):
    model, tokenizer, example, config = fixture_sampling
    setattr(tokenizer, field, value)
    with pytest.raises(ValueError):
        sample_example(model, tokenizer, example, config)


def test_encoder_decoder_rejected(fixture_sampling):
    model, tokenizer, example, config = fixture_sampling
    model.config.is_encoder_decoder = True
    with pytest.raises(ValueError, match="decoder-only"):
        sample_example(model, tokenizer, example, config)


def test_prompt_token_range(fixture_sampling):
    model, tokenizer, example, config = fixture_sampling
    tokenizer.ids = [1, 999]
    with pytest.raises(ValueError, match="vocabulary"):
        sample_example(model, tokenizer, example, config)


def test_seed_determinism_and_nonnegative():
    assert chunk_seed(42, "test:1", 0) == chunk_seed(42, "test:1", 0)
    assert 0 <= chunk_seed(42, "test:1", 0) < 2**63
    assert len({chunk_seed(42, "test:1", x) for x in range(10)}) == 10
