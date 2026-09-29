"""Real Transformers/PEFT generation with local tiny model/tokenizer files only."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import torch

from cat.data import select_math_examples
from cat.evaluation.records import read_candidate_records
from cat.evaluation.scoring import MathScoreConfig
from cat.experiments.generation import generate_and_score
from cat.generation import SamplingConfig
from cat.models.hf import HFModelConfig, LoRASettings, load_lora_model, save_lora_adapter
from cat.models.inference import load_inference_model


def run_generation_smoke(output_parent: str | Path = "outputs") -> Path:
    from cat.testing.lora_smoke import _local_base, _RECORDS
    parent = Path(output_parent)
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="generation-smoke-", dir=parent)).resolve()
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        base = root / "tiny-base"
        _local_base(base)
        model_config = HFModelConfig(str(base), attention="eager")
        lora = LoRASettings(r=4, alpha=8)
        model, tokenizer = load_lora_model(model_config, lora)
        # Nonzero random adapter fixture, explicitly not a trained checkpoint.
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    parameter.normal_(0, 0.01)
        adapter = save_lora_adapter(model, tokenizer, root / "adapter",
                                    model_config=model_config, lora=lora,
                                    provenance={"kind": "random_smoke_fixture"})
        del model
        model, tokenizer = load_inference_model(model_config, adapter=adapter)
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise AssertionError("Inference model is not fully frozen")
        frozen = {name: p.detach().clone() for name, p in model.named_parameters()}
        selection = select_math_examples(_RECORDS[:2], split="smoke_eval", shuffle=False)
        sampling = SamplingConfig(num_candidates=4, chunk_size=2, max_prompt_tokens=64,
                                  max_new_tokens=4, top_k=0)
        scoring = MathScoreConfig(budgets=(1, 2, 4), majority_trials=20)
        rng_before = torch.random.get_rng_state().clone()
        first = generate_and_score(model, tokenizer, selection, sampling, scoring, root / "first",
                                   provenance={"model": "tiny_random_mistral"}, smoke_only=True)
        if not torch.equal(torch.random.get_rng_state(), rng_before):
            raise AssertionError("Sampling changed the caller's CPU RNG state")
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
        loaded, tok2 = load_inference_model(model_config, adapter=adapter)
        second = generate_and_score(loaded, tok2, selection, sampling, scoring, root / "reloaded",
                                    provenance={"model": "tiny_random_mistral"}, smoke_only=True)
        for name in ("candidates.jsonl", "tokens.jsonl"):
            if (first / name).read_bytes() != (second / name).read_bytes():
                raise AssertionError(f"Seeded generation changed after adapter reload: {name}")
        a = json.loads((first / "scores/summary.json").read_text())
        b = json.loads((second / "scores/summary.json").read_text())
        if a != b:
            raise AssertionError("Scoring changed after adapter reload")
        records = read_candidate_records(first / "candidates.jsonl")
        if len(records) != 2 or any(len(x.completions) != 4 for x in records):
            raise AssertionError("Candidate count mismatch")
        _summary = {"kind": "smoke_only", "prompts": 2, "candidates_per_prompt": 4,
                    "frozen_weights": True, "seeded_reload_match": True, "scores_match": True}
        (root / "summary.json").write_text(json.dumps(_summary, indent=2) + "\n")
        print("2 prompts x 4 candidates; token boundaries and candidate counts OK")
        print("Frozen weights unchanged; seeded adapter reload matches; scoring/checksums OK")
        print("Generation smoke test passed (tiny random Mistral; CPU; no model or dataset downloads).")
        print(f"Local smoke outputs: {root}")
        return root
    finally:
        torch.set_num_threads(threads)
