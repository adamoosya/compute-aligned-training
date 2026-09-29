"""Reuse the generation module to collect complete, prompt-major RL batches."""
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any
import torch
from cat.data.math import MathExample
from cat.data.tokenization import EncodedExample, CausalCollator
from cat.generation.config import SamplingConfig
from cat.generation.sampling import sample_example, chunk_seed
from cat.rewards.math import extract_answer, answer_matches, normalize_answer
from cat.rewards.protein import protein_reward
from .rl import RolloutBatch


@dataclass(frozen=True)
class RLPrompt:
    example_id: str
    text: str
    answer: str | None = None
    input_h: float | None = None
    token_ids: tuple[int, ...] | None = None


def collect_rollouts(model, tokenizer, prompts: list[RLPrompt], sampling: SamplingConfig, *,
                     step: int, task: str) -> tuple[RolloutBatch, list[dict[str, Any]]]:
    if not prompts or sampling.num_candidates < 2 or type(step) is not int or step < 0:
        raise ValueError("Need prompts, >=2 rollouts per prompt and a nonnegative step")
    if task not in {"math", "protein_unconditional", "protein_conditional"}:
        raise ValueError("Unknown rollout task")
    if len({p.example_id for p in prompts}) != len(prompts):
        raise ValueError("Prompt IDs must be unique within a logical batch")
    encoded, records, rewards, keys = [], [], [], []
    settings = replace(sampling, seed=chunk_seed(sampling.seed, "rl_step", step))
    for prompt in prompts:
        if task == "math" and not prompt.answer:
            raise ValueError("MATH rollouts need a reference answer")
        if task == "protein_conditional" and prompt.input_h is None:
            raise ValueError("Conditional protein rollouts need input hydrophobicity")
        # The sampler's record carrier does not inspect the target when generating.
        example = MathExample(prompt.example_id, prompt.text, prompt.answer or "unused",
                               None, None, "rl", 0)
        candidates, details = sample_example(model, tokenizer, example, settings,
                                               prompt_ids=prompt.token_ids)
        prompt_ids = tuple(details["prompt_token_ids"])
        r, group_keys = [], []
        for i, (completion, attempt) in enumerate(zip(candidates.completions, details["attempts"])):
            target = tuple(attempt["token_ids"])
            ids = prompt_ids + target
            ex = EncodedExample(f"{prompt.example_id}:{i}", ids,
                                (-100,) * len(prompt_ids) + target, len(prompt_ids), len(ids))
            ex.validate()
            encoded.append(ex)
            if task == "math":
                parsed = extract_answer(completion).text
                r.append(float(answer_matches(parsed, prompt.answer)))
                group_keys.append(normalize_answer(parsed) if parsed is not None else None)
            else:
                r.append(protein_reward(completion, input_h=prompt.input_h if task == "protein_conditional" else None))
                group_keys.append(None)
        rewards.append(r)
        keys.append(group_keys)
        records.append({"example_id": prompt.example_id, "prompt": prompt.text,
                        "input_h": prompt.input_h, "completions": list(candidates.completions),
                        "rewards": r, "answer_keys": group_keys, "tokens": details})
    tensors = CausalCollator(tokenizer.pad_token_id)(encoded)
    batch = RolloutBatch(**tensors, rewards=torch.tensor(rewards, dtype=torch.float32),
                         answer_keys=keys if task == "math" else None)
    batch.validate()
    return batch, records
