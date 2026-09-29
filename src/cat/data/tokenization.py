"""Single-turn chat encoding with completion-only supervision and right padding."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import torch
from torch.utils.data import Dataset

from .math import MathExample, MathSelection


class ChatTokenizer(Protocol):
    def apply_chat_template(self, conversation: list[dict[str, str]], *,
                            tokenize: bool, add_generation_prompt: bool) -> list[int]: ...


@dataclass(frozen=True)
class EncodedExample:
    example_id: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    prompt_tokens: int
    original_tokens: int

    @property
    def truncated(self) -> bool:
        return len(self.input_ids) < self.original_tokens

    @property
    def target_tokens(self) -> int:
        return len(self.input_ids) - self.prompt_tokens

    def validate(self) -> None:
        if not self.input_ids or len(self.input_ids) != len(self.labels):
            raise ValueError("input_ids and labels must have the same nonzero length")
        if not 1 <= self.prompt_tokens < len(self.input_ids):
            raise ValueError("Each example needs a prompt and at least one target token")
        if self.original_tokens < len(self.input_ids):
            raise ValueError("original_tokens cannot be less than retained length")
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in self.input_ids):
            raise ValueError("input_ids must be nonnegative integers")
        if self.labels[:self.prompt_tokens] != (-100,) * self.prompt_tokens:
            raise ValueError("All prompt labels must be -100")
        if self.labels[self.prompt_tokens:] != self.input_ids[self.prompt_tokens:]:
            raise ValueError("Target labels must equal the corresponding input IDs")


def _token_list(value: Any) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError("apply_chat_template(tokenize=True) must return a nonempty list")
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in value):
        raise ValueError("Chat token IDs must be nonnegative Python integers")
    return value


def encode_math_example(
    example: MathExample, tokenizer: ChatTokenizer, *, target: str,
    max_length: int = 1024, overflow: str = "error",
    prompt_template: str = "Problem:\n{problem}",
) -> EncodedExample:
    """Encode a full chat, verifying that the prompt token prefix is unchanged.

    Prefix mismatches are rejected rather than guessing the label boundary.
    overflow='truncate_target' truncates only the end, never the prompt; no
    synthetic EOS token is appended after truncation. The event is recorded.
    """
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 2:
        raise ValueError("max_length must be an integer >= 2")
    if overflow not in {"error", "truncate_target"}:
        raise ValueError("overflow must be 'error' or 'truncate_target'")
    if not isinstance(prompt_template, str) or "{problem}" not in prompt_template:
        raise ValueError("prompt_template must include {problem}")
    content = example.target_text(target)
    # str.replace leaves unrelated LaTeX braces in the template untouched.
    prompt = prompt_template.replace("{problem}", example.problem)
    user_message = {"role": "user", "content": prompt}
    prefix = _token_list(tokenizer.apply_chat_template(
        [user_message], tokenize=True, add_generation_prompt=True,
    ))
    full = _token_list(tokenizer.apply_chat_template(
        [user_message, {"role": "assistant", "content": content}],
        tokenize=True, add_generation_prompt=False,
    ))
    if full[:len(prefix)] != prefix:
        raise ValueError(
            f"{example.example_id}: chat template/tokenizer changes the prompt prefix; "
            "cannot safely locate completion labels"
        )
    if len(prefix) >= len(full):
        raise ValueError(f"{example.example_id}: no completion tokens")
    if len(prefix) >= max_length:
        raise ValueError(f"{example.example_id}: prompt fills max_length; no target fits")
    original_tokens = len(full)
    if original_tokens > max_length:
        if overflow == "error":
            raise ValueError(f"{example.example_id}: {original_tokens} tokens exceed max_length")
        full = full[:max_length]
    labels = [-100] * len(prefix) + full[len(prefix):]
    encoded = EncodedExample(
        example.example_id, tuple(full), tuple(labels), len(prefix), original_tokens,
    )
    encoded.validate()
    return encoded


class EncodedMathDataset(Dataset):
    """Pre-tokenized records; tokenization happens once, not every epoch."""
    def __init__(self, examples: Sequence[EncodedExample]):
        if not examples:
            raise ValueError("Encoded dataset must not be empty")
        self.examples = tuple(examples)
        for example in self.examples:
            example.validate()

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> EncodedExample:
        return self.examples[index]

    def manifest(self) -> dict[str, Any]:
        return {
            "count": len(self), "truncated_count": sum(x.truncated for x in self.examples),
            "target_tokens": sum(x.target_tokens for x in self.examples),
            "records": [
                {"id": x.example_id, "prompt_tokens": x.prompt_tokens,
                 "target_tokens": x.target_tokens, "original_tokens": x.original_tokens,
                 "retained_tokens": len(x.input_ids), "truncated": x.truncated}
                for x in self.examples
            ],
        }


def encode_math_selection(selection: MathSelection, tokenizer: ChatTokenizer,
                          **encoding: Any) -> EncodedMathDataset:
    return EncodedMathDataset([
        encode_math_example(example, tokenizer, **encoding) for example in selection.examples
    ])


@dataclass(frozen=True)
class CausalCollator:
    pad_token_id: int

    def __post_init__(self) -> None:
        if (isinstance(self.pad_token_id, bool) or not isinstance(self.pad_token_id, int)
                or self.pad_token_id < 0):
            raise ValueError("pad_token_id must be a nonnegative integer")

    def __call__(self, examples: Sequence[EncodedExample]) -> dict[str, torch.Tensor]:
        if not examples:
            raise ValueError("Cannot collate an empty batch")
        for example in examples:
            example.validate()
        width = max(len(x.input_ids) for x in examples)
        ids = torch.full((len(examples), width), self.pad_token_id, dtype=torch.long)
        labels = torch.full_like(ids, -100)
        attention_mask = torch.zeros_like(ids)
        for row, example in enumerate(examples):
            length = len(example.input_ids)
            ids[row, :length] = torch.tensor(example.input_ids, dtype=torch.long)
            labels[row, :length] = torch.tensor(example.labels, dtype=torch.long)
            attention_mask[row, :length] = 1
        # Real EOS labels survive even when the model uses EOS as its padding ID.
        return {"input_ids": ids, "attention_mask": attention_mask, "labels": labels}
