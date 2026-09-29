"""Explicit independent-sampling settings, separate from test-time scoring."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from string import Formatter
from typing import Any, Mapping


def positive_int(value: Any, name: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class SamplingConfig:
    num_candidates: int = 64
    chunk_size: int = 4
    max_prompt_tokens: int = 512
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 0
    seed: int = 42
    prompt_template: str = "Problem:\n{problem}"

    def __post_init__(self) -> None:
        for field in ("num_candidates", "chunk_size", "max_prompt_tokens", "max_new_tokens"):
            positive_int(getattr(self, field), field)
        if self.chunk_size > self.num_candidates:
            raise ValueError("chunk_size cannot exceed num_candidates")
        positive_int(self.top_k, "top_k", 0)
        positive_int(self.seed, "seed", 0)
        if self.seed >= 2**63:
            raise ValueError("seed must be below 2**63")
        for field in ("temperature", "top_p"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be finite and positive")
        if self.top_p > 1:
            raise ValueError("top_p must be <= 1")
        if not isinstance(self.prompt_template, str) or not self.prompt_template.strip():
            raise ValueError("prompt_template must be nonempty")
        try:
            fields = [(field, spec, conversion) for _, field, spec, conversion
                      in Formatter().parse(self.prompt_template) if field is not None]
        except ValueError as error:
            raise ValueError("invalid prompt_template") from error
        if fields != [("problem", "", None)]:
            raise ValueError("prompt_template must contain exactly one plain {problem} field")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SamplingConfig:
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("sampling config has missing or unknown fields")
        return cls(**value)
