"""Strict JSON settings for a single SFT stage. No network or model loading."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any

from cat.models.hf import HFModelConfig, LoRASettings, _integer, require_revision
from cat.training.sft import SFTObjective


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _build(cls, value, section):
    if not isinstance(value, dict):
        raise ValueError(f"{section} must be an object")
    try:
        return cls(**value)
    except TypeError as error:
        raise ValueError(f"Invalid keys or values in {section}: {error}") from error


@dataclass(frozen=True)
class MathDataConfig:
    dataset_name: str = "nlile/hendrycks-MATH-benchmark"
    revision: str | None = None
    split: str = "train"
    max_examples: int | None = 5000
    levels: tuple[int, ...] | None = None
    target: str = "answer"
    seed: int = 42
    max_length: int = 1024
    overflow: str = "truncate_target"
    prompt_template: str = "Problem:\n{problem}"

    def __post_init__(self):
        if not isinstance(self.dataset_name, str) or not self.dataset_name.strip():
            raise ValueError("dataset_name must be a nonempty string")
        if self.revision is not None and not isinstance(self.revision, str):
            raise ValueError("data.revision must be a string or null")
        if self.split != "train":
            raise ValueError("This SFT launcher trains only on the train split")
        if self.max_examples is not None:
            _integer(self.max_examples, "data.max_examples")
        _integer(self.seed, "data.seed", 0)
        _integer(self.max_length, "data.max_length", 2)
        if self.levels is not None:
            if not isinstance(self.levels, (tuple, list)) or not self.levels:
                raise ValueError("levels must be a nonempty list or null")
            for item in self.levels:
                _integer(item, "level")
                if item > 5:
                    raise ValueError("level must be between 1 and 5")
            if len(set(self.levels)) != len(self.levels):
                raise ValueError("Duplicate levels")
            object.__setattr__(self, "levels", tuple(self.levels))
        if self.target not in {"answer", "solution"}:
            raise ValueError("data.target must be answer or solution")
        if self.overflow not in {"error", "truncate_target"}:
            raise ValueError("data.overflow must be error or truncate_target")
        if not isinstance(self.prompt_template, str) or "{problem}" not in self.prompt_template:
            raise ValueError("prompt_template must contain {problem}")


@dataclass(frozen=True)
class SFTTrainConfig:
    epochs: int = 3
    microbatch_size: int = 1
    accumulation_steps: int = 4
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    seed: int = 42

    def __post_init__(self):
        for name in ("epochs", "microbatch_size", "accumulation_steps"):
            _integer(getattr(self, name), f"training.{name}")
        _integer(self.seed, "training.seed", 0)
        for name in ("learning_rate", "weight_decay", "max_grad_norm"):
            x = getattr(self, name)
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
                raise ValueError(f"training.{name} must be finite")
            if x < 0 or (name != "weight_decay" and x == 0):
                raise ValueError(f"Invalid training.{name}")


@dataclass(frozen=True)
class SFTRunConfig:
    name: str
    model: HFModelConfig
    lora: LoRASettings
    data: MathDataConfig
    objective: SFTObjective
    training: SFTTrainConfig
    initialization: str = "adapter"
    schema_version: int = 1

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be nonempty")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if self.initialization not in {"base", "adapter"}:
            raise ValueError("initialization must be base or adapter")
        if self.initialization == "base" and self.objective.strategy != "ce":
            raise ValueError("Paper CAT branches must start from an adapter warmup")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def execution_requirements(self) -> list[str]:
        missing = []
        if not Path(self.model.name).expanduser().is_dir():
            try:
                require_revision(self.model.revision, "model.revision")
            except ValueError as error:
                missing.append(str(error))
        try:
            require_revision(self.data.revision, "data.revision")
        except ValueError as error:
            missing.append(str(error))
        if self.initialization == "adapter":
            missing.append("Provide --initial-adapter from the matching shared warmup")
        return missing


def load_sft_config(path: str | Path) -> SFTRunConfig:
    path = Path(path).expanduser()
    config = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object_pairs)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object")
    cls_by_key = {"model": HFModelConfig, "lora": LoRASettings, "data": MathDataConfig,
                  "objective": SFTObjective, "training": SFTTrainConfig}
    for key, cls in cls_by_key.items():
        if key not in config:
            raise ValueError(f"Missing configuration section: {key}")
        config[key] = _build(cls, config[key], key)
    return _build(SFTRunConfig, config, "configuration")
