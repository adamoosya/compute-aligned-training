"""Generation/evaluation run configuration. Parsing performs no network access."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from cat.evaluation.records import parse_json
from cat.evaluation.scoring import MathScoreConfig
from cat.generation.config import SamplingConfig, positive_int
from cat.models.hf import HFModelConfig, require_revision


@dataclass(frozen=True)
class GenerationDataConfig:
    dataset_name: str = "nlile/hendrycks-MATH-benchmark"
    revision: str | None = None
    split: str = "test"
    max_examples: int | None = 500
    levels: tuple[int, ...] | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        for field in ("dataset_name", "split"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"data.{field} must be nonempty")
        if self.revision is not None and not isinstance(self.revision, str):
            raise ValueError("data.revision must be a string or null")
        if self.max_examples is not None:
            positive_int(self.max_examples, "data.max_examples")
        positive_int(self.seed, "data.seed", 0)
        if self.levels is not None:
            if (not isinstance(self.levels, (list, tuple)) or not self.levels
                    or any(type(x) is not int or not 1 <= x <= 5 for x in self.levels)
                    or len(set(self.levels)) != len(self.levels)):
                raise ValueError("data.levels must be unique MATH levels 1--5")
            object.__setattr__(self, "levels", tuple(self.levels))


@dataclass(frozen=True)
class GenerationRunConfig:
    name: str
    initialization: str
    model: HFModelConfig
    data: GenerationDataConfig
    sampling: SamplingConfig
    scoring: MathScoreConfig
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be nonempty")
        if self.initialization not in {"adapter", "base"}:
            raise ValueError("initialization must be adapter or base")
        for field, cls in (("model", HFModelConfig), ("data", GenerationDataConfig),
                           ("sampling", SamplingConfig), ("scoring", MathScoreConfig)):
            if not isinstance(getattr(self, field), cls):
                raise TypeError(f"{field} must be {cls.__name__}")
        if self.model.gradient_checkpointing:
            raise ValueError("Generation requires gradient_checkpointing=false")
        if max(self.scoring.budgets) > self.sampling.num_candidates:
            raise ValueError("Scoring budget exceeds num_candidates")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.data.levels is not None:
            result["data"]["levels"] = list(self.data.levels)
        result["scoring"] = self.scoring.to_dict()
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GenerationRunConfig:
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("generation config has missing or unknown fields")
        converted = dict(value)
        for field, subcls in (("model", HFModelConfig), ("data", GenerationDataConfig)):
            section = value[field]
            if not isinstance(section, Mapping) or set(section) != set(subcls.__dataclass_fields__):
                raise ValueError(f"{field} config has missing or unknown fields")
            converted[field] = subcls(**section)
        converted["sampling"] = SamplingConfig.from_dict(value["sampling"])
        converted["scoring"] = MathScoreConfig.from_dict(value["scoring"])
        return cls(**converted)

    def execution_requirements(self) -> list[str]:
        result = []
        for value, field in ((self.model.revision, "model.revision"), (self.data.revision, "data.revision")):
            if field == "model.revision" and Path(self.model.name).expanduser().is_dir():
                continue
            try:
                require_revision(value, field)
            except ValueError as error:
                result.append(str(error))
        if self.initialization == "adapter":
            result.append("Provide --adapter pointing to a refactor-exported adapter directory")
        return result


def load_generation_config(path: str | Path) -> GenerationRunConfig:
    return GenerationRunConfig.from_dict(parse_json(Path(path).read_text(encoding="utf-8")))
