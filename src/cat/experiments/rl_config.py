"""Named paper-based RL configurations. Parsing never loads data or models."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
from pathlib import Path
from cat.evaluation.records import parse_json
from cat.generation.config import SamplingConfig, positive_int
from cat.models.hf import HFModelConfig, LoRASettings, require_revision
from cat.training.rl import CATWeightConfig, RLUpdateConfig


@dataclass(frozen=True)
class RLRunConfig:
    name: str
    task: str
    stage: str
    model: HFModelConfig
    lora: LoRASettings | None
    sampling: SamplingConfig
    weights: CATWeightConfig
    update: RLUpdateConfig
    data: dict
    training: dict
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be nonempty")
        if self.task not in {"math", "protein_unconditional", "protein_conditional"}:
            raise ValueError("Unknown experiment task")
        if self.stage not in {"rl", "warmup"}:
            raise ValueError("stage must be rl or warmup")
        if self.stage == "warmup" and self.task != "protein_conditional":
            raise ValueError("This warmup stage is for conditional protein; use train_math_sft.py for MATH")
        if self.task == "math" and self.lora is None:
            raise ValueError("MATH uses an explicitly specified LoRA configuration")
        if self.task != "math" and (self.lora is not None or self.model.quantization != "none"):
            raise ValueError("Protein experiments use full-model nonquantized training")
        if self.task == "math" and self.weights.strategy == "bon":
            raise ValueError("Use binary Pass@N/Majority weights for MATH presets")
        if self.task != "math" and self.weights.strategy not in {"standard", "bon"}:
            raise ValueError("Protein presets use standard or BoN weights")
        if self.task == "protein_conditional" and self.weights.quantile != "within_prompt":
            raise ValueError("Conditional quantiles must be within-prompt")
        if self.update.precision != self.model.precision:
            raise ValueError("Model and update precision must agree")
        if self.sampling.num_candidates < 2:
            raise ValueError("RL needs at least two candidates per prompt")
        required_training = {"epochs", "max_steps", "prompts_per_step", "learning_rate", "weight_decay", "seed", "warmup_steps"}
        if set(self.training) != required_training:
            raise ValueError("Missing or unknown training fields")
        for key in ("epochs", "prompts_per_step", "warmup_steps"):
            positive_int(self.training[key], key)
        positive_int(self.training["seed"], "seed", 0)
        if self.training["max_steps"] is not None:
            positive_int(self.training["max_steps"], "max_steps")
        for key in ("learning_rate", "weight_decay"):
            v = self.training[key]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError(f"Invalid {key}")
        if self.training["learning_rate"] == 0:
            raise ValueError("learning_rate must be positive")
        required_data = {"dataset_name", "revision", "split", "count", "offset", "levels", "seed"}
        if set(self.data) != required_data:
            raise ValueError("Missing or unknown data fields")
        positive_int(self.data["count"], "data.count")
        positive_int(self.data["offset"], "data.offset", 0)
        positive_int(self.data["seed"], "data.seed", 0)
        if self.data["split"] != "train":
            raise ValueError("Training presets require the explicit train split")
        levels = self.data["levels"]
        if levels is not None and (not isinstance(levels, (tuple, list)) or not levels or
                                  any(type(v) is not int or v not in range(1, 6) for v in levels) or len(set(levels)) != len(levels)):
            raise ValueError("Invalid MATH levels")

    def to_dict(self):
        return asdict(self)

    def requirements(self):
        missing = []
        if not Path(self.model.name).expanduser().is_dir():
            try:
                require_revision(self.model.revision, "model.revision")
            except ValueError as e:
                missing.append(str(e))
        if self.task == "math":
            try:
                require_revision(self.data["revision"], "data.revision")
            except ValueError as e:
                missing.append(str(e))
            missing.append("Provide --initial-adapter from a MATH RL warmup")
        elif self.task == "protein_conditional" and self.stage == "rl":
            missing.append("Provide --initial-model from the shared protein warmup")
        return missing


def load_rl_config(path) -> RLRunConfig:
    value = parse_json(Path(path).read_text())
    if not isinstance(value, dict) or set(value) != set(RLRunConfig.__dataclass_fields__):
        raise ValueError("Missing or unknown RL run fields")
    try:
        for k, cls in (("model", HFModelConfig), ("sampling", SamplingConfig),
                       ("weights", CATWeightConfig), ("update", RLUpdateConfig)):
            value[k] = cls(**value[k])
        value["lora"] = LoRASettings(**value["lora"]) if value["lora"] is not None else None
        return RLRunConfig(**value)
    except (TypeError, KeyError) as error:
        raise ValueError(f"Invalid RL configuration: {error}") from error
