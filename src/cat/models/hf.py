"""Single-device LoRA loading and create-only adapter exports.

The default is local-files-only. Loading a remote model requires an explicit
allow_download=True call and a pinned model revision. No Hub uploads are made.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
from typing import Any

import torch


def _integer(value: Any, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def require_revision(value: str | None, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{40}", value) is None:
        raise ValueError(f"{name} requires a 40-character Hub commit SHA, not 'main'")


@dataclass(frozen=True)
class HFModelConfig:
    name: str
    revision: str | None = None
    device: str = "cpu"
    precision: str = "fp32"
    quantization: str = "none"
    gradient_checkpointing: bool = False
    attention: str = "eager"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("model.name must be a nonempty string")
        if self.revision is not None and not isinstance(self.revision, str):
            raise ValueError("model.revision must be a string or null")
        if not isinstance(self.device, str) or not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", self.device):
            raise ValueError("This loader supports cpu or one CUDA device (e.g. cuda:0)")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")
        if self.device == "cpu" and self.precision != "fp32":
            raise ValueError("CPU training in this backend uses fp32")
        if self.quantization not in {"none", "nf4"}:
            raise ValueError("quantization must be none or nf4")
        if self.quantization == "nf4" and (self.device == "cpu" or self.precision == "fp32"):
            raise ValueError("The NF4 path in this backend requires CUDA and fp16 or bf16")
        if not isinstance(self.gradient_checkpointing, bool):
            raise ValueError("gradient_checkpointing must be a bool")
        if self.attention not in {"eager", "sdpa"}:
            raise ValueError("attention must be eager or sdpa")

    def preflight(self) -> torch.device:
        """Check actual hardware and revision before any model download."""
        local = Path(self.name).expanduser()
        if local.is_dir():
            if not (local / "config.json").is_file():
                raise ValueError("Local base model has no config.json")
        else:
            require_revision(self.revision, "model.revision")
        device = torch.device(self.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("This configuration needs CUDA; do not run it on the Mac")
            index = device.index if device.index is not None else torch.cuda.current_device()
            if index >= torch.cuda.device_count():
                raise ValueError("Requested CUDA device does not exist")
            if self.precision == "bf16":
                with torch.cuda.device(index):
                    if not torch.cuda.is_bf16_supported():
                        raise RuntimeError("This CUDA device does not support bf16; choose fp16")
            device = torch.device("cuda", index)
        return device


@dataclass(frozen=True)
class LoRASettings:
    r: int = 16
    alpha: int = 16
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    def __post_init__(self) -> None:
        _integer(self.r, "lora.r")
        _integer(self.alpha, "lora.alpha")
        if (isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float))
                or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1):
            raise ValueError("lora.dropout must be finite in [0,1)")
        if (not isinstance(self.target_modules, (tuple, list)) or not self.target_modules
                or any(not isinstance(x, str) or not x.strip() for x in self.target_modules)
                or len(set(self.target_modules)) != len(self.target_modules)):
            raise ValueError("lora.target_modules must be unique nonempty module names")
        object.__setattr__(self, "target_modules", tuple(self.target_modules))


def _dependencies():
    try:
        import transformers
        import peft
    except ImportError as error:
        raise ImportError("Install the backend: python -m pip install -e '.[hf]'") from error
    return transformers, peft


def _adapter_metadata(path: str | Path, config: HFModelConfig, lora: LoRASettings) -> dict:
    path = Path(path).expanduser()
    if not path.is_dir():
        raise ValueError("initial_adapter must be an existing local adapter directory")
    metadata_path = path / "cat_adapter.json"
    if not metadata_path.is_file():
        raise ValueError("Adapter lacks cat_adapter.json; use a warmup exported by this refactor")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ValueError("Unsupported adapter metadata schema")
    for key in ("name", "revision"):
        if metadata.get("model", {}).get(key) != getattr(config, key):
            raise ValueError(f"Adapter base-model {key} does not match this configuration")
    expected = asdict(lora)
    expected["target_modules"] = sorted(expected["target_modules"])
    actual = dict(metadata.get("lora", {}))
    actual["target_modules"] = sorted(actual.get("target_modules", []))
    if actual != expected:
        raise ValueError("Adapter LoRA settings do not match this configuration")
    # The completion marker is written last; require the exported adapter files too.
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (path / name).is_file():
            raise ValueError(f"Incomplete adapter export: missing {name}")
    return metadata


def load_lora_model(
    config: HFModelConfig, lora: LoRASettings, *,
    initial_adapter: str | Path | None = None, allow_download: bool = False,
    cache_dir: str | Path | None = None,
) -> tuple[Any, Any]:
    """Return a trainable PEFT causal model and its tokenizer.

    Only adapter parameters are trainable. An initial adapter starts a new
    optimization stage, not a resume of optimizer/scheduler state.
    """
    if not isinstance(allow_download, bool):
        raise TypeError("allow_download must be bool")
    device = config.preflight()
    if initial_adapter is not None:
        _adapter_metadata(initial_adapter, config, lora)
    transformers, peft = _dependencies()
    local_name = Path(config.name).expanduser()
    name = str(local_name) if local_name.is_dir() else config.name
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[config.precision]
    common = dict(revision=config.revision, local_files_only=not allow_download,
                  trust_remote_code=False, cache_dir=str(cache_dir) if cache_dir else None)
    tokenizer_source = str(Path(initial_adapter).expanduser()) if initial_adapter is not None else name
    tokenizer_kwargs = common if initial_adapter is None else dict(local_files_only=True, trust_remote_code=False)
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Tokenizer has no chat template; do not guess the supervision boundary")
    if getattr(tokenizer, "eos_token_id", None) is None:
        raise ValueError("Tokenizer must define an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    kwargs = dict(common, torch_dtype=dtype, attn_implementation=config.attention, use_safetensors=True)
    if config.quantization == "nf4":
        try:
            importlib.metadata.version("bitsandbytes")
        except importlib.metadata.PackageNotFoundError as error:
            raise ImportError("NF4 needs the CUDA extra: python -m pip install -e '.[hf,cuda]'") from error
        kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
        )
        # This is a training loader, not automatic inference offloading.
        kwargs["device_map"] = {"": device.index}
    model = transformers.AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if config.quantization == "none":
        model.to(device)
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        raise ValueError("Tokenizer vocabulary exceeds the base-model embeddings")
    if config.quantization == "nf4":
        model = peft.prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    elif config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if initial_adapter is None:
        module_names = {n.rsplit(".", 1)[-1] for n, _ in model.named_modules()}
        missing = set(lora.target_modules) - module_names
        if missing:
            raise ValueError(f"LoRA target modules absent from base model: {sorted(missing)}")
        peft_config = peft.LoraConfig(
            r=lora.r, lora_alpha=lora.alpha, lora_dropout=lora.dropout,
            target_modules=list(lora.target_modules), bias="none", task_type="CAUSAL_LM",
        )
        model = peft.get_peft_model(model, peft_config)
    else:
        model = peft.PeftModel.from_pretrained(
            model, str(Path(initial_adapter).expanduser()), is_trainable=True,
            local_files_only=True,
        )
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in n for n, _ in trainable):
        raise RuntimeError("Expected trainable LoRA parameters and a frozen base model")
    # AdamW and GradScaler use fp32 adapter parameters even over a half-precision base.
    for _, parameter in trainable:
        if parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()
    model.train()
    return model, tokenizer


def environment_versions() -> dict[str, str]:
    versions = {}
    for name in ("torch", "numpy", "transformers", "peft", "accelerate", "datasets", "bitsandbytes"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def save_lora_adapter(model: Any, tokenizer: Any, destination: str | Path, *,
                      model_config: HFModelConfig, lora: LoRASettings,
                      provenance: dict[str, Any] | None = None) -> Path:
    """Export adapter and tokenizer to a new folder; never overwrite a checkpoint.

    The frozen base, optimizer, and RNG state are not included. A complete export
    has cat_adapter.json, written only after PEFT and tokenizer export succeed.
    """
    if not hasattr(model, "peft_config"):
        raise TypeError("Expected a PEFT model, not a full-model snapshot")
    metadata = {"schema_version": 1, "model": asdict(model_config), "lora": asdict(lora),
                "versions": environment_versions(), "provenance": provenance or {}}
    serialized = json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path = Path(destination).expanduser()
    path.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(str(path), safe_serialization=True, save_embedding_layers=False)
    tokenizer.save_pretrained(str(path))
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (path / name).is_file():
            raise RuntimeError(f"PEFT export did not create {name}; export is incomplete")
    (path / "cat_adapter.json").write_text(serialized, encoding="utf-8")
    return path
