"""Load a frozen decoder-only base model or an explicitly supplied local LoRA adapter."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.metadata
from pathlib import Path
from typing import Any

import torch

from cat.evaluation.records import parse_json
from .hf import HFModelConfig, LoRASettings, _adapter_metadata, _dependencies


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def adapter_identity(adapter: str | Path, config: HFModelConfig) -> dict[str, Any]:
    path = Path(adapter).expanduser()
    marker = path / "cat_adapter.json"
    if not marker.is_file():
        raise ValueError("Adapter lacks cat_adapter.json; use an adapter exported by this refactor")
    metadata = parse_json(marker.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or not isinstance(metadata.get("lora"), dict):
        raise ValueError("Invalid adapter metadata")
    try:
        lora = LoRASettings(**metadata["lora"])
    except TypeError as error:
        raise ValueError("Invalid adapter LoRA settings") from error
    _adapter_metadata(path, config, lora)
    actual = parse_json((path / "adapter_config.json").read_text(encoding="utf-8"))
    expected = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": lora.r,
                "lora_alpha": lora.alpha, "lora_dropout": lora.dropout}
    if not isinstance(actual, dict) or any(actual.get(k) != v for k, v in expected.items()):
        raise ValueError("PEFT adapter configuration disagrees with cat_adapter.json")
    if sorted(actual.get("target_modules", [])) != sorted(lora.target_modules):
        raise ValueError("PEFT target_modules disagree with cat_adapter.json")
    return {"kind": "cat_lora_adapter", "metadata": metadata,
            "sha256": {name: sha256_file(path / name) for name in
                       ("cat_adapter.json", "adapter_config.json", "adapter_model.safetensors")}}


def load_inference_model(
    config: HFModelConfig, *, adapter: str | Path | None = None,
    allow_download: bool = False, cache_dir: str | Path | None = None,
) -> tuple[Any, Any]:
    """Never create a fresh adapter in place of the user's trained weights."""
    if not isinstance(config, HFModelConfig):
        raise TypeError("config must be HFModelConfig")
    if type(allow_download) is not bool:
        raise TypeError("allow_download must be bool")
    if config.gradient_checkpointing:
        raise ValueError("Set gradient_checkpointing=false for inference")
    device = config.preflight()
    if adapter is not None:
        adapter_identity(adapter, config)
    transformers, peft = _dependencies()
    local = Path(config.name).expanduser()
    name = str(local) if local.is_dir() else config.name
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[config.precision]
    common = dict(revision=config.revision, local_files_only=not allow_download,
                  trust_remote_code=False, cache_dir=str(cache_dir) if cache_dir else None)
    token_source = str(Path(adapter).expanduser()) if adapter is not None else name
    token_kwargs = dict(local_files_only=True, trust_remote_code=False) if adapter is not None else common
    tokenizer = transformers.AutoTokenizer.from_pretrained(token_source, **token_kwargs)
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Tokenizer has no chat template")
    if type(getattr(tokenizer, "eos_token_id", None)) is not int:
        raise ValueError("Tokenizer requires one EOS token ID")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    kwargs = dict(common, dtype=dtype, attn_implementation=config.attention, use_safetensors=True)
    if config.quantization == "nf4":
        try:
            importlib.metadata.version("bitsandbytes")
        except importlib.metadata.PackageNotFoundError as error:
            raise ImportError("NF4 inference needs the CUDA extra: python -m pip install -e '.[hf,cuda]'") from error
        kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        kwargs["device_map"] = {"": device.index}
    model = transformers.AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if getattr(model.config, "is_encoder_decoder", False):
        raise ValueError("Only decoder-only models are supported")
    if config.quantization == "none":
        model.to(device)
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        raise ValueError("Tokenizer vocabulary exceeds the model embeddings")
    if adapter is not None:
        model = peft.PeftModel.from_pretrained(
            model, str(Path(adapter).expanduser()), is_trainable=False, local_files_only=True,
        )
    model.requires_grad_(False)
    model.config.use_cache = True
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model, tokenizer
