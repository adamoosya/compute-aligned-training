"""Full-model ProtGPT2 loading/export, without requiring a chat template."""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import torch
from cat.models.hf import HFModelConfig, environment_versions


def _full_metadata(path: str | Path, config: HFModelConfig) -> dict:
    root = Path(path).expanduser()
    marker = root / "cat_full_model.json"
    if not marker.is_file() or not (root / "config.json").is_file():
        raise ValueError("Expected a completed refactor full-model export")
    value = json.loads(marker.read_text())
    if value.get("schema_version") != 1:
        raise ValueError("Unsupported full-model metadata")
    for key in ("name", "revision"):
        if value.get("base", {}).get(key) != getattr(config, key):
            raise ValueError("Full-model base identity does not match the configuration")
    if not list(root.glob("*.safetensors")):
        raise ValueError("Full-model export is incomplete")
    return value


def load_protein_model(config: HFModelConfig, *, initial_model=None, trainable=True,
                       allow_download=False, cache_dir=None):
    if config.quantization != "none":
        raise ValueError("Protein full-model training uses nonquantized weights")
    device = config.preflight()
    if initial_model is not None:
        _full_metadata(initial_model, config)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ImportError("Install the hf extra first") from error
    source = str(Path(initial_model).expanduser()) if initial_model else config.name
    common = {"local_files_only": True if initial_model else not allow_download,
              "trust_remote_code": False, "cache_dir": str(cache_dir) if cache_dir else None}
    if not initial_model:
        common["revision"] = config.revision
    tokenizer = AutoTokenizer.from_pretrained(source, **common)
    if type(tokenizer.eos_token_id) is not int:
        raise ValueError("Protein tokenizer needs EOS")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Master weights remain fp32 for AdamW/GradScaler. Autocast controls compute.
    # The original ProtGPT2 Hub release is .bin; use restricted tensor-only loading.
    model = AutoModelForCausalLM.from_pretrained(
        source, dtype=torch.float32, attn_implementation=config.attention,
        weights_only=True, use_safetensors=True if initial_model else None, **common)
    model.to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = not trainable
    model.requires_grad_(trainable)
    if trainable and config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train(trainable)
    return model, tokenizer


def save_protein_model(model, tokenizer, destination, *, config: HFModelConfig, provenance: dict):
    root = Path(destination).expanduser()
    metadata = {"schema_version": 1, "base": asdict(config), "provenance": provenance,
                "versions": environment_versions()}
    serialized = json.dumps(metadata, indent=2, allow_nan=False)
    root.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(root, safe_serialization=True)
    tokenizer.save_pretrained(root)
    if not (root / "config.json").is_file() or not list(root.glob("*.safetensors")):
        raise RuntimeError("Full-model export failed; completion marker not written")
    (root / "cat_full_model.json").write_text(serialized + "\n")
    return root
