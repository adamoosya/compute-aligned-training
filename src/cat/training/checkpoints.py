"""Create-only state-dict snapshots for the local SFT development loop."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


def save_sft_snapshot(path: str | Path, model: nn.Module,
                      metadata: Mapping[str, Any]) -> Path:
    """Write weights and metadata to a new directory, never an existing run.

    This is a weight snapshot, not an optimizer/RNG resume checkpoint and not
    a Hugging Face or PEFT model export. Model construction remains explicit.
    """
    encoded = json.dumps(dict(metadata), indent=2, sort_keys=True, allow_nan=False) + "\n"
    destination = Path(path).expanduser()
    if destination.is_symlink():
        raise FileExistsError(f"Refusing to follow snapshot symlink: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(state, destination / "model_state.pt")
    (destination / "metadata.json").write_text(encoded, encoding="utf-8")
    return destination


def load_sft_snapshot(path: str | Path, model: nn.Module) -> dict[str, Any]:
    """Load a local state dict with restricted weights-only deserialization."""
    source = Path(path).expanduser()
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Snapshot metadata must be a JSON object")
    state = torch.load(source / "model_state.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return metadata
