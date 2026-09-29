"""One explicitly requested MATH SFT stage; no automatic training on import."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from cat.data import (CausalCollator, assert_disjoint, encode_math_selection,
                      load_math_hf)
from cat.models.hf import (environment_versions, load_lora_model, require_revision,
                           save_lora_adapter, _adapter_metadata)
from cat.training import train_sft_epoch
from .sft_config import SFTRunConfig, load_sft_config


def _json_new(path: Path, data):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _code_identity() -> dict:
    """Record commit/dirty state, not file contents or environment credentials."""
    result = {}
    for key, args in (("commit", ["rev-parse", "HEAD"]), ("status", ["status", "--porcelain"])):
        run = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        result[key] = run.stdout.strip() if run.returncode == 0 else None
    return result


def run_sft(config: SFTRunConfig, output: str | Path, *, initial_adapter: str | Path | None,
            allow_download: bool, cache_dir: str | Path | None = None) -> Path:
    """Train using pinned data/model revisions and write a new run directory."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This launcher is single-process; do not run it through torchrun/DDP")
    if config.initialization == "adapter" and initial_adapter is None:
        raise ValueError("This branch needs --initial-adapter from the matching CE warmup")
    if config.initialization == "base" and initial_adapter is not None:
        raise ValueError("A base warmup does not accept --initial-adapter")
    config.model.preflight()
    require_revision(config.data.revision, "data.revision")
    prior = None
    if initial_adapter is not None:
        prior = _adapter_metadata(initial_adapter, config.model, config.lora)
    destination = Path(output).expanduser()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Output path already exists: {destination}")
    # Reject missing optional dependencies before reserving an output folder.
    try:
        import datasets
        import transformers
        import peft
    except ImportError as error:
        raise ImportError("Install the runner dependencies with python -m pip install -e '.[hf,data]'") from error
    if not allow_download:
        # datasets offline mode is process-global: preserve the prior setting.
        offline_before = datasets.config.HF_DATASETS_OFFLINE
        datasets.config.HF_DATASETS_OFFLINE = True
    try:
        data = load_math_hf(
            dataset_name=config.data.dataset_name, revision=config.data.revision,
            split=config.data.split, cache_dir=cache_dir, levels=config.data.levels,
            max_examples=config.data.max_examples, seed=config.data.seed,
        )
    finally:
        if not allow_download:
            datasets.config.HF_DATASETS_OFFLINE = offline_before
    assert_disjoint(data)
    manifest = data.manifest()
    if prior is not None:
        parent = prior.get("provenance", {})
        if parent.get("stage") != "warmup":
            raise ValueError("Branch initialization must be the shared CE warmup, not another branch")
        if (parent.get("selection_sha256") != manifest["selection_sha256"]
                or parent.get("data_config") != json.loads(json.dumps(asdict(config.data)))):
            raise ValueError("Warmup and branch must use identical selected records and target settings")
        if parent.get("reduction") != config.objective.reduction:
            raise ValueError("Warmup and branch must use the same loss reduction")
    random.seed(config.training.seed)
    np.random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)
    if config.model.device.startswith("cuda"):
        torch.cuda.manual_seed_all(config.training.seed)
    model, tokenizer = load_lora_model(
        config.model, config.lora, initial_adapter=initial_adapter,
        allow_download=allow_download, cache_dir=cache_dir,
    )
    encoded = encode_math_selection(
        data, tokenizer, target=config.data.target, max_length=config.data.max_length,
        overflow=config.data.overflow, prompt_template=config.data.prompt_template,
    )
    destination.mkdir(parents=True, exist_ok=False)
    _json_new(destination / "config.json", config.to_dict())
    _json_new(destination / "selection.json", manifest)
    _json_new(destination / "encoding.json", encoded.manifest())
    _json_new(destination / "environment.json", {"packages": environment_versions(),
              "python": sys.version, "git": _code_identity(), "device": config.model.device,
              "base_commit": getattr(model.config, "_commit_hash", None)})
    provenance = {"stage": "warmup" if config.initialization == "base" else "branch",
                  "run_name": config.name, "data_config": asdict(config.data),
                  "selection_sha256": manifest["selection_sha256"],
                  "reduction": config.objective.reduction}
    if initial_adapter is not None:
        parent_file = Path(initial_adapter).expanduser() / "adapter_model.safetensors"
        provenance["initial_adapter_sha256"] = hashlib.sha256(parent_file.read_bytes()).hexdigest()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.training.learning_rate,
                                weight_decay=config.training.weight_decay)
    batch_count = math.ceil(len(encoded) / config.training.microbatch_size)
    total_steps = config.training.epochs * math.ceil(batch_count / config.training.accumulation_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: max(0.0, 1.0 - step / total_steps),
    )
    scaler = torch.amp.GradScaler("cuda") if config.model.precision == "fp16" else None
    generator = torch.Generator().manual_seed(config.training.seed)
    metrics = []
    for epoch in range(config.training.epochs):
        loader = DataLoader(encoded, batch_size=config.training.microbatch_size,
                            shuffle=True, generator=generator, num_workers=0,
                            collate_fn=CausalCollator(tokenizer.pad_token_id))
        stats = train_sft_epoch(
            model, loader, optimizer, config.objective,
            accumulation_steps=config.training.accumulation_steps,
            max_grad_norm=config.training.max_grad_norm, device=config.model.device,
            scheduler=scheduler, precision=config.model.precision, scaler=scaler,
        )
        metrics.append({"epoch": epoch + 1, **asdict(stats)})
        _json_new(destination / f"metrics-epoch-{epoch + 1:03d}.json", metrics[-1])
        save_lora_adapter(model, tokenizer, destination / f"adapter-epoch-{epoch + 1:03d}",
                          model_config=config.model, lora=config.lora,
                          provenance={**provenance, "epoch": epoch + 1})
        print(f"epoch {epoch + 1}/{config.training.epochs} | loss {stats.loss:.6f} | "
              f"optimizer steps {stats.optimizer_steps}; AMP skips {stats.skipped_steps}", flush=True)
    _json_new(destination / "complete.json", {"epochs": metrics, "kind": "refactored_sft_run"})
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Print settings without downloads or training")
    mode.add_argument("--execute", action="store_true", help="Explicitly run the requested training stage")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--initial-adapter", type=Path)
    parser.add_argument("--model-revision")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--device")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        config = load_sft_config(args.config)
        model = config.model
        if args.model_revision is not None:
            model = replace(model, revision=args.model_revision)
        if args.device is not None:
            model = replace(model, device=args.device)
        data = config.data
        if args.dataset_revision is not None:
            data = replace(data, revision=args.dataset_revision)
        config = replace(config, model=model, data=data)
        if args.dry_run:
            print(json.dumps(config.to_dict(), indent=2))
            print("\nDRY RUN: no model/data loaded, no files written, no training started.")
            requirements = config.execution_requirements()
            if requirements:
                print("Before execution:")
                for item in requirements:
                    print(f"  - {item}")
            return 0
        if args.output_dir is None:
            parser.error("--execute requires a new --output-dir")
        path = run_sft(config, args.output_dir, initial_adapter=args.initial_adapter,
                       allow_download=args.allow_download, cache_dir=args.cache_dir)
        print(f"Completed SFT stage: {path}")
        return 0
    except (ValueError, FileExistsError, ImportError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
