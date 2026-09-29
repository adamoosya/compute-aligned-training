"""Generate MATH candidate pools and score them. Actual execution is opt-in."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from cat.data import assert_disjoint, load_math_hf
from cat.data.math import MathSelection
from cat.evaluation.records import CandidateRecord, parse_json
from cat.evaluation.scoring import MathScoreConfig, score_candidate_file, validate_scoring_inputs
from cat.generation import SamplingConfig, encode_prompt, sample_example
from cat.generation.sampling import _device_and_context
from cat.models.hf import environment_versions, require_revision
from cat.models.inference import adapter_identity, load_inference_model, sha256_file
from .generation_config import GenerationRunConfig, load_generation_config


def _new_json(path: Path, data: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _new_destination(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Output already exists: {path}")
    if any(parent.is_symlink() for parent in path.parents):
        raise ValueError("Output must not traverse symbolic links")
    return path


def check_training_overlap(selection: MathSelection, path: str | Path | None) -> dict:
    if path is None:
        return {"status": "not_checked", "reason": "No training-selection manifest supplied"}
    path = Path(path).expanduser()
    raw = path.read_bytes()
    prior = parse_json(raw.decode("utf-8"))
    if not isinstance(prior, dict) or not isinstance(prior.get("selected_records"), list):
        raise ValueError("Invalid training-selection manifest")
    rows = prior["selected_records"]
    expected = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if not rows or prior.get("selection_sha256") != expected:
        raise ValueError("Training-selection checksum mismatch")
    hashes = []
    for row in rows:
        value = row.get("problem_sha256") if isinstance(row, dict) else None
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("Training-selection manifest has invalid problem hashes")
        hashes.append(value)
    intersection = {x.problem_hash for x in selection.examples}.intersection(hashes)
    if intersection:
        raise ValueError(f"Evaluation overlaps the supplied training selection on {len(intersection)} problem(s)")
    return {"status": "disjoint_from_supplied_selection", "training_manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "training_selection_sha256": expected, "training_count": len(rows)}


def generate_and_score(
    model: Any, tokenizer: Any, selection: MathSelection,
    sampling: SamplingConfig, scoring: MathScoreConfig, output: str | Path,
    *, provenance: dict[str, Any], training_selection: str | Path | None = None,
    smoke_only: bool = False,
) -> Path:
    """Stream generated text/tokens to new files, then use the existing scorer.

    A failed generation leaves partial files but never a completion marker. There
    are no automatic retries, skipped prompts, continuation or overwrite modes.
    """
    if not isinstance(selection, MathSelection):
        raise TypeError("selection must be MathSelection")
    if not isinstance(sampling, SamplingConfig) or not isinstance(scoring, MathScoreConfig):
        raise TypeError("Expected SamplingConfig and MathScoreConfig")
    if type(smoke_only) is not bool or not isinstance(provenance, dict):
        raise TypeError("smoke_only must be bool and provenance must be a dictionary")
    destination = _new_destination(output)
    assert_disjoint(selection)
    if max(scoring.budgets) > sampling.num_candidates:
        raise ValueError("Scoring budget exceeds num_candidates")
    # Check references and all prompt lengths before any generation. Temporary
    # blank candidates here are schema validation only, never written as output.
    references = [CandidateRecord.from_example(x, [""] * max(scoring.budgets)) for x in selection.examples]
    validate_scoring_inputs(references, scoring)
    overlap = check_training_overlap(selection, training_selection)
    prompts = [encode_prompt(tokenizer, x.problem, sampling) for x in selection.examples]
    for ids in prompts:
        _device_and_context(model, ids, sampling)
    tokenizer_info = {
        "class": type(tokenizer).__name__, "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "chat_template_sha256": hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
    }
    info = {"schema_version": 1, "kind": "smoke_only" if smoke_only else "refactored_generation",
            "provenance": provenance, "packages": environment_versions(),
            "sampling": sampling.to_dict(), "scoring": scoring.to_dict(),
            "tokenizer": tokenizer_info, "overlap_check": overlap,
            "reproducibility": "Seeded per problem/chunk; chunk size, model and software/hardware must stay fixed."}
    json.dumps(info, allow_nan=False)  # Reject unserializable metadata before creating output.
    destination.mkdir(parents=True, exist_ok=False)
    _new_json(destination / "run.json", info)
    _new_json(destination / "selection.json", selection.manifest())
    with (destination / "candidates.jsonl").open("x", encoding="utf-8") as text_file, \
         (destination / "tokens.jsonl").open("x", encoding="utf-8") as token_file:
        for index, (example, ids) in enumerate(zip(selection.examples, prompts), 1):
            record, details = sample_example(model, tokenizer, example, sampling, prompt_ids=ids)
            text_file.write(json.dumps(record.to_dict(), ensure_ascii=False, allow_nan=False) + "\n")
            token_file.write(json.dumps(details, allow_nan=False) + "\n")
            text_file.flush()
            token_file.flush()
            print(f"generated {index}/{len(selection.examples)} | {sampling.num_candidates} candidates", flush=True)
    manifest = {"schema_version": 1, "num_problems": len(selection.examples),
                "num_candidates_per_problem": sampling.num_candidates,
                "sha256": {name: sha256_file(destination / name) for name in
                           ("run.json", "selection.json", "candidates.jsonl", "tokens.jsonl")}}
    _new_json(destination / "generation_complete.json", manifest)
    score_candidate_file(destination / "candidates.jsonl", destination / "scores", scoring)
    _new_json(destination / "complete.json", {
        "schema_version": 1, "kind": info["kind"],
        "sha256": {name: sha256_file(destination / name) for name in
                   ("generation_complete.json", "scores/complete.json")},
    })
    return destination


def _load_selection(config: GenerationRunConfig, *, allow_download: bool, cache_dir):
    try:
        import datasets
    except ImportError as error:
        raise ImportError("Loading MATH needs python -m pip install -e '.[data]'") from error
    before = datasets.config.HF_DATASETS_OFFLINE
    try:
        if not allow_download:
            datasets.config.HF_DATASETS_OFFLINE = True
        return load_math_hf(
            dataset_name=config.data.dataset_name, revision=config.data.revision,
            split=config.data.split, max_examples=config.data.max_examples,
            levels=config.data.levels, seed=config.data.seed, cache_dir=cache_dir,
        )
    finally:
        datasets.config.HF_DATASETS_OFFLINE = before


def run_generation(
    config: GenerationRunConfig, output: str | Path, *, adapter: str | Path | None = None,
    allow_download: bool = False, cache_dir: str | Path | None = None,
    training_selection: str | Path | None = None,
) -> Path:
    if not isinstance(config, GenerationRunConfig) or type(allow_download) is not bool:
        raise TypeError("Expected GenerationRunConfig and boolean allow_download")
    if os.environ.get("WORLD_SIZE", "1") != "1":
        raise ValueError("This launcher is single-process; do not use torchrun/DDP")
    _new_destination(output)
    if config.initialization == "adapter" and adapter is None:
        raise ValueError("Provide --adapter; no checkpoint is chosen automatically")
    if config.initialization == "base" and adapter is not None:
        raise ValueError("Base-model generation does not accept --adapter")
    config.model.preflight()
    require_revision(config.data.revision, "data.revision")
    identity = adapter_identity(adapter, config.model) if adapter is not None else {"kind": "base_model"}
    selection = _load_selection(config, allow_download=allow_download, cache_dir=cache_dir)
    overlap = check_training_overlap(selection, training_selection)
    trained_selection = identity.get("metadata", {}).get("provenance", {}).get("selection_sha256")
    if trained_selection and training_selection is not None and overlap["training_selection_sha256"] != trained_selection:
        raise ValueError("Supplied training selection does not match the adapter's recorded selection")
    model, tokenizer = load_inference_model(
        config.model, adapter=adapter, allow_download=allow_download, cache_dir=cache_dir,
    )
    return generate_and_score(
        model, tokenizer, selection, config.sampling, config.scoring, output,
        provenance={"config": config.to_dict(), "adapter": identity,
                    "loaded_base_commit": getattr(model.config, "_commit_hash", None),
                    "local_base_weights_hashed": False},
        training_selection=training_selection,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--training-selection", type=Path)
    parser.add_argument("--model-revision")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        config = load_generation_config(args.config)
        if args.model_revision is not None:
            config = replace(config, model=replace(config.model, revision=args.model_revision))
        if args.dataset_revision is not None:
            config = replace(config, data=replace(config.data, revision=args.dataset_revision))
        if args.dry_run:
            print(json.dumps(config.to_dict(), indent=2))
            print("\nDRY RUN: no model/data loaded, no files written, no generation started.")
            for item in config.execution_requirements():
                print(f"  - {item}")
            return 0
        if args.output_dir is None:
            parser.error("--execute requires a new --output-dir")
        result = run_generation(config, args.output_dir, adapter=args.adapter,
                                allow_download=args.allow_download, cache_dir=args.cache_dir,
                                training_selection=args.training_selection)
        print(f"Generation and scoring complete: {result}")
        return 0
    except (OSError, ValueError, TypeError, ImportError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
