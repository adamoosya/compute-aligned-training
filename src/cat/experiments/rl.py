"""Run one explicit MATH RL or protein stage. Never train at import time."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import random
import sys
import torch
from cat.data import load_math_hf, load_math_jsonl, assert_disjoint, CausalCollator
from cat.data.tokenization import EncodedExample
from cat.data.protein import make_protein_examples, protein_manifest
from cat.models.hf import (load_lora_model, save_lora_adapter, require_revision,
                           environment_versions, _adapter_metadata)
from cat.models.inference import load_inference_model, adapter_identity
from cat.models.protein import load_protein_model, save_protein_model, _full_metadata
from cat.training import train_sft_epoch, SFTObjective
from cat.training.sft import _precision_context
from cat.training.rl import update_rl, RewardHistory, disable_dropout
from cat.training.rollouts import RLPrompt, collect_rollouts
from cat.evaluation.protein import summarize_protein_rewards
from .rl_config import RLRunConfig, load_rl_config
from .sft import _code_identity


def _write(path: Path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def _new_output(path):
    root = Path(path).expanduser()
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"Output already exists: {root}")
    return root


def _math_selection(config, *, allow_download, cache_dir, local_data=None):
    d = config.data
    kwargs = dict(split=d["split"], levels=d["levels"],
                  max_examples=d["offset"] + d["count"], seed=d["seed"])
    if local_data:
        selection = load_math_jsonl(local_data, **kwargs)
    else:
        require_revision(d["revision"], "data.revision")
        try:
            import datasets
        except ImportError as error:
            raise ImportError("Real MATH loading needs python -m pip install -e '.[hf,data]'") from error
        previous = datasets.config.HF_DATASETS_OFFLINE
        try:
            if not allow_download:
                datasets.config.HF_DATASETS_OFFLINE = True
            selection = load_math_hf(dataset_name=d["dataset_name"], revision=d["revision"],
                                      cache_dir=cache_dir, **kwargs)
        finally:
            datasets.config.HF_DATASETS_OFFLINE = previous
    assert_disjoint(selection)
    examples = selection.examples[d["offset"]:]
    manifest = selection.manifest()
    manifest["rl_offset"] = d["offset"]
    manifest["rl_ids"] = [ex.example_id for ex in examples]
    return examples, manifest


def _protein_prompts(examples, tokenizer):
    return [RLPrompt(ex.example_id, ex.prompt, input_h=ex.input_h,
                     token_ids=tuple(tokenizer.encode(ex.prompt, add_special_tokens=False))) for ex in examples]


def _train_loop(model, tokenizer, prompts, config: RLRunConfig, root: Path, *, reference_model=None, collector=collect_rollouts):
    """Collect all prompts in an optimizer window before any parameter update."""
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.training["learning_rate"],
                                  weight_decay=config.training["weight_decay"])
    scaler = torch.amp.GradScaler("cuda") if config.update.precision == "fp16" else None
    history = RewardHistory(4000) if config.task == "protein_unconditional" else None
    disable_dropout(model)
    if reference_model is not None:
        disable_dropout(reference_model)
    rng = random.Random(config.training["seed"])
    b = config.training["prompts_per_step"]
    cap = config.training["max_steps"]
    # Fixed-step protein stages cycle the finite prompt list. MATH is epoch-based.
    epochs = config.training["epochs"] if cap is None else max(config.training["epochs"], math.ceil(cap / math.ceil(len(prompts) / b)))
    metrics = []
    with (root / "metrics.jsonl").open("x") as mf, (root / "rollouts.jsonl").open("x") as rf:
        for epoch in range(epochs):
            order = list(range(len(prompts)))
            rng.shuffle(order)
            for begin in range(0, len(order), b):
                window = [prompts[i] for i in order[begin:begin + b]]
                with _precision_context(next(model.parameters()).device, config.update.precision):
                    batch, records = collector(model, tokenizer, window, config.sampling,
                                               step=len(metrics), task=config.task)
                stats = update_rl(model, optimizer, batch, config.weights, config.update,
                                   reference_model=reference_model, history=history, scaler=scaler)
                stats.update(step=len(metrics) + 1, epoch=epoch + 1)
                mf.write(json.dumps(stats, allow_nan=False) + "\n"); mf.flush()
                for r in records:
                    r.update(step=stats["step"])
                    rf.write(json.dumps(r, allow_nan=False) + "\n")
                rf.flush()
                metrics.append(stats)
                print(f"step {stats['step']} | reward {stats['reward_mean']:.4f} | "
                      f"loss {stats['loss']:.5f} | weight {stats['weights']['mean']:.4f}", flush=True)
                if cap is not None and len(metrics) >= cap:
                    break
            if cap is not None and len(metrics) >= cap:
                break
    if history is not None:
        _write(root / "reward_history.json", {"capacity": 4000, "rewards": list(history.values)})
    return metrics


def _protein_warmup(model, tokenizer, config, root):
    examples = make_protein_examples(config.data["count"], seed=config.data["seed"], split="warmup")
    _write(root / "selection.json", protein_manifest(examples, seed=config.data["seed"]))
    encoded = []
    for ex in examples:
        prefix = tuple(tokenizer.encode(ex.prompt, add_special_tokens=False))
        full = tuple(tokenizer.encode(ex.prompt + " " + ex.target, add_special_tokens=False))
        if not prefix or full[:len(prefix)] != prefix:
            raise ValueError("Protein tokenizer changes the prompt prefix at the target boundary")
        full = full + (tokenizer.eos_token_id,)
        item = EncodedExample(ex.example_id, full, (-100,) * len(prefix) + full[len(prefix):], len(prefix), len(full))
        item.validate()
        encoded.append(item)
    # Paper's 25 sequences/microbatch, accumulation 4, 60 optimizer steps.
    count = config.sampling.num_candidates
    accumulation = config.training["prompts_per_step"]
    steps = config.training["warmup_steps"]
    rng = random.Random(config.training["seed"])
    def batches():
        order = list(range(len(encoded)))
        rng.shuffle(order)
        cursor = 0
        for _ in range(steps * accumulation):
            selected = []
            for _ in range(count):
                if cursor == len(order):
                    rng.shuffle(order); cursor = 0
                selected.append(encoded[order[cursor]])
                cursor += 1
            yield CausalCollator(tokenizer.pad_token_id)(selected)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config.training["learning_rate"], weight_decay=config.training["weight_decay"])
    scaler = torch.amp.GradScaler("cuda") if config.update.precision == "fp16" else None
    stats = train_sft_epoch(model, batches(), optimizer, SFTObjective(reduction="token_mean"),
                            accumulation_steps=accumulation, max_grad_norm=config.update.max_grad_norm,
                            precision=config.update.precision, scaler=scaler)
    _write(root / "warmup_metrics.json", asdict(stats))
    return asdict(stats)


def run_rl(config: RLRunConfig, output, *, initial_adapter=None, initial_model=None,
           allow_download=False, cache_dir=None, local_data=None):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This runner is single-process; do not use torchrun/DDP")
    root = _new_output(output)
    config.model.preflight()
    random.seed(config.training["seed"])
    torch.manual_seed(config.training["seed"])
    reference = None
    initial_identity = None
    if config.task == "math":
        if initial_adapter is None or initial_model is not None:
            raise ValueError("MATH RL requires --initial-adapter (not --initial-model)")
        initial_identity = adapter_identity(initial_adapter, config.model)
        prior = _adapter_metadata(initial_adapter, config.model, config.lora)
        if prior.get("provenance", {}).get("stage") != "warmup":
            raise ValueError("Each MATH RL variant must start at the same shared CE warmup")
        examples, manifest = _math_selection(config, allow_download=allow_download,
                                               cache_dir=cache_dir, local_data=local_data)
        model, tokenizer = load_lora_model(config.model, config.lora, initial_adapter=initial_adapter,
                                           allow_download=allow_download, cache_dir=cache_dir)
        if config.update.beta:
            reference, _ = load_inference_model(replace(config.model, gradient_checkpointing=False),
                                                 adapter=initial_adapter, allow_download=allow_download,
                                                 cache_dir=cache_dir)
        prompts = [RLPrompt(ex.example_id, ex.problem, answer=ex.answer) for ex in examples]
    else:
        if initial_adapter is not None or local_data is not None:
            raise ValueError("Protein pipelines do not take MATH data or LoRA adapters")
        if config.task == "protein_conditional" and config.stage == "rl":
            if initial_model is None:
                raise ValueError("Conditional RL needs --initial-model from shared warmup")
            initial_identity = _full_metadata(initial_model, config.model)
            if initial_identity.get("provenance", {}).get("stage") != "warmup":
                raise ValueError("Conditional RL branches must start from the shared warmup")
        elif initial_model is not None:
            raise ValueError("This stage starts from the pretrained base, not --initial-model")
        model, tokenizer = load_protein_model(config.model, initial_model=initial_model,
                                               allow_download=allow_download, cache_dir=cache_dir)
        if config.stage == "rl" and config.update.beta:
            reference, _ = load_protein_model(replace(config.model, gradient_checkpointing=False),
                                               initial_model=initial_model, trainable=False,
                                               allow_download=allow_download, cache_dir=cache_dir)
        if config.task == "protein_conditional":
            examples = make_protein_examples(config.data["count"], seed=config.data["seed"], split="train")
            prompts = _protein_prompts(examples, tokenizer)
            manifest = protein_manifest(examples, seed=config.data["seed"])
        else:
            bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
            prompts = [RLPrompt(f"unconditional:{i}", "[unconditional]", token_ids=(bos,))
                       for i in range(config.training["prompts_per_step"])]
            manifest = {"task": config.task, "prompt_token_ids": [bos]}
    root.mkdir(parents=True, exist_ok=False)
    _write(root / "config.json", config.to_dict())
    _write(root / "environment.json", {"packages": environment_versions(), "python": sys.version,
                                        "git": _code_identity(), "initialization": initial_identity})
    if config.stage == "warmup":
        metrics = _protein_warmup(model, tokenizer, config, root)
    else:
        _write(root / "selection.json", manifest)
        metrics = _train_loop(model, tokenizer, prompts, config, root, reference_model=reference)
    provenance = {"stage": config.stage, "run_name": config.name, "task": config.task,
                  "kind": "paper_based_refactor", "config": config.to_dict()}
    if config.task == "math":
        save_lora_adapter(model, tokenizer, root / "adapter-final", model_config=config.model,
                           lora=config.lora, provenance=provenance)
    else:
        save_protein_model(model, tokenizer, root / "model-final", config=config.model, provenance=provenance)
    _write(root / "complete.json", {"kind": "refactored_training_run", "metrics": metrics,
                                   "stage": config.stage, "not_historical_reproduction": True})
    return root


def evaluate_protein(config: RLRunConfig, checkpoint, output, *, candidates=None, num_prompts=None,
                     allow_download=False, cache_dir=None):
    if config.task == "math":
        raise ValueError("Use generate_math.py for MATH evaluation")
    root = _new_output(output)
    metadata = _full_metadata(checkpoint, config.model)
    model, tokenizer = load_protein_model(replace(config.model, gradient_checkpointing=False),
                                           initial_model=checkpoint, trainable=False,
                                           allow_download=allow_download, cache_dir=cache_dir)
    conditional = config.task == "protein_conditional"
    count = num_prompts if num_prompts is not None else (150 if conditional else 1)
    pool = candidates if candidates is not None else (32 if conditional else 512)
    if type(count) is not int or count < 1 or type(pool) is not int or pool < 2:
        raise ValueError("Invalid evaluation prompt/candidate count")
    if conditional:
        examples = make_protein_examples(count, seed=config.data["seed"], split="test")
        prompts = _protein_prompts(examples, tokenizer)
        manifest = protein_manifest(examples, seed=config.data["seed"])
    else:
        bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        prompts = [RLPrompt(f"eval:{i}", "[unconditional]", token_ids=(bos,)) for i in range(count)]
        manifest = {"task": config.task, "count": count, "prompt_token_ids": [bos]}
    sampling = replace(config.sampling, num_candidates=pool, chunk_size=min(config.sampling.chunk_size, pool))
    root.mkdir(parents=True, exist_ok=False)
    _write(root / "selection.json", manifest)
    _write(root / "config.json", {"training_config": config.to_dict(), "sampling": asdict(sampling),
                                   "checkpoint": metadata, "versions": environment_versions()})
    pools = []
    with (root / "completions.jsonl").open("x") as f:
        for prompt in prompts:
            with _precision_context(next(model.parameters()).device, config.update.precision):
                batch, records = collect_rollouts(model, tokenizer, [prompt], sampling, step=0, task=config.task)
            pools.append(batch.rewards[0].tolist())
            f.write(json.dumps(records[0], allow_nan=False) + "\n"); f.flush()
    budgets = (1, 2, 4, 8, 16, 32) if conditional else (1, 2, 4, 8, 16, 32, 64)
    summary = summarize_protein_rewards(pools, budgets)
    _write(root / "summary.json", summary)
    _write(root / "complete.json", {"kind": "refactored_protein_evaluation", "prompts": count, "pool_size": pool})
    return root


def main(argv=None, *, task_filter=None, evaluate=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--initial-adapter")
    parser.add_argument("--initial-model")
    parser.add_argument("--model-revision")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--local-data")
    parser.add_argument("--cache-dir")
    parser.add_argument("--allow-download", action="store_true")
    if evaluate:
        parser.add_argument("--checkpoint")
        parser.add_argument("--candidates", type=int)
        parser.add_argument("--prompts", type=int)
    args = parser.parse_args(argv)
    try:
        config = load_rl_config(args.config)
        if task_filter == "math" and config.task != "math" or task_filter == "protein" and config.task == "math":
            raise ValueError("Configuration is for the other task family")
        if args.model_revision:
            config = replace(config, model=replace(config.model, revision=args.model_revision))
        if args.dataset_revision:
            config = replace(config, data={**config.data, "revision": args.dataset_revision})
        if args.dry_run:
            print(json.dumps(config.to_dict(), indent=2))
            print("\nDRY RUN: no model/data loaded, no files written, no training started.")
            for requirement in config.requirements():
                print("  - " + requirement)
            if evaluate:
                print("  - Provide --checkpoint for evaluation; defaults: 150 x 32 conditional / 1 x 512 unconditional")
            return 0
        if not args.output_dir:
            raise ValueError("Execution requires a new --output-dir")
        if evaluate:
            if not args.checkpoint:
                raise ValueError("Evaluation requires --checkpoint")
            root = evaluate_protein(config, args.checkpoint, args.output_dir, candidates=args.candidates,
                                      num_prompts=args.prompts, allow_download=args.allow_download, cache_dir=args.cache_dir)
        else:
            root = run_rl(config, args.output_dir, initial_adapter=args.initial_adapter,
                            initial_model=args.initial_model, local_data=args.local_data,
                            allow_download=args.allow_download, cache_dir=args.cache_dir)
        print(f"Completed: {root}")
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
