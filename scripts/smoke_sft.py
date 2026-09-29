#!/usr/bin/env python3
"""Offline CPU integration: shared CE warmup, three SFT branches, save/reload."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import platform
import tempfile

import torch
from torch.utils.data import DataLoader

from cat.data import CausalCollator, encode_math_selection, select_math_examples
from cat.testing.tiny import ByteChatTokenizer, TinyCausalLM, tiny_math_records
from cat.training import (
    SFTObjective, evaluate_sft_loss, load_sft_snapshot, save_sft_snapshot, train_sft_epoch,
)


def run(output: Path) -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(42)
    tokenizer = ByteChatTokenizer()
    selection = select_math_examples(tiny_math_records(), split="smoke", shuffle=False)
    dataset = encode_math_selection(selection, tokenizer, target="answer", max_length=128)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=CausalCollator(tokenizer.pad_token_id))
    shared = TinyCausalLM()
    optimizer = torch.optim.AdamW(shared.parameters(), lr=0.02, weight_decay=0)
    train_sft_epoch(shared, loader, optimizer, SFTObjective(), accumulation_steps=2, device="cpu")
    initial_state = deepcopy(shared.state_dict())
    checks = {}
    # Every branch starts from the same CE-warmed weights, with a fresh optimizer.
    for name, objective in [
        ("ce", SFTObjective()),
        ("passn", SFTObjective("passn", n=4)),
        ("majority", SFTObjective("majority", n=8, k=3, reduction="token_mean")),
    ]:
        model = TinyCausalLM()
        model.load_state_dict(initial_state)
        before = evaluate_sft_loss(model, loader, objective, device="cpu").loss
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0)
        for _ in range(2):
            stats = train_sft_epoch(model, loader, optimizer, objective,
                                    accumulation_steps=2, max_grad_norm=1.0, device="cpu")
        after = evaluate_sft_loss(model, loader, objective, device="cpu").loss
        if not after < before:
            raise RuntimeError(f"{name}: offline fixture loss failed to decrease")
        if not any(not torch.equal(value, initial_state[key]) for key, value in model.state_dict().items()):
            raise RuntimeError(f"{name}: model weights did not change")
        metadata = {"kind": "smoke_only", "objective": asdict(objective),
                    "before": before, "after": after, "last_epoch": asdict(stats)}
        save_sft_snapshot(output / name, model, metadata)
        restored = TinyCausalLM()
        loaded_metadata = load_sft_snapshot(output / name, restored)
        if loaded_metadata != metadata:
            raise RuntimeError("Snapshot metadata did not round trip")
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
        checks[name] = metadata
        print(f"{name:8s} loss {before:.6f} -> {after:.6f}; save/reload OK")
    report = {"kind": "smoke_only_not_paper_results", "python": platform.python_version(),
              "torch": torch.__version__, "device": "cpu", "seed": 42,
              "data": selection.manifest(), "tokenization": dataset.manifest(),
              "checks": checks}
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("SFT smoke test passed (CPU; no model or dataset downloads).")
    print(f"Local smoke outputs: {output}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="A new directory; existing destinations are refused")
    args = parser.parse_args()
    if args.output is None:
        parent = Path("outputs")
        parent.mkdir(exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="sft-smoke-", dir=parent)).resolve()
    else:
        output = args.output.expanduser().absolute()
        if output.is_symlink():
            parser.error("Refusing an output symlink")
        output.mkdir(parents=True, exist_ok=False)
    run(output)


if __name__ == "__main__":
    main()
