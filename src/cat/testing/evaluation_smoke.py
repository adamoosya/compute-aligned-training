"""Offline scoring integration on handwritten completions, not model samples."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile

from cat.evaluation import CandidateRecord, MathScoreConfig, score_candidate_file, write_candidate_records


def fixture_candidates() -> tuple[CandidateRecord, ...]:
    return (
        CandidateRecord("fixture:0", "What is 1+1?", "2", "fixture", (
            r"Adding the two units gives \boxed{2}.", "2", r"\boxed{3}", "No answer.",
        )),
        CandidateRecord("fixture:1", "What is -3 divided by 2?", "-3/2", "fixture", (
            r"\boxed{-\frac{3}{2}}", r"\boxed{-\frac{3}{2}}", "Answer: -1.5", r"\boxed{4}",
        )),
        CandidateRecord("fixture:2", "What is 3+4?", "7", "fixture", (
            r"\boxed{1}", r"\boxed{1}", r"\boxed{2}", "No numeric answer.",
        )),
    )


def run_evaluation_smoke(output_root: str | Path = "outputs") -> Path:
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="evaluation-smoke-", dir=root))
    candidates = write_candidate_records(run / "input.jsonl", fixture_candidates())
    config = MathScoreConfig(budgets=(1, 2, 4), majority_trials=1000, seed=123)
    (run / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")
    out = score_candidate_file(candidates, run / "scored", config)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    expected = {1: 5 / 12, 2: 11 / 18, 4: 2 / 3}
    for k, value in expected.items():
        actual = summary["pass_at_k"][str(k)]["mean"]
        if not math.isclose(actual, value, rel_tol=0, abs_tol=1e-12):
            raise AssertionError(f"Pass@{k} expected {value}, got {actual}")
        print(f"Pass@{k}: {actual:.6f}; combinatorial check OK")
    if summary["invalid_candidates"] != 2:
        raise AssertionError("invalid completions were lost")
    if not math.isclose(summary["majority_at_k"]["1"]["mean"], expected[1], abs_tol=1e-12):
        raise AssertionError("Maj@1 and Pass@1 must agree")
    if summary["majority_at_k"]["4"]["mean"] != 2 / 3:
        raise AssertionError("full-pool fixture majority failed")
    other = score_candidate_file(candidates, run / "repeat", config)
    for name in ("summary.json", "per_problem.jsonl"):
        if (out / name).read_bytes() != (other / name).read_bytes():
            raise AssertionError("repeated scoring changed")
    manifest = json.loads((out / "complete.json").read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        if hashlib.sha256((out / name).read_bytes()).hexdigest() != digest:
            raise AssertionError("output checksum mismatch")
    print("Majority Vote: deterministic repeat; invalid answers retained; JSON/checksums OK")
    print("Evaluation smoke test passed (saved fixtures; no model or dataset downloads).")
    print(f"Local smoke outputs: {run}")
    return run


if __name__ == "__main__":
    run_evaluation_smoke()
