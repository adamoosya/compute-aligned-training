"""Score saved completions without loading a model or contacting a service."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any, Iterable, Mapping

from cat import __version__
from cat.rewards.math import EXTRACTION_MODES, PARSER_VERSION, answer_matches, extract_answer, normalize_answer
from .metrics import majority_at_k, pass_at_k, require_integer, summarize_problem_scores
from .records import CandidateRecord, parse_candidate_bytes, parse_json, validate_records


@dataclass(frozen=True)
class MathScoreConfig:
    budgets: tuple[int, ...] = (1, 4, 8, 16, 32, 64)
    majority_trials: int = 500
    seed: int = 42
    extraction: str = "boxed_or_last_number"
    tie_break: str = "first"
    atol: float = 1e-4
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if not isinstance(self.budgets, tuple) or not self.budgets:
            raise ValueError("budgets must be a nonempty tuple")
        for budget in self.budgets:
            require_integer(budget, "budget")
        if len(set(self.budgets)) != len(self.budgets):
            raise ValueError("budgets cannot contain duplicates")
        require_integer(self.majority_trials, "majority_trials", 2)
        require_integer(self.seed, "seed", 0)
        if self.extraction not in EXTRACTION_MODES:
            raise ValueError(f"extraction must be one of {EXTRACTION_MODES}")
        if self.tie_break not in {"first", "failure"}:
            raise ValueError("tie_break must be first or failure")
        if isinstance(self.atol, bool) or not isinstance(self.atol, (int, float)) or not math.isfinite(self.atol) or self.atol < 0:
            raise ValueError("atol must be finite and nonnegative")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MathScoreConfig:
        fields = {"schema_version", "budgets", "majority_trials", "seed", "extraction", "tie_break", "atol"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError(f"score config requires exactly these fields: {sorted(fields)}")
        if not isinstance(value["budgets"], list):
            raise ValueError("budgets must be a JSON array")
        return cls(**{**value, "budgets": tuple(value["budgets"])})

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "budgets": list(self.budgets)}


def load_score_config(path: str | Path) -> MathScoreConfig:
    return MathScoreConfig.from_dict(parse_json(Path(path).read_text(encoding="utf-8")))


def _trial_seed(seed: int, example_id: str, k: int) -> int:
    # Do not use Python's salted hash(). Reordering records or budgets must not
    # change the Monte Carlo draws for an existing (problem, budget) pair.
    payload = json.dumps([seed, example_id, k], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:16], "big")


def validate_scoring_inputs(
    records: Iterable[CandidateRecord], config: MathScoreConfig,
) -> tuple[CandidateRecord, ...]:
    if not isinstance(config, MathScoreConfig):
        raise TypeError("config must be MathScoreConfig")
    records = validate_records(records)
    for record in records:
        if max(config.budgets) > len(record.completions):
            raise ValueError(f"{record.example_id}: largest budget exceeds candidate pool size")
        # Targets must themselves be valid final answers under this verifier.
        if not answer_matches(record.answer, record.answer, atol=config.atol):
            # A boxed reference is permitted; normalize its content on both sides.
            parsed = extract_answer(record.answer, mode="answer_only")
            if not answer_matches(parsed.text, record.answer, atol=config.atol):
                raise ValueError(f"{record.example_id}: unsupported or invalid reference answer")
    return records


def score_records(
    records: Iterable[CandidateRecord], config: MathScoreConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return aggregate metrics and complete per-problem scoring records."""
    records = validate_scoring_inputs(records, config)
    problems = []
    total_candidates = 0
    invalid_candidates = 0
    for record in records:
        parsed = [extract_answer(c, mode=config.extraction) for c in record.completions]
        keys = [(normalize_answer(p.text) or None) if p.text is not None else None for p in parsed]
        correct = [answer_matches(p.text, record.answer, atol=config.atol) for p in parsed]
        n, c = len(correct), sum(correct)
        total_candidates += n
        invalid_candidates += sum(key is None for key in keys)
        # Group answers by their presentation-normalized keys, not their reward.
        # Distinct symbolic strings that both verify can still split their votes.
        scores = {}
        for k in sorted(config.budgets):
            vote = majority_at_k(
                keys, k, is_correct=lambda key: answer_matches(key, record.answer, atol=config.atol),
                trials=config.majority_trials, seed=_trial_seed(config.seed, record.example_id, k),
                tie_break=config.tie_break,
            )
            scores[str(k)] = {"pass_at_k": pass_at_k(n, c, k), "majority_at_k": asdict(vote)}
        problems.append({
            **record.to_dict(), "problem_sha256": record.problem_hash,
            "pool_size": n, "correct_count": c, "invalid_count": sum(key is None for key in keys),
            "parsed_answers": [asdict(p) for p in parsed], "vote_keys": keys,
            "correct": correct, "scores": scores,
        })
    summary: dict[str, Any] = {
        "schema_version": 1, "cat_version": __version__, "parser_version": PARSER_VERSION,
        "python_version": platform.python_version(),
        "config": config.to_dict(), "num_problems": len(problems),
        "source_split": records[0].source_split,
        "total_candidates": total_candidates, "invalid_candidates": invalid_candidates,
        "standard_error_unit": "problem", "pass_at_k": {}, "majority_at_k": {},
    }
    for k in sorted(config.budgets):
        key = str(k)
        summary["pass_at_k"][key] = summarize_problem_scores([p["scores"][key]["pass_at_k"] for p in problems])
        summary["majority_at_k"][key] = summarize_problem_scores([p["scores"][key]["majority_at_k"]["value"] for p in problems])
    return summary, problems


def score_candidate_file(
    input_path: str | Path, output_dir: str | Path, config: MathScoreConfig,
) -> Path:
    """Read one immutable input snapshot; write results to a new directory only."""
    destination = Path(output_dir).expanduser()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"output directory already exists: {destination}")
    for parent in destination.parents:
        if parent.is_symlink():
            raise ValueError("output directory must not traverse symlinks")
    input_path = Path(input_path).expanduser()
    source_bytes = input_path.read_bytes()
    records = parse_candidate_bytes(source_bytes)
    summary, problems = score_records(records, config)
    summary["input_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    # Compute everything before creating an output directory. Never fabricate a
    # summary after skipping a bad problem or failing to parse the input schema.
    summary_bytes = (json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    problem_bytes = "".join(json.dumps(p, ensure_ascii=False, allow_nan=False) + "\n" for p in problems).encode("utf-8")
    payloads = {"candidates.jsonl": source_bytes, "per_problem.jsonl": problem_bytes, "summary.json": summary_bytes}
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in payloads.items():
        with (destination / name).open("xb") as handle:
            handle.write(data)
    complete = {
        "schema_version": 1, "input_name": input_path.name,
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()},
    }
    # Last file is a completion marker; an interrupted directory is never complete.
    with (destination / "complete.json").open("x", encoding="utf-8") as handle:
        json.dump(complete, handle, indent=2)
        handle.write("\n")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="completion-only candidate JSONL")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="new output directory; never overwritten")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_score_config(args.config)
        if args.dry_run:
            records = validate_scoring_inputs(parse_candidate_bytes(args.input.read_bytes()), config)
            print(json.dumps({"config": config.to_dict(), "num_problems": len(records),
                              "source_split": records[0].source_split,
                              "pool_sizes": sorted({len(r.completions) for r in records})}, indent=2))
            print("DRY RUN: candidate schema validated; no model loaded or output written.")
            return 0
        if args.output is None:
            parser.error("--output is required unless --dry-run is used")
        output = score_candidate_file(args.input, args.output, config)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    print(f"Scored {summary['num_problems']} problems; {summary['invalid_candidates']} invalid answers kept as failed draws.")
    for k in sorted(config.budgets):
        p = summary['pass_at_k'][str(k)]['mean']
        m = summary['majority_at_k'][str(k)]['mean']
        print(f"k={k:3d}: Pass@k={p:.6f}; Maj@k={m:.6f}")
    print(f"Saved candidate copy, per-problem scores, summary and checksums: {output}")
    return 0
