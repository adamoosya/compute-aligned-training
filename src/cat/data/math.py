"""MATH records, deterministic selection, and explicit split loading."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable, Mapping, Sequence


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative Python integer")
    return value


def parse_level(value: Any) -> int | None:
    """Accept MATH's 'Level 1' form and integer levels 1--5."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("level must be an integer from 1 to 5 or 'Level 1' form")
    if isinstance(value, str):
        match = re.fullmatch(r"(?:Level\s+)?([1-5])", value.strip(), flags=re.I)
        if match:
            return int(match.group(1))
    elif isinstance(value, int) and 1 <= value <= 5:
        return value
    raise ValueError(f"Invalid MATH level: {value!r}")


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MathExample:
    """A record retains its original split index, even after selection."""
    example_id: str
    problem: str
    answer: str
    solution: str | None
    level: int | None
    source_split: str
    source_index: int

    @property
    def problem_hash(self) -> str:
        # Conservative overlap check: collapse whitespace, but not mathematics.
        return _hash_text(" ".join(self.problem.split()))

    def target_text(self, target: str) -> str:
        if target == "answer":
            return self.answer
        if target == "solution":
            if self.solution is None:
                raise ValueError(f"{self.example_id}: solution target requested but missing")
            # Do not append or synthesize an answer. MATH solutions contain their answer.
            return self.solution
        raise ValueError("target must be 'answer' or 'solution'")


@dataclass(frozen=True)
class MathSelection:
    examples: tuple[MathExample, ...]
    source: str
    revision: str | None
    split: str
    seed: int
    shuffled: bool
    levels: tuple[int, ...] | None
    requested_count: int | None
    input_count: int
    eligible_count: int
    source_fingerprint: str | None = None

    def manifest(self) -> dict[str, Any]:
        """Serializable provenance without saving full benchmark text."""
        records = [
            {"id": x.example_id, "source_index": x.source_index,
             "problem_sha256": x.problem_hash,
             "answer_sha256": _hash_text(x.answer),
             "solution_sha256": _hash_text(x.solution) if x.solution else None}
            for x in self.examples
        ]
        encoded = json.dumps(records, sort_keys=True, separators=(",", ":"))
        return {
            "source": self.source, "revision": self.revision, "split": self.split,
            "source_fingerprint": self.source_fingerprint,
            "seed": self.seed, "shuffle": self.shuffled,
            "levels": list(self.levels) if self.levels is not None else None,
            "requested_count": self.requested_count,
            "input_count": self.input_count, "eligible_count": self.eligible_count,
            "selected_count": len(records), "selected_records": records,
            "selection_sha256": _hash_text(encoded),
        }


def select_math_examples(
    records: Iterable[Mapping[str, Any]], *, split: str,
    levels: Sequence[int] | None = None, max_examples: int | None = None,
    seed: int = 42, shuffle: bool = True, source: str = "in_memory",
    revision: str | None = None, source_fingerprint: str | None = None,
) -> MathSelection:
    """Filter first, then shuffle using a private RNG, then select.

    A requested count larger than the eligible pool is an error, not a silent
    change to the experiment. Missing levels are an error when filtering.
    """
    split = _text(split, "split")
    source = _text(source, "source")
    seed = _nonnegative_int(seed, "seed")
    if not isinstance(shuffle, bool):
        raise TypeError("shuffle must be bool")
    if max_examples is not None:
        _nonnegative_int(max_examples, "max_examples")
        if max_examples == 0:
            raise ValueError("max_examples must be positive")
    allowed = None
    if levels is not None:
        if not levels:
            raise ValueError("levels cannot be empty")
        if any(isinstance(x, bool) or not isinstance(x, int) or not 1 <= x <= 5
               for x in levels):
            raise ValueError("levels must contain integers from 1 to 5")
        allowed = tuple(sorted(set(levels)))
    selected: list[MathExample] = []
    input_count = 0
    for index, row in enumerate(records):
        input_count += 1
        if not isinstance(row, Mapping):
            raise ValueError(f"{split}:{index}: record must be a mapping")
        try:
            level = parse_level(row.get("level"))
            if allowed is not None:
                if level is None:
                    raise ValueError("level is missing while level filtering is enabled")
                if level not in allowed:
                    continue
            problem = _text(row.get("problem"), "problem")
            answer = _text(row.get("answer"), "answer")
            solution_value = row.get("solution")
            solution = None if solution_value is None else _text(solution_value, "solution")
        except ValueError as error:
            raise ValueError(f"{split}:{index}: {error}") from error
        selected.append(MathExample(
            example_id=f"{split}:{index}:{_hash_text(problem)[:12]}",
            problem=problem, answer=answer, solution=solution, level=level,
            source_split=split, source_index=index,
        ))
    eligible_count = len(selected)
    if eligible_count == 0:
        raise ValueError("No eligible MATH examples")
    if max_examples is not None and max_examples > eligible_count:
        raise ValueError(f"Requested {max_examples} examples; only {eligible_count} eligible")
    if shuffle:
        random.Random(seed).shuffle(selected)
    if max_examples is not None:
        selected = selected[:max_examples]
    return MathSelection(
        examples=tuple(selected), source=source, revision=revision, split=split,
        seed=seed, shuffled=shuffle, levels=allowed, requested_count=max_examples,
        input_count=input_count, eligible_count=eligible_count,
        source_fingerprint=source_fingerprint,
    )


def load_math_jsonl(path: str | Path, *, split: str, **selection: Any) -> MathSelection:
    """Read explicit local JSONL records; blank lines are ignored."""
    path = Path(path).expanduser()
    data = path.read_bytes()
    rows = []
    for line_number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path.name}:{line_number}: invalid JSON") from error
    return select_math_examples(
        rows, split=split, source=f"jsonl:{path.name}",
        source_fingerprint=hashlib.sha256(data).hexdigest(), **selection,
    )


def load_math_hf(
    *, dataset_name: str, revision: str, split: str,
    cache_dir: str | Path | None = None, **selection: Any,
) -> MathSelection:
    """Load the requested Hugging Face split. Never fall back to another split.

    Use a dataset commit hash for a pinned revision. This optional path can
    download data; importing this module and running the smoke test cannot.
    """
    dataset_name = _text(dataset_name, "dataset_name")
    revision = _text(revision, "revision")
    split = _text(split, "split")
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError("Install dataset support with: python -m pip install -e '.[data]'") from error
    dataset = load_dataset(
        dataset_name, revision=revision, split=split,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    return select_math_examples(
        dataset, source=f"hf:{dataset_name}", revision=revision, split=split,
        source_fingerprint=getattr(dataset, "_fingerprint", None), **selection,
    )


def assert_disjoint(*selections: MathSelection) -> None:
    """Raise on duplicate problems within or across the supplied selections."""
    seen: dict[str, str] = {}
    for selection in selections:
        for example in selection.examples:
            key = example.problem_hash
            if key in seen:
                raise ValueError(f"Duplicate problem: {seen[key]} and {example.example_id}")
            seen[key] = example.example_id
