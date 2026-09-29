"""JSONL schema for generated completions and their reference answers."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from cat.data.math import MathExample


@dataclass(frozen=True)
class CandidateRecord:
    """One problem and all attempted completions; blank completions are failures."""
    example_id: str
    problem: str
    answer: str
    source_split: str
    completions: tuple[str, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        for name in ("example_id", "problem", "answer", "source_split"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if not isinstance(self.completions, tuple) or not self.completions:
            raise ValueError("completions must be a nonempty tuple")
        if any(not isinstance(c, str) for c in self.completions):
            raise ValueError("every completion must be a string; keep failed parses as text")

    @property
    def problem_hash(self) -> str:
        return hashlib.sha256(" ".join(self.problem.split()).encode("utf-8")).hexdigest()

    @classmethod
    def from_example(cls, example: MathExample, completions: Iterable[str]) -> CandidateRecord:
        if not isinstance(example, MathExample):
            raise TypeError("example must be a MathExample")
        if isinstance(completions, (str, bytes)):
            raise ValueError("completions must contain individual strings")
        return cls(example.example_id, example.problem, example.answer,
                   example.source_split, tuple(completions))

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> CandidateRecord:
        expected = {"schema_version", "example_id", "problem", "answer", "source_split", "completions"}
        if not isinstance(row, Mapping) or set(row) != expected:
            raise ValueError(f"candidate record requires exactly these fields: {sorted(expected)}")
        if not isinstance(row["completions"], list):
            raise ValueError("completions must be a JSON array, not counts or a single string")
        return cls(**{**row, "completions": tuple(row["completions"])})

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "completions": list(self.completions)}


def validate_records(records: Iterable[CandidateRecord]) -> tuple[CandidateRecord, ...]:
    result = tuple(records)
    if not result:
        raise ValueError("candidate file contains no problems")
    ids: set[str] = set()
    hashes: set[str] = set()
    for record in result:
        if not isinstance(record, CandidateRecord):
            raise TypeError("records must contain CandidateRecord objects")
        if record.example_id in ids:
            raise ValueError(f"duplicate example_id: {record.example_id}")
        if record.problem_hash in hashes:
            raise ValueError(f"duplicate problem text: {record.example_id}")
        ids.add(record.example_id)
        hashes.add(record.problem_hash)
    if len({record.source_split for record in result}) != 1:
        raise ValueError("do not combine different source splits in one evaluation")
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON value: {value}")


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def parse_json(text: str) -> Any:
    """Strict JSON: reject duplicate keys and NaN/Infinity rather than guessing."""
    return json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_unique_keys)


def parse_candidate_bytes(data: bytes) -> tuple[CandidateRecord, ...]:
    records = []
    for line_number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(CandidateRecord.from_dict(parse_json(line)))
        except (ValueError, TypeError) as error:
            raise ValueError(f"candidate line {line_number}: {error}") from error
    return validate_records(records)


def read_candidate_records(path: str | Path) -> tuple[CandidateRecord, ...]:
    return parse_candidate_bytes(Path(path).read_bytes())


def write_candidate_records(path: str | Path, records: Iterable[CandidateRecord]) -> Path:
    """Create a JSONL file without replacing an existing file or following a link."""
    records = validate_records(records)
    destination = Path(path)
    # Exclusive opening also refuses dangling symlinks.
    with destination.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, allow_nan=False) + "\n")
    return destination
