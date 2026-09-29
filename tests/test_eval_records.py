from dataclasses import replace
import json

import pytest

from cat.data.math import select_math_examples
from cat.evaluation.records import (
    CandidateRecord, parse_candidate_bytes, parse_json, read_candidate_records,
    validate_records, write_candidate_records,
)
from cat.testing.evaluation_smoke import fixture_candidates


def record():
    return fixture_candidates()[0]


def test_file_round_trip_retains_invalid_attempts(tmp_path):
    path = write_candidate_records(tmp_path / "candidates.jsonl", fixture_candidates())
    loaded = read_candidate_records(path)
    assert loaded == fixture_candidates()
    assert loaded[0].completions[-1] == "No answer."
    assert parse_candidate_bytes(b"\n" + path.read_bytes()) == loaded
    with pytest.raises(FileExistsError):
        write_candidate_records(path, loaded)


def test_from_math_example_preserves_source():
    selection = select_math_examples([{"problem": "1+1?", "answer": "2"}], split="test", shuffle=False)
    example = selection.examples[0]
    result = CandidateRecord.from_example(example, ["2", "", "wrong"])
    assert result.example_id == example.example_id
    assert result.problem_hash == example.problem_hash
    assert result.source_split == "test"
    assert len(result.completions) == 3
    with pytest.raises(ValueError):
        CandidateRecord.from_example(example, "2")
    with pytest.raises(TypeError):
        CandidateRecord.from_example({}, ["2"])


@pytest.mark.parametrize("key,value", [
    ("example_id", ""), ("example_id", 2), ("problem", " "), ("answer", ""),
    ("source_split", ""), ("completions", ()), ("completions", ["1"]),
    ("completions", (None,)), ("completions", (False,)),
    ("schema_version", 2), ("schema_version", True), ("schema_version", 1.0),
])
def test_record_validation(key, value):
    with pytest.raises(ValueError):
        replace(record(), **{key: value})


def test_strict_json_schema():
    obj = record().to_dict()
    assert CandidateRecord.from_dict(obj) == record()
    for missing in obj:
        bad = {k: v for k, v in obj.items() if k != missing}
        with pytest.raises(ValueError):
            CandidateRecord.from_dict(bad)
    with pytest.raises(ValueError):
        CandidateRecord.from_dict({**obj, "model": "not a field in this schema"})
    with pytest.raises(ValueError):
        CandidateRecord.from_dict({**obj, "completions": "not a list"})


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"x":NaN}', '{"x":Infinity}', 'not json'])
def test_json_rejects_ambiguous_or_nonfinite_values(text):
    with pytest.raises(ValueError):
        parse_json(text)


def test_json_file_reports_bad_line():
    good = json.dumps(record().to_dict())
    with pytest.raises(ValueError, match="candidate line 2"):
        parse_candidate_bytes((good + "\nnot json\n").encode())


def test_empty_file_and_duplicate_records_are_errors():
    with pytest.raises(ValueError, match="no problems"):
        parse_candidate_bytes(b"  \n")
    with pytest.raises(ValueError, match="duplicate example_id"):
        validate_records([record(), record()])
    duplicate = replace(record(), example_id="other", problem="  " + record().problem + "  ")
    with pytest.raises(ValueError, match="duplicate problem"):
        validate_records([record(), duplicate])


def test_mixed_splits_are_not_combined():
    a, b, _ = fixture_candidates()
    with pytest.raises(ValueError, match="different source splits"):
        validate_records([a, replace(b, source_split="train")])


def test_candidate_writer_rejects_symlink(tmp_path):
    target = tmp_path / "target"
    target.write_text("do not change")
    link = tmp_path / "link.jsonl"
    link.symlink_to(target)
    with pytest.raises(FileExistsError):
        write_candidate_records(link, fixture_candidates())
    assert target.read_text() == "do not change"
