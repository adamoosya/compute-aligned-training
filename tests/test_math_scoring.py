from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from cat.evaluation import MathScoreConfig, score_candidate_file, score_records, write_candidate_records
from cat.evaluation.scoring import load_score_config, main
from cat.testing.evaluation_smoke import fixture_candidates, run_evaluation_smoke


def config():
    return MathScoreConfig(budgets=(1, 2, 4), majority_trials=300)


def test_scoring_preserves_counts_and_calculates_means():
    summary, problems = score_records(fixture_candidates(), config())
    assert summary["num_problems"] == 3
    assert summary["total_candidates"] == 12
    assert summary["invalid_candidates"] == 2
    assert summary["standard_error_unit"] == "problem"
    assert summary["pass_at_k"]["1"]["mean"] == pytest.approx(5 / 12)
    assert summary["pass_at_k"]["2"]["mean"] == pytest.approx(11 / 18)
    assert summary["pass_at_k"]["4"]["mean"] == pytest.approx(2 / 3)
    assert [p["correct_count"] for p in problems] == [2, 3, 0]
    assert all(p["scores"]["1"]["pass_at_k"] == p["scores"]["1"]["majority_at_k"]["value"] for p in problems)
    assert summary["majority_at_k"]["4"]["mean"] == 2 / 3
    assert all(len(p["completions"]) == 4 for p in problems)


def test_deterministic_under_record_and_budget_reordering():
    summary, original = score_records(fixture_candidates(), config())
    other_summary, other = score_records(tuple(reversed(fixture_candidates())), replace(config(), budgets=(4, 1, 2)))
    assert sorted(original, key=lambda p: p["example_id"]) == sorted(other, key=lambda p: p["example_id"])
    assert summary["pass_at_k"] == other_summary["pass_at_k"]
    assert summary["majority_at_k"] == other_summary["majority_at_k"]
    _, fewer = score_records(fixture_candidates(), replace(config(), budgets=(2,)))
    assert [p["scores"]["2"] for p in original] == [p["scores"]["2"] for p in fewer]


def test_equal_weight_per_problem_not_candidate():
    first, second, _ = fixture_candidates()
    first = replace(first, completions=("2",) * 20)
    second = replace(second, completions=("wrong",))
    summary, _ = score_records((first, second), MathScoreConfig(budgets=(1,)))
    assert summary["pass_at_k"]["1"]["mean"] == 0.5
    assert summary["pass_at_k"]["1"]["standard_error"] == 0.5


def test_single_problem_se_is_null():
    summary, _ = score_records((fixture_candidates()[0],), config())
    assert summary["pass_at_k"]["1"]["standard_error"] is None


def test_none_of_the_failed_draws_is_dropped():
    first = replace(fixture_candidates()[0], completions=("", "", "not a number", "2"))
    summary, problems = score_records((first,), config())
    assert summary["pass_at_k"]["1"]["mean"] == 0.25
    assert problems[0]["invalid_count"] == 3
    assert problems[0]["pool_size"] == 4


def test_boxed_reference_supported_and_no_target_last_number_fallback():
    first = replace(fixture_candidates()[0], answer=r"\boxed{\frac{1}{2}}", completions=("2",) * 4)
    summary, _ = score_records((first,), config())
    assert summary["pass_at_k"]["1"]["mean"] == 0


def test_vote_does_not_pool_different_correct_forms():
    first = replace(fixture_candidates()[0], answer=".5", completions=(
        r"\boxed{1/2}", r"\boxed{1/2}", r"\boxed{0.5}", r"\boxed{0.5}",
        r"\boxed{3}", r"\boxed{3}", r"\boxed{3}",
    ))
    summary, _ = score_records((first,), MathScoreConfig(budgets=(1, 7)))
    assert summary["pass_at_k"]["7"]["mean"] == 1
    assert summary["majority_at_k"]["7"]["mean"] == 0


def test_create_only_output_and_checksums(tmp_path):
    source = write_candidate_records(tmp_path / "input.jsonl", fixture_candidates())
    output = score_candidate_file(source, tmp_path / "scored", config())
    assert (output / "candidates.jsonl").read_bytes() == source.read_bytes()
    complete = json.loads((output / "complete.json").read_text())
    for name, digest in complete["files"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    summary = json.loads((output / "summary.json").read_text())
    assert summary["input_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    original = (output / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        score_candidate_file(source, output, config())
    assert (output / "summary.json").read_bytes() == original


def test_bad_input_cannot_create_a_partial_output(tmp_path):
    source = write_candidate_records(tmp_path / "input.jsonl", fixture_candidates())
    with pytest.raises(ValueError, match="exceeds"):
        score_candidate_file(source, tmp_path / "bad", MathScoreConfig(budgets=(64,)))
    assert not (tmp_path / "bad").exists()


def test_symlink_destinations_are_refused(tmp_path):
    source = write_candidate_records(tmp_path / "input.jsonl", fixture_candidates())
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError):
        score_candidate_file(source, link, config())
    with pytest.raises(ValueError, match="symlink"):
        score_candidate_file(source, link / "subdir", config())
    assert not (target / "subdir").exists()


@pytest.mark.parametrize("field,value", [
    ("budgets", ()), ("budgets", [1, 2]), ("budgets", (1, 1)), ("budgets", (0,)),
    ("budgets", (True,)), ("budgets", (1.5,)), ("majority_trials", 0), ("majority_trials", 1),
    ("seed", -1), ("seed", True), ("extraction", "guess"), ("tie_break", "correct"),
    ("atol", -1), ("atol", float("nan")), ("atol", True),
    ("schema_version", 2), ("schema_version", True), ("schema_version", 1.0),
])
def test_config_validation(field, value):
    with pytest.raises(ValueError):
        replace(config(), **{field: value})


def test_config_json_strict_schema(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config().to_dict()))
    assert load_score_config(path) == config()
    with pytest.raises(ValueError):
        MathScoreConfig.from_dict({**config().to_dict(), "extra": "typo"})
    with pytest.raises(ValueError):
        MathScoreConfig.from_dict({})
    with pytest.raises(ValueError):
        MathScoreConfig.from_dict({**config().to_dict(), "budgets": "1,2"})


@pytest.mark.parametrize("name,maximum", [("math_passn", 64), ("math_majority", 128)])
def test_presets_parse(name, maximum):
    root = Path(__file__).resolve().parents[1]
    cfg = load_score_config(root / "configs/evaluation" / f"{name}.json")
    assert max(cfg.budgets) == maximum
    assert cfg.majority_trials == 500


def test_dry_run_and_cli(tmp_path, capsys):
    source = write_candidate_records(tmp_path / "input.jsonl", fixture_candidates())
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(config().to_dict()))
    output = tmp_path / "scores"
    args = ["--input", str(source), "--config", str(cfg), "--output", str(output)]
    assert main(args + ["--dry-run"]) == 0
    assert not output.exists()
    assert "DRY RUN" in capsys.readouterr().out
    assert main(args) == 0
    assert (output / "complete.json").exists()
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2


def test_cli_requires_output_unless_preview(tmp_path):
    source = write_candidate_records(tmp_path / "input.jsonl", fixture_candidates())
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(config().to_dict()))
    with pytest.raises(SystemExit) as exc:
        main(["--input", str(source), "--config", str(cfg)])
    assert exc.value.code == 2


def test_cli_entry_script(tmp_path):
    source = write_candidate_records(tmp_path / "input.jsonl", fixture_candidates())
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(config().to_dict()))
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([
        sys.executable, str(root / "scripts/score_math.py"), "--input", str(source),
        "--config", str(cfg), "--output", str(tmp_path / "score"),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Scored 3 problems" in result.stdout


def test_evaluation_smoke(tmp_path):
    output = run_evaluation_smoke(tmp_path)
    assert (output / "scored/complete.json").is_file()
