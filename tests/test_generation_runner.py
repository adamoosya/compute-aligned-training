from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from cat.data import select_math_examples
from cat.evaluation.scoring import MathScoreConfig
from cat.experiments.generation_config import GenerationRunConfig, load_generation_config
import cat.experiments.generation as runner
from cat.generation import SamplingConfig
from generation_support import FakeModel, FakeTokenizer, FakeGenerationConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fixture_generation(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(GenerationConfig=FakeGenerationConfig))
    selection = select_math_examples([{"problem": "First question?", "answer": "6"},
                                     {"problem": "Second question?", "answer": "7"}], split="test", shuffle=False)
    return FakeModel(), FakeTokenizer(), selection, SamplingConfig(num_candidates=4, chunk_size=2, max_new_tokens=3), MathScoreConfig(budgets=(1,2,4), majority_trials=10)


def test_generation_writes_complete_and_matching_scores(fixture_generation, tmp_path):
    model, tokenizer, selection, sampling, scoring = fixture_generation
    path = runner.generate_and_score(model, tokenizer, selection, sampling, scoring, tmp_path / "run", provenance={})
    manifest = json.loads((path / "generation_complete.json").read_text())
    for name, checksum in manifest["sha256"].items():
        assert hashlib.sha256((path / name).read_bytes()).hexdigest() == checksum
    assert (path / "scores/candidates.jsonl").read_bytes() == (path / "candidates.jsonl").read_bytes()
    assert json.loads((path / "scores/summary.json").read_text())["total_candidates"] == 8
    assert (path / "complete.json").is_file()
    assert json.loads((path / "run.json").read_text())["overlap_check"]["status"] == "not_checked"
    with pytest.raises(FileExistsError):
        runner.generate_and_score(model, tokenizer, selection, sampling, scoring, path, provenance={})


def test_failure_never_written_as_complete(fixture_generation, tmp_path):
    model, tokenizer, selection, sampling, scoring = fixture_generation
    model.behavior = "fail_second"
    with pytest.raises(RuntimeError):
        runner.generate_and_score(model, tokenizer, selection, sampling, scoring, tmp_path / "run", provenance={})
    assert not (tmp_path / "run/generation_complete.json").exists()
    assert not (tmp_path / "run/complete.json").exists()
    assert not (tmp_path / "run/scores").exists()


def test_no_reference_conditioning(fixture_generation, tmp_path):
    model, tokenizer, selection, sampling, scoring = fixture_generation
    runner.generate_and_score(model, tokenizer, selection, sampling, scoring, tmp_path / "a", provenance={})
    changed = replace(selection, examples=tuple(replace(x, answer="9999") for x in selection.examples))
    runner.generate_and_score(model, tokenizer, changed, sampling, scoring, tmp_path / "b", provenance={})
    assert (tmp_path / "a/tokens.jsonl").read_bytes() == (tmp_path / "b/tokens.jsonl").read_bytes()


def test_symlink_output_rejected(fixture_generation, tmp_path):
    model, tokenizer, selection, sampling, scoring = fixture_generation
    (tmp_path / "target").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "target", target_is_directory=True)
    with pytest.raises(ValueError):
        runner.generate_and_score(model, tokenizer, selection, sampling, scoring, tmp_path / "link/run", provenance={})


def test_all_prompts_checked_before_first_call(fixture_generation, tmp_path):
    model, tokenizer, selection, sampling, scoring = fixture_generation
    with pytest.raises(ValueError):
        runner.generate_and_score(model, tokenizer, selection, replace(sampling,max_prompt_tokens=1), scoring,
                                  tmp_path / "run", provenance={})
    assert not model.calls and not (tmp_path / "run").exists()


def test_duplicates_and_invalid_budget_before_write(fixture_generation, tmp_path):
    model, tokenizer, selection, sampling, scoring = fixture_generation
    duplicate = replace(selection, examples=selection.examples + selection.examples)
    with pytest.raises(ValueError, match="Duplicate"):
        runner.generate_and_score(model, tokenizer, duplicate, sampling, scoring, tmp_path / "a", provenance={})
    with pytest.raises(ValueError, match="budget"):
        runner.generate_and_score(model, tokenizer, selection, sampling, replace(scoring,budgets=(5,)), tmp_path / "b", provenance={})
    assert not model.calls


def test_training_overlap_and_checksum(fixture_generation, tmp_path):
    _, _, selection, _, _ = fixture_generation
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection.manifest()))
    with pytest.raises(ValueError, match="overlaps"):
        runner.check_training_overlap(selection, path)
    new = select_math_examples([{"problem":"Different training question", "answer":"8"}], split="train")
    path.write_text(json.dumps(new.manifest()))
    assert runner.check_training_overlap(selection,path)["status"] == "disjoint_from_supplied_selection"
    altered = new.manifest(); altered["selected_records"][0]["problem_sha256"] = "a" * 64
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="checksum"):
        runner.check_training_overlap(selection,path)


@pytest.mark.parametrize("family", ["passn", "majority"])
def test_config_roundtrip_and_dry_run(family,tmp_path):
    path=ROOT / f"configs/generation/math_{family}_sft.json"
    config=load_generation_config(path)
    assert GenerationRunConfig.from_dict(config.to_dict()) == config
    assert config.data.split == "test"
    result = subprocess.run([sys.executable, str(ROOT/"scripts/generate_math.py"), "--config",str(path),"--dry-run"],
                            cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    assert "no generation started" in result.stdout
    assert "40-character" in result.stdout
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("change", ["unknown", "missing", "model_unknown", "score_large", "train_gc", "bad_init", "schema_bool", "empty_name"])
def test_invalid_config(change):
    data = load_generation_config(ROOT / "configs/generation/math_passn_sft.json").to_dict()
    if change == "unknown": data["guess"] = 1
    elif change == "missing": del data["sampling"]
    elif change == "model_unknown": data["model"]["oops"] = 1
    elif change == "score_large": data["scoring"]["budgets"] = [129]
    elif change == "train_gc": data["model"]["gradient_checkpointing"] = True
    elif change == "bad_init": data["initialization"] = "latest"
    elif change == "schema_bool": data["schema_version"] = True
    elif change == "empty_name": data["name"] = ""
    with pytest.raises((ValueError, TypeError)):
        GenerationRunConfig.from_dict(data)


@pytest.mark.parametrize("change", ["name", "split", "count", "level", "seed", "revision"])
def test_invalid_data_config(change):
    data = load_generation_config(ROOT / "configs/generation/math_passn_sft.json").to_dict()
    key,value={"name":("dataset_name",""),"split":("split",False),"count":("max_examples",0),
               "level":("levels",[True]),"seed":("seed",-1),"revision":("revision",3)}[change]
    data["data"][key]=value
    with pytest.raises((ValueError,TypeError)):
        GenerationRunConfig.from_dict(data)


def test_runner_default_refuses_model_loading_without_adapter(tmp_path,monkeypatch):
    config=load_generation_config(ROOT/"configs/generation/math_passn_sft.json")
    monkeypatch.setattr(runner,"load_inference_model",lambda *a,**k:pytest.fail("Should not load"))
    with pytest.raises(ValueError,match="adapter"):
        runner.run_generation(config,tmp_path/"run")
    assert not (tmp_path/"run").exists()


def test_runner_control_flow_and_offline_restore(fixture_generation,tmp_path,monkeypatch):
    model,tokenizer,selection,_,_=fixture_generation
    config=load_generation_config(ROOT/"configs/generation/math_passn_sft.json")
    base=tmp_path/"base";base.mkdir();(base/"config.json").write_text("{}")
    config=replace(config, initialization="base",model=replace(config.model,name=str(base),device="cpu",precision="fp32",quantization="none"),
                   data=replace(config.data,revision="a"*40,max_examples=2),
                   sampling=SamplingConfig(num_candidates=4,chunk_size=2,max_new_tokens=3),
                   scoring=MathScoreConfig(budgets=(1,4),majority_trials=10))
    fake_ds=SimpleNamespace(config=SimpleNamespace(HF_DATASETS_OFFLINE=False))
    monkeypatch.setitem(sys.modules,"datasets",fake_ds)
    def load(**kwargs):
        assert fake_ds.config.HF_DATASETS_OFFLINE is True
        assert kwargs["split"]=="test"
        return selection
    monkeypatch.setattr(runner,"load_math_hf",load)
    monkeypatch.setattr(runner,"load_inference_model",lambda *a,**k:(model,tokenizer))
    runner.run_generation(config,tmp_path/"run")
    assert not fake_ds.config.HF_DATASETS_OFFLINE
    assert (tmp_path/"run/complete.json").is_file()
    monkeypatch.setenv("WORLD_SIZE","2")
    with pytest.raises(ValueError,match="single-process"):
        runner.run_generation(config,tmp_path/"run2")


def test_real_generation_and_adapter_roundtrip(tmp_path,monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE","1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE","1")
    pytest.importorskip("transformers",reason="install hf extra for real generation integration")
    pytest.importorskip("peft",reason="install hf extra for real generation integration")
    from cat.testing.generation_smoke import run_generation_smoke
    output=run_generation_smoke(tmp_path)
    assert json.loads((output/"summary.json").read_text())["seeded_reload_match"]
