"""Exercise run-directory and shared-warmup control flow without Hub calls."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from cat.data import select_math_examples
from cat.experiments.sft_config import load_sft_config
import cat.experiments.sft as runner
from cat.testing.tiny import ByteChatTokenizer, TinyCausalLM, tiny_math_records


@pytest.fixture
def fixture_run(monkeypatch,tmp_path):
    cfg=load_sft_config(Path(__file__).resolve().parents[1]/'configs/sft/passn_warmup.json')
    base=tmp_path/'base'; base.mkdir(); (base/'config.json').write_text('{}')
    cfg=replace(cfg, model=replace(cfg.model,name=str(base),device='cpu',precision='fp32',quantization='none'),
                data=replace(cfg.data,revision='a'*40,max_examples=5),training=replace(cfg.training,epochs=1))
    selection=select_math_examples(tiny_math_records(),split='train',revision='a'*40,max_examples=5)
    monkeypatch.setitem(sys.modules,'datasets',SimpleNamespace(config=SimpleNamespace(HF_DATASETS_OFFLINE=False)))
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace())
    monkeypatch.setitem(sys.modules,'peft',SimpleNamespace())
    monkeypatch.setattr(runner,'load_math_hf',lambda **kwargs:selection)
    def load(*args,**kwargs):
        model=TinyCausalLM(); model.config=SimpleNamespace()
        return model,ByteChatTokenizer()
    monkeypatch.setattr(runner,'load_lora_model',load)
    def save(model,tok,destination,**kwargs):
        destination.mkdir(exist_ok=False)
        (destination/'adapter_model.safetensors').write_bytes(b'unit fixture')
        (destination/'adapter_config.json').write_text('{}')
        metadata={'schema_version':1,'model':asdict(kwargs['model_config']),
                  'lora':asdict(kwargs['lora']),'provenance':kwargs['provenance']}
        (destination/'cat_adapter.json').write_text(json.dumps(metadata))
    monkeypatch.setattr(runner,'save_lora_adapter',save)
    return cfg


def test_runner_creates_manifests_not_test_metrics(fixture_run,tmp_path):
    output=runner.run_sft(fixture_run,tmp_path/'run',initial_adapter=None,allow_download=False)
    assert (output/'complete.json').is_file()
    assert (output/'selection.json').is_file()
    assert (output/'adapter-epoch-001/cat_adapter.json').is_file()
    values=json.loads((output/'complete.json').read_text())
    assert values['epochs'][0]['optimizer_steps']==2
    assert 'accuracy' not in values
    with pytest.raises(FileExistsError):
        runner.run_sft(fixture_run,output,initial_adapter=None,allow_download=False)


def test_runner_branch_requires_matching_warmup(fixture_run,tmp_path):
    cfg=fixture_run
    output=runner.run_sft(cfg,tmp_path/'warm',initial_adapter=None,allow_download=False)
    adapter=output/'adapter-epoch-001'
    branch=replace(cfg,initialization='adapter',objective=replace(cfg.objective,strategy='passn',n=4))
    runner.run_sft(branch,tmp_path/'branch',initial_adapter=adapter,allow_download=False)
    wrong=replace(branch,data=replace(branch.data,target='solution'))
    with pytest.raises(ValueError,match='identical'):
        runner.run_sft(wrong,tmp_path/'wrong',initial_adapter=adapter,allow_download=False)
    assert not (tmp_path/'wrong').exists()


def test_runner_requires_adapter_before_training(fixture_run,tmp_path):
    with pytest.raises(ValueError,match='initial-adapter'):
        runner.run_sft(replace(fixture_run,initialization='adapter'),tmp_path/'bad',initial_adapter=None,allow_download=False)


def test_runner_rejects_distributed(fixture_run,tmp_path,monkeypatch):
    monkeypatch.setenv('WORLD_SIZE','2')
    with pytest.raises(ValueError,match='single-process'):
        runner.run_sft(fixture_run,tmp_path/'bad',initial_adapter=None,allow_download=False)
