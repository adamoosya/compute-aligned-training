from dataclasses import replace
import json
from pathlib import Path

import pytest

from cat.experiments.sft_config import load_sft_config, MathDataConfig, SFTTrainConfig
from cat.experiments.sft import main
from cat.models.hf import HFModelConfig, LoRASettings, require_revision

CONFIGS = sorted((Path(__file__).resolve().parents[1] / 'configs/sft').glob('*.json'))


@pytest.mark.parametrize('path', CONFIGS, ids=lambda p: p.stem)
def test_config_valid_and_dry_run(path, tmp_path, monkeypatch, capsys):
    cfg = load_sft_config(path)
    assert cfg.model.quantization == 'nf4'
    assert cfg.training.microbatch_size * cfg.training.accumulation_steps == 4
    assert cfg.lora.r == 16
    assert cfg.data.revision is None  # Historical commit not inferred from a filename.
    monkeypatch.chdir(tmp_path)
    assert main(['--config', str(path), '--dry-run']) == 0
    assert 'no training started' in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_paper_variants_and_shared_warmups():
    assert len(CONFIGS) == 10
    groups = {family: [load_sft_config(p) for p in CONFIGS if p.name.startswith(family)]
              for family in ['passn', 'majority']}
    for family, cfgs in groups.items():
        warmup = next(c for c in cfgs if c.initialization == 'base')
        assert warmup.training.epochs == 2
        assert warmup.objective.strategy == 'ce'
        for c in cfgs:
            assert c.data == warmup.data
            assert c.lora == warmup.lora
            assert c.objective.reduction == warmup.objective.reduction
    assert sorted(c.objective.n for c in groups['passn'] if c.objective.strategy == 'passn') == [4,16,64]
    assert sorted((c.objective.n, c.objective.k) for c in groups['majority'] if c.objective.strategy == 'majority') == [(8,3),(16,4),(64,26)]


@pytest.mark.parametrize('kwargs', [
    {'name':''}, {'device':'mps'}, {'device':'cuda:-1'}, {'device':'auto'},
    {'precision':'fp16'}, {'precision':'float32'}, {'quantization':'nf4'},
    {'quantization':'int8'}, {'gradient_checkpointing':1}, {'attention':'unknown'},
    {'revision':123}, {'device':'cuda:0','precision':'fp32','quantization':'nf4'},
])
def test_bad_model_config(kwargs):
    with pytest.raises(ValueError):
        HFModelConfig(**({'name':'test/model'} | kwargs))


@pytest.mark.parametrize('kwargs', [
    {'r':0}, {'r':True}, {'alpha':-1}, {'dropout':1}, {'dropout':float('nan')},
    {'dropout':False}, {'target_modules':[]}, {'target_modules':['q_proj','q_proj']},
    {'target_modules':'q_proj'}, {'target_modules':['']},
])
def test_bad_lora_settings(kwargs):
    with pytest.raises(ValueError):
        LoRASettings(**kwargs)


@pytest.mark.parametrize('kwargs', [
    {'dataset_name':''}, {'split':'test'}, {'max_examples':0}, {'max_examples':True},
    {'levels':[]}, {'levels':[0]}, {'levels':[6]}, {'levels':[1,1]}, {'levels':[True]},
    {'target':'missing'}, {'overflow':'skip'}, {'seed':-1}, {'max_length':1},
    {'prompt_template':'missing field'}, {'revision':2},
])
def test_bad_data_config(kwargs):
    with pytest.raises(ValueError):
        MathDataConfig(**kwargs)


@pytest.mark.parametrize('kwargs', [
    {'epochs':0}, {'microbatch_size':False}, {'accumulation_steps':0}, {'seed':-1},
    {'learning_rate':0}, {'learning_rate':float('inf')}, {'weight_decay':-1},
    {'max_grad_norm':0}, {'learning_rate':'x'},
])
def test_bad_train_config(kwargs):
    with pytest.raises(ValueError):
        SFTTrainConfig(**kwargs)


@pytest.mark.parametrize('value', [None,'main','abcd','g'*40,42])
def test_bad_revision(value):
    with pytest.raises(ValueError):
        require_revision(value,'model.revision')


def test_strict_json(tmp_path):
    base=json.loads(CONFIGS[0].read_text())
    path=tmp_path/'config.json'
    path.write_text('{"name":"one","name":"two"}')
    with pytest.raises(ValueError,match='Duplicate'):
        load_sft_config(path)
    for mutate in [lambda x:x.update(typo=1),lambda x:x['model'].update(typo=1),lambda x:x.pop('lora')]:
        cfg=json.loads(json.dumps(base)); mutate(cfg); path.write_text(json.dumps(cfg))
        with pytest.raises(ValueError):
            load_sft_config(path)


def test_requires_explicit_mode():
    with pytest.raises(SystemExit) as e:
        main(['--config',str(CONFIGS[0])])
    assert e.value.code==2
