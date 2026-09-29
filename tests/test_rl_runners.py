import copy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from cat.experiments.rl_config import load_rl_config, RLRunConfig
from cat.experiments.rl import main, run_rl, _protein_warmup, _train_loop
from cat.models.hf import HFModelConfig
from cat.testing.rl_protein_smoke import run_rl_protein_smoke
from cat.testing.tiny import TinyCausalLM

ROOT=Path(__file__).resolve().parents[1]
CONFIGS=sorted(list((ROOT/'configs/rl').glob('*.json'))+list((ROOT/'configs/protein').glob('*.json')))


@pytest.mark.parametrize('path',CONFIGS,ids=lambda p:p.stem)
def test_rl_configs_dry_run_no_loading(path,monkeypatch,capsys):
    monkeypatch.setattr('cat.experiments.rl.run_rl',lambda *a,**k:pytest.fail('Dry run attempted training'))
    cfg=load_rl_config(path)
    assert cfg.model.revision is None and cfg.requirements()
    assert main(['--config',str(path),'--dry-run'])==0
    assert 'DRY RUN' in capsys.readouterr().out


def test_wrong_task_and_no_execute_rejected(capsys):
    p=ROOT/'configs/rl/math_passn_baseline.json'
    assert main(['--config',str(p),'--dry-run'],task_filter='protein')==1
    with pytest.raises(SystemExit):main(['--config',str(p)])


def test_output_not_overwritten_before_any_model_load(tmp_path,monkeypatch):
    c=load_rl_config(CONFIGS[0]);before=tmp_path/'old';before.mkdir();(before/'keep').write_text('original')
    monkeypatch.setattr(HFModelConfig,'preflight',lambda *a:pytest.fail('Should reject before loading'))
    with pytest.raises(FileExistsError):run_rl(c,before)
    assert (before/'keep').read_text()=='original'


def test_conditional_rejects_cross_prompt_history():
    from cat.training.rl import CATWeightConfig
    c=load_rl_config(ROOT/'configs/protein/protein_conditional_bon4.json')
    with pytest.raises(ValueError,match='within-prompt'):
        replace(c,weights=replace(c.weights,quantile='history'))


def test_warmup_executes_exact_optimizer_steps_completion_only(tmp_path):
    class Tok:
        eos_token_id=1;pad_token_id=1
        def encode(self,text,**kwargs):return [ord(c)%20+2 for c in text]
    c=load_rl_config(ROOT/'configs/protein/conditional_warmup.json')
    c=replace(c,model=replace(c.model,device='cpu',precision='fp32'),
      update=replace(c.update,precision='fp32'),data={**c.data,'count':8},
      training={**c.training,'warmup_steps':2,'prompts_per_step':2,'learning_rate':.001},
      sampling=replace(c.sampling,num_candidates=2,chunk_size=2))
    model=TinyCausalLM(vocab_size=24,hidden_size=8);before=copy.deepcopy(model.state_dict())
    stats=_protein_warmup(model,Tok(),c,tmp_path)
    assert stats['optimizer_steps']==2 and stats['sequences']==8
    assert any(not torch.equal(v,before[k]) for k,v in model.state_dict().items())
    assert (tmp_path/'selection.json').exists()


def test_complete_offline_training_smoke(tmp_path):
    root=run_rl_protein_smoke(tmp_path)
    summary=json.loads((root/'summary.json').read_text())
    assert summary['kind']=='smoke_only' and len(summary['checks'])==9
    assert all(c['updates']==3 for c in summary['checks'])
