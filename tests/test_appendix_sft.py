from dataclasses import replace
from pathlib import Path
import copy,json
import pytest
import torch
from torch.nn import functional as F
from cat.objectives.contrastive import pairwise_token_losses
from cat.training.margin import train_margin_epoch
from cat.training.sft import train_sft_epoch,SFTObjective
from cat.testing.tiny import TinyCausalLM
from cat.experiments.appendix_sft import load_appendix_sft,main
ROOT=Path(__file__).resolve().parents[1]


def batch():
    x=torch.tensor([[1,3,5,2],[1,3,4,2]])
    labels=x.clone();labels[:,:2]=-100
    return {'input_ids':x,'labels':labels,'attention_mask':torch.ones_like(x)}


def test_two_token_margin_matches_exact_formula():
    z=torch.tensor([[[2.,1.],[0.,0.]]],requires_grad=True)
    labels=torch.tensor([[-100,0]])
    value=pairwise_token_losses(z,labels,rivals=4)
    torch.testing.assert_close(value,-F.logsigmoid(torch.tensor([1.])))
    value.sum().backward()
    assert z.grad[0,0,0]<0 and z.grad[0,0,1]>0
    assert torch.equal(z.grad[0,1],torch.zeros(2))


def test_very_confident_target_does_not_break_rival_sampling():
    z=torch.tensor([[[10000.,-1000.,-1001.],[0.,0.,0.]]],requires_grad=True)
    v=pairwise_token_losses(z,torch.tensor([[-100,0]]))
    assert bool(torch.isfinite(v).all());v.sum().backward()
    assert bool(torch.isfinite(z.grad).all())


@pytest.mark.parametrize('rivals',[0,-1,True,1.2])
def test_invalid_rival_count(rivals):
    with pytest.raises(ValueError):pairwise_token_losses(torch.randn(1,3,4),torch.tensor([[-100,1,2]]),rivals=rivals)


def test_ce_reduction_matches_existing_trainer():
    torch.manual_seed(22)
    a=TinyCausalLM(vocab_size=7,hidden_size=8);b=copy.deepcopy(a)
    batches=[batch(),{k:v[:1] for k,v in batch().items()},batch()]
    x=train_sft_epoch(a,batches,torch.optim.SGD(a.parameters(),lr=.03),SFTObjective(reduction='token_mean'),accumulation_steps=2,max_grad_norm=.3)
    y=train_margin_epoch(b,batches,torch.optim.SGD(b.parameters(),lr=.03),margin_weight=0,accumulation_steps=2,max_grad_norm=.3)
    assert x.optimizer_steps==y.optimizer_steps==2
    assert y.loss==pytest.approx(x.loss)
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q)


def test_margin_updates_tiny_model():
    torch.manual_seed(7);m=TinyCausalLM(vocab_size=7,hidden_size=8)
    before=copy.deepcopy(m.state_dict())
    x=train_margin_epoch(m,[batch(),batch()],torch.optim.SGD(m.parameters(),lr=.01),accumulation_steps=2)
    assert x.optimizer_steps==1 and x.loss>0
    assert any(not torch.equal(before[k],v) for k,v in m.state_dict().items())


@pytest.mark.parametrize('name',['token_margin_warmup','token_margin_ce','token_margin_cat','token_margin_pairwise','transfer_ce_warmup','transfer_cat_warmup'])
def test_appendix_dry_run_and_config(name,capsys):
    path=ROOT/'configs/appendix'/f'{name}.json'
    c=load_appendix_sft(path);assert c.name==name
    assert main(['--config',str(path),'--dry-run'])==0
    assert 'DRY RUN' in capsys.readouterr().out


def test_appendix_no_unknown_config_fields(tmp_path):
    d=json.loads((ROOT/'configs/appendix/token_margin_warmup.json').read_text());d['made_up']=1
    p=tmp_path/'bad.json';p.write_text(json.dumps(d))
    with pytest.raises(ValueError):load_appendix_sft(p)


def test_real_mistral_pairwise_adapter_update(tmp_path):
    pytest.importorskip('transformers');pytest.importorskip('peft')
    from cat.testing.lora_smoke import _local_base
    from cat.models.hf import HFModelConfig,LoRASettings,load_lora_model
    base=tmp_path/'tiny';_local_base(base)
    m,_=load_lora_model(HFModelConfig(str(base)),LoRASettings(r=4,alpha=8))
    before={n:p.detach().clone() for n,p in m.named_parameters()}
    s=train_margin_epoch(m,[batch()],torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.01),accumulation_steps=1)
    assert s.optimizer_steps==1
    changed=[]
    for n,p in m.named_parameters():
        if not torch.equal(before[n],p):changed.append(n)
        if not p.requires_grad:torch.testing.assert_close(before[n],p,rtol=0,atol=0)
    assert changed and all('lora_' in n for n in changed)

@pytest.mark.parametrize('name',['transfer_cat_warmup','token_margin_warmup'])
def test_appendix_runner_with_local_fixture_and_mock_export(tmp_path,monkeypatch,name):
    from cat.experiments.appendix_sft import run_appendix
    from cat.models.hf import HFModelConfig
    from cat.testing.tiny import ByteChatTokenizer,tiny_math_records
    import cat.experiments.appendix_sft as runner
    c=load_appendix_sft(ROOT/'configs/appendix'/f'{name}.json')
    (tmp_path/'config.json').write_text('{}')  # local-model preflight; loading is mocked below
    b=replace(c.base,model=HFModelConfig(str(tmp_path)),data=replace(c.base.data,max_examples=5,levels=None),
              training=replace(c.base.training,epochs=1))
    c=replace(c,base=b)
    data=tmp_path/'data.jsonl';data.write_text(''.join(json.dumps(x)+'\n' for x in tiny_math_records()))
    model=TinyCausalLM(hidden_size=8)
    monkeypatch.setattr(runner,'load_lora_model',lambda *a,**kw:(model,ByteChatTokenizer()))
    def export(m,t,path,**kwargs):
        path.mkdir();(path/'metadata.json').write_text(json.dumps(kwargs['provenance']))
        return path
    monkeypatch.setattr(runner,'save_lora_adapter',export)
    out=run_appendix(c,tmp_path/'out',local_data=data)
    assert (out/'complete.json').exists()
    m=json.loads((out/'adapter-final/metadata.json').read_text())
    assert m['stage']=='warmup' and m['objective']['strategy']==c.objective.strategy
    with pytest.raises(FileExistsError):run_appendix(c,out,local_data=data)
