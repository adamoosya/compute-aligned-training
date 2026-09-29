"""Loader contracts use fakes; the separate real integration test is optional."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import cat.models.hf as hf


class FakeTokenizer:
    chat_template='template'
    eos_token_id=2
    eos_token='eos'
    pad_token_id=2
    padding_side='left'
    def __len__(self): return 64
    def save_pretrained(self,path):
        Path(path,'tokenizer_config.json').write_text('{}')


class FakeBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(64,4)
        self.q_proj=nn.Linear(4,4)
        self.k_proj=nn.Linear(4,4)
        self.v_proj=nn.Linear(4,4)
        self.o_proj=nn.Linear(4,4)
        self.config=SimpleNamespace(use_cache=True,pad_token_id=None)
    def get_input_embeddings(self): return self.emb
    def gradient_checkpointing_enable(self,**kwargs): self.gc=kwargs


class FakePeft(nn.Module):
    def __init__(self,base):
        super().__init__(); self.base=base
        for p in base.parameters(): p.requires_grad=False
        self.lora_A=nn.Parameter(torch.ones(1,dtype=torch.float16))
        self.peft_config={}
        self.config=base.config
    def save_pretrained(self,path,**kwargs):
        self.saved_kwargs=kwargs
        Path(path,'adapter_config.json').write_text('{}')
        Path(path,'adapter_model.safetensors').write_bytes(b'fake adapter for unit test only')


@pytest.fixture
def fake_hf(monkeypatch,tmp_path):
    calls={}
    def load_base(name,**kwargs):
        calls['base']=(name,kwargs); return FakeBase()
    def load_tokenizer(name,**kwargs):
        calls['tokenizer']=(name,kwargs); return FakeTokenizer()
    def make_peft(base,config):
        calls['lora']=config; return FakePeft(base)
    def load_peft(base,path,**kwargs):
        calls['adapter']=(path,kwargs); return FakePeft(base)
    transformers=SimpleNamespace(AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_base),
        AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
        BitsAndBytesConfig=lambda **kwargs:kwargs)
    peft=SimpleNamespace(LoraConfig=lambda **kwargs:kwargs,get_peft_model=make_peft,
        PeftModel=SimpleNamespace(from_pretrained=load_peft))
    monkeypatch.setattr(hf,'_dependencies',lambda:(transformers,peft))
    folder=tmp_path/'base'; folder.mkdir(); (folder/'config.json').write_text('{}')
    return hf.HFModelConfig(str(folder)), calls, transformers, peft


def test_load_local_and_frozen_base(fake_hf):
    cfg,calls,_,_=fake_hf
    model,tok=hf.load_lora_model(cfg,hf.LoRASettings())
    assert tok.padding_side=='right'
    assert not model.config.use_cache
    assert calls['base'][1]['local_files_only'] is True
    assert calls['base'][1]['use_safetensors'] is True
    assert calls['base'][1]['trust_remote_code'] is False
    assert 'device_map' not in calls['base'][1]
    assert all('lora_' in n and p.dtype==torch.float32 for n,p in model.named_parameters() if p.requires_grad)
    assert all(not p.requires_grad for p in model.base.parameters())


def test_gradient_checkpointing(fake_hf):
    cfg,_,_,_=fake_hf
    model,_=hf.load_lora_model(replace(cfg,gradient_checkpointing=True),hf.LoRASettings())
    assert model.base.gc=={'gradient_checkpointing_kwargs':{'use_reentrant':False}}


def test_missing_module_refused(fake_hf):
    cfg,_,_,_=fake_hf
    with pytest.raises(ValueError,match='absent'):
        hf.load_lora_model(cfg,hf.LoRASettings(target_modules=('not_here',)))


def test_no_guessing_chat_template(fake_hf):
    cfg,_,transformers,_=fake_hf
    tok=FakeTokenizer(); tok.chat_template=None
    transformers.AutoTokenizer.from_pretrained=lambda *a,**k:tok
    with pytest.raises(ValueError,match='chat template'):
        hf.load_lora_model(cfg,hf.LoRASettings())


def test_no_eos_refused(fake_hf):
    cfg,_,transformers,_=fake_hf
    tok=FakeTokenizer(); tok.eos_token_id=None
    transformers.AutoTokenizer.from_pretrained=lambda *a,**k:tok
    with pytest.raises(ValueError,match='EOS'):
        hf.load_lora_model(cfg,hf.LoRASettings())


def test_export_and_reload_contract(fake_hf,tmp_path):
    cfg,calls,_,_=fake_hf; lora=hf.LoRASettings()
    model,tok=hf.load_lora_model(cfg,lora)
    path=hf.save_lora_adapter(model,tok,tmp_path/'adapter',model_config=cfg,lora=lora)
    assert (path/'cat_adapter.json').is_file()
    assert model.saved_kwargs=={'safe_serialization':True,'save_embedding_layers':False}
    hf.load_lora_model(cfg,lora,initial_adapter=path)
    assert calls['adapter'][1]['is_trainable'] is True
    assert calls['tokenizer'][0]==str(path)
    with pytest.raises(FileExistsError):
        hf.save_lora_adapter(model,tok,path,model_config=cfg,lora=lora)


@pytest.mark.parametrize('change', ['model','lora','incomplete','missing_metadata'])
def test_adapter_mismatches_refused(change,fake_hf,tmp_path):
    cfg,_,_,_=fake_hf; lora=hf.LoRASettings()
    model,tok=hf.load_lora_model(cfg,lora)
    path=hf.save_lora_adapter(model,tok,tmp_path/'adapter',model_config=cfg,lora=lora)
    if change=='model':
        m=json.loads((path/'cat_adapter.json').read_text()); m['model']['name']='other'
        (path/'cat_adapter.json').write_text(json.dumps(m))
    elif change=='lora': lora=replace(lora,r=32)
    elif change=='incomplete': (path/'adapter_model.safetensors').unlink()
    else: (path/'cat_adapter.json').unlink()
    with pytest.raises(ValueError):
        hf.load_lora_model(cfg,lora,initial_adapter=path)


def test_revisions_and_hardware_checked_before_import(monkeypatch):
    def forbidden(): raise AssertionError('should not reach dependency import')
    monkeypatch.setattr(hf,'_dependencies',forbidden)
    with pytest.raises(ValueError,match='SHA'):
        hf.load_lora_model(hf.HFModelConfig('org/model'),hf.LoRASettings())
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    cfg=hf.HFModelConfig('org/model',revision='a'*40,device='cuda:0',precision='fp16',quantization='nf4')
    with pytest.raises(RuntimeError,match='CUDA'):
        hf.load_lora_model(cfg,hf.LoRASettings())


def test_nf4_loader_call_contract(fake_hf,monkeypatch):
    cfg,calls,_,peft=fake_hf
    monkeypatch.setattr(hf.HFModelConfig,'preflight',lambda self:torch.device('cuda:0'))
    original=hf.importlib.metadata.version
    monkeypatch.setattr(hf.importlib.metadata,'version',lambda n:'0.48.2' if n=='bitsandbytes' else original(n))
    def prepare(model,**kwargs): calls['prepare']=kwargs; return model
    peft.prepare_model_for_kbit_training=prepare
    cfg=replace(cfg,device='cuda:0',precision='fp16',quantization='nf4',gradient_checkpointing=True)
    hf.load_lora_model(cfg,hf.LoRASettings())
    assert calls['base'][1]['device_map']=={'':0}
    assert calls['base'][1]['quantization_config']['bnb_4bit_quant_type']=='nf4'
    assert calls['prepare']['use_gradient_checkpointing'] is True


def test_failed_export_has_no_complete_marker(fake_hf,tmp_path):
    cfg,_,_,_=fake_hf; lora=hf.LoRASettings()
    model,tok=hf.load_lora_model(cfg,lora)
    def broken(*a,**k): raise OSError('export failed')
    model.save_pretrained=broken
    with pytest.raises(OSError):
        hf.save_lora_adapter(model,tok,tmp_path/'failed',model_config=cfg,lora=lora)
    assert not (tmp_path/'failed/cat_adapter.json').exists()


def test_real_hf_lora_round_trip(tmp_path,monkeypatch):
    # This test actually imports Transformers and PEFT; the fakes above are not used.
    monkeypatch.setenv('HF_HUB_OFFLINE','1')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE','1')
    pytest.importorskip('transformers',reason='install the hf extra for real backend test')
    pytest.importorskip('peft',reason='install the hf extra for real backend test')
    from cat.testing.lora_smoke import run_lora_smoke
    path=run_lora_smoke(tmp_path)
    summary=json.loads((path/'summary.json').read_text())
    assert len(summary['results'])==3
    assert all(r['adapter_updates'] and r['base_frozen'] and r['save_reload'] for r in summary['results'])
