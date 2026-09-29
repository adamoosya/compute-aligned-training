from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest
import torch

from cat.models.hf import HFModelConfig, LoRASettings
import cat.models.inference as inference
from generation_support import FakeModel, FakeTokenizer


@pytest.fixture
def loader_fixture(monkeypatch,tmp_path):
    calls={}
    base=tmp_path/"base";base.mkdir();(base/"config.json").write_text("{}")
    config=HFModelConfig(str(base))
    tokenizer=FakeTokenizer()
    def load_base(name,**kwargs):
        calls["base"]=(name,kwargs)
        return FakeModel()
    def load_tokenizer(name,**kwargs):
        calls["tokenizer"]=(name,kwargs)
        return tokenizer
    def load_adapter(model,path,**kwargs):
        calls["adapter"]=(path,kwargs)
        return model
    transformers=SimpleNamespace(AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_base),
        AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),BitsAndBytesConfig=lambda **kwargs:kwargs)
    peft=SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=load_adapter))
    monkeypatch.setattr(inference,"_dependencies",lambda:(transformers,peft))
    adapter=tmp_path/"adapter";adapter.mkdir()
    settings=LoRASettings()
    (adapter/"cat_adapter.json").write_text(json.dumps({"schema_version":1,"model":asdict(config),"lora":asdict(settings)}))
    (adapter/"adapter_config.json").write_text(json.dumps({"peft_type":"LORA","task_type":"CAUSAL_LM","r":16,
        "lora_alpha":16,"lora_dropout":0.0,"target_modules":list(settings.target_modules)}))
    (adapter/"adapter_model.safetensors").write_bytes(b"unit fixture, not weights")
    return config,calls,adapter,tokenizer


def test_base_only_has_no_random_adapter(loader_fixture):
    config,calls,_,_=loader_fixture
    model,tokenizer=inference.load_inference_model(config)
    assert "adapter" not in calls
    assert all(not p.requires_grad for p in model.parameters())
    assert model.training is False and model.config.use_cache is True
    assert tokenizer.padding_side == "left"
    assert calls["base"][1]["local_files_only"] is True
    assert calls["base"][1]["trust_remote_code"] is False
    assert calls["base"][1]["use_safetensors"] is True
    assert calls["base"][1]["dtype"] == torch.float32
    assert "torch_dtype" not in calls["base"][1]


def test_adapter_inference_is_frozen(loader_fixture):
    config,calls,adapter,_=loader_fixture
    model,tokenizer=inference.load_inference_model(config,adapter=adapter)
    assert calls["adapter"][1] == {"is_trainable":False,"local_files_only":True}
    assert calls["tokenizer"][0] == str(adapter)
    assert all(not p.requires_grad for p in model.parameters())
    assert len(inference.adapter_identity(adapter,config)["sha256"]) == 3


@pytest.mark.parametrize("bad", ["missing_marker","missing_weights","base_name","rank","actual_rank","task_type","actual_modules"])
def test_bad_adapter_fails_before_model_load(loader_fixture,bad):
    config,calls,adapter,_=loader_fixture
    if bad=="missing_marker": (adapter/"cat_adapter.json").unlink()
    elif bad=="missing_weights": (adapter/"adapter_model.safetensors").unlink()
    elif bad in {"base_name","rank"}:
        value=json.loads((adapter/"cat_adapter.json").read_text())
        if bad=="base_name":value["model"]["name"]="different"
        else:value["lora"]["r"]=-1
        (adapter/"cat_adapter.json").write_text(json.dumps(value))
    else:
        value=json.loads((adapter/"adapter_config.json").read_text())
        if bad=="actual_rank":value["r"]=99
        elif bad=="task_type":value["task_type"]="OTHER"
        else:value["target_modules"]=["different"]
        (adapter/"adapter_config.json").write_text(json.dumps(value))
    with pytest.raises(ValueError):
        inference.load_inference_model(config,adapter=adapter)
    assert not calls


def test_no_gradient_checkpointing_in_inference(loader_fixture):
    config,calls,_,_=loader_fixture
    with pytest.raises(ValueError,match="checkpointing"):
        inference.load_inference_model(replace(config,gradient_checkpointing=True))
    assert not calls


def test_download_flag_explicit(loader_fixture):
    config,calls,_,_=loader_fixture
    inference.load_inference_model(replace(config,name="org/model",revision="b"*40),allow_download=True)
    assert calls["base"][1]["revision"]=="b"*40
    assert calls["base"][1]["local_files_only"] is False


def test_nf4_contract_without_training_preparation(loader_fixture,monkeypatch):
    config,calls,_,_=loader_fixture
    monkeypatch.setattr(HFModelConfig,"preflight",lambda self:torch.device("cuda:0"))
    monkeypatch.setattr(inference.importlib.metadata,"version",lambda name:"0.48.2")
    cfg=replace(config,device="cuda:0",precision="fp16",quantization="nf4")
    inference.load_inference_model(cfg)
    assert calls["base"][1]["device_map"]=={"":0}
    assert calls["base"][1]["quantization_config"]["bnb_4bit_compute_dtype"]==torch.float16


@pytest.mark.parametrize("missing", ["chat_template","eos_token_id"])
def test_bad_tokenizer(loader_fixture,missing):
    config,calls,_,tokenizer=loader_fixture
    setattr(tokenizer,missing,None)
    with pytest.raises(ValueError):
        inference.load_inference_model(config)
    assert "base" not in calls
