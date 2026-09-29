"""Optional real backends; tiny local architectures, never pretrained downloads."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import pytest
import torch
from cat.training.rl import CATWeightConfig, RLUpdateConfig, RolloutBatch, update_rl
from cat.models.hf import HFModelConfig, LoRASettings, load_lora_model, save_lora_adapter
from cat.models.inference import load_inference_model
from cat.models.protein import load_protein_model, save_protein_model


def _batch(vocab):
    ids=torch.tensor([[1,6,8,2],[1,6,9,2],[1,6,10,2],[1,6,11,2]])
    labels=ids.clone();labels[:,:2]=-100
    return RolloutBatch(ids,torch.ones_like(ids),labels,torch.tensor([[1.,0.],[0.,1.]]))


def test_real_lora_rl_update_and_frozen_reference(tmp_path):
    pytest.importorskip('transformers');pytest.importorskip('peft')
    from cat.testing.lora_smoke import _local_base
    base=tmp_path/'tiny';_local_base(base)
    cfg=HFModelConfig(str(base));lora=LoRASettings(r=4,alpha=8)
    model,tok=load_lora_model(cfg,lora)
    checkpoint=save_lora_adapter(model,tok,tmp_path/'warm',model_config=cfg,lora=lora,provenance={'stage':'warmup'})
    reference,_=load_inference_model(cfg,adapter=checkpoint)
    before={k:p.detach().clone() for k,p in model.named_parameters()}
    stats=update_rl(model,torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01),
       _batch(18),CATWeightConfig('passn',4),RLUpdateConfig(beta=.05,forward_batch_size=2),reference_model=reference)
    assert stats['gradient_norm']>0
    changed=[]
    for k,p in model.named_parameters():
        if not torch.equal(before[k],p):changed.append(k)
        if not p.requires_grad:torch.testing.assert_close(p,before[k],rtol=0,atol=0)
    assert changed and all('lora_' in k for k in changed)
    out=save_lora_adapter(model,tok,tmp_path/'trained',model_config=cfg,lora=lora,provenance={'stage':'rl'})
    reload,_=load_inference_model(cfg,adapter=out)
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(_batch(18).input_ids).logits,reload(_batch(18).input_ids).logits)


def test_real_gpt2_protein_warmup_rl_evaluation_pipeline(tmp_path):
    t=pytest.importorskip('transformers');pytest.importorskip('tokenizers')
    from tokenizers import Tokenizer, models, pre_tokenizers
    from cat.experiments.rl import run_rl,evaluate_protein
    from cat.experiments.rl_config import load_rl_config
    root=Path(__file__).resolve().parents[1]
    vocab=['[UNK]','[BOS]','[EOS]','[PAD]','Input:','Output:','AAAAAAAAAA','DDDDDDDDDD','AAAADDDDDD','AAAAAAADDD']
    engine=Tokenizer(models.WordLevel({s:i for i,s in enumerate(vocab)},unk_token='[UNK]'))
    engine.pre_tokenizer=pre_tokenizers.WhitespaceSplit()
    tok=t.PreTrainedTokenizerFast(tokenizer_object=engine,unk_token='[UNK]',bos_token='[BOS]',eos_token='[EOS]',pad_token='[PAD]')
    model=t.GPT2LMHeadModel(t.GPT2Config(vocab_size=len(tok),n_embd=16,n_layer=1,n_head=2,
               n_positions=64,bos_token_id=1,eos_token_id=2,pad_token_id=3,resid_pdrop=0,embd_pdrop=0,attn_pdrop=0))
    base=tmp_path/'base';model.save_pretrained(base,safe_serialization=True);tok.save_pretrained(base)
    c=load_rl_config(root/'configs/protein/conditional_warmup.json')
    c=replace(c,model=HFModelConfig(str(base)),update=replace(c.update,precision='fp32'),
              data={**c.data,'count':4},training={**c.training,'warmup_steps':1,'prompts_per_step':2},
              sampling=replace(c.sampling,num_candidates=2,chunk_size=2,max_prompt_tokens=20,max_new_tokens=2))
    warm=run_rl(c,tmp_path/'warm')
    c=replace(c,stage='rl',weights=CATWeightConfig('bon',2,normalize=True),
               update=replace(c.update,beta=.05),training={**c.training,'max_steps':1})
    trained=run_rl(c,tmp_path/'rl',initial_model=warm/'model-final')
    assert (trained/'complete.json').exists()
    out=evaluate_protein(c,trained/'model-final',tmp_path/'eval',candidates=2,num_prompts=2)
    assert (out/'complete.json').exists()
    summary=json.loads((out/'summary.json').read_text())
    assert summary['prompts']==2
