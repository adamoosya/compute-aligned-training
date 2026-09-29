from contextlib import nullcontext

import pytest
import torch
from torch import nn

import cat.training.sft as sft


@pytest.mark.parametrize('precision',['fp16','bf16','unknown'])
def test_cpu_amp_not_silently_enabled(precision):
    with pytest.raises(ValueError):
        sft._precision_context(torch.device('cpu'),precision)


def test_fp32_context():
    with sft._precision_context(torch.device('cpu'),'fp32'):
        x=torch.tensor(1.)
    assert x.dtype==torch.float32


class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.logits=nn.Parameter(torch.randn(1,3,5))
    def forward(self,**kwargs): return self.logits


def test_scaler_requires_fp16():
    model=Model(); opt=torch.optim.SGD(model.parameters(),lr=.1)
    with pytest.raises(ValueError,match='only accepted'):
        sft.train_sft_epoch(model,[],opt,sft.SFTObjective(),scaler=object())


def test_scaler_order_with_accumulation(monkeypatch):
    # Exercise scaled step order on CPU without claiming a hardware AMP test.
    monkeypatch.setattr(sft,'_precision_context',lambda *args:nullcontext())
    calls=[]
    class Scaler:
        def is_enabled(self): return True
        def get_scale(self): return 1.0
        def scale(self,x): calls.append('scale'); return x
        def unscale_(self,opt): calls.append('unscale')
        def step(self,opt): calls.append('step'); opt.step()
        def update(self): calls.append('update')
    model=Model(); opt=torch.optim.SGD(model.parameters(),lr=.1)
    batch={'input_ids':torch.tensor([[0,1,2]]),'attention_mask':torch.ones(1,3,dtype=torch.long),
           'labels':torch.tensor([[-100,1,2]])}
    stats=sft.train_sft_epoch(model,[batch,batch,batch],opt,sft.SFTObjective(),
                            accumulation_steps=2,precision='fp16',scaler=Scaler())
    assert stats.optimizer_steps==2
    assert calls==['scale','scale','unscale','step','update','scale','unscale','step','update']


def test_scaler_overflow_does_not_advance_scheduler(monkeypatch):
    monkeypatch.setattr(sft,'_precision_context',lambda *args:nullcontext())
    class Scaler:
        value=2.
        def is_enabled(self): return True
        def get_scale(self): return self.value
        def scale(self,x): return x
        def unscale_(self,opt): pass
        def step(self,opt): pass  # Simulated overflow: optimizer not called.
        def update(self): self.value/=2
    class Scheduler:
        def step(self): raise AssertionError('scheduler advanced on skipped step')
    model=Model(); before=model.logits.detach().clone(); opt=torch.optim.SGD(model.parameters(),lr=.1)
    batch={'input_ids':torch.tensor([[0,1,2]]),'attention_mask':torch.ones(1,3,dtype=torch.long),
           'labels':torch.tensor([[-100,1,2]])}
    stats=sft.train_sft_epoch(model,[batch],opt,sft.SFTObjective(),precision='fp16',
                            scaler=Scaler(),scheduler=Scheduler())
    assert stats.optimizer_steps==0 and stats.skipped_steps==1
    torch.testing.assert_close(before,model.logits)
