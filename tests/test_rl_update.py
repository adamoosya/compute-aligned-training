import copy
from dataclasses import replace
import pytest
import torch
from cat.testing.tiny import TinyCausalLM
from cat.training.rl import (RolloutBatch, CATWeightConfig, RLUpdateConfig,
                             update_rl, policy_loss_terms)


def batch():
    ids=torch.tensor([[0,2,1,0],[0,3,2,1],[0,4,1,0],[0,2,3,1],
                      [0,3,1,0],[0,3,4,1],[0,4,1,0],[0,2,1,0]])
    mask=torch.tensor([[1,1,1,0],[1,1,1,1],[1,1,1,0],[1,1,1,1],
                       [1,1,1,0],[1,1,1,1],[1,1,1,0],[1,1,1,0]])
    labels=ids.clone();labels[:,0]=-100;labels[mask==0]=-100
    r=torch.tensor([[1.,0.,0.,0.],[1.,1.,1.,0.]])
    return RolloutBatch(ids,mask,labels,r,[['yes','a','b','c'],['yes','yes','yes','a']])


@pytest.mark.parametrize('strategy,n,form',[('standard',1,'raw'),('passn',4,'raw'),('passn',4,'log'),
        ('majority',8,'raw'),('majority',8,'log'),('bon',4,'raw')])
def test_real_updates_chunk_invariant_frozen_reference(strategy,n,form):
    torch.manual_seed(14);base=TinyCausalLM(vocab_size=6,hidden_size=8)
    models=[]
    for chunk in [1,3,8]:
        model=copy.deepcopy(base);ref=copy.deepcopy(base).requires_grad_(False)
        # Nonzero reference mismatch makes KL contribute to the gradient.
        with torch.no_grad():ref.head.bias[2]+=.2
        frozen=copy.deepcopy(ref.state_dict())
        stats=update_rl(model,torch.optim.SGD(model.parameters(),lr=.001),batch(),
                         CATWeightConfig(strategy,n,form,normalize=True),
                         RLUpdateConfig(beta=.05,forward_batch_size=chunk,max_grad_norm=10000),reference_model=ref)
        assert stats['rollouts']==8 and stats['prompts']==2 and not stats['skipped']
        assert any(not torch.equal(p,q) for p,q in zip(model.parameters(),base.parameters()))
        for key,v in ref.state_dict().items():torch.testing.assert_close(v,frozen[key],rtol=0,atol=0)
        models.append(model)
    for model in models[1:]:
        for p,q in zip(model.parameters(),models[0].parameters()):
            torch.testing.assert_close(p,q,rtol=2e-6,atol=2e-7)


def test_policy_weight_and_advantage_stop_gradient_and_mask():
    curr=torch.tensor([[-2.,-3.,-1.]],requires_grad=True)
    old=curr.detach().clone().requires_grad_();a=torch.tensor([2.],requires_grad=True)
    w=torch.tensor([3.],requires_grad=True)
    mask=torch.tensor([[1,0,1]])
    loss=policy_loss_terms(curr,old,None,mask,a,w,RLUpdateConfig()).sum();loss.backward()
    torch.testing.assert_close(curr.grad,torch.tensor([[-6.,0.,-6.]]))
    assert a.grad is None and w.grad is None and old.grad is None


def test_clipped_sequence_ratio_for_both_advantage_signs():
    cur=torch.tensor([[-.1],[-2.]])
    old=torch.tensor([[-1.],[-1.]])
    a=torch.tensor([1.,-1.]); w=torch.ones(2)
    v=policy_loss_terms(cur,old,None,torch.ones_like(cur),a,w,RLUpdateConfig(surrogate='clipped'))
    torch.testing.assert_close(v,torch.tensor([-1.2,.8]))


def test_kl_zero_at_reference_and_positive_elsewhere():
    cur=torch.tensor([[-1.,-2.]],requires_grad=True);ref=torch.tensor([[-1.5,-1.5]],requires_grad=True)
    z=torch.zeros(1);one=torch.ones(1);mask=torch.ones_like(cur)
    cfg=RLUpdateConfig(beta=.3)
    loss=policy_loss_terms(cur,cur.detach(),ref,mask,z,one,cfg).sum()
    assert loss>0;loss.backward();assert ref.grad is None
    assert policy_loss_terms(cur,cur.detach(),cur.detach(),mask,z,one,cfg).item()==0


def test_failure_clears_gradients_and_does_not_step():
    model=TinyCausalLM(vocab_size=6,hidden_size=8)
    before=copy.deepcopy(model.state_dict());b=batch();b.rewards[0,0]=float('nan')
    with pytest.raises(ValueError):
        update_rl(model,torch.optim.SGD(model.parameters(),lr=.01),b,CATWeightConfig(),RLUpdateConfig())
    for key,v in model.state_dict().items():torch.testing.assert_close(v,before[key],rtol=0,atol=0)


def test_reference_must_be_separate_and_frozen():
    m=TinyCausalLM(vocab_size=6,hidden_size=8);opt=torch.optim.SGD(m.parameters(),lr=.01)
    with pytest.raises(ValueError,match='distinct'):
        update_rl(m,opt,batch(),CATWeightConfig(),RLUpdateConfig(beta=.1),reference_model=m)
    with pytest.raises(ValueError,match='frozen'):
        update_rl(m,opt,batch(),CATWeightConfig(),RLUpdateConfig(beta=.1),reference_model=copy.deepcopy(m))


def test_batch_does_not_allow_hidden_target_change():
    b=batch();b.labels[0,1]=5
    with pytest.raises(ValueError,match='actual generated'):b.validate()


def test_scheduler_once_per_optimizer_update():
    model=TinyCausalLM(vocab_size=6,hidden_size=8);opt=torch.optim.SGD(model.parameters(),lr=.01)
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:1)
    update_rl(model,opt,batch(),CATWeightConfig(),RLUpdateConfig(),scheduler=scheduler)
    assert scheduler.last_epoch==1
