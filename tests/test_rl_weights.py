import math
import pytest
import torch
from cat.training.rl import (CATWeightConfig, RLUpdateConfig, group_advantages,
    adaptive_majority_k, RewardHistory, within_prompt_quantiles, cat_weights)


def test_advantages_are_within_prompt_and_detached():
    r = torch.tensor([[0., 1., 1.], [12., 14., 16.]], requires_grad=True)
    a = group_advantages(r)
    assert not a.requires_grad
    torch.testing.assert_close(a.mean(1), torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(a[0], group_advantages(r[:1])[0])


def test_silence_penalty_constant_and_zero_variance():
    r = torch.tensor([[0., 0.], [1., 1.]])
    a = group_advantages(r, all_wrong_penalty=.5)
    torch.testing.assert_close(a, torch.tensor([[-.5, -.5], [0., 0.]]))
    with pytest.raises(ValueError):
        group_advantages(torch.tensor([[2., 2.]]), all_wrong_penalty=.5)


@pytest.mark.parametrize('n,c,m,expected', [(8,0,8,2),(8,1,8,2),(8,3,8,4),(8,8,8,5),(4,8,8,3)])
def test_adaptive_threshold(n,c,m,expected):
    keys=['bad']*c+['good']*(m-c)
    correct=[False]*c+[True]*(m-c)
    assert adaptive_majority_k(keys,correct,n)==expected


def test_invalid_answers_are_not_removed_from_threshold_denominator():
    assert adaptive_majority_k([None,None,'yes','yes'],[False,False,True,True],8)==5


@pytest.mark.parametrize('form', ['raw','log'])
@pytest.mark.parametrize('n', [2,4,8,16])
def test_pass_weights_independent_formula(n,form):
    r=torch.tensor([[1.,0.,1.,0.]],dtype=torch.float64)
    lp=torch.log(torch.tensor([[.1,.2,.5,.8]],dtype=torch.float64))
    w,_=cat_weights(r,lp,CATWeightConfig('passn',n,form,probability='sequence'))
    p=lp.exp(); expected=n*(1-p)**(n-1)
    if form=='log':expected=expected*p/(1-(1-p)**n)
    torch.testing.assert_close(w,expected,rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('form',['raw','log'])
def test_majority_weight_matches_binomial_sum(form):
    r=torch.tensor([[1.,0.,0.,0.],[1.,1.,1.,0.]],dtype=torch.float64)
    keys=[['yes','a','b','c'],['yes','yes','yes','a']]
    w,d=cat_weights(r,torch.full_like(r,-10),CATWeightConfig('majority',8,form),answer_keys=keys)
    for i,k in enumerate(d['thresholds']):
        p=(r[i].sum().item()+1)/6
        a=8*math.comb(7,k-1)*p**(k-1)*(1-p)**(8-k)
        tail=sum(math.comb(8,j)*p**j*(1-p)**(8-j) for j in range(k,9))
        assert w[i,0].item()==pytest.approx(a if form=='raw' else a*p/tail)


def test_batch_normalization_preserves_prompt_ratios_and_one_prompt_stays_raw():
    r=torch.tensor([[1.,0.,0.,0.],[1.,1.,1.,0.]])
    lp=torch.full_like(r,-1)
    cfg=CATWeightConfig('passn',4,normalize=True)
    w,d=cat_weights(r,lp,cfg)
    raw,_=cat_weights(r,lp,CATWeightConfig('passn',4))
    assert d['normalized']
    assert float(w.mean())==pytest.approx(1)
    assert float(w[0,0]/w[1,0])==pytest.approx(float(raw[0,0]/raw[1,0]))
    one,d=cat_weights(r[:1],lp[:1],cfg)
    assert not d['normalized']
    torch.testing.assert_close(one,raw[:1])


def test_history_is_finite_fifo_and_query_does_not_mutate():
    h=RewardHistory(3);r=torch.tensor([[1.,2.]])
    torch.testing.assert_close(h.quantiles(r),torch.full_like(r,.5))
    h.append(torch.tensor([[0.,2.,4.,6.]]))
    assert list(h.values)==[2.,4.,6.]
    torch.testing.assert_close(h.quantiles(r),torch.tensor([[.2,.2]]))
    assert list(h.values)==[2.,4.,6.]


def test_conditional_ranks_ties_and_other_prompts_are_independent():
    r=torch.tensor([[2.,2.,9.],[100.,200.,300.]])
    q=within_prompt_quantiles(r)
    torch.testing.assert_close(q[0],torch.tensor([.25,.25,.75]))
    torch.testing.assert_close(q[:1],within_prompt_quantiles(r[:1]))
    w,_=cat_weights(r,torch.full_like(r,-1),CATWeightConfig('bon',4))
    torch.testing.assert_close(w,4*q**3)


@pytest.mark.parametrize('kwargs',[{'strategy':'unknown'},{'n':True},{'n':0},
    {'strategy':'majority','n':1},{'strategy':'bon','form':'log'}, {'normalize':1},
    {'clip':0},{'clip':float('nan')},{'quantile':'across_prompts'}, {'form':'wrong'}])
def test_weight_config_validation(kwargs):
    with pytest.raises((ValueError,TypeError)):CATWeightConfig(**kwargs)


@pytest.mark.parametrize('kwargs',[{'surrogate':'ppoish'},{'beta':-1},{'precision':'fp8'},
    {'clip_ratio':1},{'advantage_epsilon':0},{'forward_batch_size':0},{'all_wrong_penalty':-1}])
def test_update_config_validation(kwargs):
    with pytest.raises(ValueError):RLUpdateConfig(**kwargs)


def test_zero_normalization_no_invented_unit_weight():
    # Extreme raw sequence p=1 is explicitly floored at 1-1e-10; underflows for huge N.
    r=torch.ones(2,2)
    w,d=cat_weights(r,torch.zeros_like(r),CATWeightConfig('passn',128,normalize=True,probability='sequence'))
    assert torch.equal(w,torch.zeros_like(w)) and not d['normalized']
