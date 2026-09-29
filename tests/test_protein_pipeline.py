import itertools
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from cat.data.protein import make_protein_examples, protein_manifest
from cat.rewards.protein import hydrophobicity, protein_reward
from cat.evaluation.protein import expected_max_reward, summarize_protein_rewards
from cat.training.rollouts import RLPrompt, collect_rollouts
from cat.generation import SamplingConfig


def test_synthetic_streams_reproducible_disjoint_and_gold_rounding():
    a=make_protein_examples(60,seed=123,split='train')
    assert a==make_protein_examples(60,seed=123,split='train')
    b=make_protein_examples(60,seed=123,split='test')
    assert not ({x.sequence for x in a} & {x.sequence for x in b})
    assert protein_manifest(a,seed=123)['count']==60
    for ex in a:
        assert abs(hydrophobicity(ex.target)-(1-ex.input_h)) <= .50001/len(ex.target)
        assert ex.sequence in ex.prompt


@pytest.mark.parametrize('sequence',['','BAD?','AAAA*','123','\n\t'])
def test_invalid_proteins_get_penalty(sequence):
    assert protein_reward(sequence)==-5
    with pytest.raises(ValueError):hydrophobicity(sequence)


@pytest.mark.parametrize('hydro_count',[0,3,5,6,7,10])
def test_unconditional_reward_exact_formula(hydro_count):
    h=hydro_count/10
    seq='A'*hydro_count+'D'*(10-hydro_count)
    expected=4*math.exp(-(h-.35)**2/.05)+12*math.exp(-(h-.75)**2/.015)-(2 if .45<h<.6 else 0)
    assert protein_reward(seq)==pytest.approx(expected)


def test_conditional_reward_uses_input_and_penalty():
    assert protein_reward('A'*8+'D'*2,input_h=.2)==pytest.approx(10)
    assert protein_reward('A'*8+'D'*2,input_h=.8)==pytest.approx(10*math.exp(-.36/.05)-2)
    assert protein_reward('AAAA',input_h=.2)==-5
    assert protein_reward('AAAAA',input_h=.2)>-5
    assert protein_reward('a a a a a a a a d d',input_h=.2)==pytest.approx(10)
    with pytest.raises(ValueError):protein_reward('AAAAA',input_h=float('nan'))


@pytest.mark.parametrize('pool',[[-3.,1.,2.],[1.,1.,4.],[-4.,-2.]])
@pytest.mark.parametrize('n',[1,2,4])
def test_expected_max_matches_enumeration(pool,n):
    draws=list(itertools.product(pool,repeat=n))
    exact=sum(max(x) for x in draws)/len(draws)
    assert expected_max_reward(pool,n)==pytest.approx(exact)


def test_protein_uncertainty_is_across_prompts_not_bootstrap_trials():
    s=summarize_protein_rewards([[1.,3.],[3.,5.]],[1,2])
    assert s['budgets']['1']['mean']==3
    assert s['budgets']['1']['prompt_standard_error']==1
    assert summarize_protein_rewards([[1.,3.]],[1])['budgets']['1']['prompt_standard_error'] is None


@pytest.mark.parametrize('task',['math','protein_conditional','protein_unconditional'])
def test_rollout_collection_preserves_tokens_eos_and_rewards(monkeypatch,task):
    import cat.training.rollouts as mod
    texts=['2','3',''] if task=='math' else ['AAAAAAAAAA','DDDDDDDDDD','']
    def sample(model,tok,example,config,prompt_ids=None):
        return SimpleNamespace(completions=tuple(texts)), {'prompt_token_ids':[5,6],
          'attempts':[{'token_ids':[2,1]},{'token_ids':[3,4,1]},{'token_ids':[1]}]}
    monkeypatch.setattr(mod,'sample_example',sample)
    cfg=SamplingConfig(num_candidates=3,chunk_size=3)
    prompts=[RLPrompt('a','question',answer='2',input_h=.2),RLPrompt('b','other',answer='2',input_h=.8)]
    b,records=collect_rollouts(None,SimpleNamespace(pad_token_id=1),prompts,cfg,step=0,task=task)
    assert b.input_ids.shape==(6,5) and b.rewards.shape==(2,3)
    assert b.labels[0].tolist()==[-100,-100,2,1,-100]
    assert b.labels[2].tolist()==[-100,-100,1,-100,-100]
    assert len(records)==2 and records[0]['completions'][-1]==''
    if task=='math':
        assert b.rewards[0].tolist()==[1.,0.,0.]
    else:
        assert b.rewards[0,-1]==-5
    if task=='protein_conditional':assert b.rewards[0,0]!=b.rewards[1,0]
