"""Appendix-only CE/pairwise training; main SFT and RL loops are unchanged."""
from __future__ import annotations
from itertools import islice
import math
import torch
from cat.objectives.contrastive import pairwise_token_losses
from cat.training.sft import (_batch_on_device,_check_labels,_device,_model_logits,
                              _precision_context,completion_log_probs,SFTStats)


def train_margin_epoch(model,batches,optimizer,*,margin_weight=.5,rivals=4,
                       accumulation_steps=4,max_grad_norm=.3,device=None,
                       precision='fp32',scaler=None,scheduler=None):
    if (isinstance(margin_weight,bool) or not isinstance(margin_weight,(int,float))
        or not math.isfinite(margin_weight) or not 0<=margin_weight<=1):
        raise ValueError('margin_weight must be in [0,1]')
    if type(accumulation_steps) is not int or accumulation_steps<1:raise ValueError('Invalid accumulation')
    if type(rivals) is not int or rivals<1:raise ValueError('Invalid rival count')
    if max_grad_norm<=0 or not math.isfinite(max_grad_norm):raise ValueError('Invalid clip norm')
    device=_device(model,device);_precision_context(device,precision)
    if precision=='fp16' and (scaler is None or not scaler.is_enabled()):raise ValueError('fp16 requires GradScaler')
    if precision!='fp16' and scaler is not None:raise ValueError('Unexpected GradScaler')
    params=[p for g in optimizer.param_groups for p in g['params'] if p.requires_grad]
    if not params:raise ValueError('No trainable parameters')
    iterator=iter(batches);model.train();num=0.;tokens=sequences=steps=skips=0
    try:
        while True:
            window=list(islice(iterator,accumulation_steps))
            if not window:break
            counts=[_check_labels(b['labels'],b['attention_mask']) for b in window]
            denominator=sum(int(c.sum()) for c in counts)
            optimizer.zero_grad(set_to_none=True)
            for b in window:
                b=_batch_on_device(b,device)
                with _precision_context(device,precision):z=_model_logits(model,b)
                score=completion_log_probs(z,b['labels'],b['attention_mask'])
                ce=-score.log_prob.sum()
                pair=pairwise_token_losses(z,b['labels'],b['attention_mask'],rivals=rivals).sum() if margin_weight else ce*0
                numerator=(1-margin_weight)*ce+margin_weight*pair
                if not bool(torch.isfinite(numerator)):raise ValueError('Nonfinite margin loss')
                loss=numerator/denominator
                if scaler is None:loss.backward()
                else:scaler.scale(loss).backward()
                num+=float(numerator.detach())
            if scaler is not None:scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params,max_grad_norm,error_if_nonfinite=scaler is None)
            took=True
            if scaler is None:optimizer.step()
            else:
                previous=scaler.get_scale();scaler.step(optimizer);scaler.update();took=scaler.get_scale()>=previous
            if took:
                steps+=1
                if scheduler is not None:scheduler.step()
            else:skips+=1
            tokens+=denominator;sequences+=sum(c.numel() for c in counts)
    finally:optimizer.zero_grad(set_to_none=True)
    if not tokens:raise ValueError('Empty training loader')
    return SFTStats(num/tokens,sequences,tokens,steps,skips)
