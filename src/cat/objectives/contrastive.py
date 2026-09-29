"""Sampled token-level pairwise margin from Appendix A.7."""
from __future__ import annotations
import torch
from torch.nn import functional as F
from cat.training.sft import completion_log_probs


def pairwise_token_losses(logits, labels, attention_mask=None, *, rivals=4, generator=None):
    """Return one loss per supervised token; sample rivals with replacement.

    Rival selection is detached. Gold tokens are masked *before* the softmax,
    so a very confident gold prediction cannot underflow all rival probabilities.
    """
    completion_log_probs(logits, labels, attention_mask)  # shared shape/masking validation
    if type(rivals) is not int or rivals < 1:
        raise ValueError("rivals must be a positive integer")
    work=logits.float() if logits.dtype in (torch.float16,torch.bfloat16) else logits
    valid=labels[:,1:] != -100
    z=work[:,:-1,:][valid]
    gold=labels[:,1:][valid]
    if not bool(torch.isfinite(z).all()):raise ValueError("Nonfinite logits")
    with torch.no_grad():
        negative=z.detach().clone()
        negative.scatter_(1,gold[:,None],float('-inf'))
        probabilities=negative.softmax(dim=-1)
        idx=torch.multinomial(probabilities,rivals,replacement=True,generator=generator)
    margin=z.gather(1,gold[:,None])-z.gather(1,idx)
    return -F.logsigmoid(margin).mean(dim=-1)
