# Objective reference

This module implements Table 1 of *Compute Aligned Training: Optimizing for
Test Time Inference*, with the fixed-threshold formulation in Appendix A.3
and the Best-of-N rank weight in Appendix B.2.3. It is a new implementation
of those formulas, not the recovered historical training scripts.

## Inputs

All numerical inputs are dense, nonempty PyTorch tensors. Integer budgets
`n` and thresholds `k` are Python integers, with `n >= 1` and `1 <= k <= n`.
A different dynamic threshold can be chosen by a trainer for each prompt;
this module does not estimate that threshold.

- `*_probability(p, ...)` and `*_weight(p, ...)` take probabilities in `[0,1]`.
- `*_log_probability(log_p, ...)`, `*_sft_loss(log_p, ...)`, and
  `*_sft_weight_from_logp(log_p, ...)` take finite natural log-probabilities
  `log_p <= 0`. Do not pass positive negative-log-likelihoods to these functions.
- `best_of_n_rl_weight(quantile, n)` takes a lower-tail reward probability in
  `[0,1]`, not a raw reward, an integer rank, or a sequence probability.

Float64 stays float64. Float16 and bfloat16 arithmetic is promoted to float32;
float32 stays float32. Functions preserve input device and shape and do not
modify inputs. Current tests run on CPU. Input validation reads tensor values;
compilation and GPU performance have not been tuned in this first module.

## Pass@N

For single-sample success probability `p`,

```text
S(p)       = 1 - (1-p)^n
SFT loss   = -log S(p)
SFT weight = n*p*(1-p)^(n-1) / S(p)
RL weight  = n*(1-p)^(n-1)
```

Functions: `passn_probability`, `passn_log_probability`, `passn_sft_loss`,
`passn_sft_weight`, `passn_sft_weight_from_logp`, `passn_rl_weight`.

For `n=1`, the loss is ordinary sequence NLL and both weights are one.
The SFT weight uses its continuous value one at `p=0`.
When `n*p` is below machine epsilon, the log-success implementation uses
`log(n) + log_p`; the omitted correction is below floating-point precision.
This preserves the loss and its gradient even for `log_p=-10000`.

## Fixed-threshold Majority Vote

For a fixed threshold `k`,

```text
B(p)       = sum_{i=k}^n C(n,i)*p^i*(1-p)^(n-i)
SFT loss   = -log B(p)
SFT weight = k*C(n,k)*p^k*(1-p)^(n-k) / B(p)
RL weight  = n*C(n-1,k-1)*p^(k-1)*(1-p)^(n-k)
```

Functions: `majority_probability`, `majority_log_probability`,
`majority_sft_loss`, `majority_sft_weight`,
`majority_sft_weight_from_logp`, `majority_rl_weight`.

These implement the paper's binomial surrogate, not the full plurality
probability that also depends on the rival-answer distribution. No arbitrary
probability floor is applied. Upper and lower binomial tails are evaluated
in log space; factoring out `p^k` keeps the low-probability SFT ratio stable.

The SFT weight uses its continuous value `k` at `p=0`. For `k=n`, the loss
is `-n*log_p` and the SFT weight is `n`. For `k=1`, these functions reduce to
Pass@N. Endpoint *values* are supported by probability/weight APIs; their
endpoint parameter derivatives are not a training contract. Autograd checks
cover interior probabilities and the log-loss boundary at `log_p=0`.

## Best-of-N

```text
q         = Pr[R(Y') < R(y)]
RL weight = n*q^(n-1)
```

`best_of_n_rl_weight` implements the rank weight stated in the paper.
It does not claim to compute the full expected-maximum policy gradient.
Reward ties, conditional versus historical-buffer quantiles, and any
smoothing are explicit choices for the data/training modules. They are not
silently chosen here. For SFT with a supplied best possible target, use
Pass@N as described in Appendix A.4.

## Applying weights

`weighted_policy_loss(log_probs, advantages, weights)` implements only

```text
-mean(stopgrad(advantages * weights) * log_probs).
```

The arguments must have identical shapes and devices. Completion token
masking/summing is done by the caller. Weights are detached before applying
them, so autograd does not add a derivative of the scaling coefficient itself.
This helper is not a complete GRPO/PPO objective: it has no importance ratio,
clipping, KL penalty, or advantage estimation.

SFT losses are differentiated directly. Do not multiply an already
strategy-aware SFT loss by its SFT scaling factor a second time.

## Normalization

Normalization is separate from the objective formulas and never happens
implicitly. The paper's batch-normalization discussion distinguishes
between-prompt scaling from normalization within a single prompt
(Appendix B, `app:weight_normalization`).

`normalize_prompt_weights` takes a one-dimensional tensor with one value
per **distinct prompt**, and requires at least two prompts. Normalize before
expanding each weight across that prompt's rollouts. With just one prompt,
keep the unnormalized weight; dividing it by its own mean cancels it.

`normalize_sample_weights` is for genuinely sample-specific weights, such
as BoN ranks. It must not be used with repeated copies of one prompt's
weight. Both functions return detached, nonnegative weights with mean one,
preserve relative ratios, and reject all-zero input rather than inventing
a replacement learning signal. A common rescaling before division avoids
overflow and additive-epsilon bias. There is no default clipping ceiling.

## Testing

```bash
python -m pytest -q
```

The tests use independent high-precision decimal evaluations of binomial
sums, direct differentiation of small binomial polynomials, PyTorch
finite-difference gradient checks, endpoint limits, extreme log-probabilities,
input validation, and explicit normalization/stop-gradient checks.
No tests load a checkpoint, generate answers, or access a dataset.

This numerical implementation intentionally does not copy the historical
scripts' probability/logit clamps. Any experiment-specific clamp or other
stabilization must be recorded explicitly when adding experiment configs.
