# Appendix coverage

The additional studies use the same objective, data, adapter and evaluation
code as the main pipelines. The two tables whose numerical logs are missing
are marked **manuscript transcription only** in the reference exports.

## A.7 — token-level margins

`pairwise_token_losses` samples four alternative tokens per supervised token,
masking out the target before sampling. The hybrid averages token CE and the
negative log-sigmoid margin with weight 0.5 each. Rival selection is detached.

```bash
python scripts/train_appendix_sft.py --config configs/appendix/token_margin_warmup.json \
  --model-revision "$MODEL_REVISION" --dataset-revision "$DATASET_REVISION" \
  --output-dir outputs/margin-warmup --allow-download --execute

python scripts/train_appendix_sft.py --config configs/appendix/token_margin_pairwise.json \
  --model-revision "$MODEL_REVISION" --dataset-revision "$DATASET_REVISION" \
  --initial-adapter outputs/margin-warmup/adapter-final \
  --output-dir outputs/margin-pairwise --allow-download --execute

python scripts/generate_math.py --config configs/appendix/token_margin_generation.json \
  --model-revision "$MODEL_REVISION" --dataset-revision "$DATASET_REVISION" \
  --adapter outputs/margin-pairwise/adapter-final \
  --output-dir outputs/margin-pairwise-eval --allow-download --execute
```

The CE and CAT comparisons use `token_margin_ce.json` and
`token_margin_cat.json`, from the same warmup. The paper specifies one warmup
epoch, one branch epoch, four rivals and equal loss mixing. The new presets'
2,000 examples and CAT (N=8,k=3) settings are explicit implementation choices;
they were not established from original logs. They are not a replay guarantee.

## D — CAT SFT initialization followed by standard RL

Use `train_appendix_sft.py` with `transfer_ce_warmup.json` or
`transfer_cat_warmup.json`, both starting from the same pretrained model/revision.
Each runs one epoch on 2,000 examples; the CAT branch uses Pass@16. Exports are
labelled as warmup initializers and record which objective was used.

Run `train_math_rl.py` separately on each initializer with
`configs/appendix/warmup_transfer_rl.json`, passing its `adapter-final` to
`--initial-adapter`. Both use the same standard RL config with 16 rollouts per
prompt. Evaluate with `configs/appendix/warmup_transfer_generation.json`
(100 prompts, 16 candidates). The historical table is transcribed from the paper;
its original logs were not recovered.

## I.4 and N — sweeps and weighting ablations

The Majority Vote N=64 sweep uses k=16,22,26, corresponding to fractions
0.25,0.33,0.40 with ceiling. The first two configs are in `configs/appendix/`;
the third is `configs/sft/majority_n64_k26.json`. Use the same Majority Vote
warmup and generation configuration for all three. No runtime selection of a
winner changes which configuration is presented as the paper's main result.

Appendix N uses the raw and log configurations in `configs/rl/`; it does not
need another training implementation. Their recovered results and plots are
included by `build_paper_assets.py`.

## A.11 and B — sensitivity diagnostics

The asset builder redraws the historical diagnostic datasets without rerunning
model generation. It corrects Figure 9 exactly once and leaves Figure 10's
already-correct sign convention alone.

For a new finite-support calculation, supply your own categorical probabilities:

```json
{"probabilities":[0.2,0.3,0.5],"target":0}
```

```bash
python scripts/diagonal_diagnostic.py --input distribution.json \
  --strategy plurality --n 8 --trials 5000 --seed 42 --output diagnostic.json
```

For `best_of_n`, include distinct per-category rewards. The diagnostic computes
a forced-vote sensitivity for each category, then a/(a-C). Nonpositive total
derivatives produce `rho: null`, not clipped or silently removed observations.
These are finite-support probability calculations under proportional decay, not
checks of a language model's full parameter gradients. Ties are failures for
plurality; Best-of-N here requires a distinct reward ordering.

## E, F, G — analytical calculations

The asset builder produces the SFT/RL weight curves, separate support-comparison
panels, and a uniform-grid weight-cosine table. The last is a descriptive
comparison of scalar functions, not a guarantee about model performance.

`cat.diagnostics.sensitivity.pivotal_variances(p,n)` evaluates the binary-logit
example of the Rao–Blackwell calculation. Its comparison is with the sampled
single-index pivotal estimator, not a universal variance ranking against the
fully summed leave-one-out estimator or normalized GRPO.
