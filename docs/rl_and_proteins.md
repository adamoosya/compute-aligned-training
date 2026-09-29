# MATH RL and protein experiments

This is a paper-based implementation of Sections 3.2–3.3, Appendices B/C and
J–M, and the raw-versus-log weighting ablations in Appendix N. Recovered results
and this new implementation are different artifacts. The historical training
revision is not known for every result; running a preset is not a guarantee of
recovering a printed table entry.

## Included configurations

| Family | Main configurations | Ablations |
|---|---|---|
| MATH Pass@N | `math_passn_baseline`, `math_passn_n4_log`, `math_passn_n16_log` | `math_passn_n4_raw`, `math_passn_n16_raw` |
| MATH Majority Vote | `math_majority_baseline`, `math_majority_n4_raw`, `math_majority_n8_raw` | `math_majority_n4_log`, `math_majority_n8_log` |
| Unconditional protein | `protein_unconditional_baseline`, `protein_unconditional_bon2`, `protein_unconditional_bon4`, `protein_unconditional_bon8` | — |
| Conditional protein | `protein_conditional_baseline`, `protein_conditional_bon2`, `protein_conditional_bon4`, `protein_conditional_bon8` | — |

MATH files are in `configs/rl/`; protein files are in `configs/protein/`.
All names above have a `.json` extension. One shared trainer implements the
variants. Changing the weight does not swap the baseline to another trainer.

`configs/rl_warmup/passn.json` and `configs/rl_warmup/majority.json` provide separate
CE warmups with the RL adapters' rank 16, alpha 32, and q_proj/v_proj modules.
The previous SFT experiment configurations remain unchanged. The protein
warmup is `configs/protein/conditional_warmup.json`.

## Local checks

```bash
python -m pytest -q
python scripts/smoke_rl_protein.py
```

The combined smoke uses a tiny GRU and sampled categorical completion fixtures.
It runs nine baseline/CAT branches, checks actual optimizer updates, checks the
frozen references, and round-trips local weight snapshots. It is not an accuracy
benchmark. Two additional optional pytest tests use actual Transformers/PEFT:
a tiny Mistral adapter update and a tiny GPT-2 protein warmup → RL → evaluation
pipeline. No test downloads pretrained model weights or datasets.

## Full MATH runs (GPU)

Install the `hf`, `data`, and `cuda` extras in the GPU environment. Supply full
model and dataset commit SHAs; the presets intentionally do not guess historical
revisions. CPU smoke success does not validate CUDA/NF4 execution.

```bash
python -m pip install -e '.[hf,data,cuda]'

python scripts/train_math_sft.py \
  --config configs/rl_warmup/passn.json \
  --model-revision "$MODEL_REVISION" --dataset-revision "$DATASET_REVISION" \
  --output-dir outputs/passn-rl-warmup --allow-download --execute

python scripts/train_math_rl.py \
  --config configs/rl/math_passn_n16_log.json \
  --model-revision "$MODEL_REVISION" --dataset-revision "$DATASET_REVISION" \
  --initial-adapter outputs/passn-rl-warmup/adapter-epoch-001 \
  --output-dir outputs/passn-rl-n16 --allow-download --execute

python scripts/generate_math.py \
  --config configs/generation/math_passn_rl.json \
  --model-revision "$MODEL_REVISION" --dataset-revision "$DATASET_REVISION" \
  --adapter outputs/passn-rl-n16/adapter-final \
  --output-dir outputs/passn-rl-n16-eval --allow-download --execute
```

For Majority Vote use its three-epoch warmup, `adapter-epoch-003`, and the
`math_majority_*` configurations. Each branch starts from the shared warmup,
not another branch's final adapter. MATH stage data can alternatively come from
`--local-data path.jsonl`, using the same schema as the data module.

Use `--dry-run` instead of `--execute` to print a configuration without touching
models, datasets, or output files. A real execution requires a new output path.
Run folders contain config/environment/selection records, per-step metrics,
individual rollouts with token IDs/rewards, a final adapter, and `complete.json`.
No partial run is automatically resumed or called complete.

## Full protein runs (GPU)

ProtGPT2 is trained as a full model, not LoRA. The loader does not invent a chat
template. It retains FP32 master weights and uses autocast for the configured
compute precision. A separate frozen reference is loaded when beta > 0; account
for both models in GPU memory. Public ProtGPT2 weights use a PyTorch `.bin`
checkpoint, loaded with `weights_only=True`; new exports use safetensors.

```bash
python scripts/train_protein.py \
  --config configs/protein/conditional_warmup.json \
  --model-revision "$PROTEIN_REVISION" \
  --output-dir outputs/protein-shared-warmup --allow-download --execute

python scripts/train_protein.py \
  --config configs/protein/protein_conditional_bon4.json \
  --model-revision "$PROTEIN_REVISION" \
  --initial-model outputs/protein-shared-warmup/model-final \
  --output-dir outputs/protein-conditional-bon4 --allow-download --execute

python scripts/evaluate_protein.py \
  --config configs/protein/protein_conditional_bon4.json \
  --model-revision "$PROTEIN_REVISION" \
  --checkpoint outputs/protein-conditional-bon4/model-final \
  --output-dir outputs/protein-conditional-bon4-eval --execute
```

Unconditional branches start from the pretrained model; use their configuration
without `--initial-model`. The conditional baseline and all BoN branches use
the same shared warmup. Exports contain the full model and can be large; keep
outputs outside version control. The evaluator saves individual completions and
rewards. It defaults to 150 prompts × 32 candidates for conditional experiments
and one 512-candidate pool for unconditional experiments.

## Update conventions

The trainer first collects every prompt in a logical optimizer batch. It
computes old-policy likelihoods, advantages and CAT weights without gradients.
It then backpropagates in smaller forward chunks, dividing by the total number
of completions in that logical batch. The last short batch is not underweighted.

- Training rollout count M and test-time target N are separate settings.
- Raw rewards are standardized within each prompt using sample standard deviation
  plus epsilon, before CAT weights are applied. The optional all-wrong penalty
  is -0.5 for the Majority Vote presets only, as in Appendix K.
- Pass@N presets use per-completion sequence probabilities, following Appendix J.
  Majority Vote uses Laplace-smoothed per-prompt success counts. Both probability
  choices are exposed in the weight configuration.
- The dynamic threshold is `max(2, min(N//2+1, floor(N*c_max/M)+1))`. Failed
  extraction remains an attempt; invalid answers share a wrong-answer key.
- `form=log` selects the SFT-derived p/tilde-p normalization; `raw` selects the
  RL derivative. Both use the already-tested core formulas. The legacy additive
  denominator epsilon is not reintroduced into the analytic weight.
- Mean normalization, when requested, uses the complete logical batch, not one
  prompt at a time. A one-prompt final batch keeps its prompt-level weight raw.
  All-zero weights remain zero. `clip` caps raw weights before normalization.
- Unconditional BoN uses a 4,000-reward FIFO, queried before appending the current
  batch. Conditional BoN ranks only outputs for the same prompt. Equal rewards
  receive equal strict-less ranks: q = (count of strictly lower rewards + 1)/(M+1).
- Dropout is disabled during RL so old/current likelihood comparisons are not
  contaminated by different dropout masks. Gradient checkpointing remains
  available in train mode. Weights and advantages are detached.

### Surrogate and reference penalty

`surrogate=reinforce` implements the appendix's weighted sequence-log-probability
loss. This is the default for the paper-based presets. `surrogate=clipped`
implements the sequence likelihood-ratio surrogate in Section C; it is explicit,
not a silent replacement for the appendix's update. This implementation does
not use a TRL Trainer subclass and does not claim bitwise equivalence to one.
Only one optimizer step is taken per collected batch.

The reference penalty is the mean over completion tokens of
`exp(log_pi_ref - log_pi) - (log_pi_ref - log_pi) - 1`, multiplied by beta.
It is not CAT-weighted. This is an explicitly chosen nonnegative sampled KL
surrogate, not the signed log-ratio expression in the legacy protein scripts.
It is not an exact KL evaluation over the whole sequence distribution. Sampling
uses the specified temperature/top-p settings; these practical updates are not
claimed to be unbiased evaluations of the population gradient.

## Settings and source boundaries

The code follows the paper where supplied experimental script versions disagree
with it. In particular, conditional protein rollouts share a prompt within each
rank group; the uploaded legacy version ranked outputs from different prompts.
The old Majority Vote file's scalar self-normalization is not copied. Neither
change is presented as recovery of the exact original training revision.

MATH presets use 4 rollouts/prompt for Pass@N and 8 for Majority Vote. Four prompts
form each logical optimizer batch. Pass@N uses 500 RL examples after an offset
of 2,000; Majority Vote uses the first 300 eligible examples (levels 1–3).
Counts/offsets are derived from the supplied source family; exact historical
indices were not recovered. Selected IDs and content hashes are saved. The
Pass@N warmup uses a one-epoch development preset; the exact historical warmup
revision remains unverified. Majority Vote uses the paper's three epochs.

Protein warmup uses 2,000 pairs, 60 optimizer steps, and learning rate 5e-6.
Unconditional RL uses M=32, 300 steps, learning rate 1e-5, beta=.3. Conditional
RL uses M=25, 300 steps, learning rate 2e-6, beta=.05. Four groups form an optimizer
batch; each completion is still forwarded/backpropagated in configurable chunks.
Input lengths 10–19 and the conditional short-output penalty are from the supplied
protein scripts. The synthetic generator uses separate, reproducible warmup,
train and evaluation streams rather than reusing seed-reset training inputs.

The two reward equations and hydrophobic amino-acid set are from Appendices L/M.
Whitespace is removed and invalid alphabets receive -5 instead of being silently
filtered into a valid-looking protein. These are synthetic reward tasks, not
measurements of folding, safety, binding, or biological function.

The protein evaluator computes the exact expected maximum of the *empirical*
reward pool with replacement, instead of adding resampling noise. Conditional
uncertainty is across prompts. It does not invent a population standard error
from the unconditional single pool. Historical aggregate results and their
reported errors will be handled separately by the reference-results tooling.

## API references

- [PyTorch AMP accumulation and gradient clipping](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)
- [PEFT model loading/export](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/peft_model)
- [ProtGPT2 checkpoint files](https://huggingface.co/nferruz/ProtGPT2/tree/main)
