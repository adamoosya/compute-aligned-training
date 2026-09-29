# Reproducibility boundary

## What has been checked

The core objectives, token masking, SFT, adapter exports, candidate generation,
scoring and RL/protein smoke tests passed locally on CPU. Tiny real Transformers
and PEFT checks passed on the maintainer's Mac. Full-model Mistral/ProtGPT2
training and CUDA/NF4 remain separate validation tasks. No test count is a claim
that a paper table has been reproduced by retraining.

## Historical results versus new implementations

Eight recovered numerical files supply all six main result groups, the weighting
ablations, the Majority Vote threshold sweep, and both sensitivity diagnostics.
Source hashes are verified before figures are generated. No checkpoint, data
index, confidence interval, or missing candidate is reconstructed from a mean.

The original training revision is not established for every experiment. Some
uploaded scripts were later derivatives. The refactor follows the paper's stated
methods, with differences documented in the module guides. In particular:

- The conditional protein rank comparison is within prompt, and prompt-level
  CAT weights are not divided by themselves.
- The default RL update follows the appendix's weighted sequence loss; a clipped
  surrogate is an explicit alternative, not an undocumented replacement.
- Probabilities are evaluated in numerically stable form. Historical clamps,
  sampling, parsing and normalization details are not claimed to be identical.
- Dataset/model revision fields are deliberately unset until a concrete run is
  planned. A selected current revision is not relabeled as the historical one.

The token-margin and SFT-to-RL tables are manuscript transcriptions because their
original numerical logs were not recovered. Their new commands and explicit
presets are available, but not claimed to recover the exact historical values.

## Error bars and figures

Generated reference tables report means only. Original SE calculations cannot be
validated from the available aggregate data. The new evaluator saves per-problem
scores so a newly executed study can compute its own errors without assuming a
sample size. Redrawn figures are not pixel-identical originals. Conditional
scatter plots contain only the saved first candidate per prompt.

## Platform and environment

The backend pins are for this refactor, not a reconstruction of the original
Unsloth environment. For each real run, retain config.json, environment.json,
selection IDs/hashes, initialization metadata, seeds, and outputs. Installing a
CPU PyTorch build successfully does not validate a CUDA runtime or bitsandbytes.
Do not use an HPC login node for actual GPU/model workloads.
