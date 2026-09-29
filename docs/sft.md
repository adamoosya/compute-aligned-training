# MATH data and SFT

This module connects the strategy losses to completion token probabilities.
It follows the paper's answer-only versus solution-target distinction in
Section 3.1 and the sequence likelihoods in Appendices H and I. It is a new
implementation of the paper's objectives, not a recovered historical run.

## Data selection

`select_math_examples` takes MATH-style dictionaries with `problem`, `answer`,
optional `solution`, and optional `level`. Level filtering requires a level on
each record. Records outside the chosen levels are excluded, then the eligible
pool is shuffled using a private seeded RNG. A requested sample count larger
than the eligible pool is rejected.

Each record retains its source split and original index. `selection.manifest()`
records the selection settings, IDs, content hashes, and counts. Keep that manifest
with a run. `assert_disjoint(train, evaluation)` rejects repeated problems both
within and between selections; it does not silently drop or replace them.

Two input routes are provided:

- `load_math_jsonl(path, split=..., ...)` reads a local JSONL file.
- `load_math_hf(dataset_name=..., revision=..., split=..., ...)` loads only the
  explicitly requested Hugging Face split. It requires the optional `[data]`
  dependency and may download data. Use a dataset commit hash for `revision`.
  It never substitutes the training split when an evaluation split is missing.

The JSON configurations in `configs/sft/` make the selection explicit. Model and
dataset commit revisions must be supplied before a real run. The smoke fixture
is not MATH, despite having the same schema.

## Targets and tokenization

`encode_math_selection(selection, tokenizer, target="answer")` supervises the
provided final answer. `target="solution"` uses the full provided solution;
it does not generate a reasoning trace or fall back to answer-only training if
that field is missing.

Encoding uses the tokenizer's chat template. It obtains the user-only generation
prefix and the full user/assistant training chat, then verifies that the prefix
matches exactly in token space. A mismatch raises an error instead of guessing
a supervision boundary. Tokenizer/model-specific boundary behavior must be checked
when connecting the pretrained backend.

The default user message is `Problem:\n{problem}`. It is configurable. Prompt
labels are `-100`; target tokens, including a template-provided EOS, are supervised.
Right-padding labels are also `-100`. Masking uses position, not token-ID equality,
so a real EOS is retained even if the tokenizer uses EOS for padding.

`max_length` defaults to 1024 total tokens. An overlong record is an error by
default. Explicit `overflow="truncate_target"` removes only the end and records
the original and retained lengths; it never drops the prompt or synthesizes EOS.
An example with no remaining target is rejected. This limit is a development
setting, not a claim about the exact truncation of the original scripts.

## Sequence loss and reduction

`completion_log_probs` shifts logits and labels exactly once and computes

```text
log_p[b] = sum(log P(target_token | preceding_tokens))
```

Only supervised target positions enter the sum. It is not a token-mean
log-likelihood or an empirical answer frequency.

`SFTObjective` supports:

| Strategy | Per-sequence loss |
|---|---|
| `ce` | `-log_p` |
| `passn` | `passn_sft_loss(log_p, n)` |
| `majority` | `majority_sft_loss(log_p, n, k)` |

The trainer differentiates those scalar losses directly. It does not multiply
CE by a differentiable `w(p)`, which would add an unwanted derivative of the weight.
The shared objectives retain the numerical conventions documented in
[objectives.md](objectives.md); historical logit/probability clamps are not
silently reintroduced here.

Reduction is explicit:

- `sequence_mean`: sum of sequence losses divided by sequence count.
- `token_mean`: sum of sequence losses divided by supervised target token count.

In both cases `p` is computed from the full sum before reduction. The
`token_mean` option supports the length normalization described for Majority Vote;
for a fair comparison, use the same reduction for a baseline and its CAT variant.
The smoke test intentionally exercises both options; its losses are not compared
across different objectives or normalization conventions.

## Training and accumulation

`train_sft_epoch` accepts a model, batches, optimizer, and `SFTObjective`. A batch
contains `input_ids`, `attention_mask`, and `labels`. A model returns a logits
tensor, a mapping with `logits`, or an object with `.logits`, as causal model
outputs commonly do. Model loading, model placement, and optimizer construction
belong to the calling experiment runner.

The loop divides by the actual number of examples or target tokens in each
accumulation window, including a short final window. It does not average unequal
microbatch means. It rejects nonfinite losses before each optimizer step. In full precision it
also rejects nonfinite gradients; with fp16, the GradScaler skips overflowed
updates and reduces its scale. An optional scheduler advances once per optimizer step.

`evaluate_sft_loss` reports teacher-forced loss and restores the previous model
mode. It does not compute generated-answer accuracy.

The loop is single-process and defaults to full precision. Optional CUDA fp16/bf16
uses autocast; fp16 requires a caller-owned GradScaler. The scaler unscales
before clipping, and overflow-skipped steps are recorded without advancing the
scheduler. See [mistral_sft.md](mistral_sft.md) for the optional LoRA backend.
Distributed training is not implemented.

## Local snapshots

`save_sft_snapshot` writes a state dict and JSON metadata into a **new** directory.
An existing destination is an error. `load_sft_snapshot` uses PyTorch's restricted
`weights_only=True` state-dict loader. These snapshots are for locally constructed
models, not an HF/PEFT export or an exact optimizer/RNG resume checkpoint.

## Checks

```bash
python -m pytest -q
python scripts/smoke_sft.py
```

The integration check constructs a tiny byte-tokenized GRU model, performs a CE
warmup, then starts CE, Pass@4, and Majority Vote (N=8, k=3) branches from the same
weights. It checks that each branch updates parameters, reduces its own fixture
loss, and saves/reloads identical state dicts. Outputs are tagged `smoke_only`.
All model/tokenizer fixtures are local. The optional HF dataset-loader call
contract is tested with a mock; no live MATH download or pretrained Mistral
training is claimed by these checks.

## API references

The mathematical objective source is Table 1 and Appendices A, H, and I of
*Compute Aligned Training: Optimizing for Test Time Inference*.

- PyTorch cross entropy: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html
- Hugging Face chat templates: https://huggingface.co/docs/transformers/chat_templating
- Hugging Face dataset loading and revisions: https://huggingface.co/docs/datasets/loading
