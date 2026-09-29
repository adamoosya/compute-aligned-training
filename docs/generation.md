# Generate and score MATH candidates

This stage connects a frozen base model or a saved CAT LoRA adapter to the
existing evaluator. It does not train or select a checkpoint for you.

## Local check

```bash
python -m pip install -e '.[dev,hf]'
python -m pytest -q
python scripts/smoke_generation.py
python scripts/generate_math.py --config configs/generation/math_passn_sft.json --dry-run
```

The smoke test creates a tiny random Mistral and a nonzero random LoRA fixture,
saves the adapter, reloads it for inference, and generates four candidates for
each of two handwritten questions. It checks repeatability after reloading,
frozen weights, saved candidates/tokens and scoring. It is not an accuracy test
or a trained checkpoint. It does not download a model or dataset.

## Running a prepared model

The following is a future GPU command, not part of the local smoke test:

```bash
python scripts/generate_math.py \
  --config configs/generation/math_passn_sft.json \
  --model-revision "$MODEL_REVISION" \
  --dataset-revision "$DATASET_REVISION" \
  --adapter outputs/passn-n16-run01/adapter-epoch-003 \
  --training-selection outputs/passn-n16-run01/selection.json \
  --output-dir outputs/passn-n16-evaluation-run01 \
  --allow-download --execute
```

Install the `[data,cuda]` extras on the CUDA machine before a full NF4 run.
The model and dataset revisions must be full Hub commit SHAs, as in the SFT
runner. `--allow-download` permits obtaining missing files; without it the
loader requests local model files and sets datasets offline for the data load.
A dry run does not check the hardware or load models/data.

The adapter path must be a completed export with `cat_adapter.json`,
`adapter_config.json` and `adapter_model.safetensors`. The recorded base model
and revision must match. The actual PEFT configuration is checked against the
CAT metadata. Loading freezes every parameter and never creates a fresh adapter
as a substitute for missing trained weights. This stage accepts refactor exports;
it does not silently convert historical adapter folders.

To evaluate a base model without LoRA, set `initialization` to `base` in a copied
config and omit `--adapter`. The inference path uses the same HFModelConfig as
SFT: CPU/fp32 or a single CUDA device. MPS, distributed execution and automatic
CPU/disk offloading are not implemented.

## Presets and their scope

`math_passn_sft.json` selects 500 MATH test examples and generates 64 candidates.
`math_majority_sft.json` selects 500 MATH test examples from levels 1--3 and
generates 128 candidates. Counts, length budgets and Majority Vote sampling
settings follow the paper's Appendices H/I. Both use temperature 0.8 and top-p
0.95, with the training prompt template `Problem:\n{problem}`.

Top-k is explicitly disabled (`top_k=0`), rather than inheriting a library/model
default. Chunk size four, a private seeded selection order, refusal to truncate
an overlong prompt, and the previously documented `math_basic_v1` parser are
refactor choices. They are not claims about the exact settings of every old run.
The old scripts sometimes used additional top-k filtering, different selection
orders or prompt truncation. Newly generated scores remain separate from the
recovered historical reference results.

## Sampling and boundaries

Generation processes one prompt at a time and samples up to `chunk_size`
candidates per call. This caps the active candidate batch; the last chunk may
be smaller. Sampling uses a fresh Transformers GenerationConfig with
`do_sample=true`, one beam, explicit token limits and sampling parameters.
`use_model_defaults=false` prevents unlisted model-specific settings from
silently enabling a different search strategy. No logit tensors are saved.

Only the problem enters the user chat message. The reference answer/solution
are never passed to `generate`. Chat-template tokenization supplies the exact
prompt tokens. Every returned sequence must retain that prefix. The sampler
slices by token count, not by the number of characters in a decoded prompt.
It records generated tokens through the first EOS, excluding batch padding
following EOS. Empty and unparseable completions remain attempts.

A prompt exceeding `max_prompt_tokens`, a context-window violation, a malformed
model return or an execution failure aborts the run. There are no retries,
silent skips, replacement candidates, deduplication or success-based filtering.
A short generation must end in EOS; otherwise it must reach max_new_tokens.

Each (seed, example ID, chunk start) has a deterministic derived seed. Reordering
problems does not change their random streams. Changing chunk size can change
the samples; it is saved in the manifest. Reproducibility also requires fixed
model/tokenizer files, software and hardware. No cross-platform bitwise guarantee
is made. Sampling preserves the caller's CPU and selected-CUDA RNG state and the
model's prior submodule train/eval modes, including on failure. This context is
single-threaded; it is not a concurrent-generation API.

## Output files

Each run uses a new directory:

- `run.json`: sampling/scoring settings, package versions, supplied model/adapter
  provenance, tokenizer identity and overlap-check status.
- `selection.json`: dataset identity, chosen record IDs and content hashes.
- `candidates.jsonl`: individual completion text, using the scorer's existing schema.
- `tokens.jsonl`: prompt token IDs, generated token IDs, finish reasons, chunk seeds.
- `generation_complete.json`: checksums after all requested problems are generated.
- `scores/`: the existing evaluator's unchanged candidate copy, per-problem scores,
  summary and completion marker.
- `complete.json`: written last, linking the generation and scoring completion markers.

The optional `--training-selection` rejects overlap with the supplied training
manifest and checks its internal checksum. For a refactor adapter whose metadata
records a training selection, the manifest must match that selection. Omitting
the argument records `not_checked`; a `test` split label is not proof of absence
from pretraining or other fine-tuning data. Local base model weights are not
rehashed; a local path is not a content-addressed historical-model identity.

A failed run may retain partial files without a final completion marker. Do not
count such a directory as a complete run. No resume or overwrite option is
provided in this stage. Raw text/tokens stay under the ignored `outputs/`
directory in the examples; they are not automatically published to GitHub.

## References

- Paper: *Compute Aligned Training: Optimizing for Test Time Inference*,
  Appendix H.2 (candidate-pool Pass@k), Appendix I.3 (Majority Vote evaluation).
- [Transformers generation configuration and generate](https://huggingface.co/docs/transformers/v4.56.0/main_classes/text_generation)
- [PEFT from_pretrained and frozen adapters](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/peft_model)
- [PyTorch RNG-state context](https://docs.pytorch.org/docs/stable/random.html)

API code targets the already pinned Transformers 4.57.6/PEFT 0.18.1 stack.
Real CUDA/NF4 execution and full-size experiment reproduction still need their
own checks. The optional integration test uses the real libraries, not test doubles.
