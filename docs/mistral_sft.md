# Mistral / LoRA SFT

This stage connects the shared CAT losses to Transformers and PEFT. It adds
model loading, adapter exports, and one-stage experiment commands. It does not
add reinforcement learning or generation-based evaluation.

## Offline development check

```bash
python -m pip install -e '.[dev,hf]'
python -m pytest -q
python scripts/smoke_lora.py
python scripts/train_math_sft.py --config configs/sft/passn_n16.json --dry-run
```

The integration check constructs a tiny, randomly initialized Mistral model and
a small local tokenizer. It does not download Mistral-7B weights or MATH. A CE
warmup is saved once; CE, Pass@4, and Majority Vote branches reload the same
warmup. The check verifies that adapter weights change, base weights do not,
and saved adapters reproduce evaluation logits when reloaded. Printed losses
are diagnostic values, not paper results or a convergence guarantee.

The real backend integration test is skipped when the `hf` extra is not
installed. The other loader tests use test doubles to check API arguments and
control flow; those tests are not evidence of CUDA or PEFT execution.

## Model loading

`load_lora_model` accepts `HFModelConfig` and `LoRASettings` and returns the
model and tokenizer. CPU execution uses fp32. The CUDA path supports fp16/bf16
and optional NF4 quantization. These are the devices supported by this loader;
this is not a claim about every device supported by the underlying libraries.

Only LoRA parameters are optimized; adapters use fp32 optimizer parameters.
The base model is frozen. The loader does not use automatic device offloading,
resize token embeddings, trust remote Python code, or publish to the Hub.

For NF4 it uses `prepare_model_for_kbit_training` before adding the adapter.
For a new stage it loads an exported adapter with `is_trainable=True`.
The tokenizer is restored from that adapter export, including its chat template.
Gradient checkpointing is explicit and disables generation caching.

Remote model loads require a commit SHA. `allow_download=False` is the default;
`--allow-download` must be supplied to the runner when network access is needed.
Hugging Face authentication, when needed, should use the local Hub login or an
environment variable, never a token embedded in a configuration or committed file.

## Configurations

The ten JSON files cover two experiment families:

| Family | Warmup | Branches | Targets | Training selection |
|---|---|---|---|---|
| Pass@N SFT | 2 CE epochs | CE, Pass@4, Pass@16, Pass@64; 3 epochs each | final answer | 5,000 MATH training problems |
| Majority Vote SFT | 2 CE epochs | CE, (N,k) = (8,3), (16,4), (64,26); 3 epochs each | provided solution | all eligible MATH levels 1–3 training problems |

The 2+3 epoch schedule, learning rate 5e-6, effective batch size 4, gradient
clipping 0.3 and linear decay follow the paper's SFT configuration. The majority
thresholds use its selected fractions 0.33, 0.25 and 0.40, rounded up to counts.
The maximum example count for Majority Vote is explicitly null (all eligible
records); the exact historical training selection has not been reconstructed.

LoRA r=16, alpha=16, dropout=0, on q/k/v/o projection layers matches recovered
SFT adapter metadata. This is not the r=16, alpha=32, q/v setup from a different
RL experiment. The new launcher uses the official Mistral v0.2 base with
Transformers/PEFT instead of the patched Unsloth backend. AdamW weight decay
0.01, the deterministic selection implementation, and the seed settings are
explicit refactor settings, not recovered proof of the original run settings.

The 1024-token total cap and optional target truncation are those of the shared
data module. It does not silently truncate or remove an overlong prompt.
Pass@N uses sequence-mean loss and Majority Vote uses token-mean loss. Each
family's warmup and CE comparison use the same reduction and data settings as
its CAT branches. Historical low-probability clamps are not reintroduced.

Both model and dataset revisions are null until pinned deliberately. Do not
interpret the revision of today's Hub files as the original experiment revision.
A dry run prints these unresolved requirements but neither downloads nor trains.

## GPU execution (not part of the Mac smoke test)

On the GPU machine, install an appropriate PyTorch build first, then:

```bash
python -m pip install -e '.[dev,hf,data,cuda]'
```

Pin the model and dataset commit SHAs in copies of the configurations or supply
`--model-revision` and `--dataset-revision`. These must be full commit SHAs.
Choose a new output directory for every stage. The following are examples for
a future GPU run, not commands to run during local setup:

```bash
python scripts/train_math_sft.py \
  --config configs/sft/passn_warmup.json \
  --model-revision "$MODEL_REVISION" --dataset-revision "$DATASET_REVISION" \
  --output-dir outputs/passn-warmup-run01 --allow-download --execute

python scripts/train_math_sft.py \
  --config configs/sft/passn_n16.json \
  --model-revision "$MODEL_REVISION" --dataset-revision "$DATASET_REVISION" \
  --initial-adapter outputs/passn-warmup-run01/adapter-epoch-002 \
  --output-dir outputs/passn-n16-run01 --allow-download --execute
```

Each branch starts from the same warmup, not from the preceding CAT branch.
The runner checks matching data selection, target settings, base-model identity,
LoRA settings and loss reduction. Branches start a new AdamW optimizer and
linear learning-rate schedule. This is a stage transition, not exact optimizer
or RNG resumption. The launcher is single-process; distributed execution is
rejected rather than creating several writers to the same output folder.

CUDA fp16 uses autocast and a persistent GradScaler. Gradients are unscaled
once per accumulation window before clipping. Overflow-skipped optimizer steps
do not advance the learning-rate scheduler and are counted as `skipped_steps`.
The CUDA/NF4 path still needs testing on the intended GPU environment.

## Saved outputs

The runner writes only to a new output directory:

- `config.json`, environment/package versions, selected-record hashes and
  tokenization/truncation counts;
- per-epoch training loss and optimizer-step counts (not generation accuracy);
- `adapter-epoch-NNN/` with adapter safetensors, tokenizer, and metadata;
- `complete.json` after the requested training stage succeeds.

Adapters do not include the full frozen base model. Exports are create-only and
`cat_adapter.json` is written last as a completion marker. A failed run may leave
partial files; those are not silently reused or labelled complete. Historical
results, full model weights, and old scripts are not copied into the repository.

## API references

- [PEFT model loading and save_pretrained](https://huggingface.co/docs/peft/v0.18.0/en/package_reference/peft_model)
- [PEFT quantized training](https://huggingface.co/docs/peft/developer_guides/quantization)
- [Transformers bitsandbytes integration](https://huggingface.co/docs/transformers/v4.57.1/quantization/bitsandbytes)
- [PyTorch automatic mixed precision examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)
