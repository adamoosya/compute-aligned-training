# Main experiment commands

Run commands from the checkout root in the installed environment. See
`configs/paper_experiments.json` for the table/figure map, and run
`python scripts/list_experiments.py` for all named presets.

## GPU setup

```bash
python -m pip install -e '.[hf,data,cuda]'
```

Set `MODEL_REVISION`, `DATASET_REVISION`, and `PROTEIN_REVISION` to the full Hub
commit hashes you intend to use. The configurations leave these unset rather
than invent historical revisions. All commands below can load weights/data when
`--allow-download` is present. They are real training commands, not Mac smoke
checks. Use a suitable GPU allocation, not an HPC login node.

## MATH SFT: Pass@N

```bash
python scripts/train_math_sft.py --config configs/sft/passn_warmup.json \
  --model-revision "$MODEL_REVISION" --dataset-revision "$DATASET_REVISION" \
  --output-dir outputs/passn-warmup --allow-download --execute

python scripts/train_math_sft.py --config configs/sft/passn_n16.json \
  --model-revision "$MODEL_REVISION" --dataset-revision "$DATASET_REVISION" \
  --initial-adapter outputs/passn-warmup/adapter-epoch-002 \
  --output-dir outputs/passn-n16 --allow-download --execute

python scripts/generate_math.py --config configs/generation/math_passn_sft.json \
  --model-revision "$MODEL_REVISION" --dataset-revision "$DATASET_REVISION" \
  --adapter outputs/passn-n16/adapter-epoch-003 \
  --output-dir outputs/passn-n16-eval --allow-download --execute
```

The other branches use `passn_baseline.json`, `passn_n4.json`, and
`passn_n64.json`, all starting from the same shared warmup.

## MATH SFT: Majority Vote

Use `majority_warmup.json`, then `majority_baseline.json`,
`majority_n8_k3.json`, `majority_n16_k4.json`, or `majority_n64_k26.json` in
`configs/sft/`. The generation preset is `math_majority_sft.json`. These
configurations use MATH levels 1–3 and solution targets. The Pass@N family uses
answer targets. Their results are not a controlled comparison of those two
choices because the selected task subsets also differ.

## MATH RL

```bash
python scripts/train_math_sft.py --config configs/rl_warmup/passn.json \
  --model-revision "$MODEL_REVISION" --dataset-revision "$DATASET_REVISION" \
  --output-dir outputs/rl-warmup --allow-download --execute

python scripts/train_math_rl.py --config configs/rl/math_passn_n16_log.json \
  --model-revision "$MODEL_REVISION" --dataset-revision "$DATASET_REVISION" \
  --initial-adapter outputs/rl-warmup/adapter-epoch-001 \
  --output-dir outputs/rl-passn16 --allow-download --execute

python scripts/generate_math.py --config configs/generation/math_passn_rl.json \
  --model-revision "$MODEL_REVISION" --dataset-revision "$DATASET_REVISION" \
  --adapter outputs/rl-passn16/adapter-final \
  --output-dir outputs/rl-passn16-eval --allow-download --execute
```

For Majority Vote use `configs/rl_warmup/majority.json` (three epochs), then
`math_majority_baseline`, `math_majority_n4_raw`, or `math_majority_n8_raw` from
`configs/rl/`, and `math_majority_rl.json` for generation. The log-weight
Majority Vote and raw-weight Pass@N presets are the Appendix N ablations.

These pipelines implement the appendix's weighted sequence-log-probability
update by default. The optional clipped surrogate is separately named. This
is not an unchanged TRL GRPO trainer; see [RL details](rl_and_proteins.md).

## Proteins

```bash
python scripts/train_protein.py --config configs/protein/conditional_warmup.json \
  --model-revision "$PROTEIN_REVISION" \
  --output-dir outputs/protein-warmup --allow-download --execute

python scripts/train_protein.py --config configs/protein/protein_conditional_bon4.json \
  --model-revision "$PROTEIN_REVISION" \
  --initial-model outputs/protein-warmup/model-final \
  --output-dir outputs/protein-bon4 --allow-download --execute

python scripts/evaluate_protein.py --config configs/protein/protein_conditional_bon4.json \
  --model-revision "$PROTEIN_REVISION" --checkpoint outputs/protein-bon4/model-final \
  --output-dir outputs/protein-bon4-eval --execute
```

The unconditional presets start from pretrained ProtGPT2 without the conditional
warmup or `--initial-model`. Both families provide baseline and BoN-2/4/8
presets. The tasks use synthetic hydrophobicity rewards; the scores do not
measure folding, binding, safety, or biological function.

## Output rules

Every stage uses a new directory. Initializers are read-only inputs, and no
branch starts from a different branch's final checkpoint. `complete.json`
marks a finished run. A partial directory is not automatically resumed.
Candidate scoring retains failed/empty generations as attempts. The repository
contains reference results, not pretrained checkpoints or benchmark datasets.
