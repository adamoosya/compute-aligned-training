# Compute Aligned Training

**Compute Aligned Training: Optimizing for Test Time Inference**  
Adam Ousherovitch and Ambuj Tewari · NeurIPS 2026

CAT aligns post-training with a specified test-time strategy: Pass@N, Majority
Vote, or Best-of-N. This repository provides the objective functions, MATH
SFT/RL pipelines, and two synthetic protein-generation experiments.

## Install and check

```bash
conda create -n cat-dev python=3.11 pip -y
conda activate cat-dev
python -m pip install -e '.[dev,hf,plots]'
python -m pytest -q
```

Local checks use small CPU fixtures; no pretrained model or dataset is downloaded:

```bash
python scripts/smoke_sft.py
python scripts/smoke_lora.py
python scripts/smoke_generation.py
python scripts/smoke_rl_protein.py
```

GPU experiments additionally need the `data` and `cuda` extras and pinned model
and dataset revisions. CUDA/NF4 has not yet been validated in the refactored
pipeline; CPU checks do not establish GPU behavior or reproduce full training.

## Rebuild tables and figures from the recovered results

```bash
python scripts/build_paper_assets.py --output outputs/paper-assets
```

Use a new output directory each time. This checks eight recovered result files,
creates the six main result tables (including both halves of Table 6), both
weighting-ablation tables, the corrected diagnostics, and figure redraws. It does
not train models. Main result tables contain means only: standard errors are not
invented from aggregate curves. Two other appendix tables are explicitly marked
as manuscript transcriptions. See [reference results](results/reference/README.md).

## Run an experiment

```bash
python scripts/list_experiments.py
python scripts/train_math_sft.py --config configs/sft/passn_n16.json --dry-run
python scripts/train_math_rl.py --config configs/rl/math_majority_n4_raw.json --dry-run
python scripts/train_protein.py --config configs/protein/protein_conditional_bon4.json --dry-run
```

Dry runs only print settings. Actual training requires `--execute`, explicit
model/data revisions, the appropriate warmup, and a new output path.
[Experiment commands](docs/experiments.md) cover the main paper;
[appendix commands](docs/appendix.md) cover the additional studies and diagnostics.

## Core API

```python
import torch
from cat.objectives import passn_sft_loss, majority_sft_loss, best_of_n_rl_weight

log_p = torch.tensor([-8.0, -4.0, -1.0], requires_grad=True)
loss = passn_sft_loss(log_p, n=16)
loss.backward()
majority_loss = majority_sft_loss(log_p, n=8, k=3)
rank_weights = best_of_n_rl_weight(torch.tensor([0.1, 0.5, 0.9]), n=4)
```

`log_p` is summed completion log-probability, not token-averaged NLL. The
Majority Vote objective is the fixed-threshold surrogate. The SFT and raw RL
weights differ; configurations name the choice explicitly.

## Layout

```text
src/cat/objectives/     Strategy losses and weights
src/cat/data/           MATH and synthetic protein data
src/cat/training/       SFT and weighted-RL updates
src/cat/models/         Mistral/LoRA and ProtGPT2 loading and exports
src/cat/generation/     Candidate generation
src/cat/evaluation/     Saved-candidate scoring
src/cat/diagnostics/    Finite-support sensitivity calculations
src/cat/paper/          Reference tables, figures, release checks
configs/               Named experiment settings
results/reference/     Recovered results, separate from new outputs
scripts/               Training, evaluation, plotting and check commands
```

This is a new implementation organized around the paper. Recovered historical
outputs are not relabeled as outputs of this code. Exact historical training
revisions, some appendix logs, and the original error-bar calculations were not
recovered. [Reproducibility notes](docs/reproducibility.md) describe the boundary.

## Citation and licensing

See [CITATION.cff](CITATION.cff) for the paper citation.
The code and documentation are provided under the [MIT License](LICENSE).
Third-party models and datasets retain their own licenses.
