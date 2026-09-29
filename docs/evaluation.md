# MATH evaluation

The scorer reads **saved completion text** and produces Pass@k and Majority Vote
scores. It does not load a model, generate responses, select a checkpoint, download
MATH, or import aggregate historical results. The same scorer can be used after
SFT or RL. Pretrained generation is a separate next step.

## Candidate files

One JSON object per problem, with these exact fields:

```json
{"schema_version":1,"example_id":"test:12","problem":"What is 1+1?","answer":"2","source_split":"test","completions":["The result is \\boxed{2}.","3",""]}
```

The example is a handwritten schema illustration, not a MATH record. Each entry
in `completions` is one attempt. Include empty or unparseable attempts. Do not
filter to successful parses or insert repeated answers to fill a short pool.
Completions must exclude the input prompt: its numbers must never enter answer
extraction. The scorer cannot infer whether an arbitrary text field contains a
prompt, so the generation stage must enforce this boundary.

`CandidateRecord.from_example` retains the ID, problem, answer, and source split
from the existing MATH loader. Repeated IDs/problems, mixed splits, malformed JSON,
unknown fields, empty pools, and budgets exceeding any pool size are errors, not
reasons to silently skip an example. The split field is a supplied label, not
proof that a checkpoint never saw those examples. Generation manifests will be
needed to record model revision, dataset selection and sampling settings.

## Run

```bash
python scripts/score_math.py \
  --input outputs/my-generation/candidates.jsonl \
  --config configs/evaluation/math_passn.json \
  --output outputs/my-scores
```

Use `--dry-run` to validate the file and configuration without scoring or writing.
The output directory must be new. Output files are:

- `candidates.jsonl`: an unchanged copy of the input bytes.
- `per_problem.jsonl`: original completions, parsed answers, vote keys, correctness,
  pool sizes, invalid counts and scores at each requested budget.
- `summary.json`: equally weighted problem means, standard errors, parser version,
  configuration, Python version and input checksum.
- `complete.json`: checksums of the three completed files. Written last; a directory
  without this marker is not a completed scoring run.

The scorer uses only local data and existing dependencies. No training is run.
Outputs stay under the already ignored `outputs/` directory when using the example.
Original recovered results must remain separate from these newly computed outputs.

## Conventions and their source

Appendix H.2 of the paper uses a fixed pool of generated candidates and the
without-replacement Pass@k estimator. For pool size m with c correct candidates:

```text
Pass@k = 1 - choose(m-c, k) / choose(m, k).
```

The code evaluates this without large binomial coefficients. It is different from
`1 - (1 - c/m)**k`. At k=1 it equals c/m; at k=m it is one exactly when c>0.

Appendix I.3 samples k candidates without replacement, selects the most frequent
valid answer and averages over 500 trials. That is implemented here. Different
trials independently reuse the same observed pool. Invalid responses remain in
that pool and consume draws; they are ignored only when counting valid votes.
An all-invalid group fails. Maj@1 is calculated exactly as c/m instead of adding
avoidable Monte Carlo error. This is a computational simplification of the same
single-draw quantity, not a newly generated evaluation.

The supplied legacy scripts use `Counter(...).most_common(1)`: ties go to the first
answer encountered in the sampled order. `tie_break="first"` makes that behavior
explicit. `failure` is also available for strict-plurality diagnostics. These are
different evaluation policies and must not be mixed in a comparison. Neither uses
the reference answer to break a tie. Trials use an independent seeded RNG for each
(example ID, budget), so row order and other requested budgets do not change them
within the same Python random-sampling implementation.

The JSON presets request the main table budgets (and 128 for the Majority Vote
curve). They do not fix model generation settings. The 500-trial setting comes
from Appendix I.3; using it on other saved pools is a new evaluation configuration,
not a claim about the exact historical evaluator for every experiment.

## Answer parsing and verification

The legacy scripts use boxed-answer extraction with a last-number fallback and
numeric/exact-string matching. `math_basic_v1` is a new implementation with those
basic modes. It handles nested braces and signed/scientific numbers explicitly.
Differences from old regexes can change scores; do not call re-scored outputs exact
reproductions of historical metrics.

The default `boxed_or_last_number` mode reads the last `\boxed{...}`/`\fbox{...}`
when present, and otherwise uses the last numeric token. Malformed or empty boxes
fail rather than falling back to an earlier intermediate calculation. `boxed`
disables the numeric fallback. `answer_only` treats an unboxed completion as the
whole final answer and is appropriate for bare expression/tuple outputs. Numeric
fallback can misread an intermediate calculation and does not parse an unboxed
fraction or algebraic expression as a unit; use an explicit box for those cases.

The verifier accepts exact strings after whitespace, math-delimiter and fraction-
command normalization, or numerically equal simple scalars. Scalar forms include
integers, decimals, scientific notation, a/b, and numeric LaTeX fractions. The
absolute tolerance defaults to the scripts' 1e-4 convention (strictly less than
that tolerance, with exact matches always accepted). Reference answers are never
replaced by their last number. Values are parsed using bounded rational arithmetic;
no `eval`, code execution or symbolic-expression parser is used.

This is not a general mathematical-equivalence judge. It does not simplify
expressions, reorder sets, convert units or assess reasoning. Presentation-equivalent
strings share a vote key, but different numeric forms such as `0.5` and `1/2` can
still split votes. Numeric tolerance is used for correctness only, never for
reference-dependent vote grouping. Choose one documented policy for a comparison.

## Uncertainty

The aggregate standard error is the sample standard deviation of per-problem scores
divided by sqrt(number of problems). Each problem has equal weight, regardless of
how many candidates or Monte Carlo trials it has. With one problem, SE is null.
It describes variation across the scored problems and also includes Monte Carlo
noise when the scores are estimated; it is not a training-seed error bar.

Each estimated Maj@k additionally records its Monte Carlo SE conditional on the
observed pool. This does not include uncertainty from generating a different pool.
A zero empirical Monte Carlo SE is not proof of zero estimation error. We do not
reconstruct historical standard errors from aggregate means, assume correlations
between methods, or treat 500 resampling trials as 500 new benchmark problems.

## Offline smoke check

```bash
python scripts/smoke_evaluation.py
```

It scores three handwritten problems with four attempts each, checks Pass@1/2/4
against exact combinatorial values, checks majority endpoints, confirms invalid
answers stay in the denominator, repeats the scores deterministically, and verifies
output hashes. No model or dataset download occurs. Unit tests also compare the
metrics with enumerated tiny pools. These are code checks, not paper results.
