# Reference results

`sources/` contains eight selected, unchanged numerical result files from the
recovered experiments. `manifest.json` records their original filenames, hashes,
and the archive hash. It excludes cluster usernames, credentials, training code,
checkpoints, model weights and benchmark text.

| Source | Paper output |
|---|---|
| math_sft_passn.json | Table 2, Figure 1 |
| math_sft_majority.json | Table 3, Figure 2, threshold sweep |
| math_rl_passn.json | Table 4, Figure 3, raw/log ablation |
| math_rl_majority.csv | Table 5, Figure 4, weighting ablation |
| protein_unconditional.json | Unconditional half of Table 6, distribution and scaling |
| protein_conditional.json | Conditional half of Table 6, first-candidate scatter and scaling |
| majority_diagonal.csv | Original Figure 9 values before the reviewed sign correction |
| bon_diagonal.csv | Figure 10 values as saved |

The main paper comparison checks 124 means: 122 agree at the displayed precision.
The two Table 4 differences are saved 30.2472126186% versus printed 30.3%, and
saved 25.2450137189% versus printed 25.3%. Table generation rounds the saved
number once. It never changes data to force a printed match.

Figure 9 originally saved a/(a+C); plotting converts each row once using
r/(2r-1). Figure 10 already saved a/(a-C), so it is not transformed. All 81 and
277 respective rows are retained.

Aggregate means do not recover per-problem errors or original training details.
Generated tables therefore omit SEs rather than infer them from a guessed test
set size. The protein distributions use only the saved values actually present;
conditional scatter values are the first candidate of each prompt, not full
candidate pools.

`paper_mean_checks.json` is a manuscript transcription for comparison.
`reported_appendix.json` transcribes the token-margin and SFT-to-RL tables; their
original numerical logs were not recovered. These are reported-only reference
tables, not verified training outputs. This provenance is retained in the exports.
