# Qwen3-0.6B: completed 200-question experiment

Completed October 31, 2026 at 04:31 America/Denver. All **12,000 planned
rollouts** are present: 200 questions, 20 sampled rollouts each, and three
checkpoints at 0, 77, and 251 cumulative additional GRPO steps.

The checkpoints are Qwen/Qwen3-0.6B, cmpatino/qwen-grpo-r2, and
cmpatino/qwen-grpo-r5. Exact revisions and training-lineage limitations are in
[the experiment description](../../QWEN06B_EXPERIMENT.md). The published accuracy
range motivated selection; this experiment does not measure answer accuracy.

![Checkpoint means with 95% confidence intervals](checkpoint_means.png)

[Vector PDF](checkpoint_means.pdf)

## What the measurements mean

Each score is the model's next-token entropy with the named source removed,
minus entropy with that source present. Other context and the exact generated
history stay fixed. Values are in bits per token, averaged over a 20-token
window, then over available complete rollouts within each question, then equally
across questions. The first window is Output A. A question-specific correction
is inserted before the second window, Output B.

The Input to Output A, Injection to Output B, and Output A to Output B labels
name the source and target of each comparison. The historical column name
`empowerment` means Output A to Output B in these files. This is entropy
sensitivity, not formal empowerment, channel capacity, or correction-following
accuracy. Equal entropy does not imply identical predictions or no influence.

## Means and pointwise 95% confidence intervals

| Measurement | Step 0 | Step 77 | Step 251 |
|---|---:|---:|---:|
| Input to Output A | 2.4012 [2.3511, 2.4511] | 2.5480 [2.5023, 2.5931] | 2.3188 [2.2533, 2.3855] |
| Injection to Output B | 0.0343 [0.0234, 0.0458] | 0.0129 [0.0046, 0.0211] | -0.0140 [-0.0204, -0.0075] |
| Output A to Output B | 0.0517 [0.0431, 0.0605] | -0.0027 [-0.0096, 0.0043] | -0.0034 [-0.0107, 0.0039] |

All 200 questions contribute to every primary mean. Intervals use 50,000
question-bootstrap resamples. They describe uncertainty across this selected
question set, not variation across independently trained models.

## Paired checkpoint comparisons

Differences below are later minus earlier. Two-sided t-tests pair question means;
Holm correction covers all nine primary tests. The intervals are paired
question-bootstrap intervals and are not multiplicity-adjusted.

| Measurement | Steps | Difference | 95% interval | Adjusted p |
|---|---|---:|---:|---:|
| Input to Output A | 0 to 77 | +0.146745 | [+0.098128, +0.194897] | 6.79293e-08 |
| Input to Output A | 0 to 251 | -0.082426 | [-0.149065, -0.014438] | 0.0353917 |
| Input to Output A | 77 to 251 | -0.229171 | [-0.292662, -0.164821] | 2.5309e-10 |
| Injection to Output B | 0 to 77 | -0.021376 | [-0.032880, -0.010253] | 0.000902449 |
| Injection to Output B | 0 to 251 | -0.048299 | [-0.060880, -0.036040] | 1.04987e-11 |
| Injection to Output B | 77 to 251 | -0.026923 | [-0.037082, -0.016545] | 3.03149e-06 |
| Output A to Output B | 0 to 77 | -0.054425 | [-0.064887, -0.044005] | 6.34727e-19 |
| Output A to Output B | 0 to 251 | -0.055101 | [-0.065763, -0.044586] | 6.34727e-19 |
| Output A to Output B | 77 to 251 | -0.000676 | [-0.009918, +0.008610] | 0.886794 |

## Interpretation and comparison with the earlier experiment

**Injection to Output B decreases at every measured training stage.** All three
comparisons are significant after correction. The previous 1.7B experiment also
showed a decreasing correction score. This is consistent with our reduced-
plasticity hypothesis, but the measured result is a smaller correction-related
entropy difference. We have not measured whether corrections are followed less
successfully or whether final answers improve.

**Output A to Output B drops mainly between steps 0 and 77, then shows no
statistically detectable difference between 77 and 251** (adjusted p = 0.887).
This does not establish equivalence. Its final mean is -0.0034, with a 95%
interval of [-0.0107, +0.0039], so a negative final score is not established.
It would be inaccurate to describe this score as unchanged over the entire run.

The earlier 1.7B run showed an increase in its point estimates, but none of its
three comparisons passed the adjusted 0.05 threshold. Its endpoint adjusted
p-value was 0.0541. These two runs therefore do not establish a general reversal
of empowerment under training.

Input to Output A first increases and then decreases; all three pairwise
comparisons pass the adjusted threshold. This is entropy sensitivity to the
question, not a mathematical-accuracy score.

## Selection, completion, and robustness

The first 50 question/injection pairs are preserved from the previous experiment;
those questions were screened for long responses. The added 150 were sampled
uniformly without replacement from the remaining GSM8K test questions, without
length screening. New injections use an explicit quoted phrase-replacement
format. Cohort differences can reflect both selection and correction wording.

| Checkpoint | Recorded rollouts | Complete A | Complete A+B | B early endings |
|---|---:|---:|---:|---:|
| 0 | 4,000 | 4,000 | 3,998 | 2 |
| 77 | 4,000 | 4,000 | 4,000 | 0 |
| 251 | 4,000 | 4,000 | 4,000 | 0 |

Incomplete target spans are missing values, never zeros. No question was excluded
from the primary paired analyses. The two early endings cannot explain the
main result; complete-A+B sensitivity results retain the same B-based means.

On the **same original 50 questions**, Qwen's Output A to Output B mean changes
from 0.0159 to -0.0064. That endpoint decrease is not significant after correction
(p = 0.1123). The added 150 show a decrease from 0.0636 to -0.0024
(p = 3.85e-19). The full-200 effect also survives exclusion of all five flagged
source-issue questions (195 remaining, p = 6.79e-19).

The correction-score endpoint decrease survives both separate cohort analyses:
adjusted p = 0.00643 for the original 50 and 2.64e-9 for the added 150. It also
survives source-issue exclusion (p = 7.39e-12). These are exploratory sensitivity
analyses of the same data, each with its own nine-test correction; there is no
multiplicity adjustment across sensitivity analyses.

Model family, chat framing, training regimen, generated trajectories, and the
question mix differ between the 1.7B and 0.6B experiments. The new checkpoint
series also changes training hyperparameters between stages and used some test
results to select an intermediate initialization. This is not a controlled
model-size or unique-training-dataset-size experiment, nor an independent
replication across training seeds.

## Audit and provenance

The audit validated the old 3,000 and new 12,000 raw records, recomputed all
question means, bootstrap intervals, adjusted tests, and plotted tables, and
passed 14 tests. A separate scorer replayed 15 saved trajectories across all
six model checkpoints using full-sequence forward passes without KV caches.
Its largest absolute Output A to Output B score discrepancy was 0.00210
bits/token. These spot checks support the implementation but are not bitwise
reproduction or an exhaustive inference audit.

The new run's actual runner hash matches its archived source. The previous
1.7B run's generating-script identity remains unresolved; that older limitation
is preserved rather than treated as fixed by the spot checks.

A network metadata lookup interrupted this run after the first 4,000 rollouts.
It resumed from the validated data with cached models in offline mode, with
identical code and experiment settings. No completed rollouts were discarded.

See [publication provenance and audit](../../provenance/qwen06b/README.md),
[primary statistics](checkpoint_statistics.json), and
[sensitivity checks](sensitivity_checks.json).

## Files and reproduction

- `rollouts.parquet`: 12,000 raw records, including token IDs and context entropies.
- `question_means.parquet`: 600 question/checkpoint summaries.
- `config.json`: pinned models, input/source hashes, seeds, and environment.
- `checkpoint_mean_intervals.parquet` and `checkpoint_comparisons.parquet`: plot inputs.
- `checkpoint_statistics.json` and `sensitivity_checks.json`: primary and sensitivity results.
- `checkpoint_means.png` and `checkpoint_means.pdf`: the published figures.

From the repository root, without loading any model:

```sh
python3 -m pip install -r requirements-analysis.txt
python3 analyze_checkpoint_statistics.py \
  --results-dir results/qwen06b_gsm8k200_20rollouts \
  --input data/gsm8k200/gsm8k200_injections.parquet --allow-missing-pairs
python3 plot_checkpoint_means.py --results-dir results/qwen06b_gsm8k200_20rollouts
```

The analysis command validates the raw records and grouped means before
recomputing statistics. Bootstrap resampling uses the recorded fixed seed.
