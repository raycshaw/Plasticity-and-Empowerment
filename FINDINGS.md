# Findings and evidence

## Qwen3-0.6B: completed October 31, 2026

The new 12,000-rollout experiment is published with its [full explanation,
confidence intervals, adjusted tests, and robustness checks](results/qwen06b_gsm8k200_20rollouts/README.md).
Injection to Output B declines at every stage, as in the previous experiment.
Output A to Output B declines early and then shows no detectable later change;
its final confidence interval includes zero. This supports a decrease in the
measured correction-entropy proxy, not a demonstrated behavioral loss of
plasticity or a general reversal of formal empowerment.

The [audit record](provenance/qwen06b/README.md) distinguishes exact saved-data
verification from limited model replays and the older run's unresolved source
identity. The previous results below remain separate.

## Previous 1.7B sampled checkpoint run

The October 30, 2026 run contains all 3,000 planned rollouts and 150
question/checkpoint summaries. See [methods](METHODS.md) for the estimand and
[provenance](provenance/README.md) for the unresolved historical runner hash.
Results below are internally verified saved-data findings, subject to that limit.

Differences are later checkpoint minus earlier checkpoint, in bits/token.
Intervals are paired question-bootstrap pointwise 95% intervals. P-values are
from two-sided question-paired t-tests, Holm-adjusted over these nine comparisons.

| Score | Steps | Mean difference | 95% interval | Adjusted p | p < 0.05 |
|---|---|---:|---:|---:|---|
| Prompt ablation | 0 to 50 | +0.180957 | [+0.108260, +0.256926] | 0.000142872 | yes |
| Prompt ablation | 0 to 100 | +0.197784 | [+0.093456, +0.306078] | 0.00412327 | yes |
| Prompt ablation | 50 to 100 | +0.016827 | [-0.045186, +0.080736] | 0.604139 | no |
| Correction ablation | 0 to 50 | -0.218211 | [-0.248583, -0.188437] | 4.72012e-18 | yes |
| Correction ablation | 0 to 100 | -0.261247 | [-0.289616, -0.233416] | 3.45249e-22 | yes |
| Correction ablation | 50 to 100 | -0.043036 | [-0.068701, -0.016491] | 0.0120002 | yes |
| Earlier-output ablation | 0 to 50 | +0.016354 | [-0.006328, +0.038389] | 0.327472 | no |
| Earlier-output ablation | 0 to 100 | +0.033540 | [+0.008342, +0.058899] | 0.0541447 | no |
| Earlier-output ablation | 50 to 100 | +0.017186 | [-0.002042, +0.035654] | 0.249039 | no |

The positive pointwise interval for the earlier-output 0-to-100 difference does
not overturn its adjusted p-value. These are different procedures, and the
interval has no multiplicity adjustment. Failure to pass the threshold also
does not establish that the underlying difference is zero.

### Completion and selection

| Checkpoint | Recorded | A ended early | B ended early after complete A | Complete A+B | Completion rate |
|---|---:|---:|---:|---:|---:|
| 0 | 1000 | 14 | 6 | 980 | 98.0% |
| 50 | 1000 | 20 | 33 | 947 | 94.7% |
| 100 | 1000 | 1 | 29 | 970 | 97.0% |

Input means use 986, 980, and 999 completed A spans, respectively. Correction
and earlier-output means use the 980, 947, and 970 completed A+B spans.
Question weighting is equal even when available rollout counts differ. Selection
on completion can change which trajectories enter the comparison. The 20-token
windows do not evaluate a complete mathematical answer, and lower correction
entropy difference does not demonstrate that the model ignores the correction.

### Robustness checks added during publication

Two exploratory checks recompute the same statistics. They are sensitivity
analyses of the same data, not independent replications, and their test families
are corrected separately rather than jointly with the primary analysis.

| Analysis, step 0 to 100 | Prompt difference (adjusted p) | Correction difference (adjusted p) | Earlier-output difference (adjusted p) |
|---|---:|---:|---:|
| Same complete A+B rollouts for all metrics | +0.200639 (0.003417) | -0.261247 (3.45249e-22) | +0.033540 (0.054145) |
| Exclude the two source-issue questions, 48 remaining | +0.223256 (0.000780) | -0.260753 (5.32684e-21) | +0.032085 (0.071284) |

The two excluded IDs are `test:1309` (incorrect original reference arithmetic)
and `test:255` (inconsistent stall counts in the original question). Reference
answers do not enter the model context or these scores, but the exclusion checks
whether those items drive the result. They do not drive the main directions or
the 0-to-100 adjusted significance decisions. Completion conditioning and
selection of long Qwen responses remain limitations in both checks.

Machine-readable evidence: [primary statistics](results/sampled_rlvr_50x20/checkpoint_statistics.json)
and [sensitivity checks](results/sampled_rlvr_50x20/sensitivity_checks.json).

## Earlier experiments: diagnostic lessons only

### Qwen source-prior conditional-information estimates

A separate 50-question Qwen2.5-0.5B-Instruct run used one factual 20+20-token
trajectory per question and 64 alternative source samples. It specified a model
for possible question/output/correction blocks and estimated conditional-MI
terms along observed histories. It did not use the same estimator, model, EOS
policy, or replication design as the checkpoint experiment.

The stored averages were 0.272964 bits/token for input to A, 0.032543 for A to B,
and 0.293507 for correction to B. These are **finite-particle, trajectory-conditioned
diagnostics with no demonstrated convergence**, not headline population DI results.
The stored correction is fixed experimentally rather than drawn from the assumed
correction prior, so averaging these trajectories does not generally estimate
unconditional DI under that prior.

Useful diagnostics from the saved records:

- The A-source posterior ESS fell below 2 in 8/50 questions.
- The correction-source posterior ESS fell below 2 in 25/50 questions.
- 74.3125% of sampled correction sources hit the 64-token cap.
- Mean absolute split-half differences were 0.015535 bits/token for A to B and
  0.051943 for correction to B, substantial relative to their reported means.

ESS is effective sample size. A low value means a few sampled alternatives
carry most posterior weight. Small estimated values with that behavior cannot
establish negligible information. The input calculation enumerated all 50
questions, so its posterior concentration is a different issue from finite-particle
source collapse in the two B estimators.

The [saved summary](results/earlier_diagnostics/directed_information_summary.json)
and [raw records](results/earlier_diagnostics/directed_information.parquet) are
retained to support this diagnosis. Their historical `di` field names and saved
unit-test count are not certifications of estimator convergence. The unvalidated
runner is not an active entry point in this publication.

### Answer and correction pilot

The earlier Qwen pilot had usable numerical-answer scores for only 1 of 6 GSM8K
responses. That one parsed answer was 395 against a reference of 375, despite a
reasoning JS score of 0.739788 bits. Unsupported formatting, missing markers, and
truncation limited coverage. Dependence on earlier text is not correctness.

On one correction example with three seeds, unchanged-fact reminders had greater
JS divergence than actual corrections in two seeds. Mean correction JS was
0.074705 for changed facts versus 0.075585 for unchanged reminders. Empty-content
and identical-context controls gave zero differences. Those controls validate
numerical behavior, not the interpretation of JS as successful updating.

The [pilot records](results/earlier_diagnostics/answer_pilot.json) are retained.
The pilot is too small and uses a different protocol to support a training trend
or a general behavioral conclusion.

### Excluded evidence

The single-question greedy checkpoint run was a pipeline smoke test. It suppressed
EOS, unlike the sampled experiment. It is not pooled with the sampled results or
used to infer a checkpoint trend. Earlier superseded entropy/DI artifacts that
mixed screening prefixes, EOS padding, or incompatible source definitions are
not included in the evidence set.

## What would support stronger conclusions

A behavioral study should measure whether corrections are adopted and final
answers improve, with unchanged-fact and irrelevant-text controls. It should
report completion/coverage alongside scores, use an unscreened or explicitly
stratified sample, and repeat across independent training runs. A formal
information estimate also needs an explicit source law, justified conditioning,
and demonstrated estimator convergence. Those experiments have not been done
by the current analysis.
