"""Analyze validated question means, with paired tests and completion diagnostics."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import ttest_rel

from verify_results import RESULTS, SOURCE, verify

METRICS = ('input', 'injection', 'empowerment')
BOOTSTRAPS = 50000
SEED = 20261030


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.empty(len(order))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        np.asarray(pvalues)[order] * np.arange(len(order), 0, -1)))
    return adjusted


def analyze(rows, suffix='', exclude_ids=(), allow_missing=False):
    steps = sorted({r['rlvr_step'] for r in rows})
    groups = {step: {r['id']: r for r in rows if r['rlvr_step'] == step and r['id'] not in exclude_ids} for step in steps}
    all_ids = sorted(groups[steps[0]])
    if len(all_ids) < 2 or any(set(groups[s]) != set(all_ids) for s in steps):
        raise ValueError('Need at least two paired questions at every checkpoint')
    means, tests, omitted = [], [], {}
    for metric in METRICS:
        raw_values = np.array([[groups[s][i][metric + suffix + '_mean'] for i in all_ids] for s in steps], dtype=float)
        valid = np.isfinite(raw_values).all(axis=0)
        if not valid.all() and not allow_missing:
            raise ValueError('Every question needs a finite mean; missing groups cannot be silently dropped')
        ids = [i for i, keep in zip(all_ids, valid) if keep]
        omitted[metric] = [i for i, keep in zip(all_ids, valid) if not keep]
        if len(ids) < 2:
            raise ValueError('Need at least two complete question pairs per metric')
        values = raw_values[:, valid]
        # Keep historical bootstrap draws identical across metrics and reruns.
        rng = np.random.default_rng(SEED)
        draws = rng.integers(0, len(ids), size=(BOOTSTRAPS, len(ids)))
        boot = values[:, draws].mean(axis=2)
        for j, step in enumerate(steps):
            low, high = np.quantile(boot[j], [0.025, 0.975])
            means.append(dict(metric=metric, rlvr_step=step, n_questions=len(ids), mean=float(values[j].mean()),
                              ci95_low=float(low), ci95_high=float(high)))
        for a, b in itertools.combinations(range(len(steps)), 2):
            difference = values[b] - values[a]
            test = ttest_rel(values[b], values[a])
            low, high = np.quantile(boot[b]-boot[a], [0.025, 0.975])
            tests.append(dict(metric=metric, step_from=steps[a], step_to=steps[b], n_pairs=len(ids),
                              mean_difference=float(difference.mean()), ci95_low=float(low), ci95_high=float(high),
                              t_statistic=float(test.statistic), df=len(ids)-1, p_raw=float(test.pvalue)))
    if not all(np.isfinite(r['p_raw']) for r in tests):
        raise ValueError('Degenerate paired test; cannot report a finite p-value')
    for row, p in zip(tests, holm([r['p_raw'] for r in tests])):
        row.update(p_holm=float(p), significant_005=bool(p < 0.05))
    return dict(means=means, comparisons=tests, omitted_question_ids_by_metric=omitted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=RESULTS)
    parser.add_argument('--input', type=Path, default=SOURCE)
    parser.add_argument('--allow-missing-pairs', action='store_true', help='Explicitly report and omit questions missing a mean at any checkpoint, separately per metric')
    args = parser.parse_args()
    completion = verify(args.results_dir, args.input)
    rows = pq.read_table(args.results_dir / 'question_means.parquet').to_pylist()
    result = analyze(rows, allow_missing=args.allow_missing_pairs)
    method = dict(method='50,000 question bootstrap percentile 95% CIs; two-sided paired t-tests across question means; Holm correction over all nine primary tests',
                  limitations='One training lineage; see dataset selection cohorts; fixed-length scores conditional on completion. CIs are pointwise, not simultaneous.',
                  missing_pairs='Explicit per-metric paired exclusion with omitted IDs reported' if args.allow_missing_pairs else 'Any missing question mean is an error',
                  bootstrap_seed=SEED, bootstrap_replicates=BOOTSTRAPS,
                  weighting='Equal weight per question after averaging available fixed-horizon rollouts within question/checkpoint')
    metadata = {key.encode(): str(value).encode() for key, value in method.items()}
    for key, filename in [('means', 'checkpoint_mean_intervals'), ('comparisons', 'checkpoint_comparisons')]:
        pq.write_table(pa.Table.from_pylist(result[key]).replace_schema_metadata(metadata), args.results_dir / (filename + '.parquet'))
    (args.results_dir / 'checkpoint_statistics.json').write_text(json.dumps(dict(method=method, **result), indent=2) + '\n')
    flagged = [r['id'] for r in pq.read_table(args.input).to_pylist() if r['source_issue']]
    sensitivity = dict(note='Exploratory robustness checks, not independent replications. Each analysis adjusts its own nine tests; no correction across analyses.',
                       completion_counts=completion,
                       joint_complete=analyze(rows, suffix='_joint_complete', allow_missing=args.allow_missing_pairs),
                       excluding_source_issues=dict(excluded_ids=flagged, **analyze(rows, exclude_ids=flagged, allow_missing=args.allow_missing_pairs)))
    observed_ids = {r['id'] for r in rows}
    source_rows = [r for r in pq.read_table(args.input).to_pylist() if r['id'] in observed_ids]
    cohorts = sorted({r.get('cohort') for r in source_rows if r.get('cohort')})
    if cohorts:
        sensitivity['cohorts'] = {cohort: analyze(rows, exclude_ids=[r['id'] for r in source_rows if r.get('cohort') != cohort],
                                                 allow_missing=args.allow_missing_pairs) for cohort in cohorts}
    (args.results_dir / 'sensitivity_checks.json').write_text(json.dumps(sensitivity, indent=2) + '\n')
    for r in result['comparisons']:
        print(f"{r['metric']:12} {r['step_from']:3}->{r['step_to']:3}: delta={r['mean_difference']:+.6f}, CI=[{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}], p_Holm={r['p_holm']:.6g}")


if __name__ == '__main__':
    main()
