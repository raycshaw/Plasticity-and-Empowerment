"""Compare paired JSD and entropy-ablation scores on matched saved responses."""
import argparse
import json
import csv
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METRICS = [
    ('original_plasticity_bits_per_token', 'Correction → B', 'Entropy difference (bits/token)'),
    ('original_empowerment_bits_per_token', 'A → B', 'Entropy difference (bits/token)'),
    ('correction_jsd_mean_bits', 'Correction → B', 'Mean JSD (bits)'),
    ('a_jsd_mean_bits', 'A → B', 'Mean JSD (bits)'),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    rows = pq.read_table(run / 'jsd_scores.parquet').to_pylist()
    manifest = json.loads((run / 'manifest.json').read_text())
    steps = manifest['checkpoints']
    ids = manifest['questions']
    assert len(rows) == manifest['expected_scores'] == 960
    by_key = {(r['step'], r['id'], r['trial']): r for r in rows}
    assert len(by_key) == len(rows)
    # Average trials within each question before giving every question equal weight.
    values = np.array([
        [[np.mean([by_key[(step, q, t)][metric] for t in range(5)])
          for step in steps] for q in ids]
        for metric, _, _ in METRICS
    ])
    assert values.shape == (4, 48, 4) and np.isfinite(values).all()
    rng = np.random.default_rng(20261109)
    indices = rng.integers(0, len(ids), size=(10000, len(ids)))
    bootstrap = values[:, indices, :].mean(axis=2)
    means = values.mean(axis=1)
    lows, highs = np.quantile(bootstrap, [.025, .975], axis=1)
    out = run / 'analysis'
    out.mkdir(exist_ok=True)
    table = []
    endpoints = []
    for m, (metric, _, _) in enumerate(METRICS):
        for s, step in enumerate(steps):
            table.append(dict(metric=metric, step=step, mean=float(means[m, s]),
                              ci95_low=float(lows[m, s]), ci95_high=float(highs[m, s]),
                              questions=48, responses=240))
        delta_boot = bootstrap[m, :, -1] - bootstrap[m, :, 0]
        lo, hi = np.quantile(delta_boot, [.025, .975])
        endpoints.append(dict(metric=metric, comparison='step 2600 minus step 0',
                              difference=float(means[m, -1] - means[m, 0]),
                              ci95_low=float(lo), ci95_high=float(hi)))
    correlations = []
    for s, step in enumerate(steps):
        for a, b in [(0, 2), (1, 3)]:
            rho = float(spearmanr(values[a, :, s], values[b, :, s]).statistic)
            correlations.append(dict(step=step, comparison=METRICS[a][1],
                                     question_mean_spearman=rho if np.isfinite(rho) else None))
    report = dict(
        matched_questions=48, trials_per_question=5, responses_per_checkpoint=240,
        checkpoints=steps, total_responses=len(rows),
        uncertainty='10,000 paired question-cluster bootstrap samples; pointwise 95% intervals; exploratory, unadjusted',
        aggregation='Mean tokens within response, then five trials within question, then equally weighted questions',
        scoring='First 100 available non-EOS B tokens, identical saved B history in full and ablated contexts',
        interpretation='JSD measures distribution sensitivity, not direction, correctness, formal directed information, or established empowerment.',
        selection='48 questions fully completed at every checkpoint; excludes two incomplete questions from the stopped run',
        source_run=manifest['source_run'],
        max_entropy_reproduction_error_bits=max(r['max_entropy_error_bits'] for r in rows),
        min_scored_B_tokens=min(r['scored_b_count'] for r in rows),
        max_scored_B_tokens=max(r['scored_b_count'] for r in rows),
        summary=table, endpoint_changes=endpoints, within_checkpoint_correlations=correlations,
    )
    (out / 'comparison.json').write_text(json.dumps(report, indent=2))
    with (out / 'checkpoint_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    with (out / 'paired_endpoint_changes.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(endpoints[0])); writer.writeheader(); writer.writerows(endpoints)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelcolor': '#52637a', 'xtick.color': '#52637a',
                         'ytick.color': '#52637a', 'axes.titleweight': 'bold'})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ['#326bb4', '#dd802e']
    for m, ax in enumerate(axes.flat):
        metric, title, ylabel = METRICS[m]
        color = colors[m % 2]
        ax.errorbar(steps, means[m], yerr=np.vstack((means[m]-lows[m], highs[m]-means[m])),
                    fmt='o-', color=color, linewidth=2.5, markersize=7, capsize=5)
        ax.set_xticks(steps)
        ax.set_xlabel('Additional RLVR training steps')
        ax.set_ylabel(ylabel)
        ax.set_title(('Original entropy proxy: ' if m < 2 else 'JSD: ') + title, loc='left', pad=14)
        ax.grid(axis='y', color='#e3eaf1')
        ax.set_axisbelow(True)
        if m < 2:
            ax.axhline(0, color='#9aacbe', linestyle=':', linewidth=1)
        else:
            ax.set_ylim(bottom=0)
    fig.suptitle('Do JSD and entropy differences tell the same story?',
                 fontsize=22, fontweight='bold', color='#182f53', x=.07, ha='left', y=.985)
    fig.text(.07, .94, 'OLMo 2 1B • Same 48 questions × 5 trials × 4 checkpoints • Fixed A, saved B responses',
             color='#52637a', fontsize=12)
    fig.text(.07, .065, 'Means over the first 100 available B tokens. Bars: 95% question-bootstrap intervals; separate y-axis scales.',
             color='#52637a', fontsize=10)
    fig.text(.07, .04, 'JSD measures distribution change. These are sensitivity proxies, not validated plasticity or empowerment values.',
             color='#52637a', fontsize=10)
    fig.subplots_adjust(left=.08, right=.98, top=.86, bottom=.15, hspace=.50, wspace=.26)
    fig.savefig(out / 'jsd_vs_entropy.png', dpi=180)
    fig.savefig(out / 'jsd_vs_entropy.pdf')
    plt.close(fig)
    print(json.dumps({'analysis_complete': True, 'output_directory': str(out),
                      'max_entropy_error_bits': report['max_entropy_reproduction_error_bits'],
                      'endpoint_changes': endpoints}), flush=True)


if __name__ == '__main__':
    main()
