"""Plot equal-weight question means and intervals from the checkpoint analysis."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results-dir', type=Path, default=Path(__file__).resolve().parent / 'results/sampled_rlvr_50x20')
root = parser.parse_args().results_dir
intervals = pq.read_table(root / 'checkpoint_mean_intervals.parquet').to_pylist()
steps = sorted({r['rlvr_step'] for r in intervals})
comparisons = pq.read_table(root / 'checkpoint_comparisons.parquet').to_pylist()
question_counts = {r['n_questions'] for r in intervals}
n_questions = str(next(iter(question_counts))) if len(question_counts) == 1 else f'{min(question_counts)}-{max(question_counts)} paired'
metrics = [('input_mean', 'Input to Output A', '#2563eb'),
           ('injection_mean', 'Injection to Output B', '#d97706'),
           ('empowerment_mean', 'Output A to Output B', '#0d9488')]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 3, figsize=(13, 6))
for ax, (key, title, color) in zip(axes, metrics):
    metric = key.removesuffix('_mean')
    ci = [next(r for r in intervals if r['metric'] == metric and r['rlvr_step'] == step) for step in steps]
    values = [r['mean'] for r in ci]
    ax.errorbar(steps, values, yerr=[[v-r['ci95_low'] for v,r in zip(values,ci)],
                                   [r['ci95_high']-v for v,r in zip(values,ci)]],
                color=color, linewidth=2.4, marker='o', markersize=7, capsize=5)
    notes = []
    for test in [r for r in comparisons if r['metric'] == metric]:
        p = test['p_holm']
        label = f'{p:.1e}' if p < .001 else f'{p:.3f}'
        notes.append(f"{test['step_from']} → {test['step_to']}: p = {label}" + (' *' if test['significant_005'] else ''))
    ax.text(0.5, -0.27, '\n'.join(notes), transform=ax.transAxes, ha='center', va='top', fontsize=10, linespacing=1.5)
    for x, y in zip(steps, values):
        ax.annotate(f'{y:.4f}', (x, y), xytext=(0, -19), textcoords='offset points',
                    ha='center', fontsize=10, color=color)
    ax.set(title=title, xlabel='GSM8K RLVR training step', xticks=steps)
    margin = (max(steps)-min(steps))*0.12
    ax.set_xlim(min(steps)-margin, max(steps)+margin)
    lower = min(0, min(r['ci95_low'] for r in ci))
    upper = max(0, max(r['ci95_high'] for r in ci))
    padding = (upper-lower)*0.15 or 0.01
    ax.set_ylim(lower-padding if lower < 0 else 0, upper+padding)
    ax.grid(axis='y', color='#e2e8f0', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel('Mean entropy difference (bits/token)')
fig.suptitle('Entropy measurements across RLVR checkpoints', fontsize=17, fontweight='bold', y=0.98)
fig.text(0.5, 0.025, f'Error bars: pointwise 95% bootstrap CIs across {n_questions} questions; separate y-axis scales.\n'
         'Two-sided paired t-tests; p-values Holm-adjusted across all 9 comparisons. * adjusted p < 0.05.\n'
         'Fixed-length scores conditional on completion; one training run, no across-training-seed uncertainty.',
         ha='center', fontsize=9, color='#475569', linespacing=1.5)
fig.tight_layout(rect=(0, 0.12, 1, 0.92), w_pad=2)
fig.savefig(root / 'checkpoint_means.png', dpi=200, facecolor='white')
fig.savefig(root / 'checkpoint_means.pdf', facecolor='white')
print(root / 'checkpoint_means.png')
