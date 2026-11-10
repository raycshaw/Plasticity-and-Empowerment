"""Read-only validation of the November OLMo publication, without model inference."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
FREE = ROOT / 'results/olmo2_1b_a100_full_b_50q_20261105'
FIXED = ROOT / 'results/olmo2_1b_fixed_step0_a_50q_5trials_20261106'
JSD = ROOT / 'results/olmo2_fixed_a_jsd_20261109'


def key(row):
    return row['step'], row['id'], row['trial']


def check_entropy(rows):
    assert len({key(r) for r in rows}) == len(rows)
    for r in rows:
        n = r['scored_b_count']
        assert n == min(100, r['b']['content_token_count']) and n > 0
        e = np.asarray(r['b']['entropies'][:n])
        assert e.shape == (n, 3) and np.isfinite(e).all()
        for i, label in [(1, 'plasticity'), (2, 'empowerment')]:
            assert abs(float((e[:, i]-e[:, 0]).mean())-r[label+'_bits_per_token']) < 1e-6


def main():
    free = pq.read_table(FREE/'rollouts.parquet').to_pylist()
    fixed = pq.read_table(FIXED/'rollouts.parquet').to_pylist()
    matched = pq.read_table(JSD/'matched_rollouts.parquet').to_pylist()
    scores = pq.read_table(JSD/'jsd_scores.parquet').to_pylist()
    assert len(free) == 400 and len(fixed) == 993 and len(matched) == len(scores) == 960
    for rows in [free, fixed, matched]:
        check_entropy(rows)
    originals = {key(r): r for r in fixed}
    selected = {key(r): r for r in matched}
    assert len(selected) == 960 and len({r['id'] for r in matched}) == 48
    assert {key(r) for r in scores} == set(selected)
    reference_a = {}
    for r in fixed:
        pair = r['id'], r['trial']
        a = r['a']['token_ids'][:r['a']['content_token_count']]
        if pair in reference_a:
            assert a == reference_a[pair]
        reference_a[pair] = a
    for r in matched:
        assert r == originals[key(r)]
    for s in scores:
        r = selected[key(s)]
        assert s['scored_b_count'] == r['scored_b_count']
        assert s['max_entropy_error_bits'] < .001
        for arr, mean in [('correction_jsd_bits', 'correction_jsd_mean_bits'), ('a_jsd_bits', 'a_jsd_mean_bits')]:
            vals = np.asarray(s[arr])
            assert len(vals) == s['scored_b_count'] and np.isfinite(vals).all()
            assert vals.min() >= -1e-6 and vals.max() <= 1.000001
            assert abs(vals.mean()-s[mean]) < 1e-6
        for metric in ['plasticity', 'empowerment']:
            assert s['original_'+metric+'_bits_per_token'] == r[metric+'_bits_per_token']
    report = json.loads((JSD/'analysis/comparison.json').read_text())
    for item in report['summary']:
        group = [s for s in scores if s['step'] == item['step']]
        assert len(group) == 240
        assert abs(np.mean([s[item['metric']] for s in group])-item['mean']) < 1e-10
    audit = json.loads((JSD/'analysis/injection_value_screen/summary.json').read_text())
    assert audit['screened'] == sum(not r['source_issue'] for r in matched) == 920
    assert audit['confirmed_old_value_cases'] == 7
    manifest = ROOT/'provenance/november_2026_publication.json'
    if manifest.exists():
        data = json.loads(manifest.read_text())
        for name, digest in data['sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    print('Verified: 400 fresh-A, 993 fixed-A, 960 matched/JSD responses; entropy arithmetic, fixed A, JSD bounds, published means and 920 screened responses.')


if __name__ == '__main__':
    main()
