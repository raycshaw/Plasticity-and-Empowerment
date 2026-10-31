"""Audit saved rollouts and summaries without loading models or downloading weights."""
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from experiment_config import ROOT, SOURCE, CHECKPOINTS

RESULTS = ROOT / 'results/sampled_rlvr_50x20'
METRICS = ('input', 'injection', 'empowerment')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(actual, expected, label):
    if expected is None:
        require(actual is None, label)
    elif isinstance(expected, float):
        require(actual is not None and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), label)
    else:
        require(actual == expected, label)


def validate_rows(rows, summaries, config, source):
    """Recompute the saved statistics from entropy arrays, including missing scores."""
    n, repeats = config['questions'], config['rollouts']
    checkpoints = {step: (repo, revision) for step, repo, revision in config['checkpoints']}
    expected = {(step, i, j) for step in checkpoints for i in range(n) for j in range(repeats)}
    keys = [(r['rlvr_step'], r['source_row'], r['rollout_index']) for r in rows]
    require(len(keys) == len(set(keys)) and set(keys) == expected, 'Missing, duplicate, or unexpected rollouts')
    require(len(source) >= n and len({r['id'] for r in source[:n]}) == n, 'Invalid source IDs')
    groups = {}
    for r in rows:
        step, i, j = r['rlvr_step'], r['source_row'], r['rollout_index']
        for key, value in source[i].items():
            equal(r[key], value, f'Source field {key} changed')
        equal((r['model_repo'], r['model_revision']), checkpoints[step], 'Checkpoint mismatch')
        equal(r['seed'], config['seed'] + step * 10000000 + i * 10000 + j, 'Seed mismatch')
        for span, width in [('a', 2), ('b', 3)]:
            tokens = r[f'{span}_token_ids']
            if tokens is None:
                require(span == 'b' and not r['a_complete'], 'Unexpected missing span')
                require(not r['b_complete'], 'Missing B marked complete')
                require(all(r[f'b_{k}'] is None for k in ('entropies', 'content_token_count', 'ended_at_eos',
                            'eos_probabilities', 'sampled_token_log2_probabilities')), 'Missing B has diagnostics')
                continue
            count = len(tokens) - int(r[f'{span}_ended_at_eos'])
            require(1 <= len(tokens) <= 20, 'Invalid span length')
            equal(r[f'{span}_content_token_count'], count, 'Incorrect token count')
            equal(r[f'{span}_complete'], count == 20, 'Incorrect completion flag')
            require(count == 20 or r[f'{span}_ended_at_eos'], 'Short span without EOS')
            for field in ('entropies', 'eos_probabilities', 'sampled_token_log2_probabilities'):
                values = r[f'{span}_{field}']
                require(len(values) == len(tokens), 'Diagnostic/token alignment mismatch')
                require(all(len(v) == width and all(math.isfinite(x) for x in v) for v in values), 'Invalid diagnostics')
                if field == 'eos_probabilities':
                    require(all(0 <= x <= 1 for v in values for x in v), 'Invalid EOS probability')
        status = 'complete' if r['b_complete'] else ('b_early_eos' if r['a_complete'] else 'a_early_eos')
        equal(r['status'], status, 'Incorrect status')
        for metric, span, column in [('input', 'a', 1), ('injection', 'b', 1), ('empowerment', 'b', 2)]:
            entropies = r[f'{span}_entropies'] or []
            terms = [v[column] - v[0] for v in entropies]
            total = sum(terms) if terms else None
            equal(r[f'{metric}_terms_bits'], terms, 'Entropy differences do not match raw arrays')
            equal(r[f'{metric}_scored_positions'], len(terms), 'Incorrect score count')
            equal(r[f'{metric}_observed_total_bits'], total, 'Incorrect observed total')
            equal(r[f'{metric}_observed_bits_per_token'], total / len(terms) if terms else None, 'Incorrect observed mean')
            equal(r[f'{metric}_total_20_bits'], total if r[f'{span}_complete'] else None, 'Incorrect fixed total')
            equal(r[f'{metric}_bits_per_token'], total / 20 if r[f'{span}_complete'] else None, 'Incorrect fixed mean')
        groups.setdefault((step, i), []).append(r)
    summary_keys = [(r['rlvr_step'], r['source_row']) for r in summaries]
    require(len(summary_keys) == len(set(summary_keys)) and set(summary_keys) == set(groups), 'Invalid summary groups')
    for s in summaries:
        group = groups[s['rlvr_step'], s['source_row']]
        for key in ('id', 'question', 'injection', 'model_repo', 'model_revision'):
            equal(s[key], group[0][key], f'Summary {key} mismatch')
        a = sum(r['a_complete'] for r in group)
        b = sum(r['b_complete'] for r in group)
        for key, value in dict(rollouts_expected=repeats, rollouts_recorded=repeats, aggregation_complete=True,
                              a_complete_count=a, b_complete_count=b, a_early_stop_rate=(repeats-a)/repeats,
                              b_early_stop_rate_given_reached=(a-b)/a if a else None, full_completion_rate=b/repeats).items():
            equal(s[key], value, f'Incorrect summary {key}')
        for metric in METRICS:
            for suffix in ('', '_joint_complete', '_observed'):
                field = f'{metric}_observed_bits_per_token' if suffix == '_observed' else f'{metric}_bits_per_token'
                values = [r[field] for r in group if r[field] is not None and (suffix != '_joint_complete' or r['b_complete'])]
                sd = statistics.stdev(values) if len(values) > 1 else None
                stats = dict(n=len(values), mean=statistics.mean(values) if values else None,
                             sd=sd, sem=sd / math.sqrt(len(values)) if sd is not None else None)
                for key, value in stats.items():
                    equal(s[f'{metric}{suffix}_{key}'], value, f'Incorrect {metric}{suffix}_{key}')
    return {step: dict(Counter(r['status'] for r in rows if r['rlvr_step'] == step)) for step in checkpoints}


def verify(directory=RESULTS, source_path=SOURCE):
    config = json.loads((directory / 'config.json').read_text())
    equal(hashlib.sha256(source_path.read_bytes()).hexdigest(), config['source_sha256'], 'Dataset hash mismatch')
    specification = config.get('experiment_specification')
    expected = specification['checkpoints'] if specification else [list(c) for c in CHECKPOINTS]
    equal(config['checkpoints'], expected, 'Unexpected checkpoints')
    if specification:
        equal(hashlib.sha256((ROOT / 'sampled_rlvr_experiment.py').read_bytes()).hexdigest(),
              config['script_sha256'], 'Runner source changed since run started')
    tables = [pq.read_table(directory / name) for name in ('rollouts.parquet', 'question_means.parquet')]
    for table in tables:
        equal(json.loads(table.schema.metadata[b'experiment_config']), config, 'Embedded configuration mismatch')
    return validate_rows(*(t.to_pylist() for t in tables), config, pq.read_table(source_path).to_pylist())


def verify_manifest():
    for name, digest in json.loads((ROOT / 'provenance/artifact_sha256.json').read_text()).items():
        equal(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, f'Artifact changed: {name}')


if __name__ == '__main__':
    verify_manifest()
    print(json.dumps(verify(), indent=2))
    print('Verified artifact hashes, complete rollout grid, source fields, entropy arithmetic, and all question summaries.')
    print('Historical source-code hash mismatch is documented in provenance/README.md; not certified by this audit.')
