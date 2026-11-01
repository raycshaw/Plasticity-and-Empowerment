"""Sample independent GSM8K rollouts and save raw measurements and grouped means."""
import argparse
import gc
import hashlib
import json
import math
import statistics
import platform
from importlib.metadata import version
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from experiment_config import CHECKPOINTS, SOURCE

METRICS = ('input', 'injection', 'empowerment')
HORIZON = 20


def chat(tokenizer, prompt, injection=None, chat_mode='native'):
    messages = [{'role': 'user', 'content': prompt}]
    if injection is not None:
        messages += [{'role': 'assistant', 'content': ''}, {'role': 'user', 'content': injection}]
    kwargs = {'enable_thinking': False} if chat_mode == 'qwen3_nonthinking' else {}
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False, **kwargs)


def continuation_prefix(tokenizer, prompt, injection, tokens, chat_mode='native'):
    """Splice generated IDs exactly, retaining Qwen3's forced empty thinking block."""
    full = chat(tokenizer, prompt, chat_mode=chat_mode)
    framing = chat(tokenizer, prompt, injection, chat_mode)
    boundary = len(full)
    if chat_mode == 'qwen3_nonthinking':
        thinking = tokenizer.encode('<think>\n\n</think>\n\n', add_special_tokens=False)
        if full[-len(thinking):] != thinking:
            raise ValueError('Expected the Qwen3 non-thinking generation suffix')
        # The native multi-turn template omits this forced suffix in historical
        # empty assistant messages. Keep it in A, including the empty-A baseline.
        boundary -= len(thinking)
    if framing[:boundary] != full[:boundary]:
        raise ValueError('Chat template does not support exact token splicing')
    return full + tokens + framing[boundary:]


@torch.inference_mode()
def sample_span(model, prefixes, eos, generator, cached=True, forced_tokens=None):
    """Only prefix 0 generates. Each alternative has its own independent cache."""
    if forced_tokens is not None and (len(forced_tokens) != HORIZON or any(t in eos for t in forced_tokens)):
        raise ValueError('Reference A must contain exactly 20 non-EOS tokens')
    if max(map(len, prefixes)) + HORIZON > model.config.max_position_embeddings:
        raise ValueError('Context length exceeded')
    tokens, entropies, eos_probs, token_log_probs = [], [], [], []
    caches = [None] * len(prefixes)
    eos_indices = list(eos)
    for position in range(HORIZON):
        distributions = []
        next_ids = torch.tensor([[tokens[-1]]], device=model.device) if cached and position else None
        for i, prefix in enumerate(prefixes):
            ids = next_ids if next_ids is not None else torch.tensor([prefix + tokens], device=model.device)
            prediction = model(ids,
                               past_key_values=caches[i], use_cache=cached, logits_to_keep=1)
            caches[i] = prediction.past_key_values if cached else None
            log_p = prediction.logits[0, -1].float().log_softmax(-1)
            if not torch.isfinite(log_p).all():
                raise ValueError('Nonfinite model prediction')
            distributions.append(log_p)
        # Reuse probabilities without changing any reduction or sampling arithmetic.
        probabilities = [p.exp() for p in distributions]
        entropy_values = [-(prob * p).sum() / math.log(2) for prob, p in zip(probabilities, distributions)]
        eos_values = [prob[eos_indices].sum() for prob in probabilities]
        # Temperature 1, no filtering, no EOS masking. CPU RNG is local to this rollout.
        token = (forced_tokens[position] if forced_tokens is not None else
                 int(torch.multinomial(probabilities[0].cpu(), 1, generator=generator)))
        # Transfer scalar diagnostics together, avoiding a device sync per scalar.
        diagnostics = torch.stack(entropy_values + eos_values +
                                  [p[token] / math.log(2) for p in distributions]).cpu().tolist()
        count = len(distributions)
        entropies.append(diagnostics[:count])
        eos_probs.append(diagnostics[count:2*count])
        token_log_probs.append(diagnostics[2*count:])
        tokens.append(token)
        if token in eos:
            break
    return dict(token_ids=tokens, entropies=entropies, eos_probabilities=eos_probs,
                sampled_token_log2_probabilities=token_log_probs, ended_at_eos=tokens[-1] in eos,
                content_token_count=len(tokens) - int(tokens[-1] in eos))


def measure(entropies, baseline, complete):
    terms = [row[baseline] - row[0] for row in entropies]
    total = sum(terms) if terms else None
    return dict(terms_bits=terms, scored_positions=len(terms),
                observed_total_bits=total, observed_bits_per_token=total / len(terms) if terms else None,
                total_20_bits=total if complete else None,
                bits_per_token=total / HORIZON if complete else None)


def rollout(model, tokenizer, prompt, injection, seed, cached=True, chat_mode='native',
            fixed_a=None, b_opening_ids=None):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    eos = model.generation_config.eos_token_id
    eos = tokenizer.eos_token_id if eos is None else eos
    eos = {eos} if isinstance(eos, int) else set(eos or [])
    full = chat(tokenizer, prompt, chat_mode=chat_mode)
    a = sample_span(model, [full, chat(tokenizer, '', chat_mode=chat_mode)], eos, generator, cached, forced_tokens=fixed_a)
    a_complete = a['content_token_count'] == HORIZON
    result = dict(seed=seed, a=a, a_complete=a_complete, b=None, b_complete=False,
                  input=measure(a['entropies'], 1, a_complete))
    if a_complete:
        prefixes = []
        for span, message in ((a['token_ids'], injection), (a['token_ids'], ''), ([], injection)):
            prefixes.append(continuation_prefix(tokenizer, prompt, message, span, chat_mode) + list(b_opening_ids or []))
        b = sample_span(model, prefixes, eos, generator, cached)
        result.update(b=b, b_complete=b['content_token_count'] == HORIZON)
    b_entropies = result['b']['entropies'] if result['b'] else []
    result.update(injection=measure(b_entropies, 1, result['b_complete']),
                  empowerment=measure(b_entropies, 2, result['b_complete']))
    result['status'] = 'complete' if result['b_complete'] else ('b_early_eos' if a_complete else 'a_early_eos')
    return result


def flatten(result, tokenizer):
    fields = {k: result[k] for k in ('seed', 'status', 'a_complete', 'b_complete')}
    for span_name in ('a', 'b'):
        span = result[span_name]
        fields[f'{span_name}_text'] = tokenizer.decode(span['token_ids'], skip_special_tokens=True) if span else None
        for key in ('token_ids', 'entropies', 'eos_probabilities', 'sampled_token_log2_probabilities',
                    'ended_at_eos', 'content_token_count'):
            fields[f'{span_name}_{key}'] = span[key] if span else None
    for name in METRICS:
        fields.update({f'{name}_{key}': value for key, value in result[name].items()})
    return fields


def estimate(values):
    values = [v for v in values if v is not None]
    n = len(values)
    sd = statistics.stdev(values) if n > 1 else None
    return dict(n=n, mean=statistics.mean(values) if n else None,
                sd=sd, sem=sd / math.sqrt(n) if n > 1 else None)


def summarize(rows, expected):
    summaries = []
    groups = {}
    for row in rows:
        groups.setdefault((row['rlvr_step'], row['source_row']), []).append(row)
    for key in sorted(groups):
        group = groups[key]
        first = group[0]
        n_a = sum(r['a_complete'] for r in group)
        n_b = sum(r['b_complete'] for r in group)
        summary = {k: first[k] for k in ('id', 'question', 'injection', 'source_row', 'rlvr_step', 'model_repo', 'model_revision')}
        summary.update(rollouts_expected=expected, rollouts_recorded=len(group),
                       aggregation_complete=len(group) == expected, a_complete_count=n_a, b_complete_count=n_b,
                       a_early_stop_rate=(len(group)-n_a)/len(group),
                       b_early_stop_rate_given_reached=(n_a-n_b)/n_a if n_a else None,
                       full_completion_rate=n_b/len(group))
        for name in METRICS:
            # Primary means require the full 20-token target span. Never impute zeros.
            for suffix, values in [('', [r[f'{name}_bits_per_token'] for r in group]),
                                   ('_joint_complete', [r[f'{name}_bits_per_token'] for r in group if r['b_complete']]),
                                   ('_observed', [r[f'{name}_observed_bits_per_token'] for r in group])]:
                summary.update({f'{name}{suffix}_{key}': value for key, value in estimate(values).items()})
        summaries.append(summary)
    return summaries


def save_tables(rows, directory, metadata, expected):
    for name, values in [('rollouts', rows), ('question_means', summarize(rows, expected))]:
        table = pa.Table.from_pylist(values).replace_schema_metadata(metadata)
        temporary = directory / f'{name}.tmp.parquet'
        pq.write_table(table, temporary)
        temporary.replace(directory / f'{name}.parquet')



def load_reference_a(path, questions, count, step):
    """Select existing A spans by exact case and trial identity, never resample."""
    wanted = {q['id']: q for q in questions}
    selected = {}
    for row in pq.read_table(path).to_pylist():
        key = (row['id'], row['rollout_index'])
        if row['rlvr_step'] != step or row['id'] not in wanted or not 0 <= row['rollout_index'] < count:
            continue
        if key in selected:
            raise ValueError('Duplicate reference trajectory')
        q = wanted[row['id']]
        if any(row[k] != q[k] for k in ('question', 'injection')):
            raise ValueError('Reference question or correction differs')
        ids = row['a_token_ids']
        if (not row['a_complete'] or row['a_ended_at_eos'] or len(ids) != HORIZON
                or any(not isinstance(t, int) or t < 0 for t in ids)):
            raise ValueError('Reference A must be a completed 20-token span')
        selected[key] = row
    if len(selected) != len(questions) * count:
        raise ValueError('Missing reference A spans; no silent substitution or resampling')
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-config', type=Path, help='Pinned checkpoint manifest; default is the historical experiment')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent / 'runs/sampled_rlvr_50x20')
    parser.add_argument('--rollouts', type=int, default=20)
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--seed', type=int, default=20261030)
    parser.add_argument('--b-opening', default='', help='Supplied assistant opening; excluded from B scores')
    parser.add_argument('--reference-rollouts', type=Path, help='Saved rollouts supplying identical A at every checkpoint')
    parser.add_argument('--reference-step', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    specification = json.loads(args.experiment_config.read_text()) if args.experiment_config else None
    checkpoints = specification['checkpoints'] if specification else CHECKPOINTS
    chat_mode = specification.get('chat_mode', 'native') if specification else 'native'
    if chat_mode not in ('native', 'qwen3_nonthinking'):
        parser.error('Unknown chat mode')
    if (not checkpoints or len({c[0] for c in checkpoints}) != len(checkpoints)
            or any(len(c) != 3 or not isinstance(c[0], int) or c[0] < 0
                   or len(c[2]) != 40 or any(x not in '0123456789abcdef' for x in c[2]) for c in checkpoints)):
        parser.error('Checkpoints require unique nonnegative steps and pinned 40-character revisions')
    if args.input is None:
        args.input = Path(__file__).parent / specification['source'] if specification else SOURCE
    if args.rollouts < 1 or args.limit < 1:
        parser.error('Rollouts and limit must be positive')
    source = pq.read_table(args.input)
    if args.limit > len(source):
        parser.error('Requested question count exceeds available dataset rows')
    questions = source.to_pylist()[:args.limit]
    if not questions or any(not isinstance(r.get(k), str) for r in questions for k in ('id', 'question', 'injection')):
        parser.error('Dataset must have non-null string id, question, and injection columns')
    if len({r['id'] for r in questions}) != len(questions):
        parser.error('Dataset IDs must be unique')
    if any(r.get('injection_after_tokens') != HORIZON for r in questions):
        parser.error('Dataset must specify injection_after_tokens=20 for every row')
    reference = load_reference_a(args.reference_rollouts, questions, args.rollouts, args.reference_step) if args.reference_rollouts else None
    config = dict(source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  config_module_sha256=hashlib.sha256(Path(__file__).with_name('experiment_config.py').read_bytes()).hexdigest(),
                  b_opening=args.b_opening, reference_step=args.reference_step if reference else None,
                  reference_sha256=hashlib.sha256(args.reference_rollouts.read_bytes()).hexdigest() if reference else None,
                  a_mode='teacher_forced_reference' if reference else 'sampled',
                  horizon=HORIZON, python=platform.python_version(),
                  packages={name: version(name) for name in ('torch', 'transformers', 'pyarrow')},
                  checkpoints=checkpoints, rollouts=args.rollouts, questions=len(questions), seed=args.seed,
                  experiment_specification=specification, chat_mode=chat_mode,
                  temperature=1.0, top_k=None, top_p=None, eos='allowed; stopping event included in observed diagnostics',
                  primary_means='20-content-token spans only; null for incomplete; observed means include EOS',
                  method='context-ablation entropy differences, not formal CMI',
                  entropy_context_order={'a': ['full', 'empty_prompt'], 'b': ['full', 'empty_injection', 'empty_a']},
                  device='mps' if torch.backends.mps.is_available() else 'cpu')
    config_json = json.dumps(config, sort_keys=True)
    destination = args.output_dir
    config_path = destination / 'config.json'
    if config_path.exists():
        if not args.resume or config_path.read_text() != config_json:
            parser.error('Existing run requires --resume with identical code and settings')
    elif destination.exists() and any(destination.iterdir()):
        parser.error('Output directory is not empty')
    destination.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config_json)
    metadata = {**(source.schema.metadata or {}), b'experiment_config': config_json.encode()}
    raw_path = destination / 'rollouts.parquet'
    rows = pq.read_table(raw_path).to_pylist() if raw_path.exists() else []
    done = {(r['rlvr_step'], r['source_row'], r['rollout_index']) for r in rows}
    torch.set_num_threads(4)
    if reference:
        original = json.loads(pq.read_table(args.reference_rollouts).schema.metadata[b'experiment_config'])
        if original['checkpoints'] != checkpoints or original.get('chat_mode', 'native') != chat_mode:
            raise ValueError('Reference checkpoint/tokenizer configuration differs from this experiment')
    tokenizer = AutoTokenizer.from_pretrained(checkpoints[0][1], revision=checkpoints[0][2])
    opening_ids = tokenizer.encode(args.b_opening, add_special_tokens=False) if args.b_opening else []
    if args.b_opening and not opening_ids:
        parser.error('Opening must tokenize to nonempty content')
    for step, repo, revision in checkpoints:
        if all((step, i, j) in done for i in range(len(questions)) for j in range(args.rollouts)):
            continue
        candidate = AutoTokenizer.from_pretrained(repo, revision=revision)
        if candidate.encode(args.b_opening, add_special_tokens=False) != opening_ids and args.b_opening:
            raise ValueError('Checkpoint opening token IDs differ')
        if candidate.get_vocab() != tokenizer.get_vocab():
            raise ValueError('Checkpoint tokenizer vocabulary mismatch')
        if chat_mode == 'native' and candidate.chat_template != tokenizer.chat_template:
            raise ValueError('Checkpoint chat template mismatch')
        # For Qwen3, the base template is conditional and the trained templates
        # force non-thinking. Verify actual tokenized contexts for every input.
        for question in questions:
            for prompt, injection in [(question['question'], None), ('', None),
                                      (question['question'], question['injection']), (question['question'], '')]:
                if chat(candidate, prompt, injection, chat_mode) != chat(tokenizer, prompt, injection, chat_mode):
                    raise ValueError('Checkpoint prompt tokens differ')
        model = AutoModelForCausalLM.from_pretrained(repo, revision=revision,
                    dtype=torch.float16 if config['device'] == 'mps' else torch.float32).to(config['device']).eval()
        ending_ids = model.generation_config.eos_token_id
        ending_ids = tokenizer.eos_token_id if ending_ids is None else ending_ids
        ending_ids = {ending_ids} if isinstance(ending_ids, int) else set(ending_ids or [])
        if any(t in ending_ids for t in opening_ids):
            raise ValueError('Opening must not contain EOS tokens')
        for i, question in enumerate(questions):
            for j in range(args.rollouts):
                if (step, i, j) in done:
                    continue
                seed = args.seed + step * 10000000 + i * 10000 + j
                start = time.monotonic()
                result = flatten(rollout(model, tokenizer, question['question'], question['injection'], seed,
                                         chat_mode=chat_mode,
                                         fixed_a=reference[(question['id'], j)]['a_token_ids'] if reference else None,
                                         b_opening_ids=opening_ids), tokenizer)
                rows.append({**question, **result, 'b_opening_text': args.b_opening,
                             'b_opening_token_ids': opening_ids,
                             'reference_a_seed': reference[(question['id'], j)]['seed'] if reference else None, 'source_row': i, 'rollout_index': j, 'rlvr_step': step,
                             'model_repo': repo, 'model_revision': revision, 'dtype': str(model.dtype),
                             'elapsed_seconds': time.monotonic()-start})
                done.add((step, i, j))
                if (j + 1) % 5 == 0:
                    save_tables(rows, destination, metadata, args.rollouts)
            save_tables(rows, destination, metadata, args.rollouts)
            print(f'Step {step}, question {i+1}/{len(questions)}: {len(rows)} rollouts saved', flush=True)
        del model
        gc.collect()
        if config['device'] == 'mps':
            torch.mps.empty_cache()
    save_tables(rows, destination, metadata, args.rollouts)
    print(f'Complete: {len(rows)} rollouts and {len(summarize(rows,args.rollouts))} question/checkpoint summaries', flush=True)


if __name__ == '__main__':
    main()
