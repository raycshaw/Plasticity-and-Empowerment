"""Fixed-A correction experiment with completed, explicitly marked B regions."""
import argparse
import gc
import hashlib
import json
import math
import platform
import re
import statistics
import time
from fractions import Fraction
from importlib.metadata import version
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sampled_rlvr_experiment import chat, continuation_prefix, load_reference_a, sample_span

FORMAT = ('Continue using the current information. Show only the arithmetic needed to solve the problem, '
          'followed by the final numerical answer. Do not restate the problem or add headings beyond '
          'these two labels. Use exactly this format:\n'
          'Calculation: <arithmetic steps>\nAnswer: <number>')
NUMBER = r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*/\s*[+-]?\d+(?:\.\d+)?)?'
PATTERN = re.compile(r'\s*Calculation:[ \t]*(?P<calculation>\S[\s\S]*?)\s*\nAnswer:[ \t]*(?P<answer>'+NUMBER+r')\s*\Z')


def numeric(text):
    return Fraction(text.replace(',', '').replace(' ', ''))


def region_summary(entropies, indices):
    result = {}
    for name, baseline in [('plasticity', 1), ('empowerment', 2)]:
        values = [entropies[i][baseline] - entropies[i][0] for i in indices]
        result[name + '_bits_per_token'] = statistics.mean(values) if values else None
        result[name + '_total_bits'] = sum(values) if values else None
    return result


def score_regions(tokenizer, span, eos, reference, source_issue=False):
    """Conservative parsing: null primary scores on truncation or format/alignment failure."""
    ids = span['token_ids']
    text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    result = dict(parse_status='truncated', parsed_answer=None, answer_correct=None,
                  grading_status='truncated', calculation_token_indices=[], answer_token_indices=[],
                  calculation_token_count=0, answer_token_count=0)
    for region in ('calculation', 'answer', 'first20', 'last20'):
        result.update({region+'_'+k:v for k,v in region_summary([], []).items()})
    if not span['ended_at_eos']:
        return result
    result.update(parse_status='format_mismatch', grading_status='format_mismatch')
    match = PATTERN.fullmatch(text)
    if not match or re.search(r'(?m)^\s*(Calculation|Answer):', match['calculation']):
        return result
    prefixes = [tokenizer.decode(ids[:i], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                for i in range(len(ids)+1)]
    if any(not text.startswith(p) for p in prefixes):
        result.update(parse_status='unstable_token_alignment', grading_status='unstable_token_alignment')
        return result
    regions = {}
    for name in ('calculation', 'answer'):
        start, end = match.span(name)
        regions[name] = [i for i,t in enumerate(ids) if t not in eos and len(prefixes[i]) < end
                         and len(prefixes[i+1]) > start and len(prefixes[i+1]) > len(prefixes[i])]
    if not all(regions.values()) or set(regions['calculation']) & set(regions['answer']):
        result.update(parse_status='ambiguous_region_boundary', grading_status='ambiguous_region_boundary')
        return result
    result.update(parse_status='ok', parsed_answer=match['answer'])
    for name, indices in regions.items():
        result[name+'_token_indices'] = indices
        result[name+'_token_count'] = len(indices)
        result.update({name+'_'+k:v for k,v in region_summary(span['entropies'], indices).items()})
    content = [i for i,t in enumerate(ids) if t not in eos]
    if len(content) >= 20:
        for name, indices in [('first20', content[:20]), ('last20', content[-20:])]:
            result.update({name+'_'+k:v for k,v in region_summary(span['entropies'], indices).items()})
    if source_issue:
        result['grading_status'] = 'source_issue_excluded'
    else:
        try:
            result['answer_correct'] = numeric(match['answer']) == numeric(reference)
            result['grading_status'] = 'graded'
        except (ValueError, ZeroDivisionError):
            result['grading_status'] = 'invalid_numeric_reference_or_answer'
    return result


def b_contexts(tokenizer, question, a_ids, injection, mode):
    # Same separator and FORMAT in every ablation; only factual correction or A is removed.
    return [continuation_prefix(tokenizer, question, message+'\n\n'+FORMAT, span, mode)
            for span,message in [(a_ids,injection),(a_ids,''),([],injection)]]


@torch.inference_mode()
def generate_b(model, prefixes, eos, seed, cap):
    if cap < 1 or max(map(len,prefixes))+cap > model.config.max_position_embeddings:
        raise ValueError('Invalid continuation/context budget')
    rng = torch.Generator(device='cpu').manual_seed(seed)
    tokens, entropies, caches = [], [], [None]*3
    for position in range(cap):
        distributions=[]
        for i,prefix in enumerate(prefixes):
            ids = [tokens[-1]] if position else prefix
            pred=model(torch.tensor([ids],device=model.device), past_key_values=caches[i],
                       use_cache=True, logits_to_keep=1)
            caches[i]=pred.past_key_values
            logp=pred.logits[0,-1].float().log_softmax(-1)
            if not torch.isfinite(logp).all():raise ValueError('Nonfinite prediction')
            distributions.append(logp)
        entropies.append(torch.stack([-(p.exp()*p).sum()/math.log(2) for p in distributions]).cpu().tolist())
        token=int(torch.multinomial(distributions[0].exp().cpu(),1,generator=rng))
        tokens.append(token)
        if token in eos:break
    return dict(token_ids=tokens,entropies=entropies,ended_at_eos=tokens[-1] in eos,
                content_token_count=len(tokens)-int(tokens[-1] in eos))


def summarize(rows):
    groups={}
    for row in rows:groups.setdefault((row['rlvr_step'],row['id']),[]).append(row)
    summaries=[]
    for (step,case_id), group in sorted(groups.items()):
        summary=dict(rlvr_step=step,id=case_id,recorded=len(group),
                     completed=sum(x['b_ended_at_eos'] for x in group),
                     parsed=sum(x['parse_status']=='ok' for x in group),
                     graded=sum(x['answer_correct'] is not None for x in group),
                     correct=sum(x['answer_correct'] is True for x in group))
        summary['accuracy_among_graded']=summary['correct']/summary['graded'] if summary['graded'] else None
        # Report coverage with accuracy: malformed/truncated cases are not silently treated as graded.
        for region in ('calculation','answer','first20','last20'):
            for metric in ('plasticity','empowerment'):
                key=region+'_'+metric+'_bits_per_token'
                values=[x[key] for x in group if x[key] is not None]
                summary[key+'_n']=len(values)
                summary[key+'_mean']=statistics.mean(values) if values else None
        summaries.append(summary)
    return summaries


def save(rows, output, config):
    metadata={b'experiment_config':json.dumps(config,sort_keys=True).encode()}
    for name, records in [('structured_rollouts',rows),('structured_question_means',summarize(rows))]:
        temp=output/(name+'.tmp.parquet')
        pq.write_table(pa.Table.from_pylist(records).replace_schema_metadata(metadata),temp)
        temp.replace(output/(name+'.parquet'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int,default=50)
    parser.add_argument('--rollouts',type=int,default=10)
    parser.add_argument('--max-b-tokens',type=int,default=256)
    parser.add_argument('--seed',type=int,default=20261030)
    parser.add_argument('--experiment-config',type=Path,default=Path(__file__).parent/'configs/qwen06b_gsm8k200.json')
    parser.add_argument('--reference-rollouts',type=Path,default=Path(__file__).parent/'results/qwen06b_gsm8k200_20rollouts/rollouts.parquet')
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).parent/'runs/qwen06b_structured_calculation')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--dry-run',action='store_true',help='Validate input/references only; do not load tokenizer/model or write run files')
    args=parser.parse_args()
    if min(args.limit,args.rollouts,args.max_b_tokens)<1:parser.error('Counts must be positive')
    spec=json.loads(args.experiment_config.read_text()); mode=spec['chat_mode'];checkpoints=spec['checkpoints']
    source=Path(__file__).parent/spec['source'];all_questions=pq.read_table(source).to_pylist()
    if args.limit>len(all_questions):parser.error('Too many questions')
    questions=all_questions[:args.limit]
    if len({q['id'] for q in questions})!=len(questions):parser.error('Duplicate case IDs')
    if any(q.get('injection_after_tokens')!=20 for q in questions):parser.error('Expected correction after 20 A tokens')
    reference_config=json.loads(pq.read_table(args.reference_rollouts).schema.metadata[b'experiment_config'])
    if reference_config['checkpoints']!=checkpoints or reference_config['chat_mode']!=mode:
        parser.error('Reference checkpoint/chat configuration differs')
    refs=load_reference_a(args.reference_rollouts,questions,args.rollouts,0)
    if args.dry_run:
        print(f'Validated {len(questions)} questions, {len(refs)} fixed A spans, {len(checkpoints)} checkpoints; no inference started.')
        return
    config=dict(protocol='structured_calculation_v1',specification=spec,format_instruction=FORMAT,
                questions=len(questions),rollouts=args.rollouts,max_b_tokens=args.max_b_tokens,seed=args.seed,
                device='mps' if torch.backends.mps.is_available() else 'cpu',python=platform.python_version(),
                packages={n:version(n) for n in ('torch','transformers','pyarrow')},
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                reference_sha256=hashlib.sha256(args.reference_rollouts.read_bytes()).hexdigest(),
                runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                helper_sha256=hashlib.sha256(Path(__file__).with_name('sampled_rlvr_experiment.py').read_bytes()).hexdigest(),
                context_order=['full','without_factual_correction','without_A'],temperature=1.0,
                metrics='Entropy ablations, bits/token; region means require EOS and strict format/alignment')
    output=args.output_dir;config_path=output/'config.json';encoded=json.dumps(config,sort_keys=True)
    if config_path.exists():
        if not args.resume or config_path.read_text()!=encoded:parser.error('Resume requires identical source/settings')
    elif output.exists() and any(output.iterdir()):parser.error('Output directory must be empty')
    output.mkdir(parents=True,exist_ok=True);config_path.write_text(encoded)
    path=output/'structured_rollouts.parquet';rows=pq.read_table(path).to_pylist() if path.exists() else []
    done={(x['rlvr_step'],x['id'],x['trial']) for x in rows}
    torch.set_num_threads(4)
    tokenizer=AutoTokenizer.from_pretrained(checkpoints[0][1],revision=checkpoints[0][2])
    for step,repo,revision in checkpoints:
        if all((step,q['id'],j) in done for q in questions for j in range(args.rollouts)):continue
        candidate=AutoTokenizer.from_pretrained(repo,revision=revision)
        if candidate.get_vocab()!=tokenizer.get_vocab():raise ValueError('Vocabulary mismatch')
        for q in questions:
            for message in [None,q['injection']+'\n\n'+FORMAT,'\n\n'+FORMAT]:
                if chat(candidate,q['question'],message,mode)!=chat(tokenizer,q['question'],message,mode):
                    raise ValueError('Checkpoint context token IDs differ')
        model=AutoModelForCausalLM.from_pretrained(repo,revision=revision,
              dtype=torch.float16 if config['device']=='mps' else torch.float32).to(config['device']).eval()
        endings=model.generation_config.eos_token_id
        endings=tokenizer.eos_token_id if endings is None else endings
        eos={endings} if isinstance(endings,int) else set(endings or [])
        if not eos:raise ValueError('EOS is required')
        for i,q in enumerate(questions):
            for j in range(args.rollouts):
                if (step,q['id'],j) in done:continue
                start=time.monotonic();ref=refs[q['id'],j];a=ref['a_token_ids']
                if any(t in eos for t in a):raise ValueError('EOS in reference A')
                seed=args.seed+step*10000000+i*10000+j
                b=generate_b(model,b_contexts(tokenizer,q['question'],a,q['injection'],mode),eos,seed,args.max_b_tokens)
                row=dict(id=q['id'],question=q['question'],injection=q['injection'],revised_answer=q['revised_answer'],
                         source_issue=q['source_issue'],review_note=q['review_note'],source_row=i,trial=j,
                         rlvr_step=step,model_repo=repo,model_revision=revision,seed=seed,
                         reference_a_seed=ref['seed'],a_token_ids=a,
                         b_text=tokenizer.decode(b['token_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False),
                         **{'b_'+k:v for k,v in b.items()},
                         **score_regions(tokenizer,b,eos,q['revised_answer'],q['source_issue']),
                         elapsed_seconds=time.monotonic()-start)
                rows.append(row);done.add((step,q['id'],j))
                if (j+1)%5==0:save(rows,output,config)
            save(rows,output,config)
            print(f'Step {step}, question {i+1}/{len(questions)}: {len(rows)} evaluations saved',flush=True)
        del model;gc.collect()
        if config['device']=='mps':torch.mps.empty_cache()
    save(rows,output,config);print(f'Complete: {len(rows)} evaluations',flush=True)

if __name__=='__main__':main()
