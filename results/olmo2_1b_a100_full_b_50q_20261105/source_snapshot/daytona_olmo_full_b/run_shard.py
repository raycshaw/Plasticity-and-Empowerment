"""OLMo A100/full B, whole-A entropy-ablation experiment, one checkpoint/question shard."""
import argparse,hashlib,json,os,time,statistics,math
from pathlib import Path
import torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer,AutoModelForCausalLM
from sampled_rlvr_experiment import chat,continuation_prefix
from structured_calculation_experiment import generate_b,region_summary
A_CAP=100
B_CAP=2048
SCORE_CAP=100

def contexts(tok,q,a,mode):
 return [continuation_prefix(tok,q['question'],j,span,mode) for span,j in [(a,q['injection']),(a,''),([],q['injection'])]]

def eligible(span):
 return span['content_token_count']==A_CAP

@torch.inference_mode()
def generate_full_b(model, prefixes, eos, seed):
    rng=torch.Generator(device='cpu').manual_seed(seed)
    tokens=[]; entropies=[]; caches=[None]*3
    if max(map(len,prefixes))+B_CAP > model.config.max_position_embeddings:
        raise ValueError('Context length exceeded')
    for pos in range(B_CAP):
        distributions=[]
        active=3 if pos<SCORE_CAP else 1
        for i in range(active):
            ids=[tokens[-1]] if pos else prefixes[i]
            pred=model(torch.tensor([ids],device=model.device),past_key_values=caches[i],use_cache=True,logits_to_keep=1)
            caches[i]=pred.past_key_values
            logp=pred.logits[0,-1].float().log_softmax(-1)
            if not torch.isfinite(logp).all():raise ValueError('Nonfinite model prediction')
            distributions.append(logp)
        if pos<SCORE_CAP:
            entropies.append(torch.stack([-(x.exp()*x).sum()/math.log(2) for x in distributions]).cpu().tolist())
        token=int(torch.multinomial(distributions[0].exp().cpu(),1,generator=rng));tokens.append(token)
        if token in eos:break
    return dict(token_ids=tokens,entropies=entropies,ended_at_eos=tokens[-1] in eos,content_token_count=len(tokens)-int(tokens[-1] in eos))

def atomic_json(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def main():
 p=argparse.ArgumentParser();p.add_argument('--step',type=int,required=True);p.add_argument('--shard',type=int,choices=range(3),required=True);p.add_argument('--resume',action='store_true');args=p.parse_args()
 root=Path(__file__).resolve().parent;out=root/'output';out.mkdir(exist_ok=True)
 spec=json.loads((root/'qwen06b_gsm8k200.json').read_text());step,repo,rev=next(x for x in spec['checkpoints'] if x[0]==args.step)
 allq=pq.read_table(root/'gsm8k200_injections.parquet').to_pylist()[:50];questions=[(i,q) for i,q in enumerate(allq) if args.shard==0]
 cfg=dict(protocol='olmo_a100_full_b2048_score100_v1',step=step,repo=repo,revision=rev,shard=args.shard,shards_per_checkpoint=1,question_rows=[i for i,q in questions],trials=2,a_cap=100,b_cap=2048,a_source_range=[1,100],b_target_range=[1,100],range_endpoints='inclusive; up to 100 content tokens',eligibility='A and B may end at EOS; score available B1-100 non-EOS tokens; zero scores null',chat_mode=spec['chat_mode'],injection='original correction only',temperature=1.0,top_k=None,top_p=None,device='cpu',weights_dtype='float32',entropy_dtype='float32',cpu_threads=4,seed=20261030,a_seed_offset=10**12,metric='entropy-ablation proxies, not formal DI',empty_correction_turn_retained=True,empowerment_baseline='remove all sampled A1-100; preserve empty assistant framing; same B history',source_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['run_shard.py','sampled_rlvr_experiment.py','structured_calculation_experiment.py','qwen06b_gsm8k200.json','gsm8k200_injections.parquet']})
 cp=out/'config.json';raw=out/'rollouts.jsonl';rows=[]
 if cp.exists():
  if not args.resume or json.loads(cp.read_text())!=cfg:raise ValueError('Resume requires identical config')
  if raw.exists():rows=[json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
 else:atomic_json(cp,cfg)
 done={(r['source_row'],r['trial']) for r in rows};torch.set_num_threads(4)
 tok=AutoTokenizer.from_pretrained(repo,revision=rev);model=AutoModelForCausalLM.from_pretrained(repo,revision=rev,dtype=torch.float32).to('cpu').eval()
 e=model.generation_config.eos_token_id;e=tok.eos_token_id if e is None else e;eos={e} if isinstance(e,int) else set(e)
 if not eos:raise ValueError('EOS required')
 expected=len(questions)*2
 def progress():
  atomic_json(out/'progress.json',dict(step=step,shard=args.shard,saved=len(rows),expected=expected,eligible=sum(r['included_primary'] for r in rows),a_short=sum(not r['a_complete'] for r in rows),b_short=sum(not r['b_complete'] for r in rows),updated_at_unix=time.time()))
 progress()
 with raw.open('a') as handle:
  for i,q in questions:
   for trial in range(2):
    if (i,trial) in done:continue
    seed=20261030+step*10000000+i*10000+trial;t=time.monotonic()
    a=generate_b(model,[chat(tok,q['question'],chat_mode=spec['chat_mode'])],eos,seed+10**12,A_CAP)
    a_ids=[v for v in a['token_ids'] if v not in eos];a_ok=eligible(a)
    b=generate_full_b(model,contexts(tok,q,a_ids,spec['chat_mode']),eos,seed)
    b_ok=b['content_token_count']==B_CAP;indices=list(range(min(SCORE_CAP,b['content_token_count'])));included=bool(indices)
    row=dict(step=step,shard=args.shard,source_row=i,id=q['id'],trial=trial,question=q['question'],injection=q['injection'],revised_answer=q['revised_answer'],source_issue=q['source_issue'],repo=repo,revision=rev,seed=seed,a_seed=seed+10**12,a=a,b=b,a_complete=a_ok,b_complete=b_ok,b_attempted=True,included_primary=included,selected_a_source_count=len(a_ids),scored_b_count=len(indices),a_text=tok.decode(a['token_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False),b_text=tok.decode(b['token_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False),elapsed_seconds=time.monotonic()-t,**region_summary(b['entropies'],indices))
    handle.write(json.dumps(row)+'\n');handle.flush();os.fsync(handle.fileno());rows.append(row);done.add((i,trial));progress()
    print(json.dumps({'step':step,'shard':args.shard,'saved':len(rows),'expected':expected,'eligible':included,'seconds':row['elapsed_seconds']}),flush=True)
 atomic_json(out/'DONE.json',dict(status='complete',records=len(rows),expected=expected));assert len(rows)==expected
if __name__=='__main__':main()
