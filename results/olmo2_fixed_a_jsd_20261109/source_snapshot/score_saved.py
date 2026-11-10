"""Teacher-force saved B; compute JSD without generating or changing responses."""
import argparse,json,math,time,sys,os,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from sampled_rlvr_experiment import continuation_prefix

def jsd(logp,logq):
 logm=torch.logaddexp(logp,logq)-math.log(2)
 return .5*((logp.exp()*(logp-logm)).sum(-1)+(logq.exp()*(logq-logm)).sum(-1))/math.log(2)

@torch.inference_mode()
def score(model,tok,r):
 n=min(100,r['b']['content_token_count']);assert n>0
 a=r['a']['token_ids'][:r['a']['content_token_count']];b=r['b']['token_ids'][:n]
 prefixes=[continuation_prefix(tok,r['question'],j,span,'native') for span,j in [(a,r['injection']),(a,''),([],r['injection'])]]
 ps=[]
 for prefix in prefixes:
  ids=prefix+b[:-1]
  logits=model(torch.tensor([ids],device=model.device),use_cache=False,logits_to_keep=n).logits[0].float()
  assert logits.shape[0]==n
  ps.append(logits.log_softmax(-1))
 ent=torch.stack([-(v.exp()*v).sum(-1)/math.log(2) for v in ps],dim=1)
 saved=torch.tensor(r['b']['entropies'][:n],dtype=torch.float32)
 err=(ent.cpu()-saved).abs().max().item()
 assert err<.001, f'Rescored entropy mismatch: {err}'
 j1,j2=jsd(ps[0],ps[1]),jsd(ps[0],ps[2])
 assert torch.isfinite(j1).all() and torch.isfinite(j2).all()
 assert min(j1.min().item(),j2.min().item())>=-1e-6 and max(j1.max().item(),j2.max().item())<=1.000001
 return dict(step=r['step'],id=r['id'],source_row=r['source_row'],trial=r['trial'],scored_b_count=n,prefix_lengths=[len(x) for x in prefixes],max_entropy_error_bits=err,correction_jsd_bits=j1.cpu().tolist(),a_jsd_bits=j2.cpu().tolist(),correction_jsd_mean_bits=j1.mean().item(),a_jsd_mean_bits=j2.mean().item(),original_plasticity_bits_per_token=r['plasticity_bits_per_token'],original_empowerment_bits_per_token=r['empowerment_bits_per_token'])

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--resume',action='store_true');args=p.parse_args();rows=[json.loads(x) for x in Path(args.input).read_text().splitlines() if x.strip()];assert len({(x['repo'],x['revision']) for x in rows})==1
 output=Path(args.output);config_path=output.parent/'scoring_config.json';progress_path=output.parent/'progress.json'
 cfg=dict(protocol='saved_B_jsd_teacher_forcing_v1',input_sha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),scorer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),expected=len(rows),repo=rows[0]['repo'],revision=rows[0]['revision'],threads=4,weights_dtype='float32',units='bits',context_comparisons=['full versus empty correction','full versus empty A'],B_window='first 100 available non-EOS tokens; identical saved B history',new_generation=False)
 if config_path.exists():
  assert args.resume and json.loads(config_path.read_text())==cfg,'Resume requires identical scoring configuration'
 else:config_path.write_text(json.dumps(cfg,indent=2))
 existing=[json.loads(x) for x in output.read_text().splitlines() if x.strip()] if output.exists() else []
 assert not existing or args.resume
 done={(x['step'],x['source_row'],x['trial']) for x in existing};assert len(done)==len(existing)
 assert done<={(x['step'],x['source_row'],x['trial']) for x in rows}
 torch.set_num_threads(4);start=time.monotonic();repo,rev=rows[0]['repo'],rows[0]['revision'];tok=AutoTokenizer.from_pretrained(repo,revision=rev);model=AutoModelForCausalLM.from_pretrained(repo,revision=rev,dtype=torch.float32).to('cpu').eval();print(json.dumps({'model_load_seconds':time.monotonic()-start}),flush=True)
 with output.open('a') as f:
  for i,r in enumerate(rows):
   key=(r['step'],r['source_row'],r['trial'])
   if key in done:continue
   t=time.monotonic();out=score(model,tok,r);out['elapsed_seconds']=time.monotonic()-t;f.write(json.dumps(out)+'\n');f.flush();os.fsync(f.fileno());done.add(key)
   progress=dict(saved=len(done),expected=len(rows),last_seconds=out['elapsed_seconds'],updated_at_unix=time.time())
   tmp=progress_path.with_suffix('.tmp');tmp.write_text(json.dumps(progress));tmp.replace(progress_path)
   print(json.dumps({'saved':len(done),'seconds':out['elapsed_seconds'],'max_entropy_error_bits':out['max_entropy_error_bits']}),flush=True)
 assert len(done)==len(rows)
if __name__=='__main__':main()
