"""Offline architectural proof, not a trained investment decision system."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import itertools, json, time, hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

ROOT=Path(__file__).resolve().parent
START=time.time()
torch.manual_seed(20261019)
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
MODEL='google/flan-t5-base'
tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
model=AutoModelForSeq2SeqLM.from_pretrained(MODEL,local_files_only=True).eval()
actions={
 'BUY':'Take a long position when eligible actual EPS exceeds consensus EPS.',
 'CASH':'Do not take a position when eligible actual EPS is at or below consensus EPS.',
 'WITHHOLD':'Defer the decision when required evidence is missing or conflicting.'}
cases=[
 ('positive','Eligible actual EPS is 1.34. Eligible consensus EPS is 1.07.','BUY'),
 ('negative','Eligible actual EPS is 0.85. Eligible consensus EPS is 1.07.','CASH'),
 ('equal','Eligible actual EPS is 1.07. Eligible consensus EPS is 1.07.','CASH'),
 ('missing','Actual EPS is unavailable. Eligible consensus EPS is 1.07.','WITHHOLD'),
 ('conflict','Two eligible actual EPS sources conflict: 1.34 and 0.85. Eligible consensus EPS is 1.07.','WITHHOLD')]
orders=list(itertools.permutations(actions))
manifest=dict(model=MODEL,actions=actions,cases=cases,orders=orders,repeats=5,
 seed=20261019,head='untrained shared cosine cross-attention; no learned projections',
 scope='Architectural invariance only. Policy-derived labels are not investment outcomes.')
(ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2))

@torch.inference_mode()
def encode(text):
 x=tok(text,return_tensors='pt')
 return model.encoder(**x).last_hidden_state[0]

@torch.inference_mode()
def separate(evidence,entries):
 # Each action is encoded independently; entries never share encoder context.
 H=F.normalize(encode(evidence),dim=-1)
 scores={}
 for name,desc in entries.items():
  q=F.normalize(encode(name+': '+desc).mean(0),dim=0)
  attn=torch.softmax(H@q,dim=0)
  z=attn@H
  scores[name]=float(q@z)
 # Canonical summation order removes floating-point reduction-order differences.
 names=sorted(scores)
 probs=torch.tensor([scores[n] for n in names],dtype=torch.float64).softmax(0)
 return scores,dict(zip(names,probs.tolist()))

@torch.inference_mode()
def prompted(evidence,entries):
 prompt=('Choose the action using the evidence and these definitions. '
 'Reply with exactly BUY, CASH or WITHHOLD.\nEvidence: '+evidence+
 '\nAction definitions: '+json.dumps(entries))
 names=sorted(actions)
 inputs=tok([prompt]*3,return_tensors='pt',padding=True)
 targets=tok(names,return_tensors='pt',padding=True)
 labels=targets.input_ids.clone()
 labels[targets.attention_mask==0]=-100
 logits=model(**inputs,labels=labels).logits.float().log_softmax(-1)
 ids=labels.clamp_min(0)
 lp=logits.gather(-1,ids.unsqueeze(-1)).squeeze(-1)
 scores=(lp*(labels!=-100)).sum(-1).double()
 # Includes EOS: complete answer probabilities, normalized over three answers.
 return dict(zip(names,scores.softmax(0).tolist())),prompt

rows=[]
for case_id,evidence,expected in cases:
 for oi,order in enumerate(orders):
  for repeat in range(5):
   entries={name:actions[name] for name in order}
   raw,p=separate(evidence,entries)
   baseline,prompt=prompted(evidence,entries)
   rows.append(dict(case=case_id,expected=expected,order=list(order),repeat=repeat,
    separate_scores=raw,separate_probs=p,json_prompt_probs=baseline,
    prompt=prompt))
  print(f'{case_id} order {oi+1}/6: {time.time()-START:.1f}s',flush=True)
 (ROOT/'outputs.json').write_text(json.dumps(rows,indent=2))

def tv(a,b): return sum(abs(a[k]-b[k]) for k in a)/2
def winner(p): return sorted(p,key=lambda k:(-p[k],k))[0]
summary={}
for field in ['separate_probs','json_prompt_probs']:
 order_tv=[];repeat_tv=[];flips=0;affected=0
 for case_id,_,_ in cases:
  r=[x for x in rows if x['case']==case_id]
  first=[x for x in r if x['repeat']==0]
  affected+=len({winner(x[field]) for x in first})>1
  for a,b in itertools.combinations(first,2):
   order_tv.append(tv(a[field],b[field]))
   flips+=winner(a[field])!=winner(b[field])
  for oi in range(6):
   for a,b in itertools.combinations(r[oi*5:(oi+1)*5],2):
    repeat_tv.append(tv(a[field],b[field]))
 summary[field]=dict(order_pairs=len(order_tv),mean_order_tv=sum(order_tv)/len(order_tv),
  max_order_tv=max(order_tv),order_action_changes=flips,affected_cases=affected,
  repeat_pairs=len(repeat_tv),max_repeat_tv=max(repeat_tv),
  policy_matches=sum(winner(x[field])==x['expected'] for x in rows),runs=len(rows))
assert summary['separate_probs']['max_order_tv']==0
assert summary['separate_probs']['max_repeat_tv']==0
# Evidence sensitivity control: invariant does not mean a constant representation.
summary['separate_evidence_change_max_tv']=max(tv(a['separate_probs'],b['separate_probs'])
 for a,b in itertools.combinations(rows[::30],2))
summary.update(elapsed_seconds=time.time()-START,parameters=sum(p.numel() for p in model.parameters()),
 trained_parameters=0,device='CPU',torch=torch.__version__,transformers=transformers.__version__,
 model_revision=model.config._commit_hash,
 tokenization={k:tok.encode(k) for k in actions},
 manifest_sha256=hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest())
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2),flush=True)
