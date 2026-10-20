import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
import json,random,time,hashlib,math,itertools
from pathlib import Path
import torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
P=Path(__file__).resolve().parent;start=time.time();rng=random.Random(20261024)
values=[rng.randint(1,6) for _ in range(50)];orders=[list(range(50))]
for _ in range(9):
 p=list(range(50));rng.shuffle(p);orders.append(p)
question='The following 50 numbers were observed. Predict the next number. Reply with exactly one digit from 1 to 6 and nothing else. Observations:\n'
manifest=dict(seed=20261024,values=values,orders=orders,repeats=3,conditions=['ordinary','block_local'],question=question,source_hidden_from_model='Independent fair die, pseudorandom draws; true next-value distribution is uniform',record_template='Observation: {value}.\\n',purpose='Measure order sensitivity separately from agreement with hidden data-generating distribution',model='Qwen/Qwen2.5-0.5B-Instruct',revision='7ae557604adf67be50417f59c2c2f167def9a775')
(P/'manifest.json').write_text(json.dumps(manifest,indent=2))
torch.manual_seed(20261024);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
path=Path(os.environ.get('QWEN_MODEL_PATH',str(P.parent/'alpha_framing/dictionary_invariance/qwen_model')))
tok=AutoTokenizer.from_pretrained(path,local_files_only=True);model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval()
enc=lambda s:tok.encode(s,add_special_tokens=False)
rendered=tok.apply_chat_template([{'role':'user','content':question+'RECORDS_INSERTED_HERE'}],tokenize=False,add_generation_prompt=True)
pre,suf=rendered.split('RECORDS_INSERTED_HERE');prefix=enc(pre);suffix=enc(suf);records=[enc(f'Observation: {v}.\n') for v in values]
candidates={str(i):enc(str(i)) for i in range(1,7)};assert all(len(v)==1 for v in candidates.values())
manifest.update(prefix=pre,suffix=suf,candidate_ids=candidates);(P/'manifest.json').write_text(json.dumps(manifest,indent=2))
jobs=[(mode,o,r) for mode in manifest['conditions'] for o in range(10) for r in range(3)];rng.shuffle(jobs)
rows=[];audit={};context={}
def hook(module,args,out):
 if context and out[1] is not None:
  context['max_forbidden']=max(context.get('max_forbidden',0.),float(out[1][0].masked_select(context['bad'][None]).abs().max()))
hooks=[l.self_attn.register_forward_hook(hook) for l in model.model.layers]
with torch.inference_mode(),open(P/'outputs.jsonl','w') as out:
 for k,(mode,o,r) in enumerate(jobs):
  order=orders[o];assert sorted(values[i] for i in order)==sorted(values)
  ids=prefix.copy();blocks=[-1]*len(prefix);pos=list(range(len(prefix)))
  for item in order:
   rec=records[item];ids+=rec;blocks += [item]*len(rec);pos+=list(range(len(prefix),len(prefix)+len(rec)))
  suffixstart=len(ids);ids+=suffix;blocks += [-2]*len(suffix);pos+=list(range(suffixstart,len(ids)))
  n=len(ids);idx=torch.arange(n);b=torch.tensor(blocks);causal=idx[None,:]<=idx[:,None]
  allowed=causal if mode=='ordinary' else causal & ((b[:,None]==-2)|(b[None,:]==-1)|((b[:,None]>=0)&(b[:,None]==b[None,:])))
  mask=torch.zeros(n,n).masked_fill(~allowed,float('-inf'));positions=torch.tensor(pos if mode=='block_local' else list(range(n)))[None]
  check=o==0 and r==0
  if check:context.update(bad=~allowed,max_forbidden=0.)
  result=model(input_ids=torch.tensor([ids]),position_ids=positions,attention_mask={'full_attention':mask[None,None]},use_cache=False,output_attentions=check,logits_to_keep=1)
  logits=result.logits[0,-1].float();lp=logits.log_softmax(-1);vals=torch.tensor([float(lp[candidates[str(i)][0]]) for i in range(1,7)],dtype=torch.float64);ps=vals.softmax(0).tolist();rawmass=float(vals.exp().sum());argmax=int(logits.argmax())
  row=dict(mode=mode,order=o,repeat=r,probabilities=dict(zip(map(str,range(1,7)),ps)),candidate_mass=rawmass,top_candidate=str(ps.index(max(ps))+1),full_vocab_argmax_id=argmax,full_vocab_argmax=tok.decode([argmax]),input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
  rows.append(row);out.write(json.dumps(row)+'\n');out.flush()
  if check:audit[mode]=dict(max_forbidden=context['max_forbidden'],readout_sees_all_records=bool(allowed[-1,len(prefix):suffixstart].all()));context.clear()
  if (k+1)%10==0:print(k+1,'/60',round(time.time()-start,1),'sec',flush=True)
  del result
for h in hooks:h.remove()
def tv(a,b):return .5*sum(abs(a[str(i)]-b[str(i)]) for i in range(1,7))
uniform={str(i):1/6 for i in range(1,7)};summary={}
for mode in manifest['conditions']:
 first=[next(x for x in rows if x['mode']==mode and x['order']==o and x['repeat']==0) for o in range(10)]
 repeat_tv=[]
 for o in range(10):
  group=[x for x in rows if x['mode']==mode and x['order']==o]
  repeat_tv += [tv(a['probabilities'],b['probabilities']) for a,b in itertools.combinations(group,2)]
 pairs=list(itertools.combinations(first,2))
 summary[mode]=dict(original_order_probs=first[0]['probabilities'],max_order_tv=max(tv(a['probabilities'],b['probabilities']) for a,b in pairs),mean_order_tv=sum(tv(a['probabilities'],b['probabilities']) for a,b in pairs)/len(pairs),top_candidate_changes=sum(a['top_candidate']!=b['top_candidate'] for a,b in pairs),order_pairs=len(pairs),max_repeat_tv=max(repeat_tv),min_tv_from_uniform=min(tv(x['probabilities'],uniform) for x in first),max_tv_from_uniform=max(tv(x['probabilities'],uniform) for x in first),candidate_mass_min=min(x['candidate_mass'] for x in first),candidate_mass_max=max(x['candidate_mass'] for x in first),top_candidates=sorted({x['top_candidate'] for x in first}),full_vocab_argmaxes=sorted({x['full_vocab_argmax'] for x in first}))
for o in range(10):assert len({x['input_sha256'] for x in rows if x['order']==o})==1
summary.update(audit=audit,seconds=time.time()-start,total_evaluations=len(rows),torch=torch.__version__,transformers=transformers.__version__,block_invariance_tolerance=1e-5,block_within_tolerance=summary['block_local']['max_order_tv']<1e-5)
(P/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
