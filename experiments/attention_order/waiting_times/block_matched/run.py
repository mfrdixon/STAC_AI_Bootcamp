import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
import json,time,random,hashlib
from pathlib import Path
import torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
P=Path(__file__).resolve().parent;src=json.loads((P.parent/'manifest.json').read_text());start=time.time()
torch.manual_seed(20261023);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
path=Path(os.environ.get('QWEN_MODEL_PATH',str(P.parents[1]/'alpha_framing/dictionary_invariance/qwen_model')))
tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval()
placeholder='RECORDS_INSERTED_HERE'
rendered=tok.apply_chat_template([{'role':'user','content':src['question']+'\n'+placeholder}],tokenize=False,add_generation_prompt=True)
pre,suffix=rendered.split(placeholder);enc=lambda s:tok.encode(s,add_special_tokens=False)
preids=enc(pre);suffixids=enc(suffix)
records=[enc(f'Waiting time: {v:.1f} minutes.\n') for v in src['values']]
actionids={a:enc(a) for a in ['YES','NO']};assert all(len(v)==1 for v in actionids.values())
manifest=dict(source_manifest_sha256=hashlib.sha256((P.parent/'manifest.json').read_bytes()).hexdigest(),source=src,record_template='Waiting time: {value:.1f} minutes.\\n',prefix=pre,suffix=suffix,tokenization='Each fixed segment tokenized independently, concatenated identically in every condition',conditions=['ordinary','mask_only','block_local'],positions='Record positions reset to prefix_length + within_record_index; suffix retains fixed global positions',readout='Original Qwen vocabulary output head at final assistant-prefix token; YES/NO single-token probabilities normalized over these two tokens; no EOS',action_ids=actionids)
(P/'manifest.json').write_text(json.dumps(manifest,indent=2))
jobs=[(mode,o,r) for mode in manifest['conditions'] for o in range(10) for r in range(3)];random.Random(20261023).shuffle(jobs)
rows=[];audit={};forbidden_context={};hooks=[]
def audit_hook(module,inp,out):
 if not forbidden_context:return
 w=out[1]
 if w is not None:
  bad=forbidden_context['mask'];maxbad=float(w[0].masked_select(bad[None]).abs().max())
  forbidden_context['max_weight']=max(forbidden_context.get('max_weight',0.),maxbad)
for layer in model.model.layers:hooks.append(layer.self_attn.register_forward_hook(audit_hook))
with torch.inference_mode(),open(P/'outputs.jsonl','w') as f:
 for k,(mode,oi,repeat) in enumerate(jobs):
  order=src['orders'][oi];assert sorted(src['values'][i] for i in order)==sorted(src['values'])
  ids=preids.copy();positions=list(range(len(preids)));blocks=[-1]*len(preids)
  for item in order:
   rs=records[item];ids+=rs;blocks += [item]*len(rs);positions+=list(range(len(preids),len(preids)+len(rs)))
  suffix_start=len(ids);ids+=suffixids;blocks += [-2]*len(suffixids);positions+=list(range(suffix_start,len(ids)))
  n=len(ids);idx=torch.arange(n);b=torch.tensor(blocks)
  causal=idx[None,:]<=idx[:,None]
  # Prefix: causal prefix only. Record: prefix and its own earlier tokens.
  # Readout suffix: all prior tokens. No record-to-record dependence.
  allowed=causal & ((b[:,None]==-2) | (b[None,:]==-1) | ((b[:,None]>=0)&(b[:,None]==b[None,:])))
  if mode=='ordinary':allowed=causal
  mask=torch.zeros(n,n);mask.masked_fill_(~allowed,float('-inf'))
  pos=torch.tensor(positions if mode=='block_local' else list(range(n)))[None]
  check=(oi==0 and repeat==0)
  if check:forbidden_context.update(mask=~allowed,max_weight=0.)
  output=model(input_ids=torch.tensor([ids]),position_ids=pos,attention_mask={'full_attention':mask[None,None]},use_cache=False,output_attentions=check,logits_to_keep=1)
  logits=output.logits[0,-1].float();lp=logits.log_softmax(-1)
  vals=[float(lp[actionids[a][0]]) for a in ['YES','NO']];p=float(torch.tensor(vals,dtype=torch.float64).softmax(0)[0])
  row=dict(mode=mode,order=oi,repeat=repeat,p_yes=p,yes_logprob=vals[0],no_logprob=vals[1],top_candidate='YES' if p>.5 else 'NO',raw_candidate_mass=sum(float(lp[actionids[a][0]].exp()) for a in actionids),full_vocab_argmax=tok.decode([int(logits.argmax())]),input_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
  if check:
   audit[mode]=dict(forbidden_attention_max=forbidden_context['max_weight'],all_record_tokens_can_read_prefix=True,readout_can_read_all_records=bool(allowed[-1,len(preids):suffix_start].all()),tokens=n)
   forbidden_context.clear()
  rows.append(row);f.write(json.dumps(row)+'\n');f.flush()
  if (k+1)%15==0:print(k+1,'/90',round(time.time()-start,1),'seconds',flush=True)
  del output
for h in hooks:h.remove()
summary={}
for mode in manifest['conditions']:
 first=[next(z for z in rows if z['mode']==mode and z['order']==oi and z['repeat']==0) for oi in range(10)]
 repeats=[]
 for oi in range(10):
  v=[z['p_yes'] for z in rows if z['mode']==mode and z['order']==oi];repeats.append(max(v)-min(v))
 summary[mode]=dict(p_yes_min=min(z['p_yes'] for z in first),p_yes_max=max(z['p_yes'] for z in first),p_yes_order_range=max(z['p_yes'] for z in first)-min(z['p_yes'] for z in first),max_repeat_range=max(repeats),top_candidates=sorted(set(z['top_candidate'] for z in first)),raw_mass_min=min(z['raw_candidate_mass'] for z in first),raw_mass_max=max(z['raw_candidate_mass'] for z in first))
# Ensure exactly matched input tokens, not just identical record values.
for oi in range(10):assert len({z['input_ids_sha256'] for z in rows if z['order']==oi})==1
summary.update(audit=audit,elapsed_seconds=time.time()-start,evaluations=len(rows),torch=torch.__version__,transformers=transformers.__version__,invariance_tolerance=1e-5,block_within_tolerance=summary['block_local']['p_yes_order_range']<1e-5)
(P/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
