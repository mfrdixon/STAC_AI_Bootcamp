import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
import json,random,time,re,hashlib,itertools
from pathlib import Path
import torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig
P=Path(__file__).resolve().parent;start=time.time();rng=random.Random(20261022)
values=[round(rng.uniform(1,60),1) for _ in range(50)]
orders=[list(range(50))]
for _ in range(9):
 p=list(range(50));rng.shuffle(p);orders.append(p)
count=sum(v>30 for v in values);expected='YES' if count>25 else 'NO'
question=('These are synthetic waiting times in minutes for 50 different patients. '
 'Count how many values are strictly greater than 30 minutes and calculate their percentage of all 50 patients. '
 'Did MORE THAN HALF of the patients wait more than 30 minutes? '
 'Reply on one line in exactly this format: YES; count=N; percentage=P% or NO; count=N; percentage=P%. '
 'Do not explain. Waiting times: ')
jobs=[(oi,r) for oi in range(10) for r in range(3)];rng.shuffle(jobs)
manifest=dict(seed=20261022,values=values,orders=orders,repeats=3,execution_order=jobs,question=question,expected_count=count,expected_percentage=2*count,expected_majority=expected,model='Qwen/Qwen2.5-0.5B-Instruct',revision='7ae557604adf67be50417f59c2c2f167def9a775',protocol='Ordinary unmodified model; single fixed sample; raw YES/NO prefix scoring without EOS; unconstrained greedy generation; no tools')
(P/'manifest.json').write_text(json.dumps(manifest,indent=2));print('Frozen manifest:',count,expected,flush=True)
torch.manual_seed(20261022);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
modelpath=Path(os.environ.get('QWEN_MODEL_PATH',str(P.parent/'alpha_framing/dictionary_invariance/qwen_model')))
tok=AutoTokenizer.from_pretrained(modelpath,local_files_only=True)
model=AutoModelForCausalLM.from_pretrained(modelpath,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval()
gcfg=GenerationConfig(do_sample=False,num_beams=1,repetition_penalty=1.0,max_new_tokens=48,bos_token_id=tok.bos_token_id,eos_token_id=tok.eos_token_id,pad_token_id=tok.eos_token_id)
rows=[]
with torch.inference_mode(),open(P/'outputs.jsonl','w') as out:
 for k,(oi,repeat) in enumerate(jobs):
  nums=[values[i] for i in orders[oi]];assert sorted(nums)==sorted(values)
  prompt=question+json.dumps(nums);rendered=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
  prefix=tok.encode(rendered,add_special_tokens=False);x=torch.tensor([prefix]);scores={};pieces={}
  for answer in ['YES','NO']:
   full=tok.encode(rendered+answer,add_special_tokens=False);assert full[:len(prefix)]==prefix;c=full[len(prefix):]
   lp=model(input_ids=torch.tensor([full]),use_cache=False,logits_to_keep=len(c)+1).logits[0,:-1].float().log_softmax(-1)
   scores[answer]=float(lp[torch.arange(len(c)),torch.tensor(c)].sum());pieces[answer]=c
  pyes=float(torch.tensor([scores['YES'],scores['NO']],dtype=torch.float64).softmax(0)[0])
  generated=model.generate(input_ids=x,attention_mask=torch.ones_like(x),generation_config=gcfg)
  ids=generated[0,len(prefix):].tolist();answer=tok.decode(ids,skip_special_tokens=True)
  m=re.fullmatch(r'\s*(YES|NO);\s*count=(\d+);\s*percentage=(\d+(?:\.\d+)?)%\s*',answer)
  parsed=dict(majority=m[1],count=int(m[2]),percentage=float(m[3])) if m else None
  row=dict(order=oi,repeat=repeat,prompt=prompt,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),answer=answer,parsed=parsed,p_yes=pyes,logprobs=scores,candidate_token_ids=pieces,generated_ids=ids,truncated=len(ids)>=48 and ids[-1]!=tok.eos_token_id)
  out.write(json.dumps(row)+'\n');out.flush();rows.append(row)
  print(f'{k+1}/30: order {oi} repeat {repeat}: {answer!r} | P(YES)={pyes:.6f} | {time.time()-start:.1f}s',flush=True)
first=[next(z for z in rows if z['order']==o and z['repeat']==0) for o in range(10)]
repeat_prob=max(max(z['p_yes'] for z in rows if z['order']==o)-min(z['p_yes'] for z in rows if z['order']==o) for o in range(10))
summary=dict(expected_count=count,expected_percentage=count*2,expected_majority=expected,orders=10,repeats=3,total=30,valid_format=sum(z['parsed'] is not None for z in rows),truncated=sum(z['truncated'] for z in rows),distinct_generated_outputs=len(set(z['answer'] for z in first)),distinct_majority_answers=sorted(set(z['parsed']['majority'] for z in first if z['parsed'])),distinct_counts=sorted(set(z['parsed']['count'] for z in first if z['parsed'])),distinct_percentages=sorted(set(z['parsed']['percentage'] for z in first if z['parsed'])),exact_correct=sum(z['parsed']==dict(majority=expected,count=count,percentage=float(count*2)) for z in rows),p_yes_min=min(z['p_yes'] for z in first),p_yes_max=max(z['p_yes'] for z in first),max_identical_prompt_probability_range=repeat_prob,repeat_text_changes=sum(len(set(z['answer'] for z in rows if z['order']==o))>1 for o in range(10)),elapsed_seconds=time.time()-start,torch=torch.__version__,transformers=transformers.__version__)
(P/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
