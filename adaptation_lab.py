"""Recorded LoRA audit; standard-library default. No model generation on load."""
import hashlib,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'public/decision-lab/data/recorded-adaptation.json'
def load():return json.loads(DATA.read_text())
def metrics(data,arm):
 rows=data['cases'];labels=['NEGATIVE_SIGNAL','NEUTRAL_SIGNAL','POSITIVE_SIGNAL'];cm={a:{b:0 for b in labels} for a in labels}
 for r in rows:cm[r['expected']][r[arm]['label']]+=1
 recall={a:cm[a][a]/sum(cm[a].values()) for a in labels if sum(cm[a].values())}
 return {'n':len(rows),'accuracy':sum(r[arm]['label']==r['expected'] for r in rows)/len(rows),'balanced_accuracy':statistics.mean(recall.values()),'confusion':cm}
def verify(data):
 for c in data['cases']:
  assert c['base']['prompt_sha256']==c['adapted']['prompt_sha256']==c['prompt_sha256']
  for arm in ['base','adapted']:
   assert all(math.isfinite(v) for v in c[arm]['candidate_logits'].values())
   assert max(c[arm]['candidate_logits'],key=c[arm]['candidate_logits'].get)==c[arm]['code']
 for arm in ['base','adapted']:
  for key in ['accuracy','balanced_accuracy']:assert abs(metrics(data,arm)[key]-data['summary'][arm][key])<1e-12
 return 'Verified 16 matched prompt hashes, finite candidate scores and recomputed aggregate metrics.'
def local_score(case,base_dir,adapter_dir,expected_base_hash,expected_adapter_hash):
 """Opt-in fresh inference. Uses installed packages/local weights only; no downloads."""
 import os
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
 from contextlib import nullcontext
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM
 from peft import PeftModel
 base_dir,adapter_dir=Path(base_dir),Path(adapter_dir)
 for p,expected in [(base_dir/'model.safetensors',expected_base_hash),(adapter_dir/'adapter_model.safetensors',expected_adapter_hash)]:
  if not p.exists():raise FileNotFoundError(f'Local weights required: {p}; use saved mode or provision approved artifacts.')
  if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError(f'Artifact does not match recorded run: {p}')
 device='cuda' if torch.cuda.is_available() else 'cpu';dtype=torch.float16 if device=='cuda' else torch.float32
 tok=AutoTokenizer.from_pretrained(base_dir,local_files_only=True,trust_remote_code=False)
 model=AutoModelForCausalLM.from_pretrained(base_dir,local_files_only=True,trust_remote_code=False,torch_dtype=dtype).to(device)
 model=PeftModel.from_pretrained(model,adapter_dir,local_files_only=True);model.eval()
 ids=[tok.encode(x,add_special_tokens=False) for x in 'ABC'];assert all(len(x)==1 for x in ids)
 text=tok.apply_chat_template(case['prompt'],tokenize=False,add_generation_prompt=True)
 if hashlib.sha256(text.encode()).hexdigest()!=case['prompt_sha256']:raise ValueError('Chat-template rendered prompt differs from recorded run')
 batch=tok(text,return_tensors='pt').to(device);out={}
 for arm in ['base','adapted']:
  with (model.disable_adapter() if arm=='base' else nullcontext()),torch.inference_mode():
   logits=model(**batch,logits_to_keep=1).logits[0,-1,[x[0] for x in ids]].float();assert torch.isfinite(logits).all();out[arm]={'code':'ABC'[int(logits.argmax())],'candidate_logits':dict(zip('ABC',logits.cpu().tolist()))}
 return {'execution':'fresh local inference','device':device,'outputs':out,'note':'Different hardware/precision/library versions may alter scores; not overwritten into the recorded artifact.'}
