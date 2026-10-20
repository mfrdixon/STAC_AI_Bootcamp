"""Set semantics test; no training, semantic-accuracy or randomness inference claim."""
import json,time,itertools
from pathlib import Path
import torch
from torch import nn
P=Path(__file__).resolve().parent;torch.manual_seed(20261021);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
class TinySet(nn.Module):
 def __init__(self):
  super().__init__();self.embed=nn.Linear(1,32);self.pos=nn.Embedding(50,32)
  self.layers=nn.ModuleList([nn.ModuleDict({'attn':nn.MultiheadAttention(32,4,batch_first=True,dropout=0),'n1':nn.LayerNorm(32),'n2':nn.LayerNorm(32),'ff':nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Linear(64,32))}) for _ in range(2)])
  self.head=nn.Linear(32,1)
 def forward(self,x,mode):
  # Each real is one continuous input embedding, not a digit-token string.
  pos=torch.arange(50) if mode!='block_set' else torch.zeros(50,dtype=torch.long)
  h=(self.embed(x[:,None])+self.pos(pos))[None]
  mask=torch.triu(torch.ones(50,50,dtype=torch.bool),1) if mode=='ordinary' else ~torch.eye(50,dtype=torch.bool)
  forbidden=[]
  for l in self.layers:
   z,w=l['attn'](h,h,h,attn_mask=mask,need_weights=True,average_attn_weights=False)
   forbidden.append(float(w.masked_select(mask[None,None]).abs().max()))
   h=l['n1'](h+z);h=l['n2'](h+l['ff'](h))
  # Symmetric aggregation; no last-item or list-position readout.
  return float(self.head(h[0].double().mean(0).float())),h[0],max(forbidden)
model=TinySet().eval()
g=torch.Generator().manual_seed(81234)
# Predeclare ten independent samples of size 50, N(0,1); identity + 19 permutations.
samples=[torch.randn(50,generator=g) for _ in range(10)]
orders=[torch.arange(50)]+[torch.randperm(50,generator=g) for _ in range(19)]
rows=[];start=time.time();max_equivariance=0
with torch.inference_mode():
 for si,x in enumerate(samples):
  base_h=None
  for mode in ['ordinary','block_global_positions','block_set']:
   for pi,perm in enumerate(orders):
    for repeat in range(5):
     y,h,f=model(x[perm],mode)
     rows.append(dict(sample=si,mode=mode,permutation=pi,repeat=repeat,output=y,forbidden_attention_max=f))
     if mode=='block_set':
      if pi==0 and repeat==0:base_h=h.clone()
      max_equivariance=max(max_equivariance,float((h-base_h[perm]).abs().max()))
summary={}
for mode in ['ordinary','block_global_positions','block_set']:
 ranges=[];repeat_max=0
 for si in range(10):
  r=[z for z in rows if z['mode']==mode and z['sample']==si]
  ys=[z['output'] for z in r if z['repeat']==0];ranges.append(max(ys)-min(ys))
  for pi in range(20):
   ys=[z['output'] for z in r if z['permutation']==pi];repeat_max=max(repeat_max,max(ys)-min(ys))
 summary[mode]=dict(max_output_range=max(ranges),mean_output_range=sum(ranges)/10,first_sample_range=ranges[0],max_repeat_range=repeat_max)
# Check sensitivity to actual values, not just constant output.
with torch.inference_mode():
 x=samples[0];a,_,_=model(x,'block_set');b,_,_=model(x+1,'block_set')
assert summary['block_set']['max_output_range']<1e-6
assert max_equivariance<1e-6
assert summary['ordinary']['max_output_range']>1e-5
assert abs(a-b)>1e-5
summary.update(parameters=sum(p.numel() for p in model.parameters()),trained=False,seed=20261021,data_seed=81234,distribution='standard normal, independent pseudorandom draws',samples=10,items_per_sample=50,permutations=20,repeats=5,evaluations=len(rows),block_max_hidden_equivariance_error=max_equivariance,block_forbidden_attention_max=max(z['forbidden_attention_max'] for z in rows if z['mode']=='block_set'),first_sample_original_output=a,first_sample_shifted_by_one_output=b,value_change_absolute_output=abs(a-b),seconds=time.time()-start,tolerance=1e-6)
(P/'iid_set_outputs.json').write_text(json.dumps(rows,indent=2));(P/'iid_set_summary.json').write_text(json.dumps(summary,indent=2));(P/'iid_set_inputs.json').write_text(json.dumps(dict(samples=[x.tolist() for x in samples],permutations=[o.tolist() for o in orders]),indent=2));print(json.dumps(summary,indent=2))
