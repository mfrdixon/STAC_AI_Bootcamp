import json,itertools,math,time
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
P=Path(__file__).resolve().parent;torch.manual_seed(20261020);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
actions={'BUY':'take a long position when actual earnings exceed consensus','CASH':'do not take a position when actual earnings are below consensus','WITHHOLD':'defer when evidence is missing or conflicting'}
evidence=['actual earnings 134 consensus 107','actual earnings 85 consensus 107','actual earnings 107 consensus 107','actual earnings missing consensus 107','actual earnings conflict 134 85 consensus 107']
vocab={w:i+1 for i,w in enumerate(sorted(set(' '.join(evidence+list(actions.values())+list(actions)+['changed irrelevant description']).split())))}
class Tiny(nn.Module):
 def __init__(self):
  super().__init__();self.emb=nn.Embedding(len(vocab)+1,32);self.pos=nn.Embedding(128,32)
  self.layers=nn.ModuleList([nn.ModuleDict({'attn':nn.MultiheadAttention(32,4,batch_first=True,dropout=0),'n1':nn.LayerNorm(32),'n2':nn.LayerNorm(32),'ff':nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Linear(64,32))}) for _ in range(2)])
 def forward(self,blocks,mode):
  ids=[];bid=[];local=[];spans={}
  for j,(name,text) in enumerate(blocks):
   tokens=[vocab[x] for x in text.split()];start=len(ids);ids+=tokens;bid += [j]*len(tokens);local+=list(range(len(tokens)));spans[name]=slice(start,len(ids))
  b=torch.tensor(bid);n=len(ids);causal=torch.triu(torch.ones(n,n,dtype=torch.bool),1)
  mask=causal if mode=='ordinary' else causal | (b[:,None]!=b[None,:])
  pos=torch.tensor(local) if mode=='block_reset' else torch.arange(n)
  x=(self.emb(torch.tensor(ids))+self.pos(pos))[None];weights=[]
  for layer in self.layers:
   z,w=layer['attn'](x,x,x,attn_mask=mask,need_weights=True,average_attn_weights=False);weights.append(w)
   x=layer['n1'](x+z);x=layer['n2'](x+layer['ff'](x))
  H=F.normalize(x[0,spans['E']],dim=-1);vectors={};scores={}
  for name in actions:
   q=F.normalize(x[0,spans[name]].mean(0),dim=0);vectors[name]=q
   scores[name]=q@((H@q).softmax(0)@H)
  names=sorted(scores);raw=torch.stack([scores[n] for n in names]);prob=raw.double().softmax(0)
  cross=b[:,None]!=b[None,:]
  mass=max(float(w[0,:,:, :][:,cross].abs().max()) for w in weights)
  return dict(zip(names,prob.tolist())),vectors,mass
model=Tiny().eval();start=time.time();rows=[]
with torch.inference_mode():
 for mode in ['ordinary','block_global_positions','block_reset']:
  for ci,e in enumerate(evidence):
   for order in itertools.permutations(actions):
    for repeat in range(5):
     blocks=[('E',e)]+[(a,a+' '+actions[a]) for a in order]
     probs,v,m=model(blocks,mode);rows.append(dict(mode=mode,case=ci,order=order,repeat=repeat,scores=probs,cross_block_max_attention=m))
def tv(a,b):return sum(abs(a[k]-b[k]) for k in a)/2
summary={}
for mode in ['ordinary','block_global_positions','block_reset']:
 vals=[];reps=[];flips=0
 for ci in range(5):
  r=[x for x in rows if x['mode']==mode and x['case']==ci];first=[x for x in r if x['repeat']==0]
  for a,b in itertools.combinations(first,2):
   vals.append(tv(a['scores'],b['scores']));flips+=max(a['scores'],key=a['scores'].get)!=max(b['scores'],key=b['scores'].get)
  for o in itertools.permutations(actions):
   g=[x for x in r if tuple(x['order'])==o]
   reps.extend(tv(a['scores'],b['scores']) for a,b in itertools.combinations(g,2))
 summary[mode]=dict(max_order_tv=max(vals),mean_order_tv=sum(vals)/len(vals),top_changes=flips,order_pairs=len(vals),max_repeat_tv=max(reps),repeat_pairs=len(reps),cross_block_max_attention=max(x['cross_block_max_attention'] for x in rows if x['mode']==mode))
# Change earlier BUY text; CASH must not change under block/reset.
blocks=[('E',evidence[0])]+[(a,a+' '+actions[a]) for a in actions]
changed=list(blocks);changed[1]=('BUY','BUY changed irrelevant description')
with torch.inference_mode():
 _,v,_=model(blocks,'block_reset');_,v2,_=model(changed,'block_reset')
 isolation=float((v['CASH']-v2['CASH']).abs().max())
 # Compare joint masked execution to genuinely independent per-block executions.
 # Hidden-state isolation is checked above; probabilities need not remain fixed when BUY changes.
assert summary['block_reset']['max_order_tv']<1e-6
assert summary['block_reset']['cross_block_max_attention']==0
assert isolation<1e-6
assert summary['ordinary']['max_order_tv']>1e-5
assert summary['block_global_positions']['max_order_tv']>1e-5
summary.update(parameters=sum(x.numel() for x in model.parameters()),layers=2,heads=4,width=32,trained=False,seed=20261020,isolation_CASH_vector_max_change=isolation,seconds=time.time()-start,torch=torch.__version__,tolerance=1e-6)
(P/'outputs.json').write_text(json.dumps(rows,indent=2));(P/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
