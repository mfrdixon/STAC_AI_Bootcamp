"""Learned numerical task, not a pretrained language model. Frozen test split."""
import os
os.environ['OMP_NUM_THREADS']='2'
import torch,json,time,copy,math
from torch import nn
import torch.nn.functional as F
from pathlib import Path
P=Path(__file__).resolve().parent;start=time.time();torch.manual_seed(20261025);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
N=50
class Model(nn.Module):
 def __init__(self):
  super().__init__();self.embed=nn.Linear(1,32);self.pos=nn.Embedding(N,32)
  self.layers=nn.ModuleList([nn.ModuleDict({'attn':nn.MultiheadAttention(32,4,dropout=0,batch_first=True),'n1':nn.LayerNorm(32),'n2':nn.LayerNorm(32),'ff':nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Linear(64,32))}) for _ in range(2)])
  self.head=nn.Linear(32,2)
  self.register_buffer('causal',torch.triu(torch.ones(N,N,dtype=torch.bool),1));self.register_buffer('block',~torch.eye(N,dtype=torch.bool))
 def forward(self,x,blocked=False):
  positions=torch.zeros(N,dtype=torch.long) if blocked else torch.arange(N)
  h=self.embed((x/30-1).unsqueeze(-1))+self.pos(positions)[None]
  for l in self.layers:
   z,_=l['attn'](h,h,h,attn_mask=self.block if blocked else self.causal,need_weights=False)
   h=l['n1'](h+z);h=l['n2'](h+l['ff'](h))
  o=self.head(h.mean(1))
  return o[:,0].sigmoid(),o[:,1]
def dataset(size,seed):
 g=torch.Generator().manual_seed(seed)
 p=torch.rand(size,1,generator=g)*.8+.1
 positive=torch.rand(size,N,generator=g)<p
 low=torch.rand(size,N,generator=g)*29.9
 high=30.1+torch.rand(size,N,generator=g)*29.9
 x=torch.where(positive,high,low)
 count=(x>30).sum(1);fraction=count.float()/N;majority=(count>25).float()
 return x,fraction,majority
train=dataset(8000,81001);test=dataset(1000,81002)
manifest=dict(seed=20261025,train_seed=81001,test_seed=81002,train_sets=8000,test_sets=1000,items=50,base_steps=500,adapt_steps=300,batch_size=64,learning_rate=.002,task='Estimate fraction above 30 minutes; classify strictly more than half',input='continuous numerical values, not text',training='all tiny-model weights; no pretrained LM and no LoRA',evaluation='10 fixed permutations per test set; three identical repeats of first permutation',success_criteria='report all results; no selection based on held-out performance')
(P/'manifest.json').write_text(json.dumps(manifest,indent=2));model=Model();initial=copy.deepcopy(model.state_dict())
def fit(model,blocked,steps,seed):
 model.train();opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.001);g=torch.Generator().manual_seed(seed);logs=[]
 for step in range(steps):
  idx=torch.randint(len(train[0]),(64,),generator=g);x,t,y=[a[idx] for a in train]
  f,logit=model(x,blocked);loss=20*F.mse_loss(f,t)+F.binary_cross_entropy_with_logits(logit,y)
  opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
  if (step+1)%100==0:
   logs.append(dict(step=step+1,loss=float(loss.detach())));print('train',blocked,step+1,round(float(loss.detach()),4),round(time.time()-start,1),flush=True)
 return logs
base_logs=fit(model,False,500,81003);base=copy.deepcopy(model).eval();adapt_logs=fit(model,True,300,81004);adapt=model.eval()
# Fixed evaluation permutations; never used to select checkpoints.
g=torch.Generator().manual_seed(81005);perms=[torch.arange(N)]+[torch.randperm(N,generator=g) for _ in range(9)]
results={}
with torch.inference_mode():
 for name,m,blocked in [('ordinary_trained',base,False),('block_without_adaptation',base,True),('block_after_adaptation',adapt,True)]:
  fractions=[];probabilities=[]
  for perm in perms:
   fs=[];ps=[]
   for lo in range(0,1000,100):
    f,l=m(test[0][lo:lo+100,perm],blocked);fs.append(f);ps.append(l.sigmoid())
   fractions.append(torch.cat(fs));probabilities.append(torch.cat(ps))
  fs=torch.stack(fractions);ps=torch.stack(probabilities);labels=ps>.5;truth=test[2].bool()[None]
  # Recompute identical first inputs twice, with the same batching as primary evaluation.
  repeat_delta=0.
  for repeat in range(2):
   for lo in range(0,1000,100):
    f,l=m(test[0][lo:lo+100],blocked)
    repeat_delta=max(repeat_delta,float((f-fs[0,lo:lo+100]).abs().max()),float((l.sigmoid()-ps[0,lo:lo+100]).abs().max()))
  boundary=(test[1]>.45)&(test[1]<.56)
  results[name]=dict(majority_accuracy=float((labels==truth).float().mean()),majority_accuracy_first_order=float((labels[0]==truth[0]).float().mean()),fraction_mae_percentage_points=float((fs-test[1][None]).abs().mean()*100),fraction_mae_first_order_pp=float((fs[0]-test[1]).abs().mean()*100),max_fraction_order_range_pp=float((fs.max(0).values-fs.min(0).values).max()*100),max_yes_probability_order_range=float((ps.max(0).values-ps.min(0).values).max()),sets_with_answer_flip=int((labels.any(0)!=labels.all(0)).sum()),test_sets=1000,max_identical_repeat_difference=repeat_delta,boundary_sets=int(boundary.sum()),boundary_accuracy=float((labels[:,boundary]==truth[:,boundary]).float().mean()))
  torch.save(dict(fractions=fs,yes_probabilities=ps,truth_fraction=test[1],truth_majority=test[2]),P/(name+'_outputs.pt'))
  print(name,json.dumps(results[name]),flush=True)
summary=dict(results=results,parameters=sum(p.numel() for p in model.parameters()),seconds=time.time()-start,base_training=base_logs,adaptation_training=adapt_logs,torch=torch.__version__)
(P/'summary.json').write_text(json.dumps(summary,indent=2));torch.save(dict(base=base.state_dict(),adapted=adapt.state_dict(),test_inputs=test[0],permutations=perms),P/'checkpoints.pt');print('COMPLETE',json.dumps(summary,indent=2),flush=True)
