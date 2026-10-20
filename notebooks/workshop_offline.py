"""Portable workshop calculations. Python standard library; no model calls.
Graph evaluator reuses the tested graph_repair experiment implementation.
Other functions mirror the declared browser teaching logic and are parity-tested.
"""
from pathlib import Path
import json, math, collections
from datetime import datetime

def load_fixture(name):
    return json.loads((Path(__file__).parent / "data" / name).read_text())

def show_table(rows):
    """Readable tables in Jupyter; plain text fallback outside a notebook."""
    if not rows:
        print("No rows"); return
    from html import escape
    columns=list(rows[0])
    html="<table><thead><tr>"+"".join("<th style='text-align:left;padding:8px'>"+escape(str(k))+"</th>" for k in columns)+"</tr></thead><tbody>"
    for row in rows:
        html += "<tr>"+"".join("<td style='text-align:left;padding:8px;border-top:1px solid #ddd'>"+escape(str(row.get(k,'')))+"</td>" for k in columns)+"</tr>"
    html += "</tbody></table>"
    try:
        from IPython.display import display, HTML
        display(HTML(html))
    except ImportError:
        for row in rows: print(" | ".join(str(row.get(k,'')) for k in columns))

def architecture(case="simple", branch=False):
    if case not in ("simple", "annex", "missing"): raise ValueError(case)
    required=case!="simple"; available=case=="annex"
    correct_cash=(10 if available else None) if required else 15
    rows=[]
    for id,label in [("rag","Fixed RAG"),("fixed","Fixed RAG + calculator"),("agent","Agent-selected tools")]:
        calculator=id!="rag"
        resolved=(not required or (branch and available)) if id=="fixed" else (not required or available) if id=="agent" else False
        supported=calculator and resolved and correct_cash is not None
        cash=(correct_cash if resolved else 15) if calculator else None
        leverage=((75-correct_cash)/20 if supported else 3) if calculator else None
        rows.append(dict(id=id,label=label,cash=cash,leverage=leverage,supported=supported,missing=required and not resolved,status="CALCULATION SUPPORTED" if supported else "HOLD CALCULATION CLAIM"))
    return rows

def rag_case(case="baseline"):
    fixture=load_fixture("rag-reference.json")[0]
    sources=[dict(s) for s in fixture["sources"]]
    cutoff=datetime.fromisoformat("2026-10-20T12:55:00+00:00")
    if case not in ("baseline","future","missing"): raise ValueError(case)
    for s in sources:
        s["eligible"]=all(datetime.fromisoformat(s[k].replace("Z","+00:00"))<=cutoff for k in ("published","available"))
        s["inContext"]=s["id"] in ({"E1","E2","E3","E4"} if case=="future" else {"E1","E3"} if case=="missing" else {"E1","E2","E3"})
    eligible={s["id"] for s in sources if s["eligible"] and s["inContext"]}
    bound={"E1","E2"}<=eligible
    return {"sources":sources,"bound":bound,"current_ratio":3.0 if bound else None,"current_claim":"SUPPORTED IN FIXTURE" if bound else "UNRESOLVED","comfort_claim":"UNRESOLVED","gate":"HOLD"}

def masked_nll(probabilities,mask):
    if len(probabilities)!=len(mask) or not sum(mask): raise ValueError("No target tokens")
    if any(not 0<p<=1 for p in probabilities) or any(m not in (0,1) for m in mask): raise ValueError("Invalid token scores")
    return -sum(m*math.log(p) for p,m in zip(probabilities,mask))/sum(mask)

def lora_count(d_in,d_out,rank):
    if not all(isinstance(v,int) for v in (d_in,d_out,rank)) or not 1<=rank<=min(d_in,d_out): raise ValueError("Invalid rank")
    return rank*(d_in+d_out)

def dpo(chosen,rejected,ref_chosen=-8,ref_rejected=-6,beta=.1):
    z=beta*((chosen-rejected)-(ref_chosen-ref_rejected))
    return {"margin":z,"loss":max(0,-z)+math.log1p(math.exp(-abs(z)))}

def source_value(records,spec):
 rs=[r for r in records if all(r.get(k)==v for k,v in spec['where'].items())]
 eligible=[r for r in rs if r.get('eligible') is True and r.get('operative') is True]
 vals={r['value'] for r in eligible}
 if not vals:return {'state':'blocked','reason':'no_eligible_source','value':None,'support':[]}
 if len(vals)>1:return {'state':'blocked','reason':'conflicting_sources','value':None,'support':[r['id'] for r in eligible]}
 return {'state':'ok','value':next(iter(vals)),'support':[r['id'] for r in eligible]}

def evaluate(graph,records):
 result={};pending=dict(graph)
 while pending:
  progressed=False
  for id,node in list(pending.items()):
   deps=node.get('deps',[])
   if any(d not in result for d in deps):continue
   typ=node['type'];vals=[result[d] for d in deps];good=[x for x in vals if x['state']=='ok']
   if typ=='source':out=source_value(records,node)
   elif typ=='constant':out={'state':'ok','value':node['value'],'support':[]}
   elif typ=='or':
    out={'state':'ok','value':True,'support':[d for d in deps if result[d]['state']=='ok' and result[d]['value'] is True]} if any(x['value'] is True for x in good) else ({'state':'ok','value':False,'support':deps} if len(good)==len(vals) else {'state':'blocked','reason':'no_alternative_support','value':None,'support':[]})
   elif len(good)!=len(vals):out={'state':'blocked','reason':'dependency_failed','value':None,'support':[]}
   else:
    v=[x['value'] for x in vals];op=node['op']
    if op=='subtract':value=v[0]-v[1]
    elif op=='positive':value=v[0]>0
    elif op=='action':value='buy' if v[0] else 'cash'
    elif op=='above':value=v[0]>node['threshold']
    elif op=='identity':value=v[0]
    else:raise ValueError(op)
    out={'state':'ok','value':value,'support':deps}
   result[id]=out;del pending[id];progressed=True
  if not progressed:raise ValueError('Cycle or missing dependency')
 return result

def descendants(graph,seeds):
 children=collections.defaultdict(list)
 for n,s in graph.items():
  for d in s.get('deps',[]):children[d].append(n)
 seen=set(seeds);queue=list(seeds)
 while queue:
  for n in children[queue.pop()]:
   if n not in seen:seen.add(n);queue.append(n)
 return sorted(seen-set(seeds))

def diagnose(graph,state):
 roots=[n for n,s in state.items() if graph[n]['type']=='source' and s['state']=='blocked']
 requests=[]
 for n in roots:
  reason=state[n]['reason'];requests.append({'node':n,'reason':reason,'constraints':graph[n]['where']|{'eligible':True,'operative':True},'action':'obtain_authoritative_resolution' if reason=='conflicting_sources' else 'obtain_eligible_evidence','required_record':'resolution with issuer authority and superseded source IDs' if reason=='conflicting_sources' else 'source meeting every declared constraint','structural_potentially_affected':descendants(graph,[n])})
 return {'failed_source_dependencies':roots,'requests':requests,'structural_potentially_affected':descendants(graph,roots),'actually_blocked':[n for n,s in state.items() if s['state']=='blocked']}

def apply_repair(records,repair):
 out=[dict(r) for r in records]; audit=[]
 for res in repair.get('resolutions',[]):
  if res.get('authority')!='issuer' or not res.get('document_id') or not res.get('supersedes'):
   audit.append({'accepted':False,'reason':'invalid_resolution_authority'});continue
  targets=[r for r in out if r['id'] in res['supersedes']]
  if len(targets)!=len(res['supersedes']) or any(r.get('issuer')!=res.get('issuer') or r.get('metric')!=res.get('metric') for r in targets):
   audit.append({'accepted':False,'reason':'resolution_scope_mismatch'});continue
  for r in targets:r['operative']=False
  audit.append({'accepted':True,'document_id':res['document_id'],'superseded':res['supersedes']})
 out.extend(repair.get('add',[]));return out,audit


def financial_replay(data,mode='valid',gate=True,fee_bps=None):
    fee=data['fee_bps_default'] if fee_bps is None else fee_bps
    if mode not in ('valid','leaked') or not math.isfinite(fee) or fee<0: raise ValueError('Invalid replay settings')
    rows=[]
    for r in data['observations']:
        signal=r[mode+'_signal'];available=r[mode+'_available_at']
        eligible=datetime.fromisoformat(available)<=datetime.fromisoformat(r['decision_time'])
        blocked=gate and not eligible;trade=not blocked and signal=='BUY'
        net=r['realized_return']-2*fee/10000 if trade else 0
        rows.append(dict(r,signal=signal,available=available,eligible=eligible,blocked=blocked,trade=trade,net=net,pnl=net*data['capital_per_opportunity']))
    return dict(rows=rows,trades=sum(r['trade'] for r in rows),blocked=sum(r['blocked'] for r in rows),net_pnl=sum(r['pnl'] for r in rows),net_bps=10000*sum(r['net'] for r in rows)/len(rows),worst_pnl=min(r['pnl'] for r in rows))

def check_framing_evidence(case,records=None):
    if records is None: records=json.loads(case['evidence'])
    eligible=[r for r in records if r['issuer']==case['issuer'] and r['period']=='2026-Q2' and r['asof_eligible'] is True and r['status']=='operative']
    values={}
    for key in ('actual_eps','consensus_eps'):
        values[key]=list(dict.fromkeys(r['value'] for r in eligible if r['kind']==key and isinstance(r['value'],(int,float)) and not isinstance(r['value'],bool) and math.isfinite(r['value'])))
    failed=[k+(':conflicting' if v else ':missing') for k,v in values.items() if len(v)!=1]
    action='WITHHOLD' if failed else 'BUY' if values['actual_eps'][0]>values['consensus_eps'][0] else 'CASH'
    return dict(action=action,failed=failed,values=values)

def framing_metrics(data,model,frame):
    rows=[r for r in data['outputs'] if r['model']==model and r['framing']==frame]
    return dict(n=len(rows),correct=sum(r['action']==r['expected'] for r in rows),unsupported_buys=sum(r['action']=='BUY' and r['expected']!='BUY' for r in rows))

class CommitController:
    """In-memory teaching controller; no external side effects or durable transactions."""
    def __init__(self,initial):
        self.state=dict(initial);self.proposal=None;self.next=1;self.used=set();self.audit=[];self.orders=[]
    def _log(self,event,**details):
        import copy
        self.audit.append(dict(sequence=len(self.audit)+1,event=event,**copy.deepcopy(details)))
    def prepare(self,notional=5000):
        self.proposal=dict(id='proposal-'+str(self.next),snapshot_version=self.state['snapshot_version'],pending_orders=self.state['pending_orders'],grant_version=self.state['grant_version'],notional=notional)
        self.next+=1;self._log('PREPARED',proposal=self.proposal);return dict(self.proposal)
    def change(self,kind):
        if kind=='snapshot':self.state['snapshot_version']+=1
        elif kind=='pending':self.state['pending_orders']+=1
        elif kind=='grant':self.state['grant_active']=False;self.state['grant_version']+=1
        elif kind=='evidence':self.state['evidence_valid']=False
        elif kind=='approval':self.state['reviewer_approved']=False
        elif kind!='unchanged':raise ValueError('Unknown transition')
        self._log('STATE_CHANGED',kind=kind,state=self.state)
    def commit(self):
        p=self.proposal;s=self.state
        checks=dict(proposal_present=p is not None,snapshot_current=bool(p) and p['snapshot_version']==s['snapshot_version'],pending_orders_current=bool(p) and p['pending_orders']==s['pending_orders'],grant_current=bool(p) and p['grant_version']==s['grant_version'] and s['grant_active'],within_limit=bool(p) and isinstance(p['notional'],(int,float)) and math.isfinite(p['notional']) and 0<p['notional']<=s['grant_limit'],evidence_valid=s['evidence_valid'],reviewer_approved=s['reviewer_approved'],not_already_committed=bool(p) and p['id'] not in self.used)
        permitted=all(checks.values())
        if permitted:
            self.used.add(p['id']);self.orders.append(dict(id='mock-order-'+str(len(self.orders)),proposal_id=p['id'],notional=p['notional']))
        result=dict(decision='PERMITTED' if permitted else 'BLOCKED',checks=checks,dispatched=permitted)
        self._log('COMMIT_CHECK',proposal_id=p['id'] if p else None,**result);return result
