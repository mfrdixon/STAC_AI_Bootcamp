"""Controlled workflow diagnosis. No model generation, training or future returns."""
import copy,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
GRAPH=[
 {'id':'source','label':'Retrieved document','type':'evidence','parents':[]},
 {'id':'selected','label':'Selected reported EPS','type':'extraction','parents':['source']},
 {'id':'tool_request','label':'Calculator arguments','type':'tool_request','parents':['selected']},
 {'id':'tool_result','label':'Calculated surprise','type':'tool_result','parents':['tool_request']},
 {'id':'interpretation','label':'Signal interpretation','type':'claim','parents':['tool_result']},
 {'id':'recommendation','label':'Proposed action','type':'action','parents':['interpretation']},
 {'id':'report','label':'Reported earnings summary','type':'claim','parents':['selected']},
 {'id':'risk','label':'Negative-surprise flag','type':'claim','parents':['tool_result']},
 {'id':'authority','label':'Delegated order permission','type':'permission','parents':[]},
 {'id':'execution','label':'Execution gate','type':'gate','parents':['source','selected','tool_request','tool_result','interpretation','recommendation','authority']},
]
BASE={'issuer':'Harbor Metals','period':'2026 Q2','reported_eps':1.2,'forecast_eps':.8,'consensus_eps':1.0,'published_at':'2026-07-01','id':'HM-Q2-release'}
CASES=[
 {'id':'period','title':'Wrong reporting period','faults':{'source':'prior_period'},'expected_first':'source','repair':'source'},
 {'id':'basis','title':'Forecast mistaken for reported EPS','faults':{'selected':'forecast'},'expected_first':'selected','repair':'selected'},
 {'id':'arguments','title':'Calculator receives the wrong consensus','faults':{'tool_request':'wrong_consensus'},'expected_first':'tool_request','repair':'tool_request'},
 {'id':'action','title':'Buy proposal contradicts the verified signal','faults':{'recommendation':'buy'},'reported_eps':.8,'expected_first':'recommendation','repair':'recommendation'},
 {'id':'two_faults','title':'Extraction repair leaves a tool-input failure','faults':{'selected':'forecast','tool_request':'wrong_consensus'},'expected_first':'selected','repair':'selected'},
]
INTERVENTIONS=[{'id':'none','label':'No intervention','nodes':[]},{'id':'source','label':'Supply verified current-period source','nodes':['source']},{'id':'selected','label':'Replace selected figure with reported EPS','nodes':['selected']},{'id':'tool_request','label':'Rebuild calculator arguments from checked evidence','nodes':['tool_request']},{'id':'recommendation','label':'Apply the declared signal-to-action rule','nodes':['recommendation']},{'id':'all','label':'Repair all four observable stages','nodes':['source','selected','tool_request','recommendation']}]
def descendants(nodes):
 found=set(nodes);changed=True
 while changed:
  changed=False
  for n in GRAPH:
   if n['id'] not in found and any(x in found for x in n['parents']):found.add(n['id']);changed=True
 return [n['id'] for n in GRAPH if n['id'] in found and n['id'] not in nodes]
def evaluate(case,repairs=()):
 ref=copy.deepcopy(BASE);ref['reported_eps']=case.get('reported_eps',1.2)
 faults={k:v for k,v in case['faults'].items() if k not in repairs};nodes={}
 def put(id,value,ok,explanation):nodes[id]={'value':value,'check':'PASS' if ok else 'FAIL','explanation':explanation,'parents':next(n['parents'] for n in GRAPH if n['id']==id)}
 source=copy.deepcopy(ref)
 if faults.get('source'):source.update(period='2026 Q1',reported_eps=.8,id='HM-Q1-release')
 put('source',source,source['period']==ref['period'],'Require issuer Harbor Metals and target period 2026 Q2. The supplied exercise document is checked against this declared requirement.')
 selected={'value':source['forecast_eps'] if faults.get('selected') else source['reported_eps'],'basis':'forecast' if faults.get('selected') else 'reported','source_id':source['id']}
 put('selected',selected,selected['basis']=='reported' and selected['value']==source['reported_eps'],'Select reported EPS, not the forecast figure. Check the selected value and basis against the supplied document fields.')
 args={'actual':selected['value'],'consensus':2.0 if faults.get('tool_request') else source['consensus_eps']}
 put('tool_request',args,args['actual']==selected['value'] and args['consensus']==source['consensus_eps'],'Calculator arguments must equal the selected reported EPS and the supplied consensus EPS.')
 surprise=(args['actual']-args['consensus'])/abs(args['consensus']);put('tool_result',round(surprise,8),True,'Independently recompute (actual − consensus) / |consensus|. Correct arithmetic alone does not validate its inputs.')
 signal='POSITIVE' if surprise>.1 else 'NON_POSITIVE';put('interpretation',signal,True,'Declared demonstration signal: POSITIVE only when surprise exceeds 10%. This threshold is a teaching rule, not a validated alpha model.')
 wanted='BUY' if signal=='POSITIVE' else 'HOLD';action='BUY' if faults.get('recommendation') else wanted
 put('recommendation',action,action==wanted,'Declared action rule: BUY for POSITIVE, otherwise HOLD. Compare the actual proposal with this rule.')
 put('report',f"Reported EPS: {selected['value']:.2f}",nodes['source']['check']=='PASS' and nodes['selected']['check']=='PASS','This parallel report depends on source eligibility and figure selection, not on the calculator.')
 put('risk',surprise<0,all(nodes[x]['check']=='PASS' for x in ['source','selected','tool_request','tool_result']),'This parallel flag is supported only if the inputs and calculation pass; it is not a portfolio risk estimate.')
 put('authority',True,True,'Independent fixed exercise permission: proposed instrument and notional are within the delegated grant.')
 failed=[x for x in ['source','selected','tool_request','tool_result','interpretation','recommendation','authority'] if nodes[x]['check']=='FAIL']
 put('execution','WITHHOLD' if failed else action,not failed,'Block execution if any required observable check fails. Passing means this declared workflow permits its action, not investment merit.')
 return {'nodes':nodes,'first_observable_failure':failed[0] if failed else None,'failed_checks':failed,'terminal_action':nodes['execution']['value'],'proposed_action':action}
def run(case,intervention):
 before=evaluate(case);after=evaluate(case,intervention['nodes']);changed=[n['id'] for n in GRAPH if before['nodes'][n['id']]!=after['nodes'][n['id']]]
 return {'intervention':intervention['id'],'replaced_nodes':intervention['nodes'],'potential_descendants':descendants(intervention['nodes']),'actually_changed_nodes':changed,'remaining_failed_checks':after['failed_checks'],'restored':not after['failed_checks'],'after':after}
def build():
 payload={'version':'1.0','scope':'Executed controlled synthetic workflow; no LLM intermediate states inferred from the saved LoRA experiment.','method':'Policy-defined observable checks plus supplied verified replacements and graph re-evaluation. Conditional localization, not proof of a neural cause.','graph':GRAPH,'reference_document':BASE,'interventions':INTERVENTIONS,'cases':[]}
 checks=0
 for c in CASES:
  before=evaluate(c);assert before['first_observable_failure']==c['expected_first'];checks+=1
  runs=[run(c,i) for i in INTERVENTIONS]
  for r in runs:
   assert r['after']['nodes']['authority']==before['nodes']['authority'];checks+=1
  chosen=next(r for r in runs if r['intervention']==c['repair'])
  assert chosen['restored']==(c['id']!='two_faults');checks+=1
  assert next(r for r in runs if r['intervention']=='all')['restored'];checks+=1
  repaired=next(r for r in runs if r['intervention']=='all')
  assert repaired['after']['terminal_action']==('HOLD' if c['id']=='action' else 'BUY');checks+=1
  assert repaired['after']['nodes']['tool_result']['value']==(-.2 if c['id']=='action' else .2);checks+=1
  if c['id']=='two_faults':assert chosen['remaining_failed_checks']==['tool_request'];checks+=1
  payload['cases'].append({**c,'before':before,'runs':runs})
 # Wrong repair cannot masquerade as resolving an unrelated upstream check.
 for c in CASES[:3]:assert not run(c,INTERVENTIONS[4])['restored'];checks+=1
 payload['verification']={'assertions_passed':checks,'controlled_cases':len(CASES),'intervention_replays':len(CASES)*len(INTERVENTIONS),'negative_control':'Changing only the action cannot resolve source, extraction or tool-argument failures.','independent_branch':'authority unchanged in every intervention replay'}
 return payload
if __name__=='__main__':
 p=build();out=ROOT/'apps/belieflens-decision-lab/public/decision-lab/data/workflow-diagnosis.json';out.write_text(json.dumps(p,indent=2));(Path(__file__).parent/'results.json').write_text(json.dumps(p,indent=2));print(json.dumps(p['verification'],indent=2))
