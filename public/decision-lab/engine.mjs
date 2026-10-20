import {evidenceItems} from './evidence-items.mjs';
export const VERSION='1.0.0';
export const sections=[
['Financial language-model architectures','09:30–10:15',['Compare deployment choices','Distinguish retrieval from adaptation','Inspect exact model context','Choose controls for a workflow'],'Which failure requires new evidence rather than training?','A missing current covenant requires retrieval. Consistent formatting failures may motivate adaptation; execution permissions require a controller.'],
['Building grounded financial RAG','10:15–11:15',['Enforce point-in-time eligibility','Trace source provenance','Identify conflicts and missing coverage','Test retrieval dependence'],'Can a relevant document still be inadmissible?','Yes. A document published or available after the decision time cannot support that historical decision.'],
['Post-training for financial applications','11:30–12:30',['Explain SFT and low-rank updates','Inspect preference objectives','Separate behaviour from knowledge','Distinguish preference from calibration'],'Does preference training establish calibrated confidence?','No. Preference optimization changes relative output likelihood. Calibration requires appropriate labelled outcomes and independent evaluation.'],
['Hands-on adaptation lab','13:15–14:15',['Prepare instruction data','Separate train and test groups','Compare matched configurations','Identify generalization limits'],'Is better training-set performance enough?','No. Test on held-out issuers, time periods and failure types. The notebook trains a small numerical classifier to expose this distinction; it is not an adapted LLM.'],
['LLMs and SLMs for alpha and risk','14:15–15:00',['Define a testable signal','Respect availability timestamps','Detect vintage leakage','Account for costs and baselines'],'Does a good retrospective narrative prove predictive value?','No. Use point-in-time replay, held-out evaluation and financial baselines. The notebook demonstrates how future labels inflate apparent performance.'],
['Evaluation and evidence attribution','15:15–16:15',['Trace evidence to decisions','Run source-removal tests','Measure response stability','Separate confidence from reliability'],'Does repeated agreement prove an answer is correct?','No. Stability can coexist with a consistently wrong answer. Independent labelled cases and source checks address different risks.'],
['Deployment and governance clinic','16:15–17:00',['Define release and escalation rules','Enforce delegated permissions','Compare fixed controls with sequential decisions','Connect consequences to intervention'],'When does a fixed gate stop being a complete decision model?','When actions alter future information, state or costs. An MDP models observed sequential state; a POMDP adds uncertainty about hidden state. A hard permission boundary remains necessary.']
].map((s,i)=>({id:i+1,title:s[0],time:s[1],objectives:s[2],question:s[3],answer:s[4]}));
export function prepare(input={}){
 const c={mode:'rag_tools',cash:15,ebitda:20,dropCovenant:false,futureLeak:false,disruption:true,approved:false,revoked:false,wording:'standard',...input};
 if(!['model_only','rag','rag_tools'].includes(c.mode))throw Error('Unknown architecture');
 for(const k of ['cash','ebitda'])if(typeof c[k]!=='number'||!Number.isFinite(c[k])||c[k]<0||c[k]>1000||(k==='ebitda'&&c[k]===0))throw Error('Invalid '+k);
 for(const k of ['dropCovenant','futureLeak','disruption','approved','revoked'])if(typeof c[k]!=='boolean')throw Error('Invalid '+k);
 if(!['standard','rephrase'].includes(c.wording))throw Error('Invalid wording');
 const sources=[
 {id:'E1',type:'evidence',title:'Harbor Metals filing',published:'2026-10-20T12:00:00Z',available:'2026-10-20T12:02:00Z',text:`Debt USD75m; cash USD${c.cash}m; covenant EBITDA USD${c.ebitda}m. Same reporting period and covenant definitions.`,eligible:true},
 {id:'E2',type:'evidence',title:'Signed covenant',published:'2026-10-20T12:08:00Z',available:'2026-10-20T12:10:00Z',text:'Net leverage = (debt - cash) / covenant EBITDA. Ceiling 3.5x.',eligible:true},
 {id:'E3',type:'evidence',title:'Operations update',published:'2026-10-20T12:40:00Z',available:'2026-10-20T12:42:00Z',text:'Supply disruption. A USD3m EBITDA reduction is a scenario, not a realized result. Credit review remains unresolved.',eligible:true},
 {id:'E4',type:'evidence',title:'Later covenant amendment',published:'2026-10-20T13:05:00Z',available:'2026-10-20T13:07:00Z',text:'Ceiling amended to 4.0x.',eligible:false}
 ];
 for(const source of sources)source.data_points=evidenceItems(source);
 const evidence=c.mode==='model_only'?[]:sources.filter(s=>(s.id!=='E2'||!c.dropCovenant)&&(s.id!=='E3'||c.disruption)&&(s.eligible||c.futureLeak));
 const calculation=c.mode==='rag_tools'?{formula:'(75 - cash) / EBITDA',current:(75-c.cash)/c.ebitda,scenario:c.ebitda>3?(75-c.cash)/(c.ebitda-3):null,ceiling:3.5,inputs:['E1','E2'],note:'Scenario EBITDA falls by USD3m; this is not an observed outcome.'}:null;
 const question=c.wording==='standard'?'Can we release the Harbor Metals credit brief stating comfortable covenant headroom?':'Is the evidence sufficient to publish a comfortable-headroom assessment for Harbor Metals?';
 const messages=[{role:'system',content:'You assess a fictional credit brief. Use only the supplied context. Distinguish current measurements from scenarios. Never invent evidence or permission. Return only JSON with keys claim (string), citations (array of evidence IDs), proposed_action (RELEASE, REVIEW or HOLD).'},
 {role:'user',content:JSON.stringify({question,as_of:'2026-10-20T12:55:00Z',sources:evidence,calculator:calculation})}];
 return {config:c,sources,evidence,calculation,messages};
}
export function evaluate(prepared,answer=null,meta={}){
 const {config:c,evidence,calculation}=prepared, ids=evidence.map(e=>e.id);
 const simulator=answer===null;
 if(simulator)answer={claim:c.mode==='model_only'?'Insufficient supplied evidence to assess headroom.':c.mode==='rag'?'Current documents suggest headroom; calculation and disruption review are needed.':`Current net leverage is ${calculation.current.toFixed(2)}x. ${c.disruption?'The downside scenario needs credit review.':(calculation.current<=3.5?'The current calculation is within the 3.5x ceiling.':'The current calculation exceeds the 3.5x ceiling.')}`,citations:ids.filter(id=>id!=='E4'),proposed_action:'REVIEW'};
 const valid=answer&&typeof answer.claim==='string'&&Array.isArray(answer.citations)&&answer.citations.every(x=>typeof x==='string')&&['RELEASE','REVIEW','HOLD'].includes(answer.proposed_action);
 const check=(id,name,pass,reason,owner)=>({id,name,status:pass?'PASS':'UNRESOLVED',reason,owner});
 const checks=[
 check('R1','Eligible evidence',ids.length>0&&!ids.includes('E4'),ids.includes('E4')?'Future amendment entered context. Remove it and regenerate.':ids.length?'Supplied sources precede the decision cutoff.':'No source context supplied.','Data steward'),
 check('R2','Bound calculation',ids.includes('E1')&&ids.includes('E2')&&!!calculation,calculation&&ids.includes('E2')?'Deterministic calculation bound to filing and covenant.':'Missing covenant or calculator output.','Credit analyst'),
 check('R3','Sufficient support',valid&&answer.citations.includes('E1')&&answer.citations.includes('E2')&&answer.citations.every(id=>ids.includes(id))&&!c.disruption&&!!calculation&&calculation.current<=3.5,'Workshop rule: require filing and covenant citations, a ratio at or below 3.5x, and no unresolved disruption. This is a bounded rule check, not general semantic verification.','Credit risk owner'),
 check('R4','Current authority',c.approved&&!c.revoked,c.revoked?'Simulated approval grant revoked.':c.approved?'Simulated named reviewer approval is present.':'Simulated reviewer approval is absent.','Accountable approver')];
 const gate=checks.every(x=>x.status==='PASS')?'RELEASE':'HOLD';
 const nodes=[...evidence.map(e=>({id:e.id,type:'evidence',label:e.title})),...(calculation?[{id:'T1',type:'tool',label:'Net leverage calculator'}]:[]),{id:'W1',type:'language',label:valid?answer.claim:'Invalid model output'}, {id:'C1',type:'claim',label:'Target: comfortable headroom'},...checks.map(r=>({id:r.id,type:'rule',label:r.name})),{id:'D1',type:'decision',label:gate}];
 const edges=[...evidence.map(e=>({from:e.id,to:'W1',label:'supplied context'})),{from:'W1',to:'C1',label:'assesses target'},...evidence.map(e=>({from:e.id,to:'R1',label:'eligibility checked'})),...['E1','E2'].filter(id=>ids.includes(id)).map(id=>({from:id,to:calculation?'T1':'R2',label:'binds input'})),...(calculation?[{from:'T1',to:'R2',label:'checked calculation'},{from:'T1',to:'R3',label:'supports numerical check'}]:[]),{from:'C1',to:'R3',label:'requires support'},...checks.map(r=>({from:r.id,to:'D1',label:r.status==='PASS'?'permits':'blocks'}))];
 return {version:VERSION,run_id:globalThis.crypto?.randomUUID?.()||String(Date.now()),timestamp:new Date().toISOString(),execution:simulator?'deterministic teaching simulator':'live Together AI',...prepared,answer:valid?answer:{claim:'Invalid structured output',citations:[],proposed_action:'HOLD'},checks,gate,executed_action:'NONE — teaching environment',graph:{nodes,edges},trace:['Prepare point-in-time context','Generate proposed assessment','Apply instructor-defined checks','Check simulated current authority','Record permitted action; execute no financial action'],...meta};
}
export function run(input={}){return evaluate(prepare(input));}
