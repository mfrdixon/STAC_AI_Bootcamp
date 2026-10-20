import {enrichTrace} from './evidence-items.mjs';
// Case-independent inspection of a declared, typed trace. No LLM judge.
export const TRACE_VERSION='belieflens.trace.v1';
export function descendants(nodes,edges,start){
 const seen=new Set(),queue=[start];while(queue.length){const from=queue.shift();for(const e of edges.filter(e=>e.from===from)){if(!seen.has(e.to)&&e.to!==start){seen.add(e.to);queue.push(e.to)}}}return [...seen];
}
function pathTo(edges,start,end){const q=[[start]],seen=new Set([start]);while(q.length){const p=q.shift();if(p.at(-1)===end)return p;for(const e of edges.filter(e=>e.from===p.at(-1))){if(!seen.has(e.to)){seen.add(e.to);q.push([...p,e.to])}}}return [];}
export function validateTrace(t){
 if(t?.schema!==TRACE_VERSION||!Array.isArray(t.nodes)||!Array.isArray(t.edges)||!t.policy||!Number.isFinite(Date.parse(t.as_of)))throw Error('Expected belieflens.trace.v1 with nodes, edges, policy and an ISO as_of timestamp.');
 if(t.nodes.length>300||t.edges.length>1500)throw Error('Trace is too large for this local inspector.');
 const ids=new Set();for(const n of t.nodes){if(typeof n.id!=='string'||!n.id||ids.has(n.id)||!['evidence','tool','language','claim','decision'].includes(n.type)||typeof n.label!=='string')throw Error('Invalid or duplicate node.');ids.add(n.id)}
 for(const e of t.edges)if(!ids.has(e.from)||!ids.has(e.to)||!['observed','declared','context'].includes(e.basis))throw Error('Invalid edge endpoint or basis.');
 const degree=new Map([...ids].map(id=>[id,0]));for(const e of t.edges)degree.set(e.to,degree.get(e.to)+1);const q=[...ids].filter(id=>degree.get(id)===0);let count=0;while(q.length){const id=q.shift();count++;for(const e of t.edges.filter(e=>e.from===id)){degree.set(e.to,degree.get(e.to)-1);if(degree.get(e.to)===0)q.push(e.to)}}if(count!==ids.size)throw Error('Trace must be a directed acyclic graph.');
 return t;
}
export function profileTrace(trace){
 const t=validateTrace(trace),findings=[],nodes=t.nodes,edges=t.edges;
 const add=(rule,node,reason,owner,action)=>findings.push({id:'F'+(findings.length+1),rule,node,reason,owner,action});
 for(const n of nodes){
  if(n.type==='evidence'){
   const dates=['published','available'].map(k=>Date.parse(n[k]));
   if(dates.some(x=>!Number.isFinite(x)))add('temporal_eligibility',n.id,'A publication or availability timestamp is missing or invalid.','Data steward','Establish source timing before relying on it.');
   else if(dates.some(x=>x>Date.parse(t.as_of)))add('temporal_eligibility',n.id,'Source was published or available after the decision cutoff.','Data steward','Exclude this source and revalidate dependent claims.');
   if(!n.provenance?.origin||!n.provenance?.version)add('source_provenance',n.id,'Source origin or version is missing.','Data steward','Obtain a reproducible source reference.');
  }
  if(n.type==='claim'&&!edges.some(e=>e.to===n.id&&e.basis!=='context'&&['evidence','tool','claim'].includes(nodes.find(n=>n.id===e.from)?.type)))add('claim_provenance',n.id,'No evidence, calculation or prior claim is linked as support. A link would establish declared provenance, not factual truth.','Domain reviewer','Request supporting evidence or withdraw the claim.');
  if(n.type==='decision'&&t.policy.require_approval&&n.approval!=='current')add('authority',n.id,'No current approval is recorded in this trace. This does not verify a real permission system.','Accountable approver','Withhold execution pending an authenticated approval.');
  if(n.type==='tool'&&n.error)add('tool_execution',n.id,'Tool returned an error: '+n.error,'Workflow owner','Repair the failed tool call before using its output.');
  if(n.type==='tool'&&!edges.some(e=>e.to===n.id&&e.basis==='observed'))add('tool_provenance',n.id,'The tool has no recorded input-source dependency.','Workflow owner','Record the origins of the inputs.');
 }
 for(const req of t.policy.required_tools||[])if(!nodes.some(n=>n.type==='tool'&&n.tool===req&&!n.error)){const d=nodes.find(n=>n.type==='decision');if(d)add('required_tool',d.id,'Required tool absent or failed: '+req,'Workflow owner','Run the required tool and inspect its result.');}
 for(const finding of findings){
  const linked=edges.filter(e=>e.basis!=='context');
  finding.affected=descendants(nodes,linked,finding.node);
  finding.possible_context_effects=descendants(nodes,edges,finding.node).filter(id=>!finding.affected.includes(id));
  finding.paths=finding.affected.filter(id=>['claim','decision'].includes(nodes.find(n=>n.id===id)?.type)).map(id=>pathTo(linked,finding.node,id));
 }
 return {schema:'belieflens.profile.v1',trace_id:t.id,findings,scope:'Checks declared provenance, timing and trace structure. It does not certify semantic support or infer unrecorded dependencies.',flat_alerts:findings.map(f=>f.rule+': '+f.reason),revalidation:[...new Set(findings.flatMap(f=>[f.node,...f.affected]))],potential_context_review:[...new Set(findings.flatMap(f=>f.possible_context_effects))]};
}
export function traceFromAgent(result){
 const nodes=[],edges=[],add=(from,to,label,basis)=>{if(nodes.some(n=>n.id===from)&&nodes.some(n=>n.id===to))edges.push({from,to,label,basis})};
 for(const e of result.evidence)nodes.push({id:e.id,type:'evidence',label:e.title,published:e.published,available:e.available,provenance:{origin:'synthetic-workshop-fixture',version:'harbor-v1'},text:e.text});
 const toolIds=[];
 for(const [i,event] of (result.tool_events||[]).entries())if(event.tool!=='retrieve_financial_documents'){
  const id='T'+(i+1);toolIds.push(id);nodes.push({id,type:'tool',tool:event.tool,label:event.tool,inputs:event.arguments,output:event.result,error:event.result?.error});for(const source of event.arguments?.source_ids||[])add(source,id,'recorded input','observed');
 }
 nodes.push({id:'W1',type:'language',label:result.answer.claim});
 for(const e of result.evidence)add(e.id,'W1','available in model context','context');
 for(const id of toolIds)add(id,'W1','returned to model','context');
 let claims=Array.isArray(result.answer.claims)?result.answer.claims.filter(c=>typeof c.text==='string'&&Array.isArray(c.source_ids)).slice(0,20):[];
 if(!claims.length)claims=[{text:result.answer.claim,source_ids:result.answer.citations||[]}];
 claims.forEach((c,i)=>{const id='C'+(i+1);nodes.push({id,type:'claim',label:c.text});add('W1',id,'expressed in response','context');for(const source of c.source_ids)add(source,id,'agent-declared support','declared');if(c.uses_calculator===true)for(const tool of toolIds)add(tool,id,'agent-declared calculation dependency','declared')});
 nodes.push({id:'D1',type:'decision',label:result.answer.proposed_action,approval:result.config.approved&&!result.config.revoked?'current':'absent'});
 for(const c of nodes.filter(n=>n.type==='claim'))add(c.id,'D1','assessment informs proposed action','declared');
 return enrichTrace({schema:TRACE_VERSION,id:result.run_id,as_of:'2026-10-20T12:55:00Z',nodes,edges,policy:{require_approval:true,required_tools:['calculate_leverage']},construction:'Observed tool arguments and outputs; agent-declared claim citations. Context exposure is recorded separately and is not a proven dependency.'},result);
}
// Expert defines the target population, required strata, and comparable metric.
export function coverageAudit(input){
 const {requirements,observations}=input;
 if(!Array.isArray(requirements)||!Array.isArray(observations))throw Error('Requirements and observations must be arrays.');
 const rows=requirements.map(r=>{const matches=observations.filter(o=>Object.entries(r.match).every(([k,v])=>o[k]===v));return {...r,present:matches.length>0,source_ids:matches.map(o=>o.id)}});
 const flags=rows.filter(r=>!r.present).map(r=>({type:'coverage_gap',reason:'Missing required stratum: '+r.label,owner:r.owner||'Market research specialist'}));
 const metrics=new Set(observations.map(o=>o.metric));if(metrics.size>1)flags.push({type:'metric_mismatch',reason:'Sales, ratings or other metrics are not interchangeable measures of consumer preference.',owner:'Measurement specialist'});
 if(observations.some(o=>o.selection!=='probability_sample'))flags.push({type:'selection_risk',reason:'Some observations use convenience or voluntary samples; representativeness is not established.',owner:'Survey statistician'});
 if(observations.some(o=>o.metric==='sales_units'&&!o.denominator))flags.push({type:'denominator_missing',reason:'Sales counts lack a declared exposure denominator. Population scaling alone would not establish taste preference.',owner:'Market research specialist'});
 return {rows,flags,decision:flags.length?'ESCALATE':'REVIEW',scope:'Coverage is assessed against the declared requirements, not all possible sources of bias. No automatic release or unbiasedness certificate.'};
}
export function partialIdentification(observedMean,missingWeight){
 if(!Number.isFinite(observedMean)||observedMean<0||observedMean>1||!Number.isFinite(missingWeight)||missingWeight<0||missingWeight>1)throw Error('Mean and missing weight must lie in [0,1].');
 return {lower:(1-missingWeight)*observedMean,upper:(1-missingWeight)*observedMean+missingWeight,interpretation:'Worst-case identification interval, not a confidence interval. Assumes a bounded [0,1] outcome, known missing population share and representative observed strata; sampling error is not included.'};
}
// Transparent relevance and stratified comparison: deterministic reasoning + statistics.
export function relevanceProfile(source,question){
 const reasons=[], dimensions={entity:question.entities.includes(source.entity),metric:source.metric===question.metric,period:source.period===question.period,population:source.population===question.population};
 for(const [k,v] of Object.entries(dimensions))if(!v)reasons.push(k+' does not match the question');
 return {dimensions,score:Object.values(dimensions).filter(Boolean).length/4,eligible:Object.values(dimensions).every(Boolean),reasons,meaning:'Fraction of four declared relevance criteria met. This is not a truth or relevance probability.'};
}
export function compareStrata(groups,targetWeights){
 const entities=[...new Set(groups.map(g=>g.entity))];if(entities.length!==2)throw Error('Exactly two entities required.');
 if(Object.values(targetWeights).some(w=>!Number.isFinite(w)||w<0)||Math.abs(Object.values(targetWeights).reduce((a,b)=>a+b,0)-1)>1e-8)throw Error('Target weights must sum to one.');
 const statistics=entities.map(entity=>{const rows=groups.filter(g=>g.entity===entity);let count=0,wins=0,standardized=0,complete=true;for(const row of rows){if(!Number.isInteger(row.n)||row.n<=0||!Number.isInteger(row.wins)||row.wins<0||row.wins>row.n)throw Error('Invalid binomial counts.');count+=row.n;wins+=row.wins;}
 for(const [stratum,weight]of Object.entries(targetWeights)){const matched=rows.filter(g=>g.stratum===stratum);if(matched.length!==1){complete=false;continue}standardized+=weight*matched[0].wins/matched[0].n;}
 return {entity,n:count,raw:wins/count,standardized:complete?standardized:null,complete};});
 const raw_difference=statistics[0].raw-statistics[1].raw,adjusted_difference=statistics.every(s=>s.complete)?statistics[0].standardized-statistics[1].standardized:null;
 return {statistics,raw_difference,adjusted_difference,reversal:adjusted_difference!==null&&raw_difference*adjusted_difference<0,scope:'Descriptive standardization to expert-selected opponent weights. It does not establish causal team quality; games may be dependent, and sampling uncertainty is not estimated here.'};
}
export function wilson(successes,n,z=1.96){
 if(!Number.isInteger(n)||n<=0||!Number.isInteger(successes)||successes<0||successes>n)throw Error('Invalid counts.');
 const p=successes/n,d=1+z*z/n,c=(p+z*z/(2*n))/d,h=z*Math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;
 return {lower:c-h,upper:c+h};
}
export function reportingBias({wins,losses,coveredWins,coveredLosses}){
 const a=wilson(coveredWins,wins),b=wilson(coveredLosses,losses);
 return {win_coverage:coveredWins/wins,loss_coverage:coveredLosses/losses,gap:coveredWins/wins-coveredLosses/losses,win_interval:a,loss_interval:b,scope:'Coverage disparity against a complete reference schedule. Wilson intervals are approximate marginal 95% intervals under binomial sampling, not a joint interval for the gap. For a census of this period, the gap is descriptive and needs no sampling interval.'};
}
export function robustOutliers(values,threshold=3.5){
 if(!Array.isArray(values)||values.length<3||values.some(v=>!Number.isFinite(v)))throw Error('Provide at least three finite observations.');
 const median=a=>{const s=[...a].sort((x,y)=>x-y),m=Math.floor(s.length/2);return s.length%2?s[m]:(s[m-1]+s[m])/2};
 const center=median(values),mad=median(values.map(v=>Math.abs(v-center)));
 return {median:center,mad,threshold,rows:values.map((value,i)=>({index:i,value,score:mad?0.67448975*(value-center)/mad:null,flag:mad?Math.abs(0.67448975*(value-center)/mad)>threshold:null})),scope:'Modified MAD score is a screening heuristic, not a p-value or evidence of bias. Zero MAD yields unknown scores. Investigate provenance and competing explanations before changing the frame.'};
}
