import {descendants} from './repair-engine.mjs';
const copy=x=>JSON.parse(JSON.stringify(x));
export function evaluateWorkflow(spec,fixture,repairs=[]){
 const known=new Set(spec.graph.map(n=>n.id));for(const r of repairs)if(!known.has(r))throw Error('Unknown repair node');
 const ref={...copy(spec.reference_document),reported_eps:fixture.reported_eps??spec.reference_document.reported_eps};
 const faults=Object.fromEntries(Object.entries(fixture.faults).filter(([k])=>!repairs.includes(k))),nodes={},pending=new Map(spec.graph.map(n=>[n.id,n]));
 const pass=id=>nodes[id].check==='PASS';
 const explain=id=>fixture.before.nodes[id].explanation;
 while(pending.size){let advanced=false;for(const [id,n]of pending){if(n.parents.some(p=>!nodes[p]))continue;let value,ok=true;switch(id){
 case 'source':value=copy(ref);if(faults.source)Object.assign(value,{period:'2026 Q1',reported_eps:.8,id:'HM-Q1-release'});ok=value.period===ref.period&&value.issuer===ref.issuer;break;
 case 'selected':{const s=nodes.source.value;value={value:faults.selected?s.forecast_eps:s.reported_eps,basis:faults.selected?'forecast':'reported',source_id:s.id};ok=value.basis==='reported'&&value.value===s.reported_eps;break;}
 case 'tool_request':value={actual:nodes.selected.value.value,consensus:faults.tool_request?2:nodes.source.value.consensus_eps};ok=value.actual===nodes.selected.value.value&&value.consensus===nodes.source.value.consensus_eps;break;
 case 'tool_result':{const a=nodes.tool_request.value;if(a.consensus===0)throw Error('Zero consensus denominator');value=Number(((a.actual-a.consensus)/Math.abs(a.consensus)).toFixed(8));break;}
 case 'interpretation':value=nodes.tool_result.value>.1?'POSITIVE':'NON_POSITIVE';break;
 case 'recommendation':{const wanted=nodes.interpretation.value==='POSITIVE'?'BUY':'HOLD';value=faults.recommendation?'BUY':wanted;ok=value===wanted;break;}
 case 'report':value=`Reported EPS: ${nodes.selected.value.value.toFixed(2)}`;ok=pass('source')&&pass('selected');break;
 case 'risk':value=nodes.tool_result.value<0;ok=['source','selected','tool_request','tool_result'].every(pass);break;
 case 'authority':value=true;break;
 case 'execution':ok=n.parents.every(pass);value=ok?nodes.recommendation.value:'WITHHOLD';break;
 default:throw Error('Unknown workflow operator '+id);
 }
 nodes[id]={value,check:ok?'PASS':'FAIL',explanation:explain(id),parents:n.parents};pending.delete(id);advanced=true;
 }if(!advanced)throw Error('Cycle or missing dependency');}
 const order=['source','selected','tool_request','tool_result','interpretation','recommendation','authority'],failed=order.filter(n=>!pass(n));
 return {nodes,first_observable_failure:failed[0]??null,failed_checks:failed,terminal_action:nodes.execution.value,proposed_action:nodes.recommendation.value};
}
export function interveneWorkflow(spec,fixture,intervention){
 const before=evaluateWorkflow(spec,fixture),after=evaluateWorkflow(spec,fixture,intervention.nodes);
 const g=Object.fromEntries(spec.graph.map(n=>[n.id,{deps:n.parents}]));const potential=new Set(descendants(g,intervention.nodes));
 return {intervention:intervention.id,replaced_nodes:intervention.nodes,potential_descendants:spec.graph.filter(n=>potential.has(n.id)).map(n=>n.id),actually_changed_nodes:spec.graph.filter(n=>JSON.stringify(before.nodes[n.id])!==JSON.stringify(after.nodes[n.id])).map(n=>n.id),remaining_failed_checks:after.failed_checks,restored:after.failed_checks.length===0,after};
}
