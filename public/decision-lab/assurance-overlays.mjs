import {verifySemanticEdge} from './semantic-matching.mjs';
// Finite, declared support calculus from Evidence Obligations and Uncertainty in Typed Derivation Graphs.
// This evaluates a supplied specification, not unrestricted semantic truth.
export function minimalSupports(target,primitives,rules){
 const base=new Set(primitives),memo=new Map(),visiting=new Set();
 const minimize=sets=>sets.filter((s,i)=>!sets.some((t,j)=>j!==i&&t.every(x=>s.includes(x))&&(t.length<s.length||j<i)));
 function visit(id){if(base.has(id))return [[id]];if(memo.has(id))return memo.get(id);if(visiting.has(id))throw Error('Cyclic derivation specification');visiting.add(id);let supports=[];
 for(const rule of rules.filter(r=>r.conclusion===id)){let products=[[]];for(const p of rule.premises){products=products.flatMap(a=>visit(p).map(b=>[...new Set([...a,...b])].sort()));if(products.length>4096)throw Error('Support enumeration limit exceeded');products=minimize(products)}supports.push(...products)}
 visiting.delete(id);supports=minimize(supports);memo.set(id,supports);return supports;}return visit(target);
}
export function evaluateSpecification(spec){
 const supports=minimalSupports(spec.target,spec.obligations.map(o=>o.id),spec.rules),admitted=new Set(spec.obligations.filter(o=>o.admitted).map(o=>o.id));
 const missing=supports.map(s=>s.filter(x=>!admitted.has(x))).filter((s,i,a)=>!a.some((t,j)=>j!==i&&t.every(x=>s.includes(x))&&(t.length<s.length||j<i)));
 return {supports,missing,derivable:supports.some(s=>s.every(x=>admitted.has(x)))};
}
export function buildOverlays(result){
 const g=result.graph,spec=result.assurance_specification||g.assurance_specification||null,assessment=spec?evaluateSpecification(spec):null;
 const cutoff=result.evidence_trace?.as_of||g.as_of;const nodes=g.nodes,edges=g.edges;
 const linked=edges.filter(e=>e.basis!=='context');
 function descendants(id){const seen=new Set(),queue=[id];while(queue.length){const from=queue.shift();for(const e of linked.filter(e=>e.from===from))if(!seen.has(e.to)){seen.add(e.to);queue.push(e.to)}}return [...seen]}
 const byNode={};
 for(const n of nodes){const details={};
 if(n.type==='evidence'){
  const time=[n.published,n.available].map(Date.parse);const unknown=!cutoff||!Number.isFinite(Date.parse(cutoff))||time.some(x=>!Number.isFinite(x)),late=!unknown&&time.some(x=>x>Date.parse(cutoff));
  details.timing={status:unknown?'unknown':late?'review':'pass',reason:unknown?'Timing cannot be assessed from this trace.':late?'Published or available after assessment cutoff.':'Published and available by cutoff.',owner:'Data steward',action:late?'Exclude and revalidate dependent claims.':unknown?'Supply cutoff and source timestamps.':'Retain timestamp record.',cutoff,published:n.published,available:n.available};
  const mandatory=spec?.obligations.filter(o=>o.node_ids?.includes(n.id))||[];
  details.relevance={status:'unknown',reason:'Source-to-claim semantic matching is not verified. Recorded connections identify candidate relevance only.',semantic_profile:n.relevance_metadata||'Entity, metric, period and population require an approved mapping; no scalar score inferred from citations.',derivational_relevance:mandatory.map(o=>o.label),recorded_downstream:descendants(n.id),decision_relevance:spec?{release_currently_derivable:assessment.derivable,release_after_source_withdrawal:evaluateSpecification({...spec,obligations:spec.obligations.map(o=>o.node_ids?.includes(n.id)?{...o,admitted:false}:o)}).derivable,method:'Conservative withdrawal of directly bound case obligations, not a causal model rerun.'}:'No declared decision specification',owner:'Domain reviewer',action:'Verify source-to-claim entity, metric, period and scenario compatibility.'};
  const family=n.provenance?.source_family;const peers=family?nodes.filter(p=>p.type==='evidence'&&p.provenance?.source_family===family).map(p=>p.id):[];
  details.sources={status:peers.length>1?'review':'unknown',reason:peers.length>1?'Several source records share a declared upstream family.':'Independence is unverified: no established upstream ancestry or reference registry.',source_family:family||'Unknown',shared_records:peers,statistical_bias:'Not identified: no eligible-event registry or matched reference measurements supplied.',origin:n.provenance,owner:'Data steward / statistical reviewer',action:'Establish upstream origin and a suitable reference before estimating reporting disparity or measurement bias.'};
 }
 const unresolved=spec?.obligations.filter(o=>!o.admitted&&o.node_ids?.includes(n.id))||[];
 details.gaps={status:unresolved.length?'review':spec?'pass':'unknown',reason:unresolved.length?'Unresolved mandatory case obligations affect this node.':spec?'No unresolved obligation is bound directly to this node; this does not certify its claims.':'No declared obligations supplied.',obligations:unresolved,owner:unresolved.map(o=>o.owner).join('; ')||'Workflow owner',action:unresolved.map(o=>o.action).join(' ')||'Review the declared specification and unrepresented assumptions.'};
 const incoming=linked.filter(e=>e.to===n.id),outgoing=linked.filter(e=>e.from===n.id);
 details.dependencies={status:incoming.length||outgoing.length?'pass':'unknown',label:incoming.length||outgoing.length?'linked':'no support link',reason:'Recorded non-context dependencies; this is structural relevance, not verified semantic support.',incoming:incoming.map(e=>({from:e.from,relationship:e.label,basis:e.basis})),outgoing:outgoing.map(e=>({to:e.to,relationship:e.label,basis:e.basis})),affected_downstream:descendants(n.id),owner:'Domain reviewer',action:'Inspect whether each declared relationship is substantively justified.'};
 const bound=spec?.obligations.filter(o=>o.node_ids?.includes(n.id))||[];
 const without=spec?evaluateSpecification({...spec,obligations:spec.obligations.map(o=>bound.some(b=>b.id===o.id)?{...o,admitted:false}:o)}):null;
 const changed=assessment&&without?assessment.derivable!==without.derivable:false;
 details.sensitivity={status:!spec?'unknown':changed?'review':'pass',label:!spec?'no policy':changed?'release → hold':!assessment.derivable?'already held':'no gate change',reason:!spec?'A decision specification is required.':changed?'Withdrawing this node’s bound premises changes the declared release warrant.':!assessment.derivable?'The release warrant is already blocked. Withdrawal cannot make an already-held decision more held; this does not mean the evidence is irrelevant.':'Withdrawal does not change the current declared gate.',bound_obligations:bound.map(o=>o.label),before:assessment?.derivable?'RELEASE WARRANT':'HOLD',after:without?.derivable?'RELEASE WARRANT':'HOLD',structurally_required:bound.filter(o=>assessment?.supports.length&&assessment.supports.every(s=>s.includes(o.id))).map(o=>o.label),method:'Conservative withdrawal of directly bound policy premises. No model rerun, causal claim or probability of truth.',owner:'Risk owner',action:changed?'Require these premises before release.':'Inspect unresolved obligations and repeat after resolving them.'};
 byNode[n.id]=details;
 }
 const byEdge=edges.map(e=>({relevance:verifySemanticEdge(e,g)}));
 for(const n of nodes){const assessed=edges.map((e,i)=>e.from===n.id||e.to===n.id?byEdge[i].relevance:null).filter(v=>v&&v.label!=='context only'&&v.label!=='not assessed');const checked=assessed.filter(v=>v.status!=='unknown');if(checked.length)byNode[n.id].relevance={status:checked.some(v=>v.status==='review')?'review':'pass',label:checked.some(v=>v.status==='review')?'mismatch':'bounded match',reason:'Edge-specific quantitative checks available. Click each edge for its witnesses and limitations.',checked_relationships:checked.length,unverified_relationships:assessed.length-checked.length};}
 return {spec,assessment,byNode,byEdge,scope:'Paper-based finite support checks. Semantic validity, unknown source bias and open-world completeness are not certified.'};
}
