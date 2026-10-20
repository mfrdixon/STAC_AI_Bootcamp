// Case-specific policy adapter. Not part of the generic BeliefLens inference engine.
export function harborSpecification(result){
 if(!result.config||!result.checks)return null;
 const nodes=result.graph.nodes,has=id=>nodes.some(n=>n.id===id),check=id=>result.checks.find(c=>c.id===id)?.status==='PASS';
 const obs=[
 ['O1','Filing available',has('E1'),['E1'],'Data steward','Retrieve the applicable filing.'],
 ['O2','Signed covenant available',has('E2'),['E2'],'Credit specialist','Obtain the signed covenant and verify applicability.'],
 ['O3','Calculator inputs verified',check('R2'),nodes.filter(n=>n.type==='tool').map(n=>n.id),'Workflow owner','Verify the numeric inputs, definition and calculator result.'],
 ['O4','Evidence meets cutoff',check('R1'),nodes.filter(n=>n.type==='evidence').map(n=>n.id),'Data steward','Exclude future-dated evidence and rerun the assessment.'],
 ['O5','Credit support sufficient',check('R3'),nodes.filter(n=>n.type==='claim').map(n=>n.id),'Credit risk owner','Resolve scenario impact, citations and leverage limits under the case policy.'],
 ['O6','Current simulated approval',check('R4'),['D1'],'Accountable approver','Obtain approval; a workshop flag is not an authenticated grant.']
 ].map(([id,label,admitted,node_ids,owner,action])=>({id,label,admitted,node_ids,owner,action,admission_method:'Harbor Metals case adapter using recorded inputs and explicit workshop rules'}));
 return {version:'harbor-release-v1',question:'May this credit brief pass the workshop release policy?',target:'release_warrant',obligations:obs,rules:[{id:'release-and',premises:obs.map(o=>o.id),conclusion:'release_warrant'}],scope:'All six obligations are jointly required. This warrant concerns the declared workshop release policy, not truth of the phrase comfortable headroom, nor permission to execute a financial action.'};
}
