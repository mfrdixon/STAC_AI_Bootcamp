// Declared teaching policy and hypothetical records; not real team or survey data.
export const sportsQuestion = 'Which is better: New York Yankees or New England Patriots?';
export const sportsCases = [
 {id:'unclear',label:'1 · Ask “which is better?”',objective:false,coverage:false,comparable:false,approval:false},
 {id:'coverage',label:'2 · Define campaign; use one fan poll',objective:true,coverage:false,comparable:false,approval:false},
 {id:'mismatch',label:'3 · Add sources; retain mismatched measures',objective:true,coverage:true,comparable:false,approval:false},
 {id:'review',label:'4 · Resolve evidence; await approval',objective:true,coverage:true,comparable:true,approval:false},
 {id:'release',label:'5 · Record current approval',objective:true,coverage:true,comparable:true,approval:true}
];
export function evaluateSports(input){
 const s=typeof input==='string'?sportsCases.find(c=>c.id===input):input;
 if(!s)throw Error('Unknown sports case');
 const rules=[
  {id:'R1',name:'Define the decision',node:'O1',passed:s.objective===true,owner:'Business owner',action:'Clarify objective, audience, market and period',reason:s.objective?'Campaign objective and scope recorded.':'“Better” has no approved comparison criterion.',record:'Signed objective and target population',path:['O1','M1','C1','A1']},
  {id:'R2',name:'Check the evidence',node:'E1 / E2',passed:s.coverage===true,owner:'Data steward + survey specialist',action:'Obtain coverage and source-method records',reason:s.coverage?'Both teams have source-method and target-population records in this fixture.':'A voluntary Yankees fan poll and missing equivalent Patriots evidence cannot support a general-market comparison.',record:'Provenance, sampling frame and coverage gaps',path:['E1 / E2','M1','C1','A1']},
  {id:'R3',name:'Validate the inference',node:'E → M → C',passed:s.comparable===true,owner:'Statistician + commercial expert',action:'Match measures and qualify the claim',reason:s.comparable?'Reviewed measures share scope, period and denominator; claim limited to a comparison memo.':'Audience counts and stated purchase interest are different outcomes; neither alone establishes campaign return.',record:'Matched metrics, assumptions and uncertainty review',path:['M1','C1','A1']},
  {id:'R4',name:'Authorize the action',node:'A1',passed:s.approval===true,owner:'Named approver',action:'Verify current approval after material checks pass',reason:s.approval?'Current simulated approval recorded.':'No current approval to release the memo.',record:'Policy version, check results and signed decision',path:['A1']}
 ];
 const failed=rules.filter(r=>!r.passed);
 const decision=!rules[0].passed?'CLARIFY':!rules[1].passed||!rules[2].passed?'ESCALATE / WITHHOLD RANKING':!rules[3].passed?'AWAIT APPROVAL':'RELEASE SCOPED MEMO';
 return {case:s.id,decision,release:failed.length===0,rules,affected:[...new Set(failed.flatMap(r=>r.path))],policy:'sports-sponsorship-v1',scope:'Hypothetical metadata and expert-review states. No real sports ranking, survey validation, autonomous bias discovery or authenticated release is measured.'};
}
