// Bounded, deterministic semantic checks for financial quantities.
// Grammar matches and arithmetic witnesses are explicit; arbitrary language is not certified.
const numeric='([0-9]+(?:\\.[0-9]+)?)';
export function extractClaim(text){
 if(/\b(not|never|incorrect|wrong|false|isn't|isn’t)\b/i.test(text)||/[?]/.test(text))return {error:'Negated or interrogative wording is outside the affirmative-claim grammar.'};
 const patterns=[['current_leverage',new RegExp('current (?:net )?leverage(?: ratio)?\\s*(?:is|of|=|:)?\\s*'+numeric+'\\s*[x×]','ig')],['scenario_leverage',new RegExp('(?:downside|scenario|stressed) (?:net )?leverage(?: ratio)?\\s*(?:is|of|=|:)?\\s*'+numeric+'\\s*[x×]','ig')],['ceiling',new RegExp('(?:covenant (?:ceiling|limit)|(?:applicable )?ceiling)\\s*(?:is|of|=|:)?\\s*'+numeric+'\\s*[x×]','ig')]];
 const matches=patterns.flatMap(([metric,re])=>[...text.matchAll(re)].map(m=>({metric,value:Number(m[1]),quote:m[0],start:m.index,end:m.index+m[0].length,unit:'times'})));
 if(matches.length!==1)return {error:matches.length?'More than one quantitative proposition: split the claim before verification.':'No supported quantitative proposition found; qualitative or unparsed claims require review.'};
 return matches[0];
}
function base(status,label,reason,extra={}){return {status,label,reason,method:'financial-quantity-grammar-v1 + deterministic arithmetic',scope:'Verifies a bounded quantity relationship only. Does not verify source truth, legal applicability, qualitative judgment or action permission.',owner:'Credit specialist',action:status==='pass'?'Retain witnesses and separately check time, source validity and mandatory policy obligations.':'Review the claim and evidence; do not treat this relationship as verified.',...extra};}
function calculate(tool,metric){const a=tool.inputs||{},field=metric==='current_leverage'?'current':'scenario';if(tool.error||![a.debt,a.cash,a.ebitda].every(Number.isFinite)||a.ebitda<=0)return {error:'Missing or invalid calculator inputs.'};if(metric==='scenario_leverage'&&(!Number.isFinite(a.downside_reduction)||a.downside_reduction<0||a.ebitda<=a.downside_reduction))return {error:'Scenario inputs do not define positive EBITDA.'};const expected=(a.debt-a.cash)/(a.ebitda-(metric==='scenario_leverage'?a.downside_reduction:0));const recorded=tool.output?.[field];if(!Number.isFinite(recorded))return {error:'Requested current/scenario output is absent.'};if(Math.abs(recorded-expected)>1e-8)return {contradiction:'Recorded tool output disagrees with recomputed arithmetic.',expected,recorded};return {expected,recorded,formula:metric==='current_leverage'?'(debt − cash) / EBITDA':'(debt − cash) / (EBITDA − downside reduction)',inputs:a,field};}
function numericFields(source,tool,metric){const patterns={debt:/\bDebt USD([\d.]+)m/i,cash:/\bcash USD([\d.]+)m/i,ebitda:/\bcovenant EBITDA USD([\d.]+)m/i,...(metric==='scenario_leverage'?{downside_reduction:/USD([\d.]+)m EBITDA reduction/i}:{})};return Object.entries(patterns).flatMap(([field,re])=>{const m=source.text?.match(re);return m?[{field,source_value:Number(m[1]),tool_value:tool.inputs?.[field],quote:m[0],start:m.index,end:m.index+m[0].length,match:Number(m[1])===tool.inputs?.[field]}]:[]});}
export function verifySemanticEdge(edge,graph){
 const from=graph.nodes.find(n=>n.id===edge.from),to=graph.nodes.find(n=>n.id===edge.to);
 if(edge.basis==='context')return base('unknown','context only','Availability in context does not establish use or support.');
 if(from?.type==='evidence'&&to?.type==='tool'){
  const witnesses=numericFields(from,to,'scenario_leverage');if(!witnesses.length)return base('unknown','not established','No supported numeric field mapping exists for this input link.');const all=witnesses.every(w=>w.match);return base(all?'pass':'review',all?'input match':'input mismatch',all?'Source numeric fields match the recorded calculator arguments.':'At least one recorded calculator input differs from its cited source.',{witnesses,source_id:from.id});
 }
 if(to?.type!=='claim')return base('unknown','not assessed','This is a process or policy link, not a source-to-claim quantitative assertion.');
 const claim=extractClaim(to.label);if(claim.error)return base('unknown','not established',claim.error);
 if(claim.metric==='ceiling'){
  if(from?.type!=='evidence')return base('unknown','not established','The contract ceiling requires a source statement, not an arithmetic result.',{claim});
  const matches=[...(from.text||'').matchAll(/\bCeiling\s+(?:amended\s+to\s+|is\s+)?([\d.]+)\s*x/ig)];
  if(matches.length!==1)return base('unknown','not established','No unique contractual ceiling was extracted from this source.',{claim});
  const m=matches[0],reference=Number(m[1]),same=Math.abs(reference-claim.value)<1e-9;
  return base(same?'pass':'review',same?'quantity match':'contradiction',same?'The stated ceiling matches the cited source value. Its temporal and contractual applicability are separate checks.':'The claim’s ceiling differs from the cited source.',{claim,reference,witnesses:[{source_id:from.id,quote:m[0],start:m.index,end:m.index+m[0].length}]});
 }
 let tools=[];
 if(from?.type==='tool')tools=[from];
 else if(from?.type==='evidence')tools=graph.nodes.filter(n=>n.type==='tool'&&graph.edges.some(e=>e.from===from.id&&e.to===n.id&&e.basis==='observed')&&graph.edges.some(e=>e.from===n.id&&e.to===to.id&&e.basis==='declared'));
 if(tools.length!==1)return base('unknown','not established',tools.length?'Multiple recorded calculator paths: disambiguate the claim dependency.':'No unique recorded calculator path links this source to the quantitative claim.',{claim});
 const tool=tools[0],calculation=calculate(tool,claim.metric);if(calculation.error)return base('unknown','not established',calculation.error,{claim});if(calculation.contradiction)return base('review','contradiction',calculation.contradiction,{claim,calculation});
 // Fixed tolerance corresponds to rounding to two decimal places; never silently use broad integer rounding.
 if(Math.abs(calculation.expected-claim.value)>0.005000001)return base('review','contradiction','The claim differs from independently recomputed leverage (tolerance 0.005×).',{claim,calculation});
 if(from.type==='evidence'){const witnesses=numericFields(from,tool,claim.metric);if(!witnesses.length)return base('unknown','not established','This source is linked, but no relevant quantitative input was matched.',{claim,calculation});if(witnesses.some(w=>!w.match))return base('review','input mismatch','Calculator arguments differ from the linked source.',{claim,calculation,witnesses});return base('pass','input match','Matched source inputs contribute to a calculation consistent with this claim. Other required inputs and source validity must be checked separately.',{claim,calculation,witnesses,scope:'Verified contribution of the listed source fields; not an assertion that this source alone entails the claim.'});}
 return base('pass','arithmetic match','The stated current/scenario leverage agrees with arithmetic recomputed from recorded inputs.',{claim,calculation,scope:'Arithmetic verified conditional on the recorded inputs; input accuracy and source applicability remain separate obligations.'});
}
