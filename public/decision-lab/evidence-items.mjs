// Deterministic, source-anchored itemization for this workshop's short documents.
// A unit is one semicolon/sentence-delimited evidence item, not an independent observation.
export function evidenceItems(doc){
 const items=[];const re=/[^;]+?(?:;|\.(?=\s+[A-Z]|$)|$)/g;
 for(const match of doc.text.matchAll(re)){const raw=match[0],quote=raw.trim();if(!quote)continue;
 const start=match.index+raw.indexOf(quote);items.push({id:`${doc.id}:P${items.length+1}`,source_id:doc.id,quote,start,end:start+quote.length,kind:/\d/.test(quote)?'numeric fact or scenario':'clause',method:'deterministic clause segmentation v1'});
 }return items;
}
export function enrichTrace(trace,result){
 const byId=new Map(trace.nodes.map(n=>[n.id,n]));
 for(const n of trace.nodes){
  if(n.type==='evidence')n.data_points=evidenceItems(n);
  if(n.type==='tool'&&!n.error)n.data_points=['current','scenario'].filter(k=>Number.isFinite(n.output?.[k])).map(k=>({id:`${n.id}:${k}`,source_id:n.id,field:k,value:n.output[k],kind:'computed output',method:'recorded calculator output'}));
 }
 const claims=(result.answer.claims||[]).filter(c=>typeof c.text==='string'&&Array.isArray(c.source_ids)).slice(0,20);
 for(const e of trace.edges){
  const from=byId.get(e.from),to=byId.get(e.to);let items=null,method='No item-level link recorded';
  if(e.basis==='context'&&to?.type==='language'&&from?.data_points){items=from.data_points;method='Items supplied in context; exposure is not use or support';}
  if(e.basis==='observed'&&to?.type==='tool'&&from?.type==='evidence'){
   const patterns={debt:/\bDebt USD([\d.]+)m/i,cash:/\bcash USD([\d.]+)m/i,ebitda:/\bcovenant EBITDA USD([\d.]+)m/i,downside_reduction:/USD([\d.]+)m EBITDA reduction/i};
   items=from.data_points.filter(p=>Object.entries(patterns).some(([field,re])=>{const m=p.quote.match(re);return m&&Number(m[1])===to.inputs?.[field]}));
   method='Exact numeric field/value match to recorded calculator arguments; fixture-specific parser. Non-numeric definitions are not counted as arithmetic inputs.';
  }
  if(e.basis==='declared'&&to?.type==='claim'&&from?.data_points){
   const c=claims[Number(to.id.slice(1))-1],ids=c?.evidence_point_ids;
   if(Array.isArray(ids)){items=from.data_points.filter(p=>ids.includes(p.id));method='Model-declared item references, validated for existence and source membership; semantic support is not verified';}
  }
  e.data_points=items;e.data_point_count=items===null?null:items.length;e.count_method=method;
 }
 for(const e of trace.edges){
  const peers=trace.edges.filter(p=>p.to===e.to&&p.label===e.label&&p.basis===e.basis);
  const known=peers.every(p=>p.data_point_count!==null);const total=known?peers.reduce((s,p)=>s+p.data_point_count,0):null;
  e.normalization={scope:'Same target, relationship and basis',total,complete:known};
  e.connection_share=known&&total>0?e.data_point_count/total:null;
 }
 return trace;
}
