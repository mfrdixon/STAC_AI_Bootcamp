export const CUTOFF='2025-01-12T16:00:00Z';
const url='https://harbor.example/investors/q4-results';
export const VERSIONS={
 archived:{id:'v1',url,published:'2025-01-10T12:00:00Z',firstObserved:'2025-01-10T12:05:00Z',retrieved:'2025-02-01T12:00:00Z',eps:0.80,text:'Harbor Metals reports Q4 adjusted EPS of $0.80. Comparable consensus is $1.00.'},
 revised:{id:'v2',url,published:'2025-01-20T12:00:00Z',firstObserved:'2025-01-20T12:05:00Z',retrieved:'2025-02-01T12:00:00Z',eps:1.20,text:'Correction, January 20: Q4 adjusted EPS is $1.20, replacing the previously reported $0.80. Comparable consensus is $1.00.'},
 unknown:{id:'unverified',url,published:'2025-01-10T12:00:00Z',firstObserved:null,retrieved:'2025-02-01T12:00:00Z',eps:1.20,text:'Harbor Metals reports Q4 adjusted EPS of $1.20. Comparable consensus is $1.00. Page header: January 10.'}
};
export function checkVersion(v,cutoff=CUTOFF){
 const t=Date.parse(cutoff),p=Date.parse(v.published),o=Date.parse(v.firstObserved);
 if(!Number.isFinite(t))throw Error('Invalid decision cutoff');
 if(!Number.isFinite(o))return {pass:false,reason:'Historical availability unverified: no first-observed record for these exact bytes.'};
 if(!Number.isFinite(p)||p>t||o>t)return {pass:false,reason:'This content version became available after the decision cutoff.'};
 return {pass:true,reason:'This exact version was published and first observed before the cutoff.'};
}
export function replay(id,cutoff=CUTOFF){
 if(!Object.hasOwn(VERSIONS,id))throw Error('Unknown version');
 const v=VERSIONS[id],check=checkVersion(v,cutoff);
 const context=check.pass?[{id:v.id,text:v.text,eps:v.eps}]:[];
 // The deterministic mock consumes ONLY the filtered context. No live fetch/fallback.
 const mock=context.length?(context[0].eps>1?'BUY':'NO BUY'):'WITHHOLD';
 return {version:v,check,context,mock,naive:v.eps>1?'BUY':'NO BUY'};
}
export async function contentHash(text){const bytes=await globalThis.crypto.subtle.digest('SHA-256',new TextEncoder().encode(text));return Array.from(new Uint8Array(bytes),x=>x.toString(16).padStart(2,'0')).join('');}
