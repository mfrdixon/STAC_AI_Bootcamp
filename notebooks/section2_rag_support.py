"""Offline retrieval and readable display helpers for the Section 2 workbook."""
import math
import re
from collections import Counter
from html import escape

DOCUMENTS = [
 dict(id='FILING', title='Harbor Metals: quarterly filing', issuer='Harbor Metals', available='2026-10-20 08:00', role='filing', text='Harbor Metals reports debt of $75 million, cash of $15 million and EBITDA of $20 million. Figures are for the quarter ended 30 September 2026.'),
 dict(id='COVENANT', title='Harbor Metals: signed credit agreement', issuer='Harbor Metals', available='2026-10-19 16:00', role='covenant', text='The signed covenant limits net leverage to 3.5 times EBITDA. Net leverage is (debt minus eligible cash) divided by EBITDA. Eligible cash means total cash less restricted cash. The restricted cash amount is specified in Annex A.'),
 dict(id='ANNEX', title='Harbor Metals: Annex A', issuer='Harbor Metals', available='2026-10-20 08:30', role='annex', text='Annex A to the signed credit agreement: restricted cash is $5 million for the quarter ended 30 September 2026. This amount is unavailable for debt repayment and must be excluded from eligible cash.'),
 dict(id='AMENDMENT', title='Harbor Metals: covenant amendment', issuer='Harbor Metals', available='2026-10-20 09:07', role='amendment', text='The covenant amendment permits all cash, including restricted cash, to be deducted when calculating net leverage. The ceiling remains 3.5 times EBITDA. Released at 09:07 on 20 October 2026.'),
 dict(id='OTHER', title='Bay Copper: restricted cash annex', issuer='Bay Copper', available='2026-10-20 08:00', role='annex', text='Restricted cash in the Bay Copper covenant annex is zero. All cash is eligible for the net leverage covenant calculation.'),
 dict(id='PRESENTATION', title='Harbor Metals: management presentation', issuer='Harbor Metals', available='2026-10-20 08:10', role='presentation', text='Management describes liquidity and covenant headroom as comfortable. This presentation does not provide a downside EBITDA scenario or replace the signed credit agreement.'),
]
STOP = set('the a an of is are in to for and by from what which does do its it be'.split())
def tokens(text):
    return [w for w in re.findall(r'[a-z0-9]+', text.lower()) if w not in STOP]

def search_documents(query, documents, top_k=4):
    """Real lexical TF-IDF cosine retrieval; not an embedding model."""
    counts = [Counter(tokens(d['text'])) for d in documents]
    terms = set().union(*(set(c) for c in counts)) if counts else set()
    idf = {w: math.log((1+len(counts))/(1+sum(w in c for c in counts)))+1 for w in terms}
    def vector(c): return {w:n*idf[w] for w,n in c.items() if w in idf}
    q = vector(Counter(tokens(query))); qnorm = math.sqrt(sum(v*v for v in q.values()))
    ranked=[]
    for d,c in zip(documents,counts):
        v=vector(c); norm=math.sqrt(sum(x*x for x in v.values()))
        score=sum(x*v.get(w,0) for w,x in q.items())/(qnorm*norm) if norm*qnorm else 0
        if score>0: ranked.append(dict(d, score=score, matches=', '.join(sorted(set(q)&set(v)))))
    return sorted(ranked,key=lambda x:(-x['score'],x['id']))[:top_k]

def table(rows):
    if not rows:
        print('No rows.'); return
    from IPython.display import HTML, display
    keys=list(rows[0]); html='<table style="border-collapse:collapse;text-align:left">'
    html+='<tr>'+''.join('<th style="padding:8px;border-bottom:2px solid #345">'+escape(str(k))+'</th>' for k in keys)+'</tr>'
    for r in rows:
        html+='<tr>'+''.join('<td style="padding:8px;border-bottom:1px solid #ddd;vertical-align:top">'+escape(str(r.get(k,'')))+'</td>' for k in keys)+'</tr>'
    display(HTML(html+'</table>'))

def show_results(results):
    table([{'Rank':i+1,'ID':d['id'],'Lexical score':round(d['score'],3),'Matched words':d['matches'],'Returned passage':d['text']} for i,d in enumerate(results)])

def filter_documents(documents, issuer, cutoff):
    allowed=[]; audit=[]
    for d in documents:
        reasons=[]
        if d['issuer']!=issuer: reasons.append('wrong issuer')
        if d['available']>cutoff: reasons.append('not yet available')
        if not reasons: allowed.append(d)
        audit.append({'ID':d['id'],'Available (New York)':d['available'],'Decision':'INCLUDE' if not reasons else 'EXCLUDE','Reason':'; '.join(reasons) or 'issuer and availability pass'})
    return allowed,audit

def assemble_context(question, passages):
    evidence='\n\n'.join('['+d['id']+'] '+d['text'] for d in passages)
    return 'TASK: Answer using the supplied evidence. Cite source IDs. Identify missing evidence.\n\nQUESTION: '+question+'\n\nEVIDENCE:\n'+evidence

def check_requirements(passages):
    # Declared requirements are instructional policy, not inferred truth.
    ids={d['id'] for d in passages}
    return [dict(Claim=claim, Required=', '.join(required), Missing=', '.join(sorted(set(required)-ids)) or 'none', Status='READY TO CHECK' if set(required)<=ids else 'BLOCKED') for claim,required in [('Current covenant ratio',['FILING','COVENANT','ANNEX']),('Comfortable headroom',['FILING','COVENANT','ANNEX','DOWNSIDE'])]]

def show_graph(passages):
    from IPython.display import SVG,display
    ids={d['id'] for d in passages}
    nodes=[('FILING',20,20),('COVENANT',20,100),('ANNEX',20,180),('CALCULATOR',260,100),('RATIO CLAIM',510,100)]
    ready={'FILING','COVENANT','ANNEX'}<=ids
    svg='<svg xmlns="http://www.w3.org/2000/svg" width="730" height="260" viewBox="0 0 730 260"><defs><marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#555"/></marker></defs>'
    for sy,label in [(45,'financial inputs'),(125,'formula / definition'),(205,'cash restriction')]:
        svg+=f'<path d="M190,{sy} L260,125" stroke="#555" fill="none" marker-end="url(#arr)"/>'
    svg+='<path d="M430,125 L510,125" stroke="#555" marker-end="url(#arr)"/><text x="445" y="110" font-size="12">supports</text>'
    for key,x,y in nodes:
        ok=(key in ids) if x==20 else ready
        color='#e0f2e9' if ok else '#ffe0d6'
        svg+=f'<rect x="{x}" y="{y}" width="170" height="50" rx="8" fill="{color}" stroke="#456"/><text x="{x+85}" y="{y+22}" text-anchor="middle" font-family="sans-serif" font-size="13">{key}</text><text x="{x+85}" y="{y+40}" text-anchor="middle" font-family="sans-serif" font-size="11">{"available" if x==20 and ok else "missing" if x==20 else "checkable" if ok else "blocked"}</text>'
    svg+='<text x="20" y="248" font-family="sans-serif" font-size="12">Source documents → defined calculation → claim. Green: available/checkable; orange: unresolved.</text></svg>'
    display(SVG(svg))

def evaluate_queries(queries, documents, k=3):
    rows=[]
    for query,relevant in queries:
        found={d['id'] for d in search_documents(query,documents,k)}
        hits=found&set(relevant)
        rows.append({'Query':query,'Retrieved':', '.join(sorted(found)),'Required relevant IDs':', '.join(sorted(relevant)),f'Precision@{k}':f'{len(hits)/k:.2f}',f'Recall@{k}':f'{len(hits)/len(relevant):.2f}','Missed':', '.join(sorted(set(relevant)-found)) or 'none'})
    return rows
