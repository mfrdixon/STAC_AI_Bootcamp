import json, math, re, copy, uuid
from pprint import pprint

QUESTION = "Does Harbor Metals have comfortable covenant headroom? Establish current covenant leverage, cite sources, and identify what remains unresolved."
SYSTEM = """You assess the fictional Harbor Metals case. Use only supplied documents and tool results.
For an agent run, retrieve the initial packet first. Follow any annex reference before choosing eligible cash.
Request calculate_leverage for arithmetic. If evidence is unavailable, state the gap; do not invent an amount.
A current ratio alone does not establish comfortable headroom or authorization to act.
Give a concise answer with document IDs. Tools cannot trade or release a brief."""

def make_store(restricted=True, annex_available=True, debt=75, cash=15, ebitda=20):
    docs = {
        "FILING": {"id": "FILING", "text": f"Harbor Metals: debt {debt} million; cash {cash} million; EBITDA {ebitda} million."},
        "COVENANT": {"id": "COVENANT", "text": "Ceiling 3.5 times. " + (
            "Exclude restricted cash specified in Annex A." if restricted else "All reported cash is eligible.")}}
    if restricted and annex_available:
        docs["ANNEX_A"] = {"id": "ANNEX_A", "text": "Restricted cash: 5 million."}
    return docs

def retrieve(store, document_id):
    # Deliberately small document-ID retriever. A production retriever can replace it.
    if document_id == "INITIAL":
        return {"documents": [copy.deepcopy(store[k]) for k in ("FILING", "COVENANT")]}
    return {"documents": [copy.deepcopy(store[document_id])]} if document_id in store else {"documents": [], "error": "Document unavailable", "requested": document_id}

def calculate_leverage(debt, eligible_cash, ebitda):
    values = [debt, eligible_cash, ebitda]
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values):
        raise ValueError("Finite numeric inputs required")
    if debt < 0 or eligible_cash < 0 or ebitda <= 0:
        raise ValueError("Debt/cash must be nonnegative and EBITDA positive")
    return {"debt": debt, "eligible_cash": eligible_cash, "ebitda": ebitda,
            "formula": "(debt - eligible_cash) / ebitda", "leverage": (debt-eligible_cash)/ebitda}

TOOLS = [
    {"type": "function", "function": {"name": "retrieve", "description": "Retrieve INITIAL for filing and covenant, or ANNEX_A for the referenced annex.",
        "parameters": {"type": "object", "properties": {"document_id": {"type": "string"}}, "required": ["document_id"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "calculate_leverage", "description": "Calculate leverage using evidence-supported eligible cash, in millions.",
        "parameters": {"type": "object", "properties": {k: {"type": "number"} for k in ["debt", "eligible_cash", "ebitda"]}, "required": ["debt", "eligible_cash", "ebitda"], "additionalProperties": False}}}
]

def number(text, field):
    return float(re.search(re.escape(field) + r"[: ]+([0-9.]+)", text, re.I).group(1))

def received_payloads(messages):
    payloads=[]
    for m in messages:
        if m["role"] in ("tool", "user"):
            try:
                value=json.loads(m.get("content") or "")
                if isinstance(value, dict): payloads.append(value)
            except (ValueError, TypeError): pass
    return payloads

class SyntheticLLM:
    def __init__(self, skip_annex=False): self.skip_annex=skip_annex
    def complete(self, messages, tools=None):
        payloads=received_payloads(messages)
        docs={d["id"]:d for p in payloads for d in p.get("documents", [])}
        calculations=[p for p in payloads if "leverage" in p]
        def call(name, args):
            return {"role":"assistant", "content":None, "tool_calls":[{
                "id":"call_"+uuid.uuid4().hex[:10], "type":"function",
                "function":{"name":name,"arguments":json.dumps(args)}}]}
        if tools and "FILING" not in docs:
            return call("retrieve", {"document_id":"INITIAL"})
        needs_annex="Annex A" in docs.get("COVENANT",{}).get("text","")
        tried_annex=any(p.get("requested")=="ANNEX_A" for p in payloads)
        if tools and needs_annex and "ANNEX_A" not in docs and not self.skip_annex:
            if tried_annex:
                return {"role":"assistant","content":"HOLD calculation claim: Annex A is unavailable; eligible cash is unresolved [COVENANT]."}
            return call("retrieve", {"document_id":"ANNEX_A"})
        if tools and not calculations:
            f=docs["FILING"]["text"]
            cash=number(f,"cash") - (number(docs["ANNEX_A"]["text"],"Restricted cash") if "ANNEX_A" in docs else 0)
            return call("calculate_leverage", {"debt":number(f,"debt"),"eligible_cash":cash,"ebitda":number(f,"EBITDA")})
        if calculations:
            result=calculations[-1]
            text=f"Calculator reports {result['leverage']:.2f}× using cash {result['eligible_cash']}m. Sources: {', '.join(docs)}. Comfort and release still need scenario review and approval."
        else:
            text="Retrieved the filing and covenant, but this workflow supplies no executed calculation. Current headroom needs verification; any referenced annex must be resolved."
        return {"role":"assistant","content":text}

import urllib.request, urllib.error
from getpass import getpass

class HostedLLM:
    ENDPOINTS={"openai":"https://api.openai.com/v1/chat/completions",
               "together":"https://api.together.ai/v1/chat/completions"}
    def __init__(self, provider, model, api_key):
        self.endpoint=self.ENDPOINTS[provider]; self.provider=provider
        self.model=model; self._key=api_key
    def complete(self,messages,tools=None):
        body={"model":self.model,"messages":messages}
        body["max_completion_tokens" if self.provider=="openai" else "max_tokens"]=2048
        if tools: body.update(tools=tools,tool_choice="auto")
        request=urllib.request.Request(self.endpoint,data=json.dumps(body).encode(),headers={
            "Authorization":"Bearer "+self._key,"Content-Type":"application/json","User-Agent":"BeliefLens-Workshop/1.0"})
        try:
            with urllib.request.urlopen(request,timeout=90) as response: result=json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Provider HTTP {exc.code}. Check key, model access, tool support and quota. No synthetic fallback.") from None
        except urllib.error.URLError:
            raise RuntimeError("Provider network request failed. No synthetic fallback.") from None
        choice=result["choices"][0]
        if choice.get("finish_reason")=="length": raise RuntimeError("Model hit its token budget; increase it explicitly or choose another model.")
        message=choice["message"]
        # Keep the provider tool-call IDs for subsequent tool messages.
        return {k:message[k] for k in ("role","content","tool_calls") if k in message}

# Presentation and provider adapters; workflow orchestration stays in the notebook.
from types import SimpleNamespace
from html import escape

DOCUMENTS=make_store()
INITIAL_DOCUMENTS=["FILING", "COVENANT"]
QUESTION="Does Harbor Metals have comfortable covenant headroom? Establish current covenant leverage, cite sources, and identify what remains unresolved."

class DocumentSearch:
    def __init__(self, documents): self.documents=copy.deepcopy(documents)
    def search(self, query, document_ids):
        # Transparent keyword retrieval over this small synthetic collection.
        terms=set(re.findall(r"[a-z]+",query.lower()))
        found=[]
        for id in document_ids:
            doc=self.documents.get(id)
            if doc and terms.intersection(re.findall(r"[a-z]+",(id+' '+doc['text']).lower())):
                found.append(copy.deepcopy(doc))
        return {"query":query,"searched":list(document_ids),"documents":found}

SEARCH_TOOL={"type":"function","function":{"name":"retrieve","description":"Keyword search, not semantic search: query words must occur in the passage. Start with FILING and COVENANT. If Annex A is referenced, search ANNEX_A using query Annex A restricted cash. Empty results can mean no keyword match; refine the query before concluding evidence is unavailable.","parameters":{"type":"object","properties":{"query":{"type":"string"},"document_ids":{"type":"array","items":{"type":"string","enum":["FILING","COVENANT","ANNEX_A"]}}},"required":["query","document_ids"],"additionalProperties":False}}}

class ModelSession:
    def __init__(self, provider="synthetic", model="", skip_annex=False):
        if provider not in ("synthetic", "openai", "together") or (provider!="synthetic" and not model.strip()):
            raise ValueError("Choose synthetic, or specify a provider and tool-capable model ID")
        self.synthetic=provider=="synthetic"
        self.client=SyntheticLLM(skip_annex) if self.synthetic else HostedLLM(provider,model,getpass(f"{provider} API key (hidden): "))
        self.messages=[{"role":"system","content":SYSTEM},{"role":"user","content":QUESTION}]
    def add_context(self, documents, calculation=None):
        self.messages.append({"role":"user","content":json.dumps(documents)})
        if calculation is not None: self.messages.append({"role":"user","content":json.dumps(calculation)})
    def next(self, allow_tools=False):
        response=self.client.complete(self.messages,tools=[SEARCH_TOOL,TOOLS[1]] if allow_tools else None)
        calls=[]
        for raw in response.get("tool_calls",[]):
            args=json.loads(raw['function']['arguments'])
            if self.synthetic and raw['function']['name']=='retrieve':
                id=args['document_id']
                args={"query":"Harbor Metals debt cash EBITDA covenant" if id=='INITIAL' else "Annex A restricted cash", "document_ids":INITIAL_DOCUMENTS if id=='INITIAL' else [id]}
                raw['function']['arguments']=json.dumps(args)
            calls.append(SimpleNamespace(id=raw['id'],name=raw['function']['name'],arguments=args))
        self.messages.append(response)
        return SimpleNamespace(text=response.get('content') or '', calls=calls)
    def observe(self,call,result):
        if call.name=='retrieve' and not result['documents'] and 'ANNEX_A' in call.arguments.get('document_ids',[]):
            result={**result,'requested':'ANNEX_A','error':'No matching passage returned; refine query or verify document availability'}
        self.messages.append({'role':'tool','tool_call_id':call.id,'content':json.dumps(result)})

def financial_inputs(packet):
    filing=next(d['text'] for d in packet['documents'] if d['id']=='FILING')
    return dict(debt=number(filing,'debt'),eligible_cash=number(filing,'cash'),ebitda=number(filing,'EBITDA'))

def panel(title,value):
    if hasattr(value,'calls'):
        text=value.text or '\n'.join(c.name+'('+', '.join(f'{k}={v}' for k,v in c.arguments.items())+')' for c in value.calls)
    elif isinstance(value,dict) and 'documents' in value:
        text='Search query: '+value.get('query','')+'\nDocuments searched: '+', '.join(value.get('searched',[]))+'\n\n'+('\n\n'.join(d['id']+': '+d['text'] for d in value['documents']) or 'No document returned.')
    elif isinstance(value,dict): text='\n'.join(f'{k}: {v}' for k,v in value.items())
    else:text=str(value)
    try:
        from IPython.display import display,HTML
        display(HTML('<div style="border:1px solid #bdd3d5;border-radius:8px;padding:12px;margin:8px 0"><b>'+escape(title)+'</b><pre style="white-space:pre-wrap;margin-bottom:0">'+escape(text)+'</pre></div>'))
    except ImportError: print(title+'\n'+text)

def restricted_cash(packet):
    """Read the amount from the controlled annex format."""
    return number(packet["documents"][0]["text"], "Restricted cash")
