"""Fictional Harbor Metals tools; local teaching permissions, no external data."""
import math
DOCUMENTS = {
    "FILING": "Harbor Metals: debt 75; cash 15; EBITDA 20 (USD millions).",
    "COVENANT": "Net leverage ceiling: 3.5 times. Exclude restricted cash in Annex A.",
    "ANNEX_A": "Restricted cash: 5 million. Eligible cash = reported cash minus restricted cash."
}
GRANTS = {
    "research": {"search_documents", "calculate_leverage"},
    "reviewer": {"search_documents", "calculate_leverage", "release_brief"}
}
AUDIT = []
RELEASES = []
def search_documents(query):
    """Tiny keyword retriever: returns source IDs and text; not vector search."""
    words=set(query.lower().split())
    selected = ["ANNEX_A"] if "annex" in words or "restricted" in words else ["FILING", "COVENANT"]
    return [{"source": k, "text": DOCUMENTS[k]} for k in selected]
def calculate_leverage(debt, eligible_cash, ebitda):
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in (debt,eligible_cash,ebitda)):
        raise ValueError("Finite numeric inputs required")
    if debt < 0 or eligible_cash < 0 or ebitda <= 0:
        raise ValueError("Nonnegative debt/cash and positive EBITDA required")
    return {"formula": "(debt - eligible_cash) / ebitda", "leverage": (debt-eligible_cash)/ebitda}
def release_brief(brief):
    RELEASES.append(brief)
    return {"status": "SIMULATED RELEASE", "brief": brief}
def dispatch(principal, name, arguments):
    # Trusted application/server supplies principal, never the model's arguments.
    permitted = name in GRANTS.get(principal,set())
    AUDIT.append({"principal":principal,"tool":name,"allowed":permitted})
    if not permitted:
        raise PermissionError(f"{principal} cannot call {name}")
    return {"search_documents":search_documents,"calculate_leverage":calculate_leverage,
            "release_brief":release_brief}[name](**arguments)
