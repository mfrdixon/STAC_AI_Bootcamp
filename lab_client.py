"""Shared browser/Node engine client. No third-party package required."""
import json, os, shutil, subprocess, urllib.request, urllib.error
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def run(config=None, *, live=False, base_url=None, key=None, model=None):
    payload=dict(config or {})
    if not live:
        node=os.getenv('NODE_BINARY') or shutil.which('node')
        if not node:
            raise RuntimeError('Install Node.js 20+ or set NODE_BINARY to its absolute path.')
        result=subprocess.run([node,str(ROOT/'scripts/run.mjs')],input=json.dumps(payload),text=True,capture_output=True,check=True)
        return json.loads(result.stdout)
    payload.update(live=True)
    if model: payload['model']=model
    url=(base_url or os.getenv('LAB_URL','http://127.0.0.1:8787/decision-lab')).rstrip('/')+'/api/run'
    request=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+(key or os.getenv('WORKSHOP_KEY',''))})
    try:
        with urllib.request.urlopen(request,timeout=60) as response:return json.load(response)
    except urllib.error.HTTPError as e:
        raise RuntimeError(e.read().decode()) from None

def compact(result):
    return {'mode':result['config']['mode'],'execution':result['execution'],'proposed':result['answer']['proposed_action'],'permitted':result['gate'],'checks':{x['id']:x['status'] for x in result['checks']}}

def save(result,path='experiment.json'):
    Path(path).write_text(json.dumps(result,indent=2))
