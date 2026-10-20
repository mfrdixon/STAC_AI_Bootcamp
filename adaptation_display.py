"""Portable, escaped HTML presentation of recorded adaptation evidence."""
import html,json
from IPython.display import HTML,display

def esc(x):return html.escape(str(x))
def value(x):
 if isinstance(x,(dict,list)):return json.dumps(x,ensure_ascii=False,indent=2)
 if isinstance(x,float):return f'{x:.6g}'
 return str(x)
def table(headers,rows):
 return '<table style="width:100%;border-collapse:collapse;text-align:left"><thead><tr>'+''.join('<th style="padding:9px;background:#e7f3f3;border-bottom:2px solid #087b83">'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td style="padding:9px;border-bottom:1px solid #dce5e8;vertical-align:top;white-space:pre-wrap;overflow-wrap:anywhere">'+esc(value(c))+'</td>' for c in r)+'</tr>' for r in rows)+'</tbody></table>'
def details(title,content):return '<details style="margin:12px 0"><summary style="cursor:pointer;color:#087b83;font-weight:600">'+esc(title)+'</summary><pre style="white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f7f8;padding:14px;max-height:440px;overflow:auto;font-size:13px">'+esc(value(content))+'</pre></details>'
def panel(title,body,note=''):
 display(HTML('<section style="font-family:system-ui,sans-serif;font-size:15px;line-height:1.5;color:#19333f;max-width:1100px;padding:18px;border:1px solid #dce5e8;border-radius:12px;margin:10px 0;background:#fff"><h3 style="margin:0 0 12px;color:#087b83;font-size:21px">'+esc(title)+'</h3>'+body+('<p style="background:#fff6df;padding:12px;border-left:4px solid #b88013">'+esc(note)+'</p>' if note else '')+'</section>'))
def bullets(items):return '<ul>'+''.join('<li style="margin:7px 0">'+esc(x)+'</li>' for x in items)+'</ul>'
def status(message,execution):panel('Recorded experiment · ready', '<p>'+esc(message)+'</p><p>'+esc(execution)+'</p>','Selecting a case replays measured outputs. It does not generate a new answer.')
def model(data):
 m=data['metadata'];a=m['adapter_config'];e=data['training_examples'][0]
 panel('What was trained?',table(['Property','Recorded configuration'],[['Base model',m['model']],['Frozen base parameters',f"{m['base_parameters']:,}"],['Trainable adapter parameters',f"{m['adapter_parameters']:,} ({100*m['adapter_parameters']/m['base_parameters']:.4f}% of base)"],['Combined parameters',f"{m['combined_parameters']:,}"],['LoRA targets',', '.join(a['target_modules'])],['Rank / alpha',f"{a['r']} / {a['lora_alpha']}"],['Selected checkpoint',m['selected_checkpoint']]])+details('Inspect one actual supervised training example',e),'The adapter adds trainable matrices inside selected attention projections; the base model is not made smaller.')
def losses(data):
 m=data['metadata'];rows=data['losses'];panel('Training fit and validation',table(['Epoch','Mean logged training loss','Validation loss'],[[r['epoch'],f"{r['mean_logged_training_loss']:.4f}",f"{r['validation_loss']:.4f}"] for r in rows])+bullets([f"Recorded runtime: {m['training_runtime_seconds']/60:.1f} minutes.",f"Non-finite gradient diagnostics logged: {m['nonfinite_grad_norm_entries']}.",'Epoch 2 was selected using validation loss. Held-out forecasting is evaluated separately.']),'Lower training loss is not evidence of improved investment performance.')
def case_view(c):
 rows=[]
 for arm in ['base','adapted']:
  a=c[arm];rows.append([arm.title(),a['label'],a['code'],' | '.join(f'{k}: {v:.4f}' for k,v in a['candidate_logits'].items())])
 panel(f"Matched case · {c['ticker']} · {c['as_of']}",'<p>'+esc(c['question'])+'</p>'+table(['Model','Predicted class','Actual token','Candidate logits (not calibrated probabilities)'],rows)+details('Exact shared model prompt · expand to inspect all evidence',c['prompt'])+details('Prompt integrity',{'case_id':c['id'],'shared_prompt_sha256':c['prompt_sha256']}),'Both models received identical evidence. These are actual class-token outputs, not generated narratives.')
def reveal(c):
 rows=[]
 for arm in ['base','adapted']:
  a=c[arm];rows.append([arm.title(),'Correct' if a['label']==c['expected'] else 'Incorrect',a['checks']['decision'],str(a['eligible'])])
 panel('Reveal the outcome and inspect the checks', '<p><strong>Observed evaluation label: '+esc(c['expected'])+'</strong></p>'+table(['Model','Forecast outcome','Evidence decision','Eligible'],rows)+details('Future realized outcome · evaluation only',c['outcome_only']),'Future outcomes determine evaluation labels. They were not inputs to the recorded forecast.')
 for arm in ['base','adapted']:
  check=c[arm]['checks'];rows=[]
  for x in check['transformation_checks']:rows.append([x.get('id'),x.get('status'),x.get('reported','—'),x.get('recomputed','—'),x.get('scope','')])
  panel(arm.title()+' · arithmetic and evidence checks',table(['Check','Status','Reported','Recomputed','Scope'],rows)+details('Evidence flags · complete recorded list',check['evidence_checks']),check['limitation'])
 rows=[[d.get('source',d.get('id')),d.get('type','—'),d.get('published_at','Unknown'),d.get('retrieved_at','Unknown'),d.get('provenance_status',d.get('availability_basis','Unknown'))] for d in c['documents']]
 panel('Source timing and provenance',table(['Source','Type','Published','Retrieved','Status / limitation'],rows)+''.join(details('Read source · '+str(d.get('id')),d) for d in c['documents'])+details('Known evidence gaps and owners',c['source_audit']),'Publication dates are proxies. Later downloads do not prove what exact text was available at the historical decision time.')
def results(data,metrics):
 rows=[]
 for arm in ['base','adapted']:
  r=metrics(data,arm);rows.append([arm.title(),f"{round(r['accuracy']*r['n'])}/{r['n']}",f"{r['accuracy']:.1%}",f"{r['balanced_accuracy']:.1%}"])
 labels=['NEGATIVE_SIGNAL','NEUTRAL_SIGNAL','POSITIVE_SIGNAL'];counts=[[x,sum(c['expected']==x for c in data['cases'])] for x in labels]
 panel('All held-out cases · no winning subset',table(['Model','Correct','Accuracy','Balanced accuracy'],rows)+table(['Observed class','Count'],counts),'The adapter corrects 3 base errors but introduces 6. A single neutral outcome strongly affects balanced accuracy; these results do not establish better forecasting or alpha.')
 for arm in ['base','adapted']:
  cm=metrics(data,arm)['confusion'];panel(arm.title()+' · confusion matrix',table(['Observed / predicted',*labels],[[a,*[cm[a][b] for b in labels]] for a in labels]))
def protocol(data):
 p=data['metadata']['protocol'];keys=['train_outcomes_before','validation_as_of_from','validation_outcomes_before','test_decision_dates','universe','target','objective']
 panel('What this experiment can support',table(['Protocol element','Recorded value'],[[k.replace('_',' ').capitalize(),p[k]] for k in keys])+bullets(data['metadata']['limitations'])+details('Full frozen protocol',p),'The split holds out later dates, not new issuers. Retrospective source snapshots limit historical validity.')
def fresh(result=None):
 if result is None:panel('Fresh inference · disabled','<p>The saved results above remain unchanged. Enable RUN_FRESH only after provisioning the exact local model and adapter.</p>')
 else:panel('Fresh local inference',details('Actual new outputs and environment',result),'Fresh scores are separate from the frozen recorded evaluation.')
