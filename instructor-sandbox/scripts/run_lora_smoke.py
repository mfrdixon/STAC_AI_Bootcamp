"""Prepare isolated demo configuration; training runs only with --run."""
import argparse,datetime,hashlib,json,os,re,subprocess,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--direct',action='store_true',help='Run inside the isolated GPU container without Docker');p.add_argument('--steps',type=int,default=10);a=p.parse_args();assert 1<=a.steps<=100
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6];folder=ROOT/'artifacts/instructor_smoke'/stamp;folder.mkdir(parents=True,exist_ok=False)
 source=ROOT/'config/alpha_v2_lora.yaml';text=source.read_text();changes={'output_dir':f'/app/artifacts/instructor_smoke/{stamp}/adapter','save_strategy':'"no"','eval_strategy':'"no"','load_best_model_at_end':'false','plot_loss':'false'}
 for k,v in changes.items():text=re.sub(r'^'+k+r':.*$',f'{k}: {v}',text,flags=re.M)
 text+='\nmax_steps: '+str(a.steps)+'\nmax_samples: 64\n';cfg=folder/'smoke.yaml';cfg.write_text(text)
 manifest={'source_config_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'smoke_config_sha256':hashlib.sha256(cfg.read_bytes()).hexdigest(),'steps':a.steps,'status':'prepared, not trained','training_data_sha256':hashlib.sha256((ROOT/'data/alpha_v2/alpha_v2_train.json').read_bytes()).hexdigest()};(folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
 env=os.environ.copy();env.update(LAB_UID=str(os.getuid()),LAB_GID=str(os.getgid()))
 command=['docker','compose','--profile','train','run','--rm','train','llamafactory-cli','train',f'/app/artifacts/instructor_smoke/{stamp}/smoke.yaml'];print('Prepared:',cfg);print('Command:', ' '.join(command))
 if not a.run:print('No training launched. Re-run with --run only on approved, already provisioned GPU infrastructure.');return
 if a.direct:
  subprocess.run(['python',str(ROOT/'scripts/preflight_alpha_v2.py')],cwd=ROOT,check=True)
  with open(folder/'training.log','w') as log:subprocess.run(['llamafactory-cli','train',str(cfg)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  manifest['status']='completed smoke training; no held-out improvement established';(folder/'manifest.json').write_text(json.dumps(manifest,indent=2));print(manifest['status']);return
 subprocess.run(['docker','compose','--profile','train','--profile','evaluate','stop','train','evaluate'],cwd=ROOT,env=env,check=True)
 subprocess.run(['docker','compose','--profile','train','run','--rm','train','python','scripts/preflight_alpha_v2.py'],cwd=ROOT,env=env,check=True)
 with open(folder/'training.log','w') as log:subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 manifest['status']='completed smoke training; no held-out improvement established';(folder/'manifest.json').write_text(json.dumps(manifest,indent=2));print(manifest['status'])
if __name__=='__main__':main()
