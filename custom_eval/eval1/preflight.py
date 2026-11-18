from __future__ import annotations
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
from eval1.audit_registry import audit
from eval1.build_label_blind_manifests import EXPECTED,build,sha
ROOT=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser();p.add_argument('--skip-tests',action='store_true');a=p.parse_args();checks=[]
 for ds,(digest,n) in EXPECTED.items():
  source=ROOT/'outputs/manifests'/f'{ds}.jsonl';blind=ROOT/'outputs/label_blind'/f'{ds}.jsonl';checks.append(build(source,blind,ds))
 registry=audit(ROOT/'model_registry.yaml',ROOT/'outputs/registry_audit.json');failed=[m for m in registry['models'] if not m['eligible']]
 if failed:raise RuntimeError({'failed_gates':[(m['id'],m['reasons']) for m in failed]})
 for m in registry['models']:
  cp=Path(m['checkpoint']);payload=__import__('torch').load(cp,map_location='cpu',weights_only=False)
  if not isinstance(payload.get('model'),dict) or not payload['model']:raise RuntimeError(f'invalid checkpoint model state: {m["id"]}')
  checks.append({'model':m['id'],'state_keys':len(payload['model']),'checkpoint_sha256':m['checkpoint_sha256']})
 if not a.skip_tests:subprocess.run([sys.executable,'-m','pytest','-q',str(ROOT/'tests')],cwd=ROOT.parent,check=True)
 result={'status':'PASS','models':len(registry['models']),'all_eligible':True,'checks':checks};out=ROOT/'outputs/preflight/_SUCCESS.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
