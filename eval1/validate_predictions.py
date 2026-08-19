from __future__ import annotations
import argparse,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def validate(manifest,predictions,expected_audio=None):
 m=[json.loads(x) for x in Path(manifest).read_text().splitlines() if x.strip()];p=[json.loads(x) for x in Path(predictions).read_text().splitlines() if x.strip()];expected={r['eval_audio_id'] for r in m};by=defaultdict(list)
 for r in p:by[r['eval_audio_id']].append(int(r['beam_rank']))
 if expected_audio is not None and len(expected)!=expected_audio:raise RuntimeError(('manifest coverage',len(expected),expected_audio))
 if set(by)!=expected:raise RuntimeError({'missing':len(expected-set(by)),'extra':len(set(by)-expected)})
 bad={k:v for k,v in by.items() if sorted(v)!=[0,1,2]}
 if bad:raise RuntimeError(f'invalid beam ranks for {len(bad)} IDs')
 return {'audio_count':len(expected),'prediction_rows':len(p),'predictions_sha256':sha(predictions),'manifest_sha256':sha(manifest)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--predictions',required=True);p.add_argument('--expected-audio',type=int);a=p.parse_args();print(json.dumps(validate(a.manifest,a.predictions,a.expected_audio),indent=2))
if __name__=='__main__':main()
