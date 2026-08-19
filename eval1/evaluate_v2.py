from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from eval1.phonology_v2 import *
from eval1.retrieval import evaluate_retrieval,human_ceiling
from eval1.bootstrap import class_stratified_mean_ci
def load(p,fmt):
 rows=[json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]; out={}
 for r in rows:
  aid=str(r.get('eval_audio_id') or r.get('audio_id')); rank=int(r.get('beam_rank',0))
  if aid in out and rank>=out[aid]['rank']: continue
  text=str(r.get('prediction',r.get('text',''))).strip(); out[aid]={'raw_text':text,'rank':rank}
 if fmt=='english_text': seq=english_to_phones_batch([x['raw_text'] for x in out.values()])
 elif fmt=='rwcp_repo_phone': seq=[rwcp_repo_to_phones(x['raw_text']) for x in out.values()]
 elif fmt=='ipa_tokens': seq=[temporal_view(x['raw_text'].split()) for x in out.values()]
 else: raise ValueError(fmt)
 for x,s in zip(out.values(),seq): x['temporal']=s;x['segmental']=segmental_view(s)
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--predictions',required=True);p.add_argument('--format',required=True,choices=['english_text','rwcp_repo_phone','ipa_tokens']);p.add_argument('--output',required=True);a=p.parse_args()
 rows=[json.loads(x) for x in Path(a.manifest).read_text().splitlines() if x.strip() and json.loads(x)['split']=='test']; pred=load(a.predictions,a.format); ids={r['eval_audio_id'] for r in rows}; missing=ids-set(pred)
 if missing: raise RuntimeError(f'missing predictions: {len(missing)}')
 paired=[]
 for r in rows:
  x=pred[r['eval_audio_id']]; tl=[z['phone_tokens_temporal'] for z in r['labels']]; sl=[z['phone_tokens_segmental'] for z in r['labels']]; td=[phone_edit_distance(x['temporal'],z) for z in tl]; sd=[phone_edit_distance(x['segmental'],z) for z in sl]; dd=[duration_distance(x['temporal'],z) for z in tl]
  paired.append({'eval_audio_id':r['eval_audio_id'],'class_key':r['class_key'],'prediction':x['raw_text'],'paired_min_ped_temporal':min(td),'paired_mean_ped_temporal':float(np.mean(td)),'paired_min_ped_segmental':min(sd),'paired_mean_ped_segmental':float(np.mean(sd)),'paired_min_duration_distance':min(dd),'valid_output':bool(x['temporal'])})
 views={v:{k:x[v] for k,x in pred.items()} for v in ('temporal','segmental')}; metrics={'schema':'eval1_phonology_v2','dataset':rows[0]['eval_dataset'],'n_evaluated':len(rows),'valid_output_rate':float(np.mean([x['valid_output'] for x in paired])),'paired':{m:class_stratified_mean_ci(p,m,replicates=10000,seed=1337) for m in ('paired_min_ped_temporal','paired_mean_ped_temporal','paired_min_ped_segmental','paired_mean_ped_segmental','paired_min_duration_distance')},'retrieval':{v:evaluate_retrieval(views[v],rows,view=v,replicates=100,seed=1337,labels_per_candidate=5) for v in views},'human_ceiling':{v:human_ceiling(rows,view=v,replicates=100,seed=1337,labels_per_candidate=5) for v in views}}
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);(out/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n');(out/'per_query.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in paired));(out/'_SUCCESS.json').write_text(json.dumps({'status':'COMPLETE','n':len(rows),'schema':'eval1_phonology_v2'})+'\n');print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
