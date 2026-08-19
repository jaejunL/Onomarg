from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
import numpy as np,yaml
from eval1.bootstrap import class_stratified_mean_ci
from eval1.phonology import SCHEMA,duration_distance,english_to_phones_batch,phone_edit_distance,romanized_korean_to_phones_batch,rwcp_repo_to_phones,segmental_view,temporal_view
from eval1.retrieval import evaluate_retrieval,human_ceiling
ROOT=Path(__file__).resolve().parent
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read_jsonl(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def load_predictions(path,prediction_format,lexicon_path=None):
 selected={}
 for row in read_jsonl(path):
  rank=int(row.get('beam_rank',0));aid=str(row.get('eval_audio_id') or row.get('audio_id') or '')
  if not aid:raise ValueError('prediction eval_audio_id missing')
  if aid in selected and rank>selected[aid]['beam_rank']:continue
  if aid in selected and rank==selected[aid]['beam_rank']:raise ValueError(f'duplicate prediction rank: {aid}:{rank}')
  text=str(row.get('prediction',row.get('raw_text',row.get('text','')))).strip();selected[aid]={'raw_text':text,'beam_rank':rank,'source':row}
 ids=list(selected);texts=[selected[k]['raw_text'] for k in ids];metadata=[{} for _ in ids]
 if prediction_format=='english_text':seq=english_to_phones_batch(texts)
 elif prediction_format=='rwcp_repo_phone':seq=[rwcp_repo_to_phones(x) for x in texts]
 elif prediction_format=='ipa_tokens':seq=[temporal_view(x.split()) for x in texts]
 elif prediction_format=='firat_romanized_korean':
  if not lexicon_path:raise ValueError('romanization lexicon required')
  lex=json.loads(Path(lexicon_path).read_text())['lexicon'];seq,metadata=romanized_korean_to_phones_batch(texts,lex,return_metadata=True)
 else:raise ValueError(prediction_format)
 for aid,s,m in zip(ids,seq,metadata):selected[aid].update({'temporal':s,'segmental':segmental_view(s),**m})
 return selected
def _bootstrap_metric(args):
 metric, paired, replicates = args
 return metric, class_stratified_mean_ci(paired, metric, replicates=replicates, seed=1337)


def evaluate(manifest,prediction_path,output,*,prediction_format,cfg,lexicon_path=None,allow_missing=False):
 all_rows=read_jsonl(manifest);test=[r for r in all_rows if r['split']=='test'];pred=load_predictions(prediction_path,prediction_format,lexicon_path);expected={r['eval_audio_id'] for r in test};observed=set(pred);missing=sorted(expected-observed);extra=sorted(observed-expected)
 if extra:raise RuntimeError({'extra_prediction_ids':extra[:20],'count':len(extra)})
 if missing and not allow_missing:raise RuntimeError({'missing_prediction_ids':missing[:20],'count':len(missing)})
 test=[r for r in test if r['eval_audio_id'] in pred];paired=[]
 for r in test:
  x=pred[r['eval_audio_id']];tl=[z['phone_tokens_temporal'] for z in r['labels'] if z['phone_tokens_temporal']];sl=[z['phone_tokens_segmental'] for z in r['labels'] if z['phone_tokens_segmental']];td=[phone_edit_distance(x['temporal'],z) for z in tl];sd=[phone_edit_distance(x['segmental'],z) for z in sl];dd=[duration_distance(x['temporal'],z) for z in tl]
  row={'eval_audio_id':r['eval_audio_id'],'class_key':r['class_key'],'prediction':x['raw_text'],'phone_tokens_temporal':x['temporal'],'phone_tokens_segmental':x['segmental'],'paired_min_ped_temporal':min(td,default=1.),'paired_mean_ped_temporal':float(np.mean(td)) if td else 1.,'paired_min_ped_segmental':min(sd,default=1.),'paired_mean_ped_segmental':float(np.mean(sd)) if sd else 1.,'paired_min_duration_distance':min(dd,default=1.),'valid_output':bool(x['temporal']),'adjacent_repetitions':sum(a==b for a,b in zip(x['temporal'],x['temporal'][1:]))}
  for k in ('romanization_oov_tokens','romanization_oov_count','reconstructed_hangul_runs'):
   if k in x:row[k]=x[k]
  paired.append(row)
 views={v:{k:x[v] for k,x in pred.items() if k in expected} for v in ('temporal','segmental')};metrics_names=('paired_min_ped_temporal','paired_mean_ped_temporal','paired_min_ped_segmental','paired_mean_ped_segmental','paired_min_duration_distance')
 metrics={'schema':SCHEMA,'evaluator_sha256':sha(Path(__file__)),'phonology_sha256':sha(ROOT/'phonology.py'),'dataset':test[0]['eval_dataset'] if test else None,'split':'test','prediction_format':prediction_format,'n_expected':len(expected),'n_evaluated':len(test),'n_missing':len(missing),'class_count':len({r['class_key'] for r in test}),'valid_output_rate':float(np.mean([r['valid_output'] for r in paired])) if paired else 0.,'empty_output_rate':float(np.mean([not r['valid_output'] for r in paired])) if paired else 1.,'paired':dict(__import__('multiprocessing').get_context('fork').Pool(processes=min(int(__import__('os').environ.get('EVAL1_WORKERS','1')),len(metrics_names))).map(_bootstrap_metric,[(m,paired,int(cfg['bootstrap_replicates'])) for m in metrics_names])) if int(__import__('os').environ.get('EVAL1_WORKERS','1'))>1 else {m:class_stratified_mean_ci(paired,m,replicates=int(cfg['bootstrap_replicates']),seed=1337) for m in metrics_names},'retrieval':{v:evaluate_retrieval(views[v],test,view=v,replicates=int(cfg['label_replicates']),seed=1337,labels_per_candidate=int(cfg['labels_per_candidate'])) for v in views},'human_ceiling':{v:human_ceiling(test,view=v,replicates=int(cfg['label_replicates']),seed=1337,labels_per_candidate=int(cfg['labels_per_candidate'])) for v in views},'exclusions':{'missing_predictions':missing}}
 if prediction_format=='firat_romanized_korean':metrics['romanization_oov']={'occurrences':sum(r.get('romanization_oov_count',0) for r in paired),'rate':sum(r.get('romanization_oov_count',0) for r in paired)/max(1,sum(len(str(r['prediction']).split()) for r in paired))}
 output.mkdir(parents=True,exist_ok=True);(output/'per_query.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in paired));(output/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2,sort_keys=True)+'\n');(output/'_SUCCESS.json').write_text(json.dumps({'status':'COMPLETE','n':len(paired),'schema':SCHEMA},indent=2)+'\n');return metrics
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default=str(ROOT/'config.yaml'));p.add_argument('--manifest',required=True);p.add_argument('--predictions',required=True);p.add_argument('--output',required=True);p.add_argument('--prediction-format',choices=('english_text','rwcp_repo_phone','ipa_tokens','firat_romanized_korean'),default='english_text');p.add_argument('--lexicon',default=str(ROOT/'outputs/phonology/firat_train_romanization_lexicon.json'));p.add_argument('--allow-missing',action='store_true');a=p.parse_args();cfg=yaml.safe_load(Path(a.config).read_text());print(json.dumps(evaluate(Path(a.manifest),Path(a.predictions),Path(a.output),prediction_format=a.prediction_format,cfg=cfg,lexicon_path=Path(a.lexicon),allow_missing=a.allow_missing),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
