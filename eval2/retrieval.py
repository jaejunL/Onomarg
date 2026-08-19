from __future__ import annotations
import hashlib,os
from collections import defaultdict
from typing import Any,Iterable,Sequence
import numpy as np
from eval1.phonology import phone_edit_distance
def _stable_key(seed,replicate,audio_id,label_id):return hashlib.sha256(f'{seed}|labels|{replicate}|{audio_id}|{label_id}'.encode()).hexdigest()
def unique_labels(row,view):
 key=f'phone_tokens_{view}';by_sequence={}
 for label in row['labels']:
  sequence=tuple(label[key])
  if sequence and sequence not in by_sequence:by_sequence[sequence]=label
 return sorted(by_sequence.values(),key=lambda label:label['label_id'])
def sampled_label_sets(rows,*,view,replicate,seed,k):
 output={};token_key=f'phone_tokens_{view}'
 for row in rows:
  labels=unique_labels(row,view)
  if len(labels)<k:continue
  selected=sorted(labels,key=lambda label:_stable_key(seed,replicate,row['eval_audio_id'],label['label_id']))[:k];output[row['eval_audio_id']]=[tuple(label[token_key]) for label in selected]
 return output
def _rank(query,true_id,candidate_ids,label_sets):
 scored=[]
 for candidate_id in candidate_ids:
  distance=min(phone_edit_distance(query,label) for label in label_sets[candidate_id]);scored.append((distance,candidate_id))
 scored.sort(key=lambda item:(item[0],item[1]));return [x[1] for x in scored].index(true_id)+1,len(scored)
def _metrics(ranks):
 if not ranks:return {'n':0,'R@1':0.,'R@5':0.,'R@10':0.,'MRR':0.,'median_rank':float('nan')}
 a=np.asarray(ranks,dtype=np.float64);return {'n':int(len(ranks)),'R@1':float(np.mean(a<=1)),'R@5':float(np.mean(a<=5)),'R@10':float(np.mean(a<=10)),'MRR':float(np.mean(1./a)),'median_rank':float(np.median(a))}
def _summarize(values):
 if not values:return {'replicates':0}
 output={'replicates':len(values),'n':int(values[0]['n'])}
 for key in ('R@1','R@5','R@10','MRR','median_rank'):
  data=np.asarray([value[key] for value in values],dtype=np.float64);output[key]={'mean':float(np.nanmean(data)),'std':float(np.nanstd(data)),'ci95':[float(np.nanpercentile(data,2.5)),float(np.nanpercentile(data,97.5))]}
 return output
def _retrieval_one(args):
 predictions,rows,view,replicate,seed,k=args;by_id={row['eval_audio_id']:row for row in rows};label_sets=sampled_label_sets(rows,view=view,replicate=replicate,seed=seed,k=k);eligible=sorted(set(label_sets)&set(predictions));by_class=defaultdict(list)
 for audio_id in eligible:by_class[str(by_id[audio_id]['class_key'])].append(audio_id)
 global_ranks=[];within_ranks=[]
 for audio_id in eligible:
  query=tuple(predictions[audio_id]);global_ranks.append(_rank(query,audio_id,eligible,label_sets)[0]);class_ids=sorted(by_class[str(by_id[audio_id]['class_key'])])
  if len(class_ids)>=2:within_ranks.append(_rank(query,audio_id,class_ids,label_sets)[0])
 return _metrics(global_ranks),_metrics(within_ranks),len(rows)-len(eligible)
def evaluate_retrieval(predictions,rows,*,view,replicates=100,seed=1337,labels_per_candidate=5):
 args=[(predictions,rows,view,replicate,seed,labels_per_candidate) for replicate in range(replicates)];workers=int(os.environ.get('EVAL1_WORKERS','1'))
 if workers>1 and replicates>1:
  import multiprocessing as mp
  with mp.get_context('fork').Pool(processes=workers) as pool:values=pool.map(_retrieval_one,args,chunksize=1)
 else:values=[_retrieval_one(x) for x in args]
 return {'view':view,'labels_per_candidate':labels_per_candidate,'global':_summarize([x[0] for x in values]),'within_class':_summarize([x[1] for x in values]),'excluded_audio_mean':float(np.mean([x[2] for x in values])) if values else 0.}
def _human_one(args):
 replicate,predictions,modified_rows,view,seed,candidate_k=args
 return replicate,evaluate_retrieval(predictions,modified_rows,view=view,replicates=1,seed=seed+replicate,labels_per_candidate=candidate_k)
def human_ceiling(rows,*,view,replicates=100,seed=1337,labels_per_candidate=5):
 token_key=f'phone_tokens_{view}';predictions_by_replicate=[];candidate_k=max(1,labels_per_candidate-1)
 for replicate in range(replicates):
  predictions={};modified_rows=[]
  for row in rows:
   labels=unique_labels(row,view)
   if len(labels)<labels_per_candidate:continue
   ordered=sorted(labels,key=lambda label:_stable_key(seed,replicate,row['eval_audio_id'],label['label_id']));predictions[row['eval_audio_id']]=tuple(ordered[0][token_key]);copy=dict(row);copy['labels']=ordered[1:];modified_rows.append(copy)
  predictions_by_replicate.append((predictions,modified_rows))
 workers=int(os.environ.get('EVAL1_WORKERS','1'))
 args=[(i,item[0],item[1],view,seed,candidate_k) for i,item in enumerate(predictions_by_replicate)]
 if workers>1 and replicates>1:
  import multiprocessing as mp
  with mp.get_context('fork').Pool(processes=min(workers,replicates)) as pool:
   values=pool.map(_human_one,args,chunksize=1)
 else:
  values=[_human_one(x) for x in args]
 results=[value for _,value in values]
 def collapse(section):
  entries=[]
  for value in results:
   flattened={'n':value[section]['n']}
   for key in ('R@1','R@5','R@10','MRR','median_rank'):flattened[key]=value[section][key]['mean']
   entries.append(flattened)
  return _summarize(entries)
 return {'view':view,'global':collapse('global'),'within_class':collapse('within_class')}
