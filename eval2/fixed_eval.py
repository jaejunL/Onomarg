from __future__ import annotations
import hashlib,json,math,re
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from eval1.phonology import phone_edit_distance,duration_distance,segmental_view,temporal_view,english_to_phones_batch,romanized_korean_to_phones_batch,rwcp_repo_to_phones
from eval2.bootstrap import class_stratified_indices,percentile_ci

def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_rows(path,split=None):
    rows=[json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()];return [r for r in rows if split is None or r.get('split')==split]
def beam0(rows):
    out={}
    for r in rows:
        aid=r.get('audio_id',r.get('eval_audio_id'))
        if aid not in out or int(r.get('beam_rank',0))<int(out[aid].get('beam_rank',0)):out[aid]=r
    return out
def prediction_phones(rows,fmt,dataset,mapping_path,lexicon_path=None):
    chosen=beam0(rows);ids=sorted(chosen)
    if fmt=='english_text':seqs=english_to_phones_batch([chosen[i].get('prediction',chosen[i].get('text','')) for i in ids])
    elif fmt=='firat_romanized_korean':
        payload=json.loads(Path(lexicon_path).read_text());lex=payload.get('lexicon',payload);seqs=romanized_korean_to_phones_batch([chosen[i].get('prediction','') for i in ids],lex)
    elif fmt=='rwcp_repo_phone':seqs=[rwcp_repo_to_phones(chosen[i].get('prediction','').split()) for i in ids]
    elif fmt=='ipa_tokens':seqs=[chosen[i]['phone_tokens_temporal'] for i in ids]
    elif fmt=='arpabet_runs':
        mapping=json.loads(Path(mapping_path).read_text());seqs=[]
        for i in ids:
            seq=[]
            for run in chosen[i].get('runs',[]):
                p=re.sub(r'\d+$','',run['phone'].upper());mapped=mapping.get(p,'<unk>');count=max(1,round(float(run['duration_sec'])*100));seq.extend([mapped]*count)
            seqs.append(seq)
    else:raise ValueError(fmt)
    return {i:temporal_view(s) for i,s in zip(ids,seqs)},chosen
def candidate_labels(rows,view):
    key=f'phone_tokens_{view}';return {r['eval_audio_id']:[tuple(x[key]) for x in r['labels'] if x.get(key)] for r in rows}
def scalar_distance_matrix(preds,rows,view,cache_path=None):
    ids=sorted(preds);labels=candidate_labels(rows,view);cids=sorted(labels);key=hashlib.sha256(json.dumps({'preds':[(i,preds[i]) for i in ids],'labels':[(i,labels[i]) for i in cids],'view':view,'code':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},sort_keys=True).encode()).hexdigest()
    if cache_path is not None:
        cache_path=Path(cache_path)
        if cache_path.is_file():
            z=np.load(cache_path,allow_pickle=False)
            if str(z['key'])==key and z['matrix'].shape==(len(ids),len(cids)):return ids,cids,z['matrix']
    a=np.empty((len(ids),len(cids)),np.float32)
    for i,qid in enumerate(ids):
        q=segmental_view(preds[qid]) if view=='segmental' else temporal_view(preds[qid])
        for j,cid in enumerate(cids):a[i,j]=min(phone_edit_distance(q,x) for x in labels[cid])
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(cache_path,key=key,matrix=a);cache_path.with_suffix('.json').write_text(json.dumps({'sha256_key':key,'shape':list(a.shape),'dtype':str(a.dtype),'view':view},indent=2)+'\n')
    return ids,cids,a
def ranks_from_matrix(query_ids,candidate_ids,matrix,true_ids=None):
    true_ids=query_ids if true_ids is None else true_ids;index={x:i for i,x in enumerate(candidate_ids)};r=[]
    for i,t in enumerate(true_ids):
        order=sorted(range(len(candidate_ids)),key=lambda j:(float(matrix[i,j]),candidate_ids[j]));r.append(order.index(index[t])+1)
    return np.asarray(r,dtype=np.int64)
def retrieval_metrics(ranks):
    x=np.asarray(ranks);n=len(x);return {'n':n,'R@1':float(np.mean(x<=1)) if n else 0.,'R@5':float(np.mean(x<=5)) if n else 0.,'R@10':float(np.mean(x<=10)) if n else 0.,'MRR':float(np.mean(1/x)) if n else 0.,'median_rank':float(np.median(x)) if n else float('nan'),'mean_rank_percentile':float(np.mean(1-(x-1)/max(n-1,1))) if n else 0.}
def evaluate(preds,manifest_rows,reps=10000,seed=1337,cache_root=None):
    test=[r for r in manifest_rows if r['split']=='test'];by={r['eval_audio_id']:r for r in test};expected=set(by)
    if set(preds)!=expected:raise RuntimeError({'missing':sorted(expected-set(preds))[:5],'extra':sorted(set(preds)-expected)[:5]})
    result={'count':len(test),'views':{}}
    for view in ('temporal','segmental'):
        qids,cids,matrix=scalar_distance_matrix(preds,test,view,(Path(cache_root)/f'{view}.npz') if cache_root else None);ranks=ranks_from_matrix(qids,cids,matrix);global_metrics=retrieval_metrics(ranks);classes=[by[i]['class_key'] for i in qids];within=[]
        for i,qid in enumerate(qids):
            candidates=[j for j,c in enumerate(cids) if by[c]['class_key']==by[qid]['class_key']];order=sorted(candidates,key=lambda j:(float(matrix[i,j]),cids[j]));within.append(order.index(cids.index(qid))+1)
        boot=class_stratified_indices(classes,reps,seed);mrr=np.asarray([np.mean(1/ranks[x]) for x in boot]);global_metrics['MRR_ci95']=percentile_ci(mrr)
        result['views'][view]={'global':global_metrics,'within_class':retrieval_metrics(within)}
        if view=='temporal':result['per_query_temporal']=[{'audio_id':i,'class_key':by[i]['class_key'],'rank':int(r)} for i,r in zip(qids,ranks)]
        labs=candidate_labels(test,view);mins=[];means=[];dur=[]
        for qid in qids:
            query=segmental_view(preds[qid]) if view=='segmental' else temporal_view(preds[qid])
            distances=[phone_edit_distance(query,x) for x in labs[qid]]
            mins.append(min(distances));means.append(float(np.mean(distances)))
            if view=='temporal':dur.append(min(duration_distance(query,x) for x in labs[qid]))
        if view=='temporal':
            result.update({'min_PED_temporal':float(np.mean(mins)),'mean_PED_temporal':float(np.mean(means)),'duration_distance':float(np.mean(dur))})
        else:
            result.update({'min_PED_segmental':float(np.mean(mins)),'mean_PED_segmental':float(np.mean(means))})
    n=len(test);result['chance']={'R@1':1/n,'R@5':min(5,n)/n,'R@10':min(10,n)/n,'expected_MRR':sum(1/i for i in range(1,n+1))/n};return result
def mode_collapse(raw_outputs,classes):
    n=len(raw_outputs);c=Counter(raw_outputs);ent=-sum((v/n)*math.log(v/n) for v in c.values()) if n else 0.;by=defaultdict(list)
    for o,k in zip(raw_outputs,classes):by[k].append(o)
    return {'unique_output_ratio':len(c)/n if n else 0.,'top1_mass':max(c.values(),default=0)/n if n else 0.,'top5_mass':sum(sorted(c.values(),reverse=True)[:5])/n if n else 0.,'normalized_entropy':ent/math.log(len(c)) if len(c)>1 else 0.,'per_class_unique_ratio_macro':float(np.mean([len(set(v))/len(v) for v in by.values()])) if by else 0.}

def paired_shuffle_comparison(original,shuffled,manifest_rows,reps=10000,seed=1337):
    test=[r for r in manifest_rows if r['split']=='test'];by={r['eval_audio_id']:r for r in test};qids,cids,a=scalar_distance_matrix(original,test,'temporal');sq,sc,b=scalar_distance_matrix(shuffled,test,'temporal')
    if qids!=sq or cids!=sc:raise RuntimeError('shuffle coverage/order mismatch')
    ra=ranks_from_matrix(qids,cids,a);rb=ranks_from_matrix(qids,cids,b);classes=[by[i]['class_key'] for i in qids];indices=class_stratified_indices(classes,reps,seed);delta=np.asarray([np.mean(1/ra[x]-1/rb[x]) for x in indices]);equal=np.mean([tuple(original[i])==tuple(shuffled[i]) for i in qids])
    col={cid:i for i,cid in enumerate(cids)};paired_a=np.asarray([a[i,col[qid]] for i,qid in enumerate(qids)]);paired_b=np.asarray([b[i,col[qid]] for i,qid in enumerate(qids)])
    return {'original_minus_shuffled_MRR':float(np.mean(1/ra-1/rb)),'MRR_delta_ci95':percentile_ci(delta),'original_minus_shuffled_min_PED':float(np.mean(paired_a-paired_b)),'output_equality_rate':float(equal)}
def annotation_sensitivity(preds,manifest_rows,view='temporal',ks=(1,5),reps=100,seed=1337):
    test=[r for r in manifest_rows if r['split']=='test'];by={r['eval_audio_id']:r for r in test};ids=sorted(preds);cids=sorted(by)
    key=f'phone_tokens_{view}'
    labels={cid:[x for x in by[cid]['labels'] if x.get(key)] for cid in cids}
    if any(not x for x in labels.values()):raise RuntimeError('annotation sensitivity candidate has no valid labels')
    max_labels=max(len(x) for x in labels.values())
    distances=np.full((len(ids),len(cids),max_labels),np.inf,dtype=np.float32)
    for i,qid in enumerate(ids):
        query=segmental_view(preds[qid]) if view=='segmental' else temporal_view(preds[qid])
        for j,cid in enumerate(cids):
            for li,label in enumerate(labels[cid]):
                distances[i,j,li]=phone_edit_distance(query,label[key])
    true_cols=np.asarray([cids.index(i) for i in ids],dtype=np.int64);out={}
    for k in ks:
        vals=[]
        for rep in range(reps):
            matrix=np.empty((len(ids),len(cids)),dtype=np.float32)
            for j,cid in enumerate(cids):
                positions={x['label_id']:i for i,x in enumerate(labels[cid])}
                chosen=sorted(by[cid]['labels'],key=lambda x:hashlib.sha256(f"{seed}|annotations|{rep}|{cid}|{x['label_id']}".encode()).hexdigest())[:k]
                selected=[positions[x['label_id']] for x in chosen if x.get(key)]
                if not selected:raise RuntimeError('annotation replicate selected no valid labels')
                matrix[:,j]=distances[:,j,selected].min(axis=1)
            order=np.argsort(matrix,axis=1,kind='stable')
            inverse=np.empty_like(order)
            inverse[np.arange(len(ids))[:,None],order]=np.arange(len(cids))[None,:]
            ranks=inverse[np.arange(len(ids)),true_cols]+1
            vals.append(float(np.mean(1/ranks)))
        out[str(k)]={'mean':float(np.mean(vals)),'std':float(np.std(vals)),'replicates':reps}
    return out

def human_heldout_consistency(manifest_rows,reps=10000,seed=1337):
    test=[r for r in manifest_rows if r['split']=='test'];ids=sorted(r['eval_audio_id'] for r in test);by={r['eval_audio_id']:r for r in test};queries={};labels={}
    for aid in ids:
        ordered=sorted(by[aid]['labels'],key=lambda x:x['label_id']);queries[aid]=tuple(ordered[0]['phone_tokens_temporal']);labels[aid]=[tuple(x['phone_tokens_temporal']) for x in ordered[1:] if x.get('phone_tokens_temporal')]
    eligible=[i for i in ids if labels[i]];matrix=np.asarray([[min(phone_edit_distance(queries[q],x) for x in labels[c]) for c in eligible] for q in eligible],dtype=np.float32);ranks=ranks_from_matrix(eligible,eligible,matrix);classes=[by[i]['class_key'] for i in eligible];boot=class_stratified_indices(classes,reps,seed);vals=[np.mean(1/ranks[x]) for x in boot];return {'n':len(eligible),'MRR':float(np.mean(1/ranks)),'MRR_ci95':percentile_ci(vals)}
def detailed_mode_collapse(raw_outputs,phone_outputs,classes):
    out=mode_collapse(raw_outputs,classes);unique={tuple(x) for x in phone_outputs};pairs=[];u=sorted(unique)
    for i in range(len(u)):
        for j in range(i+1,len(u)):pairs.append(phone_edit_distance(segmental_view(u[i]),segmental_view(u[j])))
    out['mean_pairwise_segmental_PED_unique']=float(np.mean(pairs)) if pairs else 0.;out['adjacent_repetition_count_mean']=float(np.mean([sum(a==b for a,b in zip(x,x[1:])) for x in phone_outputs])) if phone_outputs else 0.;return out
