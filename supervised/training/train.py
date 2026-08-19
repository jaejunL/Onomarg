from __future__ import annotations
import argparse, hashlib, json, os, random, time
from pathlib import Path
import numpy as np, torch, yaml
from torch.utils.data import DataLoader
from data.dataset import OnomaDataset, train_collate, eval_collate
from models.onomacap_bart import OnomaCapModel
from metrics.coco_caption_adapter import evaluate as metric_eval

ROOT=Path(__file__).resolve().parents[1]
def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    try: torch.use_deterministic_algorithms(True,warn_only=True)
    except Exception: pass
def sha(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def select_best(old,new):
    if old is None:return True
    for k in ('METEOR','BLEU1'):
        if abs(new[k]-old[k])>1e-12:return new[k]>old[k]
    return new['epoch']<old['epoch']
def save_atomic(obj,path):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp'); torch.save(obj,tmp); os.replace(tmp,path)
def load_manifest(cfg): return str(ROOT/'outputs/splits/split_manifest.jsonl')
def evaluate_model(model,loader,device,out_dir,epoch):
    model.eval(); preds=[]; refs=[]; rows=[]
    with torch.no_grad():
        for batch in loader:
            ids,texts=model.generate_text(batch['audio'].to(device)); decoded=[x.strip().lower() for x in texts]
            for j,(meta,pred) in enumerate(zip(batch['meta'],decoded)):
                preds.append(pred); refs.append(meta['references']); rows.append({'audio_id':meta['audio_id'],'class_id':meta['class_id'],'prediction':pred,'generated_token_ids':ids[j].detach().cpu().tolist(),'references':meta['references']})
    metrics=metric_eval(preds,refs); metrics={k:float(v) for k,v in metrics.items()}; metrics['epoch']=epoch
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True); (out_dir/f'epoch_{epoch:02d}_predictions.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows)); (out_dir/f'epoch_{epoch:02d}_metrics.json').write_text(json.dumps(metrics,indent=2)); return metrics
def run(cfg,smoke=False,resume=None):
    seed_all(int(cfg['seed'])); device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1',os.environ.get('CUDA_VISIBLE_DEVICES')
    model=OnomaCapModel(cfg,device=device); audit=model.load_htsat(); (ROOT/'outputs/preflight/htsat_load_audit.json').write_text(json.dumps(audit,indent=2))
    tok=model.tokenizer; manifest=load_manifest(cfg)
    tr=OnomaDataset(manifest,'train',tok,train=True,max_length=cfg['max_length'],max_audio_sec=cfg['max_audio_sec'],seed=cfg['seed'])
    va=OnomaDataset(manifest,'validation',tok,train=False,max_length=cfg['max_length'],max_audio_sec=cfg['max_audio_sec'],seed=cfg['seed'])
    if smoke:
        tr.items=tr.items[:160]
        va.items=va.items[:8]
    batch=4 if smoke else cfg['batch_size']; workers=0 if smoke else cfg['num_workers']
    tl=DataLoader(tr,batch_size=batch,shuffle=True,num_workers=workers,drop_last=True,collate_fn=lambda x:train_collate(x,tok.pad_token_id),generator=torch.Generator().manual_seed(cfg['seed']))
    vl=DataLoader(va,batch_size=4 if smoke else cfg['batch_size'],shuffle=False,num_workers=workers,collate_fn=eval_collate)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],betas=(.9,.999),eps=1e-8,weight_decay=cfg['weight_decay'])
    total_steps=(len(tl)*cfg['epochs']) if not smoke else 10; warm=max(1,round(total_steps*cfg['warmup_fraction']))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda step: min((step+1)/warm,1.0)*(.5*(1+np.cos(np.pi*max(0,step-warm)/max(1,total_steps-warm)))))
    start_epoch=1; best=None; global_step=0; resume_batch=0
    run_dir=ROOT/'outputs/runs/seed20'; run_dir.mkdir(parents=True,exist_ok=True)
    if resume:
        cp=torch.load(resume,map_location=device,weights_only=False); model.load_state_dict(cp['model']); opt.load_state_dict(cp['optimizer']); sched.load_state_dict(cp['scheduler']); start_epoch=cp['epoch']; global_step=cp['global_step']; best=cp['best']; resume_batch=cp.get('batch_index',0);
        if cp.get('torch_rng') is not None: torch.set_rng_state(cp['torch_rng']);
        if device.type=='cuda' and cp.get('cuda_rng') is not None: torch.cuda.set_rng_state_all(cp['cuda_rng'])
    max_epochs=1 if smoke else cfg['epochs']
    for epoch in range(start_epoch,max_epochs+1):
        model.train(); losses=[]; t0=time.time()
        for bi,b in enumerate(tl):
            if epoch==start_epoch and bi<resume_batch: continue
            opt.zero_grad(set_to_none=True); loss,_=model(b['audio'].to(device),b['labels'].to(device),b['attention_mask'].to(device)); assert torch.isfinite(loss),loss; loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip']); opt.step(); sched.step(); global_step+=1; losses.append(float(loss.detach()))
            if global_step%250==0 or (smoke and global_step==10): save_atomic({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'epoch':epoch,'batch_index':bi+1,'global_step':global_step,'best':best,'config':cfg,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None},run_dir/f'checkpoint_step_{global_step}.pt')
            if smoke and global_step>=10: break
        vm=evaluate_model(model,vl,device,run_dir/'validation',epoch); selected={k:vm[k] for k in ('METEOR','BLEU1')}; selected['epoch']=epoch
        if select_best(best,selected): best=selected; save_atomic({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'epoch':epoch,'batch_index':0,'global_step':global_step,'best':best,'config':cfg,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None,'htsat_audit':audit},run_dir/'best.pt')
        print(json.dumps({'epoch':epoch,'step':global_step,'loss':sum(losses)/max(1,len(losses)),'validation':vm,'best':best,'seconds':time.time()-t0}),flush=True)
        if smoke:
            (ROOT/'outputs/smoke/_SUCCESS.json').write_text(json.dumps({'status':'COMPLETE','steps':global_step,'best':best,'determinism':'warn_only_b1'}),encoding='utf-8')
            break
    return best
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--smoke',action='store_true'); ap.add_argument('--resume'); a=ap.parse_args(); cfg=yaml.safe_load(Path(a.config).read_text()); run(cfg,a.smoke,a.resume)
