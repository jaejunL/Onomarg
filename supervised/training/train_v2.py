from __future__ import annotations
import argparse, hashlib, json, os, random, time
from pathlib import Path
import numpy as np, torch, yaml
from torch.utils.data import DataLoader, Sampler
from data.dataset import OnomaDataset, train_collate, eval_collate
from models.onomacap_bart import OnomaCapModel
from metrics.coco_caption_adapter import evaluate as metric_eval
ROOT=Path(__file__).resolve().parents[1]

def seed_all(s):
 random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
 torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
 torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
 torch.use_deterministic_algorithms(True,warn_only=True)

def hash_state(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()): h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def save_atomic(obj,path):
 path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp'); torch.save(obj,tmp); os.replace(tmp,path)

def make_batches(n,batch_size,epochs,seed,drop_last=True):
 g=torch.Generator().manual_seed(seed); result=[]
 for _ in range(epochs):
  perm=torch.randperm(n,generator=g).tolist(); batches=[perm[i:i+batch_size] for i in range(0,len(perm),batch_size)]
  if drop_last: batches=[x for x in batches if len(x)==batch_size]
  result.append(batches)
 return result

class FixedBatchSampler(Sampler):
 def __init__(self,batches): self.batches=batches
 def __iter__(self): return iter(self.batches)
 def __len__(self): return len(self.batches)

def evaluate_model(model,loader,device,out_dir,epoch):
 model.eval(); preds=[]; refs=[]; rows=[]
 with torch.no_grad():
  for batch in loader:
   ids,texts=model.generate_text(batch['audio'].to(device)); decoded=[x.strip().lower() for x in texts]
   for j,(meta,pred) in enumerate(zip(batch['meta'],decoded)):
    preds.append(pred); refs.append(meta['references']); rows.append({'audio_id':meta['audio_id'],'class_id':meta['class_id'],'prediction':pred,'generated_token_ids':ids[j].detach().cpu().tolist(),'references':meta['references']})
 metrics={k:float(v) for k,v in metric_eval(preds,refs).items()}; metrics['epoch']=epoch
 out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True); (out_dir/f'epoch_{epoch:02d}_predictions.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows)); (out_dir/f'epoch_{epoch:02d}_metrics.json').write_text(json.dumps(metrics,indent=2)); return metrics

def run(cfg,smoke=False,resume=None):
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='1': raise RuntimeError('physical GPU 1 required')
 seed_all(int(cfg['seed'])); device=torch.device('cuda:0'); model=OnomaCapModel(cfg,device=device); audit=model.load_htsat()
 manifest=str(ROOT/'outputs/splits/split_manifest.jsonl'); tok=model.tokenizer
 tr=OnomaDataset(manifest,'train',tok,train=True,max_length=cfg['max_length'],max_audio_sec=cfg['max_audio_sec'],seed=cfg['seed']); va=OnomaDataset(manifest,'validation',tok,train=False,max_length=cfg['max_length'],max_audio_sec=cfg['max_audio_sec'],seed=cfg['seed'])
 if smoke: tr.items=tr.items[:160]; va.items=va.items[:8]
 batch=4 if smoke else cfg['batch_size']; workers=0 if smoke else cfg['num_workers']; epochs=1 if smoke else cfg['epochs']
 all_batches=make_batches(len(tr),batch,epochs,int(cfg['seed']),drop_last=True)
 opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],betas=(.9,.999),eps=1e-8,weight_decay=cfg['weight_decay'])
 total_steps=(len(all_batches[0])*epochs) if smoke else sum(len(x) for x in all_batches); warm=max(1,round(total_steps*cfg['warmup_fraction']))
 sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:min((step+1)/warm,1.0)*(.5*(1+np.cos(np.pi*max(0,step-warm)/max(1,total_steps-warm)))))
 start_epoch=0; resume_batch=0; global_step=0; best=None
 run_dir=ROOT/'outputs/runs/seed20'; run_dir.mkdir(parents=True,exist_ok=True)
 if resume:
  cp=torch.load(resume,map_location='cpu',weights_only=False); model.load_state_dict(cp['model']); opt.load_state_dict(cp['optimizer']); sched.load_state_dict(cp['scheduler']); start_epoch=int(cp['epoch']); resume_batch=int(cp['batch_index']); global_step=int(cp['global_step']); best=cp['best']
  random.setstate(cp['python_rng']); np.random.set_state(cp['numpy_rng']); torch.set_rng_state(cp['torch_rng']); torch.cuda.set_rng_state_all(cp['cuda_rng'])
 for ei in range(start_epoch,epochs):
  model.train(); losses=[]; t0=time.time(); batches=all_batches[ei]; first=resume_batch if ei==start_epoch else 0
  dl=DataLoader(tr,batch_sampler=FixedBatchSampler(batches[first:]),num_workers=workers,collate_fn=lambda x:train_collate(x,tok.pad_token_id))
  for local_b,b in enumerate(dl):
   bi=first+local_b; opt.zero_grad(set_to_none=True); loss,_=model(b['audio'].to(device),b['labels'].to(device),b['attention_mask'].to(device)); assert torch.isfinite(loss),loss; loss.backward(); gn=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip']); opt.step(); sched.step(); global_step+=1; losses.append(float(loss.detach()))
   if global_step%250==0 or (smoke and global_step==10):
    save_atomic({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'epoch':ei,'batch_index':bi+1,'global_step':global_step,'best':best,'sampler_batches':all_batches,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'config':cfg,'htsat_audit':audit},run_dir/f'checkpoint_step_{global_step}.pt')
   if smoke and global_step>=10: break
  vm=evaluate_model(model,DataLoader(va,batch_size=4 if smoke else cfg['batch_size'],shuffle=False,num_workers=workers,collate_fn=eval_collate),device,run_dir/'validation',ei+1)
  selected={k:vm[k] for k in ('METEOR','BLEU1')}; selected['epoch']=ei+1
  if best is None or selected['METEOR']>best['METEOR']+1e-12 or (abs(selected['METEOR']-best['METEOR'])<=1e-12 and (selected['BLEU1']>best['BLEU1']+1e-12 or (abs(selected['BLEU1']-best['BLEU1'])<=1e-12 and selected['epoch']<best['epoch']))):
   best=selected; save_atomic({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'epoch':ei+1,'batch_index':0,'global_step':global_step,'best':best,'sampler_batches':all_batches,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'config':cfg,'htsat_audit':audit},run_dir/'best.pt')
  print(json.dumps({'epoch':ei+1,'step':global_step,'loss':sum(losses)/max(1,len(losses)),'validation':vm,'best':best,'seconds':time.time()-t0,'sdp':{'flash':torch.backends.cuda.flash_sdp_enabled(),'mem_efficient':torch.backends.cuda.mem_efficient_sdp_enabled(),'math':torch.backends.cuda.math_sdp_enabled()}}),flush=True)
  resume_batch=0
  if smoke:
   (ROOT/'outputs/smoke/_SUCCESS.json').write_text(json.dumps({'status':'COMPLETE','steps':global_step,'best':best,'determinism':'math_sdp_bicubic_warn_only'})); break
 return best
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--smoke',action='store_true'); ap.add_argument('--resume'); a=ap.parse_args(); cfg=yaml.safe_load(Path(a.config).read_text()); run(cfg,a.smoke,a.resume)
