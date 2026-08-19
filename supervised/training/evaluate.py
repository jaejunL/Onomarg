from pathlib import Path
import argparse,json,yaml,torch
from torch.utils.data import DataLoader
from data.dataset import OnomaDataset,eval_collate
from models.onomacap_bart import OnomaCapModel
from metrics.coco_caption_adapter import evaluate
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--checkpoint',required=True); a=ap.parse_args(); cfg=yaml.safe_load(Path(a.config).read_text()); device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); m=OnomaCapModel(cfg,device=device); m.load_htsat(); cp=torch.load(a.checkpoint,map_location=device,weights_only=False); m.load_state_dict(cp['model']); ds=OnomaDataset(str(ROOT/'outputs/splits/split_manifest.jsonl'),'test',m.tokenizer,train=False,max_length=30,max_audio_sec=10,seed=20); dl=DataLoader(ds,batch_size=cfg['batch_size'],shuffle=False,num_workers=cfg['num_workers'],collate_fn=eval_collate); m.eval(); p=[];r=[];rows=[]
 with torch.no_grad():
  for b in dl:
   ids,txt=m.generate_text(b['audio'].to(device))
   for i,(meta,t) in enumerate(zip(b['meta'],txt)):
    p.append(t.strip().lower());r.append(meta['references']);rows.append({'audio_id':meta['audio_id'],'class_id':meta['class_id'],'prediction':t.strip().lower(),'generated_token_ids':ids[i].cpu().tolist(),'references':meta['references']})
 metrics=evaluate(p,r); out=ROOT/'outputs/runs/seed20/test';out.mkdir(parents=True,exist_ok=True); (out/'predictions.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows)); (out/'metrics.json').write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics))
if __name__=='__main__': main()
