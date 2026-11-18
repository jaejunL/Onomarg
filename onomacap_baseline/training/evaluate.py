from pathlib import Path
import argparse,json,torch
from torch.utils.data import DataLoader
from config_utils import load_config, project_path
from data.dataset import OnomaDataset,eval_collate
from models.onomacap_bart import OnomaCapModel
from metrics.coco_caption_adapter import evaluate
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--checkpoint'); ap.add_argument('--manifest'); ap.add_argument('--output-dir'); a=ap.parse_args(); cfg=load_config(a.config); device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); checkpoint=project_path(a.checkpoint) if a.checkpoint else Path(cfg['run_dir'])/'best.pt'; manifest=project_path(a.manifest) if a.manifest else Path(cfg['manifest_path']); output=project_path(a.output_dir) if a.output_dir else Path(cfg['test_output_dir']); m=OnomaCapModel(cfg,device=device); m.load_htsat(); cp=torch.load(checkpoint,map_location=device,weights_only=False); m.load_state_dict(cp['model']); ds=OnomaDataset(manifest,'test',m.tokenizer,train=False,max_length=cfg['max_length'],max_audio_sec=cfg['max_audio_sec'],seed=cfg['seed']); dl=DataLoader(ds,batch_size=cfg['batch_size'],shuffle=False,num_workers=cfg['num_workers'],collate_fn=eval_collate); m.eval(); p=[];r=[];rows=[]
 with torch.no_grad():
  for b in dl:
   ids,txt=m.generate_text(b['audio'].to(device))
   for i,(meta,t) in enumerate(zip(b['meta'],txt)):
    p.append(t.strip().lower());r.append(meta['references']);rows.append({'audio_id':meta['audio_id'],'class_id':meta['class_id'],'prediction':t.strip().lower(),'generated_token_ids':ids[i].cpu().tolist(),'references':meta['references']})
 metrics=evaluate(p,r); output.mkdir(parents=True,exist_ok=True); (output/'predictions.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows)); (output/'metrics.json').write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics))
if __name__=='__main__': main()
