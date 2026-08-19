from __future__ import annotations
import hashlib, json
from pathlib import Path
import torch
import torch.nn as nn
from .htsat import HTSAT_Swin_Transformer

class _HTSATConfig(dict):
    def to_dict(self): return dict(self)

class HTSATEncoder(nn.Module):
    """WavCaps HTSAT temporal encoder, vendored at commit a5a9649."""
    def __init__(self, cfg, train_spec_augment=True):
        super().__init__(); self.cfg=cfg; self.train_spec_augment=train_spec_augment
        audio_args={'sr':cfg['sample_rate'],'n_fft':cfg['n_fft'],'hop_length':cfg['hop_length'],'f_min':cfg['f_min'],'f_max':cfg['f_max'],'n_mels':cfg['n_mels']}
        hcfg=_HTSATConfig(audio_args=audio_args)
        self.enc=HTSAT_Swin_Transformer(spec_size=256,patch_size=4,patch_stride=(4,4),num_classes=527,embed_dim=96,depths=[2,2,6,2],num_heads=[4,8,16,32],window_size=8,config=hcfg)
        self.enc.is_spec_augment=True
        self.load_audit={}
    def load_audioset(self,path,expected_sha):
        h=hashlib.sha256(Path(path).read_bytes()).hexdigest(); assert h==expected_sha,(h,expected_sha)
        ck=torch.load(path,map_location='cpu',weights_only=False).get('state_dict',{})
        clean={}; discarded=[]
        for k,v in ck.items():
            if k.startswith('sed_model.'):
                nk=k[len('sed_model.'):]
                if nk.startswith(('spectrogram_extractor.','logmel_extractor.','tscam_conv.','head.')): discarded.append(k); continue
                clean[nk]=v
            else: discarded.append(k)
        expected=self.enc.state_dict(); loaded=[]; missing=[]; unexpected=[]; mismatch=[]
        for k,v in clean.items():
            if k not in expected: unexpected.append(k)
            elif expected[k].shape!=v.shape: mismatch.append({'key':k,'expected':list(expected[k].shape),'got':list(v.shape)})
            else: loaded.append(k)
        missing=[k for k in expected if k not in clean]
        fatal=[k for k in missing if not k.startswith(('audio_feats_extractor.','spectrogram_extractor.','logmel_extractor.','tscam_conv.','head.'))]
        if mismatch or fatal: raise RuntimeError(json.dumps({'shape_mismatch':mismatch[:5],'missing_backbone':fatal[:10]}))
        self.enc.load_state_dict({k:clean[k] for k in loaded},strict=False)
        self.load_audit={'checkpoint_sha256':h,'loaded':loaded,'missing':missing,'unexpected':unexpected,'discarded':discarded,'shape_mismatch':mismatch}
        return self.load_audit
    def forward(self,wave):
        self.enc.is_spec_augment=self.train_spec_augment
        return self.enc(wave)
