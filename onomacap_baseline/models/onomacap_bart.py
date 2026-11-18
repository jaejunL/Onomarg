from __future__ import annotations
import torch
import torch.nn as nn
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer
from transformers.models.bart.modeling_bart import shift_tokens_right
from .audio_encoder import HTSATEncoder

class OnomaCapModel(nn.Module):
    def __init__(self,cfg,device=None):
        super().__init__(); self.cfg=cfg
        self.tokenizer=BartTokenizer.from_pretrained(cfg['bart_model_id'],revision=cfg['bart_revision'])
        self.htsat=HTSATEncoder(cfg,train_spec_augment=True)
        bc=BartConfig.from_pretrained(cfg['bart_model_id'],revision=cfg['bart_revision'])
        bc.decoder_start_token_id=self.tokenizer.bos_token_id
        bc.forced_eos_token_id=self.tokenizer.eos_token_id
        self.bart=BartForConditionalGeneration(bc)
        self.audio_projection=nn.Linear(768,bc.d_model)
        self.loss_fn=nn.CrossEntropyLoss(ignore_index=-100,label_smoothing=float(cfg['label_smoothing']))
        if device is not None: self.to(device)
    def load_htsat(self): return self.htsat.load_audioset(self.cfg['htsat_checkpoint'],self.cfg['htsat_sha256'])
    def audio_embeds(self,wave):
        z=self.htsat(wave); assert z.ndim==3 and z.shape[-1]==768 and z.shape[1]>1,z.shape
        return self.audio_projection(z)
    def forward(self,wave,labels,decoder_attention_mask=None):
        if decoder_attention_mask is None:
            raise ValueError("v2 requires decoder_attention_mask")
        if decoder_attention_mask.shape != labels.shape:
            raise ValueError((decoder_attention_mask.shape, labels.shape))
        if decoder_attention_mask.dtype not in (torch.bool, torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError(decoder_attention_mask.dtype)
        emb=self.audio_embeds(wave)
        enc=self.bart.model.encoder(inputs_embeds=emb,return_dict=True)
        decoder_input_ids=shift_tokens_right(
            labels,
            self.bart.config.pad_token_id,
            self.bart.config.decoder_start_token_id,
        )
        out=self.bart(
            encoder_outputs=enc,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=True,
            use_cache=False,
        )
        logits=out.logits
        return self.loss_fn(logits.reshape(-1,logits.size(-1)),labels.reshape(-1)), logits
    @torch.no_grad()
    def generate_text(self,wave):
        emb=self.audio_embeds(wave)
        enc=self.bart.model.encoder(inputs_embeds=emb,return_dict=True)
        ids=self.bart.generate(encoder_outputs=enc,num_beams=3,num_return_sequences=1,min_length=2,max_length=30,length_penalty=1.0,repetition_penalty=1.0,do_sample=False)
        return ids,self.tokenizer.batch_decode(ids,skip_special_tokens=True)
