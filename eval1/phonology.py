from __future__ import annotations
import functools,re,unicodedata
from typing import Iterable,Sequence
from panphon.distance import Distance
from phonemizer import phonemize
from phonemizer.separator import Separator
SCHEMA='eval1_phonology_v2';_SEP=Separator(phone=' ',word='|');_DIST=Distance();_BOUNDARIES={'|','<pause>'}
def temporal_view(tokens:Iterable[str])->list[str]:
 out=[]
 for token in tokens:
  token=unicodedata.normalize('NFC',str(token).strip())
  if not token or token=='|':continue
  out.extend((token[:-1],token[:-1]) if token.endswith('ː') and len(token)>1 else (token,))
 return out
def segmental_view(tokens:Iterable[str])->list[str]:
 out=[]
 for token in temporal_view(tokens):
  if token in {'<pause>','ʔ'}:continue
  token='ɴ' if token=='N' else token
  if not out or out[-1]!=token:out.append(token)
 return out
def _phonemize_batch(texts:Sequence[str],language:str)->list[list[str]]:
 clean=[unicodedata.normalize('NFC',str(x or '')).strip() for x in texts]; rendered=phonemize(clean,language=language,backend='espeak',strip=True,preserve_punctuation=False,with_stress=False,njobs=1,separator=_SEP)
 if isinstance(rendered,str):rendered=[rendered]
 result=[temporal_view(re.sub(r'[ˈˌ]','',str(x)).replace('|',' | ').split()) for x in rendered]
 if any('|' in token for seq in result for token in seq):raise RuntimeError('embedded eSpeak word boundary')
 return result
def english_to_phones_batch(texts):return _phonemize_batch(texts,'en-us')
def korean_to_phones_batch(texts):return _phonemize_batch([str(x).replace(' ','') for x in texts],'ko')
def english_to_phones(text):return english_to_phones_batch([text])[0]
def korean_to_phones(text):return korean_to_phones_batch([text])[0]
RWCP_TO_IPA={':':('<length>',),'N':('ɴ',),'N:':('ɴ','ɴ'),'a':('a',),'a:':('a','a'),'i':('i',),'i:':('i','i'),'u':('ɯ',),'u:':('ɯ','ɯ'),'e':('e',),'e:':('e','e'),'o':('o',),'o:':('o','o'),'b':('b',),'by':('bʲ',),'ch':('tɕ',),'d':('d',),'dy':('dʲ',),'f':('ɸ',),'g':('ɡ',),'gy':('ɡʲ',),'h':('h',),'hy':('ç',),'j':('dʑ',),'k':('k',),'ky':('kʲ',),'m':('m',),'my':('mʲ',),'n':('n',),'ny':('ɲ',),'p':('p',),'py':('pʲ',),'q':('ʔ',),'q:':('ʔ','ʔ'),'r':('ɾ',),'ry':('ɾʲ',),'s':('s',),'sh':('ɕ',),'t':('t',),'ts':('ts',),'ty':('tʲ',),'w':('w',),'y':('j',),'z':('z',),'zy':('ʑ',),'sp':('<pause>',),'isp':('i','<pause>'),'ir':('i','ɾ'),'katya':('k','a','t','j','a')}
def rwcp_repo_to_phones(text_or_tokens):
 out=[]
 for token in (str(text_or_tokens).split() if isinstance(text_or_tokens,str) else list(text_or_tokens)):
  if token not in RWCP_TO_IPA:raise ValueError(f'unknown RWCP phone token: {token!r}')
  mapped=RWCP_TO_IPA[token]
  if mapped==('<length>',):
   if out and out[-1] not in _BOUNDARIES:out.append(out[-1])
  else:out.extend(mapped)
 return temporal_view(out)
def rwcp_phone_audit_flags(text_or_tokens):
 src=str(text_or_tokens).split() if isinstance(text_or_tokens,str) else list(text_or_tokens);flags=[]
 if src and src[0]==':':flags.append('orphan_initial_length_marker_dropped')
 for token in ('isp','ir','katya'):
  if token in src:flags.append(f'repository_compound_or_malformed_token_expanded:{token}')
 return flags
def romanized_korean_to_phones_batch(texts,lexicon,return_metadata=False):
 outputs=[];metadata=[]
 for text in texts:
  toks=unicodedata.normalize('NFC',str(text or '')).lower().strip().split();pieces=[];known=[];oov=[]
  def flush():
   if known:pieces.append(('ko',''.join(lexicon[t] for t in known)));known.clear()
  for token in toks:
   if token in lexicon:known.append(token)
   else:flush();oov.append(token);pieces.append(('en',token))
  flush();phones=[];runs=[]
  for kind,piece in pieces:
   if kind=='ko':runs.append(piece);phones.extend(korean_to_phones(piece))
   else:phones.extend(english_to_phones(piece))
  outputs.append(temporal_view(phones));metadata.append({'romanization_oov_tokens':oov,'romanization_oov_count':len(oov),'reconstructed_hangul_runs':runs})
 return (outputs,metadata) if return_metadata else outputs
@functools.lru_cache(maxsize=200000)
def _sub(a,b):
 if a==b:return 0.0
 if a in _BOUNDARIES or b in _BOUNDARIES:return 1.0
 try:return min(1.0,float(_DIST.weighted_feature_edit_distance(a,b)))
 except Exception:return 1.0
@functools.lru_cache(maxsize=500000)
def _edit(left,right):
 if not left and not right:return 0.0
 prev=list(map(float,range(len(right)+1)))
 for i,a in enumerate(left,1):
  cur=[float(i)]
  for j,b in enumerate(right,1):cur.append(min(prev[j]+1,cur[j-1]+1,prev[j-1]+_sub(a,b)))
  prev=cur
 return prev[-1]/max(len(left),len(right),1)
def phone_edit_distance(a,b):return _edit(tuple(a),tuple(b))
def duration_distance(a,b):return abs(len(a)-len(b))/max(len(a),len(b),1)
def distance_bundle(left,right):
 lt,rt=temporal_view(left),temporal_view(right);ls,rs=segmental_view(lt),segmental_view(rt);return {'ped_temporal':phone_edit_distance(lt,rt),'ped_segmental':phone_edit_distance(ls,rs),'duration_distance':duration_distance(lt,rt)}
