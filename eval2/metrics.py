from __future__ import annotations
import math,collections
def _norm(x): return str(x or '').strip().lower()
def score_file(predictions,labels):
 n=len(predictions); empty=sum(not _norm(x) for x in predictions)
 counts=collections.Counter(_norm(x) for x in predictions)
 ent=0.0
 for c in counts.values():
  p=c/n
  ent-=p*math.log(p) if p else 0
 return {"count":n,"valid_rate":(n-empty)/n if n else 0,"empty_rate":empty/n if n else 0,"unique_output_ratio":len(counts)/n if n else 0,"normalized_entropy":ent/math.log(len(counts)) if len(counts)>1 else 0}
