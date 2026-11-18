import collections,math
def summarize(outputs):
 n=len(outputs); c=collections.Counter(outputs); ent=0
 for v in c.values():
  p=v/n;ent-=p*math.log(p) if p else 0
 return {'unique_output_ratio':len(c)/n if n else 0,'top1_mass':max(c.values(),default=0)/n if n else 0,'top5_mass':sum(sorted(c.values(),reverse=True)[:5])/n if n else 0,'normalized_entropy':ent/math.log(len(c)) if len(c)>1 else 0}
