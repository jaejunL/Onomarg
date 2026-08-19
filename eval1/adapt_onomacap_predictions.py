from __future__ import annotations
import argparse,json
from pathlib import Path
from eval1.phonology_v2 import romanized_korean_to_phones_batch
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--lexicon',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 lex=json.loads(Path(a.lexicon).read_text())['lexicon']; rows=[json.loads(x) for x in Path(a.input).read_text().splitlines() if x.strip()]; texts=[str(r.get('prediction',r.get('text',''))) for r in rows]; phones,meta=romanized_korean_to_phones_batch(texts,lex,return_metadata=True); out=[]
 for r,pseq,m in zip(rows,phones,meta):
  q=dict(r); q['prediction']=' '.join(pseq); q['prediction_format']='ipa_tokens'; q.update(m); out.append(q)
 Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out))
if __name__=='__main__':main()
