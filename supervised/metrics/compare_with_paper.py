import argparse,json
from pathlib import Path
PAPER={'BLEU1':.656,'BLEU2':.550,'BLEU3':.453,'BLEU4':.367,'METEOR':.251,'ROUGE-L':.543}
def compare(metrics):
    rows={k:{'reproduced':float(metrics[k]),'paper':v,'delta':float(metrics[k])-v,'abs_delta':abs(float(metrics[k])-v),'relative_delta':(float(metrics[k])-v)/v} for k,v in PAPER.items()}
    ae=[x['abs_delta'] for x in rows.values()]; mae=sum(ae)/len(ae); mx=max(ae)
    status='STRONG_REPRODUCTION_MATCH' if mae<=.020 and mx<=.040 else 'ACCEPTABLE_REPRODUCTION_MATCH' if mae<=.030 and mx<=.050 else 'PAPER_METRIC_MISMATCH'
    return {'metrics':rows,'MAE6':mae,'MAXAE6':mx,'classification':status}
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('metrics'); ap.add_argument('--out',required=True); a=ap.parse_args(); d=compare(json.loads(Path(a.metrics).read_text())); Path(a.out).write_text(json.dumps(d,indent=2))
