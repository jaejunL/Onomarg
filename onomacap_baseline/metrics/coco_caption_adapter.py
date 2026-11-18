"""Python-3 COCO-caption metric adapter (pycocoevalcap 1.2 semantics)."""
from __future__ import annotations
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge

def evaluate(predictions, references):
    assert len(predictions)==len(references) and all(len(x)==5 for x in references)
    gts={i:list(refs) for i,refs in enumerate(references)}
    res={i:[predictions[i]] for i in range(len(predictions))}
    b,_=Bleu(4).compute_score(gts,res)
    m,_=Meteor().compute_score(gts,res)
    r,_=Rouge().compute_score(gts,res)
    return {'BLEU1':float(b[0]),'BLEU2':float(b[1]),'BLEU3':float(b[2]),'BLEU4':float(b[3]),'METEOR':float(m),'ROUGE-L':float(r)}
