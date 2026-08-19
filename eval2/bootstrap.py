from __future__ import annotations
import numpy as np
def class_stratified_indices(classes,reps=10000,seed=1337):
 rng=np.random.default_rng(seed); groups={}
 for i,c in enumerate(classes):groups.setdefault(c,[]).append(i)
 return [np.concatenate([rng.choice(v,len(v),replace=True) for v in groups.values()]) for _ in range(reps)]
def percentile_ci(values): return [float(np.percentile(values,2.5)),float(np.percentile(values,97.5))]
