from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np


def class_stratified_mean_ci(
    rows: Sequence[dict[str, Any]], metric: str, *, replicates: int = 10_000, seed: int = 1337
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["class_key"])].append(float(row[metric]))
    rng = np.random.Generator(np.random.PCG64(seed))
    observed = float(np.mean([value for values in grouped.values() for value in values]))
    samples = np.empty(replicates, dtype=np.float64)
    classes = sorted(grouped)
    for replicate in range(replicates):
        collected = []
        for class_key in classes:
            values = np.asarray(grouped[class_key], dtype=np.float64)
            collected.extend(rng.choice(values, size=len(values), replace=True).tolist())
        samples[replicate] = np.mean(collected)
    return {
        "mean": observed,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
    }

