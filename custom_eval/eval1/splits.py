from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np


def largest_remainder_allocation(rows: list[dict[str, Any]], field: str, target: int) -> dict[str, int]:
    counts = Counter(str(row[field]) for row in rows)
    quotas = {key: value * target / len(rows) for key, value in counts.items()}
    allocation = {key: math.floor(value) for key, value in quotas.items()}
    remaining = target - sum(allocation.values())
    order = sorted(counts, key=lambda key: (-(quotas[key] - allocation[key]), key))
    for key in order[:remaining]:
        allocation[key] += 1
    return allocation


def assign_stratified_80_10_10(
    rows: list[dict[str, Any]], *, seed: int = 20, stratum_field: str = "class_key"
) -> list[dict[str, Any]]:
    """Match the frozen Firat split algorithm: global 10% targets, PCG64, class strata."""
    total = len(rows)
    n_test = round(total * 0.1)
    n_validation = round(total * 0.1)
    test_alloc = largest_remainder_allocation(rows, stratum_field, n_test)
    validation_alloc = largest_remainder_allocation(rows, stratum_field, n_validation)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[stratum_field])].append(row)
    rng = np.random.Generator(np.random.PCG64(seed))
    for key in sorted(grouped):
        values = sorted(grouped[key], key=lambda row: str(row["eval_audio_id"]))
        permutation = rng.permutation(len(values))
        nt, nv = test_alloc[key], validation_alloc[key]
        for rank, index in enumerate(permutation):
            values[int(index)]["split"] = "test" if rank < nt else "validation" if rank < nt + nv else "train"
    actual = Counter(row["split"] for row in rows)
    expected = {"test": n_test, "validation": n_validation, "train": total - n_test - n_validation}
    if dict(actual) != expected:
        raise RuntimeError({"actual": dict(actual), "expected": expected})
    return rows

