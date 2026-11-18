from __future__ import annotations

import json
import multiprocessing as mp
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval1.phonology import phone_edit_distance
from eval1.retrieval import _metrics, _rank, sampled_label_sets, unique_labels

ROOT = Path(__file__).resolve().parent
PAIRS = [
    ("EXP4_E0B", "EXP4_E0A"),
    ("EXP5_E1", "EXP4_E0B"),
    ("EXP5_E1", "EXP5_E1S"),
    ("EXP5_E2", "EXP4_E0B"),
    ("EXP5_E2", "EXP5_E2S"),
    ("EXP6_E3P_B", "EXP4_E0B"),
    ("EXP6_E3T_B", "EXP6_E3P_B"),
    ("EXP6_E3T_B", "EXP4_E0B"),
    ("EXP6_E3T_B", "EXP6_E3J"),
    ("EXP7_VIM_ONLY", "EXP6_E3T_B"),
    ("EXP7_ESC_ONLY", "EXP6_E3T_B"),
]
BRIDGES = [
    "EXP4_E0A", "EXP4_E0B", "EXP5_E1", "EXP5_E1S", "EXP5_E2",
    "EXP5_E2S", "EXP6_E3P_B", "EXP6_E3T_B", "EXP6_E3J",
    "EXP7_VIM_ONLY", "EXP7_ESC_ONLY",
]
METRICS = (
    "paired_min_ped_temporal", "paired_mean_ped_temporal",
    "paired_min_ped_segmental", "paired_mean_ped_segmental",
    "paired_min_duration_distance",
)
RETRIEVAL_METRICS = ("R@1", "R@5", "R@10", "MRR", "median_rank")


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def paired_delta(left, right, metric):
    lm = {x["eval_audio_id"]: x for x in left}
    rm = {x["eval_audio_id"]: x for x in right}
    ids = sorted(set(lm) & set(rm))
    groups = defaultdict(list)
    for aid in ids:
        groups[str(lm[aid]["class_key"])].append(float(lm[aid][metric]) - float(rm[aid][metric]))
    observed = float(np.mean([v for group in groups.values() for v in group]))
    rng = np.random.Generator(np.random.PCG64(1337))
    samples = []
    for _ in range(10_000):
        samples.append(np.mean([x for key in sorted(groups) for x in rng.choice(groups[key], size=len(groups[key]), replace=True)]))
    return {"mean_delta": observed, "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))], "replicates": 10_000, "seed": 1337}


def _paired_job(args):
    key, left, right, metric = args
    return key, metric, paired_delta(left, right, metric)


def one_retrieval(pred, manifest, view, replicate):
    label_sets = sampled_label_sets(manifest, view=view, replicate=replicate, seed=1337, k=5)
    eligible = sorted(set(label_sets) & set(pred))
    by_class = defaultdict(list)
    by_id = {x["eval_audio_id"]: x for x in manifest}
    for aid in eligible:
        by_class[str(by_id[aid]["class_key"])].append(aid)
    global_ranks = []
    within = []
    for aid in eligible:
        global_ranks.append(_rank(tuple(pred[aid]), aid, eligible, label_sets)[0])
        class_ids = sorted(by_class[str(by_id[aid]["class_key"])])
        if len(class_ids) >= 2:
            within.append(_rank(tuple(pred[aid]), aid, class_ids, label_sets)[0])
    return {"global": _metrics(global_ranks), "within_class": _metrics(within)}


_DISTANCE_LABELS = []


def _distance_row(query):
    return [phone_edit_distance(query, label) for label in _DISTANCE_LABELS]


_FAST_DISTANCE = None
_FAST_LABEL_INDEX = {}
_FAST_MANIFEST = []
_FAST_VIEW = ""
_FAST_AUDIO_IDS = []
_FAST_QUERY_INDEX = {}
_FAST_CLASSES = None
_FAST_LOWER = None


def _fast_retrieval_replicate(replicate):
    label_sets = sampled_label_sets(_FAST_MANIFEST, view=_FAST_VIEW, replicate=replicate, seed=1337, k=5)
    label_indices = np.asarray(
        [[_FAST_LABEL_INDEX[tuple(label)] for label in label_sets[aid]] for aid in _FAST_AUDIO_IDS],
        dtype=np.int32,
    )
    output = {}
    same_class = _FAST_CLASSES[:, None] == _FAST_CLASSES[None, :]
    class_sizes = same_class.sum(axis=1)
    within_rows = class_sizes >= 2
    for model, query_indices in _FAST_QUERY_INDEX.items():
        scores = _FAST_DISTANCE[query_indices[:, None, None], label_indices[None, :, :]].min(axis=2)
        true_scores = np.diag(scores)
        lower_ties = (scores == true_scores[:, None]) & _FAST_LOWER
        global_ranks = 1 + (scores < true_scores[:, None]).sum(axis=1) + lower_ties.sum(axis=1)
        within_ranks = 1 + ((scores < true_scores[:, None]) & same_class).sum(axis=1) + (lower_ties & same_class).sum(axis=1)
        output[model] = {
            "global": _metrics(global_ranks.tolist()),
            "within_class": _metrics(within_ranks[within_rows].tolist()),
        }
    return replicate, output


def retrieval_cache(model_rows, manifest, view, workers):
    global _DISTANCE_LABELS, _FAST_DISTANCE, _FAST_LABEL_INDEX, _FAST_MANIFEST
    global _FAST_VIEW, _FAST_AUDIO_IDS, _FAST_QUERY_INDEX, _FAST_CLASSES, _FAST_LOWER

    predictions = {
        model: {row["eval_audio_id"]: tuple(row[f"phone_tokens_{view}"]) for row in values}
        for model, values in model_rows.items()
    }
    sampled_zero = sampled_label_sets(manifest, view=view, replicate=0, seed=1337, k=5)
    common_ids = set(sampled_zero)
    for values in predictions.values():
        common_ids &= set(values)
    audio_ids = sorted(common_ids)

    queries = sorted({values[aid] for values in predictions.values() for aid in audio_ids})
    labels = sorted({tuple(label[f"phone_tokens_{view}"]) for row in manifest for label in row["labels"] if label[f"phone_tokens_{view}"]})
    _DISTANCE_LABELS = labels
    if workers > 1:
        with mp.get_context("fork").Pool(processes=min(workers, len(queries))) as pool:
            distance_rows = pool.map(_distance_row, queries, chunksize=1)
    else:
        distance_rows = [_distance_row(query) for query in queries]
    distance = np.asarray(distance_rows, dtype=np.float64)
    query_map = {query: index for index, query in enumerate(queries)}
    label_map = {label: index for index, label in enumerate(labels)}
    by_id = {row["eval_audio_id"]: row for row in manifest}

    _FAST_DISTANCE = distance
    _FAST_LABEL_INDEX = label_map
    _FAST_MANIFEST = manifest
    _FAST_VIEW = view
    _FAST_AUDIO_IDS = audio_ids
    _FAST_QUERY_INDEX = {
        model: np.asarray([query_map[values[aid]] for aid in audio_ids], dtype=np.int32)
        for model, values in predictions.items()
    }
    _FAST_CLASSES = np.asarray([str(by_id[aid]["class_key"]) for aid in audio_ids], dtype=object)
    _FAST_LOWER = np.tri(len(audio_ids), len(audio_ids), k=-1, dtype=bool)

    jobs = list(range(100))
    if workers > 1:
        with mp.get_context("fork").Pool(processes=min(workers, len(jobs))) as pool:
            values = pool.map(_fast_retrieval_replicate, jobs, chunksize=1)
    else:
        values = [_fast_retrieval_replicate(job) for job in jobs]
    cache = defaultdict(dict)
    for replicate, model_values in values:
        for model, value in model_values.items():
            cache[model][replicate] = value
    return cache

def retrieval_delta_from_cache(cache, left, right):
    result = {}
    for section in ("global", "within_class"):
        result[section] = {}
        for metric in RETRIEVAL_METRICS:
            values = np.asarray([cache[left][rep][section][metric] - cache[right][rep][section][metric] for rep in range(100)])
            result[section][metric] = {
                "mean_delta": float(np.mean(values)),
                "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
            }
    return {"replicates": 100, "seed": 1337, **result}


def main():
    audit = json.loads((ROOT / "outputs/registry_audit.json").read_text())
    eligible = {x["id"] for x in audit["models"] if x["eligible"]}
    workers = int(os.environ.get("EVAL1_WORKERS", "128"))
    out = {}
    for dataset_name in ("firat", "rwcp"):
        manifest = [x for x in rows(ROOT / "outputs/manifests" / f"{dataset_name}.jsonl") if x["split"] == "test"]
        pairs = list(PAIRS)
        anchor = "ONOMACAP_FIRAT_SUPERVISED" if dataset_name == "firat" else "EXP8_R3"
        pairs += [(model, anchor) for model in BRIDGES]
        pairs.append(("ONOMACAP_FIRAT_SUPERVISED", "EXP8_R3"))
        valid_pairs = []
        model_rows = {}
        for left, right in pairs:
            left_path = ROOT / "outputs/evaluations" / left / dataset_name / "per_query.jsonl"
            right_path = ROOT / "outputs/evaluations" / right / dataset_name / "per_query.jsonl"
            if left not in eligible or right not in eligible or not left_path.is_file() or not right_path.is_file():
                continue
            if left not in model_rows:
                model_rows[left] = rows(left_path)
            if right not in model_rows:
                model_rows[right] = rows(right_path)
            valid_pairs.append((left, right))

        paired_jobs = []
        for left, right in valid_pairs:
            key = f"{left}_minus_{right}"
            paired_jobs.extend((key, model_rows[left], model_rows[right], metric) for metric in METRICS)
        if workers > 1:
            with mp.get_context("fork").Pool(processes=min(workers, len(paired_jobs))) as pool:
                paired_values = pool.map(_paired_job, paired_jobs, chunksize=1)
        else:
            paired_values = [_paired_job(job) for job in paired_jobs]
        paired_by_key = defaultdict(dict)
        for key, metric, value in paired_values:
            paired_by_key[key][metric] = value

        retrieval_by_view = {
            view: retrieval_cache(model_rows, manifest, view, workers)
            for view in ("temporal", "segmental")
        }
        dataset = {}
        for left, right in valid_pairs:
            key = f"{left}_minus_{right}"
            dataset[key] = {
                "paired": paired_by_key[key],
                "retrieval": {
                    view: retrieval_delta_from_cache(retrieval_by_view[view], left, right)
                    for view in ("temporal", "segmental")
                },
            }
        out[dataset_name] = dataset

    target = ROOT / "outputs/comparisons"
    target.mkdir(parents=True, exist_ok=True)
    (target / "comparisons.json").write_text(json.dumps({"status": "COMPLETE", "comparisons": out}, indent=2, sort_keys=True) + "\n")
    (target / "_SUCCESS.json").write_text(json.dumps({"status": "COMPLETE", "datasets": {k: len(v) for k, v in out.items()}}, indent=2) + "\n")


if __name__ == "__main__":
    main()
