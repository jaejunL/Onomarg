"""Build the deterministic FiratESC split used by the OnomaCap reproduction."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from config_utils import load_config, project_path
from data.text_normalization import normalize_text


BAD_AUDIO_FILES = {"38_68.wav", "38_95.wav"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _class_allocation(rows, target):
    classes = sorted({int(row["class_id"]) for row in rows})
    counts = {class_id: sum(int(row["class_id"]) == class_id for row in rows) for class_id in classes}
    quota = {class_id: counts[class_id] * target / len(rows) for class_id in counts}
    allocation = {class_id: math.floor(value) for class_id, value in quota.items()}
    remaining = target - sum(allocation.values())
    order = sorted(counts, key=lambda class_id: (-(quota[class_id] - allocation[class_id]), class_id))
    for class_id in order[:remaining]:
        allocation[class_id] += 1
    return allocation


def build(config_path: str):
    config = load_config(config_path)
    output = project_path(config.get("output_root", "outputs"))
    annotation_csv = Path(config["annotation_csv"])
    audio_root = Path(config["audio_root"])
    with annotation_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        source = list(csv.DictReader(handle))
    required = ["audio_file", "class", "id"] + [f"candidate{index}_en" for index in range(1, 6)]
    if not source or any(key not in source[0] for key in required):
        raise ValueError(f"annotation CSV must contain columns: {required}")
    excluded = [row for row in source if row["audio_file"] in BAD_AUDIO_FILES]
    rows = [row for row in source if row["audio_file"] not in BAD_AUDIO_FILES]
    if len(rows) != 7962 or len({int(row["class"]) for row in rows}) != 41:
        raise ValueError("expected 7,962 valid annotations from 41 classes")
    if len({row["audio_file"] for row in rows}) != len(rows):
        raise ValueError("audio_file values must be unique")

    records = []
    for row in rows:
        audio_path = audio_root / row["audio_file"]
        references = [row[f"candidate{index}_en"] for index in range(1, 6)]
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        normalized = [normalize_text(value) for value in references]
        if not all(normalized):
            raise ValueError(f"empty normalized reference: {row['audio_file']}")
        records.append({
            "audio_id": row["audio_file"],
            "audio_file": row["audio_file"],
            "audio_path": str(audio_path),
            "audio_sha256": sha256_file(audio_path),
            "class_id": int(row["class"]),
            "source_id": row["id"],
            "references_raw_en": references,
            "references_normalized": normalized,
        })

    records.sort(key=lambda item: (item["class_id"], item["audio_file"]))
    generator = np.random.Generator(np.random.PCG64(int(config["seed"])))
    validation_allocation = _class_allocation(records, 796)
    test_allocation = validation_allocation.copy()
    by_class = {
        class_id: [row for row in records if row["class_id"] == class_id]
        for class_id in sorted({row["class_id"] for row in records})
    }
    for class_id, values in by_class.items():
        permutation = generator.permutation(len(values))
        test_count = test_allocation[class_id]
        validation_count = validation_allocation[class_id]
        for position, index in enumerate(permutation):
            values[index]["split"] = (
                "test" if position < test_count else
                "validation" if position < test_count + validation_count else
                "train"
            )

    manifest = output / "preflight/manifest.jsonl"
    split_manifest = output / "splits/split_manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    split_manifest.parent.mkdir(parents=True, exist_ok=True)
    serialized = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records)
    manifest.write_text(serialized, encoding="utf-8")
    split_manifest.write_text(serialized, encoding="utf-8")
    counts = {split: sum(row["split"] == split for row in records) for split in ("train", "validation", "test")}
    (output / "preflight/dataset_summary.json").write_text(json.dumps({
        "records": len(records), "classes": len(by_class), "counts": counts,
        "excluded": excluded, "csv_sha256": sha256_file(annotation_csv),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "splits/split_provenance.json").write_text(json.dumps({
        "seed": config["seed"], "rng": "numpy.PCG64",
        "train": 6370, "validation": 796, "test": 796,
        "filter": sorted(BAD_AUDIO_FILES),
    }, indent=2), encoding="utf-8")
    (output / "splits/split_counts_by_class.csv").write_text(
        "class_id,train,validation,test\n" + "".join(
            f"{class_id},"
            f"{sum(row['class_id'] == class_id and row['split'] == 'train' for row in records)},"
            f"{sum(row['class_id'] == class_id and row['split'] == 'validation' for row in records)},"
            f"{sum(row['class_id'] == class_id and row['split'] == 'test' for row in records)}\n"
            for class_id in sorted(by_class)
        ),
        encoding="utf-8",
    )
    (output / "preflight/_SUCCESS.json").write_text(json.dumps({
        "status": "COMPLETE", "records": len(records),
        "excluded": len(excluded), "counts": counts,
    }, indent=2), encoding="utf-8")
    print(json.dumps({
        "valid": len(records), "excluded": len(excluded),
        "classes": len(by_class), **counts,
        "expanded_train": counts["train"] * 5,
    }))
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    build(arguments.config)
