from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class OnomaDataset(Dataset):
    def __init__(self, manifest, split, tokenizer, train=False, max_length=30, max_audio_sec=10, seed=20):
        rows = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.rows = [row for row in rows if row["split"] == split]
        self.train = train
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_samples = 32000 * max_audio_sec
        self.seed = seed
        self.items = []
        for row in self.rows:
            if train:
                for index, text in enumerate(row["references_normalized"]):
                    self.items.append((row, index, text))
            else:
                self.items.append((row, None, row["references_normalized"]))

    def __len__(self):
        return len(self.items)

    def _wave(self, path, index):
        wave, _ = librosa.load(path, sr=32000, mono=True)
        if len(wave) > self.max_samples:
            if self.train:
                generator = np.random.default_rng(self.seed + index)
                start = int(generator.integers(0, len(wave) - self.max_samples + 1))
            else:
                start = 0
            wave = wave[start:start + self.max_samples]
        return torch.from_numpy(np.asarray(wave, dtype=np.float32))

    def __getitem__(self, index):
        row, target_index, target = self.items[index]
        wave = self._wave(row["audio_path"], index)
        if self.train:
            tokens = self.tokenizer(
                target, padding=False, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            return {
                "audio": wave,
                "audio_id": row["audio_id"],
                "class_id": row["class_id"],
                "target_index": target_index + 1,
                "input_ids": tokens.input_ids[0],
                "attention_mask": tokens.attention_mask[0],
                "target": target,
            }
        return {
            "audio": wave,
            "audio_id": row["audio_id"],
            "class_id": row["class_id"],
            "references": target,
        }


def train_collate(batch, pad_id):
    audio_length = max(item["audio"].numel() for item in batch)
    waves = [F.pad(item["audio"], (0, audio_length - item["audio"].numel())) for item in batch]
    text_length = max(item["input_ids"].numel() for item in batch)
    ids, masks = [], []
    for item in batch:
        difference = text_length - item["input_ids"].numel()
        ids.append(F.pad(item["input_ids"], (0, difference), value=pad_id))
        masks.append(F.pad(item["attention_mask"], (0, difference), value=0))
    labels = torch.stack(ids)
    labels[labels == pad_id] = -100
    return {
        "audio": torch.stack(waves),
        "labels": labels,
        "attention_mask": torch.stack(masks),
        "meta": batch,
    }


def eval_collate(batch):
    length = max(item["audio"].numel() for item in batch)
    return {
        "audio": torch.stack([F.pad(item["audio"], (0, length - item["audio"].numel())) for item in batch]),
        "meta": batch,
    }
