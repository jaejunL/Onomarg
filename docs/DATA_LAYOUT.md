# Data and manifest layout

No dataset, annotation, checkpoint, or generated prediction is included in this repository. Keep them outside the git worktree.

The evaluator expects one JSONL manifest per corpus. Each row includes at least:

```json
{"eval_audio_id":"...","eval_dataset":"firat","class_key":"...","audio_path":"/path/to/audio.wav","split":"test","labels":[...]}
```

Official human-label evaluation additionally requires `labels` with `label_id`, `phone_tokens_temporal`, and `phone_tokens_segmental`. Use the corresponding release builder to construct manifests; do not hand-edit test labels.

For model-only inference, use a label-blind manifest containing only audio identity, dataset, class key, audio path, and split.
