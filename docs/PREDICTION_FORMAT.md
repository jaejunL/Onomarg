# Prediction-file contract

Every model writes a UTF-8 JSONL file for each target dataset. The evaluator joins rows by `eval_audio_id`; never depend on file order.

## Required fields

```json
{"eval_audio_id":"firat:10_ (22).wav","eval_dataset":"firat","beam_rank":0,"prediction":"boooooonk"}
```

| Field | Requirement |
|---|---|
| `eval_audio_id` | Exact identifier from the released target manifest. |
| `eval_dataset` | `firat` or `rwcp`. |
| `beam_rank` | Integer. Use `0` for a single decoded hypothesis. If exporting three beams, write exactly ranks `0`, `1`, and `2` for every audio. Scoring uses rank 0. |
| `prediction` | A Unicode string in the declared `prediction-format`. |

Extra provenance fields such as `checkpoint_sha256`, `generated_token_ids`, `decode_config`, and `model_id` are encouraged and ignored by the evaluator.

## Supported prediction formats

- `english_text`: ordinary English/Latin onomatopoeic text, e.g. `brrr`, `boooooonk`, `ding ding`.
- `firat_romanized_korean`: Firat-style romanized Korean output; requires the released Firat train romanization lexicon.
- `rwcp_repo_phone`: space-separated RWCP/Julius phone tokens, e.g. `b u w a:`.
- `ipa_tokens`: precomputed whitespace-separated IPA-like temporal tokens.
- `arpabet_runs`: a `runs` field rather than `prediction`; each run needs `phone` and `duration_sec`.

## Freeze before labels

Generate all target-test predictions using audio and a public label-blind manifest only, checksum the JSONL, then score against the human-label manifest. Do not tune checkpoints, decoding, or post-processing after reading target test labels.

For a three-beam submission:

```bash
python -m eval1.validate_predictions --manifest /path/to/firat_manifest.jsonl --predictions submissions/my_model/firat.jsonl
```

For rank-0-only output, use the evaluator directly; `validate_predictions.py` intentionally validates the historical three-beam contract.
