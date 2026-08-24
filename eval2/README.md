# Eval2: fixed-all-label cross-lingual retrieval evaluation

Eval2 is the recommended evaluation protocol in this repository. It evaluates a
model's transcription of an unseen target audio without treating Korean or
Japanese spelling as directly comparable to English pseudo-label text.

## Firat ESC50 fixed test split

`manifests/firat_test_audio_paths_seed20.csv` is the public, label-blind list
of the **796** Firat test audio files used by Eval2. It contains the exact
`eval_audio_id`, class key, SHA-256, and audio path **relative to the root of
the downloaded Firat ESC50 dataset**. It does not redistribute audio or human
annotations.

The split is fixed by original audio (not by human annotation), using seed 20
and class-stratified 80/10/10 allocation:

| split | audio files | human labels |
| --- | ---: | ---: |
| train | 6,370 | 31,850 |
| validation | 796 | 3,980 |
| test | 796 | 3,980 |

The 7,962 annotated Firat audios span 41 annotated classes. Every audio in the
frozen manifest has five eligible human onomatopoeia labels. The CSV is for
inference and prediction-file construction only; official scoring additionally
needs the privately held human-label manifest built from the annotation source.

## Protocol

1. Run inference exactly once for every listed test audio. Write one rank-0
   prediction per `eval_audio_id` in the JSONL schema in
   [`docs/PREDICTION_FORMAT.md`](../docs/PREDICTION_FORMAT.md).
2. Freeze the checkpoint, decoder settings, and prediction JSONL. Record its
   SHA-256 **before** opening test human labels.
3. Convert the prediction and human labels to the common IPA-like phone space.
   `english_text` uses eSpeak English; `firat_romanized_korean` uses the Firat
   romanization lexicon; `rwcp_repo_phone`, `ipa_tokens`, and `arpabet_runs`
   have their corresponding deterministic mappings.
4. For each query audio, compare its predicted temporal phone sequence against
   every candidate test audio. A candidate's distance is the minimum normalized
   phone edit distance to **any of its five human labels**. Thus the primary
   protocol does not randomly select one label, nor does it make a bank of one
   exemplar per class.
5. Rank all 796 candidate audios globally (and, separately, candidates from the
   query's class). Ties are broken by `eval_audio_id`, so the point estimate is
   deterministic.

The primary result is **temporal global MRR** (larger is better). Report
R@1/R@5/R@10, median rank, min/mean normalized temporal and segmental PED, and
duration distance alongside it. Temporal sequences retain adjacent repeated
phones; segmental sequences collapse them before distance calculation.

Eval2 reports a 95% uncertainty interval using 10,000 seed-1337
class-stratified bootstrap resamples of the fixed 796 **query audios**. The
candidate bank and each query's rank stay fixed during this bootstrap. It is
test-set uncertainty, not stochastic re-decoding or label sampling. A separate
secondary annotation-sensitivity diagnostic samples k=1 and k=5 labels per
candidate for 100 deterministic seed-1337 replicates; it is not the primary
confidence interval.

The evaluator also reports output-collapse statistics (unique-output ratio,
top-output mass, entropy, and repeated-phone behavior). Do not use target test
labels for checkpoint selection, decoding choices, lexicon construction, or
post-processing.

## Run

```bash
python -m eval2.score_predictions \
  --manifest /secure/path/firat_human_label_manifest.jsonl \
  --predictions submissions/model/firat.jsonl \
  --prediction-format english_text \
  --output results/model/firat_eval2
```

The scorer verifies exact test-ID coverage and writes `metrics.json`, including
the prediction and manifest SHA-256 values. See
[`docs/EVALUATION.md`](../docs/EVALUATION.md) for the concise Eval1/Eval2
comparison.
