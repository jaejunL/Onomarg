# Evaluation guide

Eval1 and Eval2 share phone conversion and target manifests, but answer different uncertainty questions.

- **Eval1** samples five stable human labels per candidate over 100 deterministic annotation replicates. It is the historical compatibility evaluator.
- **Eval2** uses every eligible human label per candidate and reports 10,000 class-stratified query-bootstrap replicates. It is the recommended primary result.

Both map cross-lingual outputs into a common IPA-like phone space, report temporal and segmental views, paired phone edit distance, duration distance, and closed-set retrieval. They are not Korean/Japanese orthographic WER/CER.

## Eval1

```bash
python -m eval1.evaluate --manifest /path/to/firat_manifest.jsonl --predictions submissions/my_model/firat.jsonl --prediction-format english_text --output results/my_model/eval1/firat
```

## Eval2

```bash
python -m eval2.score_predictions --manifest /path/to/firat_manifest.jsonl --predictions submissions/my_model/firat.jsonl --prediction-format english_text --output results/my_model/eval2/firat
```

For `firat_romanized_korean`, add `--firat-lexicon /path/to/firat_train_romanization_lexicon.json`. For `arpabet_runs`, keep the default ARPAbet map or pass a compatible map with `--arpabet-map`.
