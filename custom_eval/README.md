# Custom onomatopoeia evaluation

This directory preserves the original Onomarg evaluation package. It is
independent of the supervised OnomaCap baseline in `../onomacap_baseline/` and
was not used to select or validate that baseline checkpoint.

## Contents

- `eval1/`: historical five-label-sampling evaluator.
- `eval2/`: recommended all-human-label evaluator with class-stratified
  bootstrap uncertainty and output-collapse diagnostics.
- `docs/`: data layout, prediction schema, and evaluation protocol.
- `examples/`: example Firat and RWCP prediction files.

No audio, human annotation, checkpoint, or experiment output is committed.

## Installation

Run commands from this directory so that the original `eval1` and `eval2`
module paths remain valid:

```bash
cd custom_eval
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the `espeak-ng` system package as required by `phonemizer`, including
Korean voice support for Firat conversion.

## Evaluation

Prepare prediction JSONL according to
[`docs/PREDICTION_FORMAT.md`](docs/PREDICTION_FORMAT.md), then run:

```bash
python -m eval2.score_predictions \
  --manifest /path/to/firat_manifest.jsonl \
  --predictions /path/to/firat_predictions.jsonl \
  --prediction-format english_text \
  --output /path/to/results/firat
```

For reproducible Firat inference,
[`eval2/manifests/firat_test_audio_paths_seed20.csv`](eval2/manifests/firat_test_audio_paths_seed20.csv)
contains the fixed 796 Eval2 test paths relative to the downloaded dataset
root. See [`eval2/README.md`](eval2/README.md) and
[`docs/EVALUATION.md`](docs/EVALUATION.md) for the complete protocol.

## Tests

```bash
pytest -q eval1/tests eval2/tests
```
