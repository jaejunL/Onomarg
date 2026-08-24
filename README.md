# Onomarg

Reference code for supervised onomatopoeia transcription and label-aware, cross-lingual phonological evaluation.

## Contents

- `supervised/`: an OnomaCap-style supervised model: HTSAT acoustic encoder, learned audio projection, randomly initialized BART decoder, and beam-search transcription.
- `eval1/`: historical five-label-sampling evaluator.
- `eval2/`: recommended all-human-label evaluator with class-stratified bootstrap uncertainty and output-collapse diagnostics.

No audio, human annotation, checkpoint, prediction, or local experiment output is committed. These artifacts are intentionally ignored by git.

FiratESC test audio is available from the [Firat ESC-50 Kaggle dataset](https://www.kaggle.com/datasets/buraktaci/firat-esc50), and its human onomatopoeia label annotations are sourced from [jspirit01/sound-to-onomatopoeia](https://github.com/jspirit01/sound-to-onomatopoeia). Those data are not redistributed here.

For reproducible Firat inference, [`eval2/manifests/firat_test_audio_paths_seed20.csv`](eval2/manifests/firat_test_audio_paths_seed20.csv) lists the fixed 796 Eval2 test files as paths relative to the downloaded dataset root. The complete protocol is in [`eval2/README.md`](eval2/README.md).

The trained supervised OnomaCap-style checkpoint (`best.pt`, 1.9 GB) is available in the [Onomarg Dropbox checkpoint folder](https://www.dropbox.com/scl/fo/h8zgfrhbuwp22zpx2hl8i/AMbd31Qf_cr1YuivhLmGRwY?rlkey=8en6y3jpq3k8t0f32hz4bmhzh&st=7yj1o4g0&dl=0).

## Quick start: evaluate any model

```bash
git clone https://github.com/jaejunL/Onomarg.git
cd Onomarg
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Install the `espeak-ng` system package as required by `phonemizer` (including Korean voice support for Firat conversion). Then write one JSONL prediction file per target corpus according to [docs/PREDICTION_FORMAT.md](docs/PREDICTION_FORMAT.md).

```bash
python -m eval2.score_predictions --manifest /path/to/firat_manifest.jsonl --predictions submissions/my_model/firat.jsonl --prediction-format english_text --output results/my_model/firat
```

Use the same command with the RWCP manifest and RWCP predictions. See [docs/EVALUATION.md](docs/EVALUATION.md) for Eval1 versus Eval2 and all formats.

## Sharing predictions across lab models

The only required interchange artifact is one prediction JSONL per target corpus. Use the target manifest's exact `eval_audio_id`, write `beam_rank: 0`, and put decoded text in `prediction`:

```json
{"eval_audio_id":"firat:10_ (22).wav","eval_dataset":"firat","beam_rank":0,"prediction":"brrrroooom"}
```

Do not store a class label in place of transcription, do not rename audio IDs, and do not select a checkpoint after inspecting target test labels. Optional provenance fields are encouraged. The full schema and three-beam rule are in [docs/PREDICTION_FORMAT.md](docs/PREDICTION_FORMAT.md).

## Supervised OnomaCap-style baseline

`supervised/` is a compact deterministic reproduction of the supervised HTSAT+BART baseline. Copy `supervised/configs/onomacap.example.yaml`, fill only local data/checkpoint paths, then build the manifest and train from inside `supervised/`:

```bash
cd supervised
python -m data.build_manifest --config configs/onomacap.local.yaml
CUDA_VISIBLE_DEVICES=0 python -m training.train_v2 --config configs/onomacap.local.yaml
```

The example config intentionally has placeholders. Checkpoints, datasets, and their licenses must be acquired separately. Validation selection is METEOR first, BLEU-1 tie-break.

## Testing

```bash
pytest -q eval1/tests eval2/tests
PYTHONPATH=supervised pytest -q supervised/tests
```

Some supervised tests require data/checkpoint fixtures; Eval1/Eval2 tests are the fresh-clone CI target. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before redistributing or training with external assets.
