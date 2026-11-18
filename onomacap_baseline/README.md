# OnomaCap baseline reproduction

This directory contains the supervised environmental-sound-to-onomatopoeia
baseline used in the OnomaCap reproduction: an AudioSet-pretrained HTSAT
encoder, a learned audio projection, and a randomly initialized BART-base
encoder-decoder trained on five English onomatopoeia references per sound.

The code is independent of `../custom_eval/`. Model selection uses validation
METEOR, with BLEU-1 and earlier epoch as deterministic tie breakers. Final
metrics use the COCO-caption BLEU, METEOR, and ROUGE implementations.

## What is not included

This repository contains code plus the small BART-base configuration and
tokenizer assets required for offline initialization. It contains no model
weights. You must obtain the FiratESC audio, the sound-to-onomatopoeia
annotation CSV, the AudioSet HTSAT checkpoint, and the reproduced `best.pt`
separately and comply with their licenses.

## Expected local layout

All paths in `configs/onomacap_seed20.yaml` are relative to this directory.
Place external artifacts as follows:

```text
onomacap_baseline/
├── datasets/
│   └── firat_esc50/
│       ├── annotations/
│       │   └── sound-to-onomatopoeia_annotation.csv
│       └── audio/
│           ├── <audio_file values referenced by the CSV>
│           └── ...
├── checkpoints/
│   ├── pretrained/
│   │   └── HTSAT_AudioSet_Saved_1.ckpt
│   └── trained/
│       └── seed20/
│           └── best.pt
├── outputs/                         # generated locally; ignored by Git
├── assets/
│   └── bart-base/                   # config/tokenizer only; no model weights
├── configs/
├── data/                            # Python code, not dataset content
├── metrics/
├── models/
└── training/
```

The expected HTSAT checkpoint SHA256 is recorded in the config and verified
before training. The dataset manifest builder expects the annotation columns
`audio_file`, `class`, `id`, and `candidate1_en` through `candidate5_en`.

## Environment

```bash
cd onomacap_baseline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

METEOR is implemented by `pycocoevalcap` and requires Java on `PATH`. Java is
a system dependency and cannot be installed by `pip`. Install one of:

```bash
# Ubuntu/Debian
sudo apt-get install openjdk-17-jre-headless

# Conda
conda install -c conda-forge openjdk=17

java -version
```

The vendored BART config and tokenizer are loaded with
`local_files_only=True`; training and evaluation do not download BART assets
from Hugging Face. The BART model weights are initialized randomly, matching
the reproduction protocol.

The reproduced run used one NVIDIA RTX 3090 (24 GB), PyTorch 2.6.0, CUDA 12.4,
and seed 20. Other compatible CUDA GPUs can be selected through
`CUDA_VISIBLE_DEVICES`; the code contains no server-specific GPU index.

## Build the deterministic split

```bash
python -m data.build_manifest --config configs/onomacap_seed20.yaml
```

This validates the source files, removes the two frozen exclusions
`38_68.wav` and `38_95.wav`, and writes a deterministic class-stratified split:

- train: 6,370 audio files, expanded to 31,850 text targets
- validation: 796 audio files
- test: 796 audio files

Generated manifests are written under `outputs/` and are not committed.

## Train

Select one physical GPU by exposing it as logical `cuda:0`:

```bash
CUDA_VISIBLE_DEVICES=0 python -m training.train_v2 \
  --config configs/onomacap_seed20.yaml
```

Checkpoints are written to `outputs/runs/seed20/`. The default release config
saves `checkpoint_step_<N>.pt` every 5,000 optimizer steps and keeps the
validation-selected `best.pt`. Resume with:

```bash
CUDA_VISIBLE_DEVICES=0 python -m training.train_v2 \
  --config configs/onomacap_seed20.yaml \
  --resume outputs/runs/seed20/checkpoint_step_20000.pt
```

## Evaluate

```bash
CUDA_VISIBLE_DEVICES=0 python -m training.evaluate \
  --config configs/onomacap_seed20.yaml
```

The default released checkpoint is `checkpoints/trained/seed20/best.pt`;
predictions and test metrics are written to `outputs/runs/seed20/test/`.
Newly trained checkpoints remain under `outputs/runs/seed20/`. Evaluation
paths can be overridden with `--checkpoint`, `--manifest`, and `--output-dir`.
To evaluate a newly trained checkpoint instead of the distributed reproduction
checkpoint, run:

```bash
CUDA_VISIBLE_DEVICES=0 python -m training.evaluate \
  --config configs/onomacap_seed20.yaml \
  --checkpoint outputs/runs/seed20/best.pt
```

## Reproduction result

The released code was verified with a 20-epoch, 39,800-step seed-20 run. Epoch
9 was selected on validation. The selected checkpoint produced 796 test
predictions and was classified as a strong reproduction match.

| Metric | OnomaCap paper | Reproduced `best.pt` | Difference |
|---|---:|---:|---:|
| BLEU-1 | 0.6560 | 0.6447 | -0.0113 |
| BLEU-2 | 0.5500 | 0.5477 | -0.0023 |
| BLEU-3 | 0.4530 | 0.4555 | +0.0025 |
| BLEU-4 | 0.3670 | 0.3693 | +0.0023 |
| METEOR | 0.2510 | 0.2463 | -0.0047 |
| ROUGE-L | 0.5430 | 0.5313 | -0.0117 |

- Mean absolute error across the six metrics: 0.0058
- Maximum absolute error: 0.0117
- Reproduced checkpoint SHA256:
  `1a1a608fd5feffceac4c81540f645ace78d01c85be84bd4208b2c85a26941657`

The checkpoint itself is not committed to this repository.

## Tests

```bash
PYTHONPATH=. pytest -q tests
```
