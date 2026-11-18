# Onomarg

Onomarg contains two independent codebases for research on environmental-sound
onomatopoeia transcription:

- [`onomacap_baseline/`](onomacap_baseline/): a deterministic reproduction of
  the supervised HTSAT+BART baseline reported in
  [OnomaCap](https://jspirit01.github.io/onomacap/).
- [`custom_eval/`](custom_eval/): the original label-aware, cross-lingual
  phonological evaluation code previously published in this repository.

The custom evaluator is retained for existing users and experiments, but it is
not a dependency of the OnomaCap baseline and was not used for baseline model
selection or reproduction validation. Each directory is a self-contained
project with its own README and requirements file.

## Repository policy

This repository distributes source code only. It does not contain datasets,
audio, human annotations, pretrained weights, trained checkpoints, prediction
files, or local experiment outputs. Follow the layout documented in each
subproject after obtaining those artifacts from their original sources.

## Quick links

- [Train and evaluate the OnomaCap baseline](onomacap_baseline/README.md)
- [Run the custom evaluator](custom_eval/README.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
