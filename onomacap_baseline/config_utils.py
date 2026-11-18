"""Load repository-relative configuration without embedding machine paths."""
from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
PATH_KEYS = {
    "annotation_csv",
    "audio_root",
    "htsat_checkpoint",
    "manifest_path",
    "run_dir",
    "test_output_dir",
}


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path) -> dict:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in PATH_KEYS:
        if config.get(key):
            config[key] = str(project_path(config[key]))
    return config
