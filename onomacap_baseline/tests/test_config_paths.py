from config_utils import PROJECT_ROOT, load_config


def test_release_config_paths_are_project_relative():
    config = load_config(PROJECT_ROOT / "configs/onomacap_seed20.yaml")
    for key in ("annotation_csv", "audio_root", "htsat_checkpoint", "manifest_path", "run_dir", "test_output_dir"):
        assert str(config[key]).startswith(str(PROJECT_ROOT))


def test_release_config_uses_sparse_checkpoints():
    config = load_config(PROJECT_ROOT / "configs/onomacap_seed20.yaml")
    assert config["checkpoint_interval"] == 5000
