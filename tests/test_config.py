from pathlib import Path

import pytest
from pydantic import ValidationError

from distill_sql.config import PROJECT_ROOT, AppConfig, load_config

CONFIGS = PROJECT_ROOT / "configs"


def test_base_config_loads_and_anchors_paths() -> None:
    cfg = load_config(CONFIGS / "base.yaml")
    assert isinstance(cfg, AppConfig)
    assert cfg.model.base_id == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    assert cfg.paths.raw_dir == PROJECT_ROOT / "data" / "raw"
    assert cfg.paths.raw_dir.is_absolute()


def test_ci_config_extends_base() -> None:
    cfg = load_config(CONFIGS / "ci.yaml")
    assert cfg.seed == 42
    assert cfg.eval_regression is not None
    assert 0.0 <= cfg.eval_regression.min_execution_accuracy <= 1.0


def test_train_and_eval_splits_are_disjoint() -> None:
    cfg = load_config(CONFIGS / "base.yaml")
    for name, ds in cfg.datasets.items():
        assert not set(ds.train_splits) & set(ds.eval_splits), name


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(f"extends: {CONFIGS / 'base.yaml'}\nsede: 1\n")
    with pytest.raises(ValidationError):
        load_config(bad)


def test_no_secrets_in_yaml() -> None:
    for path in CONFIGS.glob("*.yaml"):
        text = path.read_text().lower()
        assert "api_key" not in text, f"{path.name} looks like it contains a secret"


def test_disabled_teachers_are_not_active_but_still_configured() -> None:
    cfg = load_config(CONFIGS / "base.yaml")
    active = {t.name for t in cfg.active_teachers}
    assert "gemini-flash" not in active and "gemini-flash-lite" in active
    assert cfg.teacher("gemini-flash").enabled is False  # still reportable
