from __future__ import annotations

from pathlib import Path

import pytest

from distill_sql.config import PROJECT_ROOT, AppConfig, DatasetConfig, PathsConfig, load_config
from distill_sql.data.datasets import Registry
from distill_sql.data.prepare import SplitLeakError, prepare
from distill_sql.utils.io import iter_jsonl, read_json
from tests.conftest import make_spider_layout


def _cfg(tmp_path: Path) -> AppConfig:
    base = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    spider = base.datasets["spider"].model_copy(
        update={"train_splits": ["train_spider"], "eval_splits": ["dev"]}
    )
    paths = PathsConfig(
        data_dir=Path("data"),
        raw_dir=Path("data/raw"),
        processed_dir=Path("data/processed"),
        cache_dir=Path("data/cache"),
        runs_dir=Path("runs"),
    ).resolved(tmp_path)
    datasets: dict[str, DatasetConfig] = {"spider": spider}
    return base.model_copy(update={"paths": paths, "datasets": datasets})


def test_prepare_builds_registry_with_roles(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    make_spider_layout(
        cfg.paths.raw_dir / "spider",
        {
            "train_spider": [("pets_1", "How many pets?", "SELECT count(*) FROM pets")],
            "dev": [("concert_singer", "How many singers?", "SELECT count(*) FROM singer")],
        },
    )
    registry = prepare(cfg, tmp_path)

    roles = {d.db_id: d.role for d in registry.databases}
    assert roles == {"pets_1": "train", "concert_singer": "eval"}
    processed = cfg.paths.processed_dir
    assert Registry.model_validate(read_json(processed / "registry.json")) == registry
    dev = list(iter_jsonl(processed / "examples" / "spider" / "dev.jsonl"))
    assert dev[0]["uid"] == "spider/dev/0"
    assert (processed / "schemas" / "spider" / "pets_1.json").exists()
    assert (processed / "manifest.json").exists()
    assert (
        "| spider | dev | eval | 1 | 1 |" in (processed / "reports" / "data_report.md").read_text()
    )
    # Paths in the registry are relative, so the registry is portable across machines.
    assert not Path(registry.databases[0].path).is_absolute()


def test_prepare_refuses_db_shared_by_train_and_eval(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    make_spider_layout(
        cfg.paths.raw_dir / "spider",
        {
            "train_spider": [("pets_1", "How many pets?", "SELECT count(*) FROM pets")],
            "dev": [("pets_1", "Heaviest pet?", "SELECT max(weight) FROM pets")],
        },
    )
    with pytest.raises(SplitLeakError, match="pets_1"):
        prepare(cfg, tmp_path)


def test_prepare_excludes_empty_and_configured_train_dbs(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    spider = cfg.datasets["spider"].model_copy(
        update={"exclude_dbs": {"pets_copy": "same database as an eval DB"}}
    )
    cfg = cfg.model_copy(update={"datasets": {"spider": spider}})
    make_spider_layout(
        cfg.paths.raw_dir / "spider",
        {
            "train_spider": [
                ("pets_1", "How many pets?", "SELECT count(*) FROM pets"),
                ("pets_copy", "How many pets?", "SELECT count(*) FROM pets"),
                ("empty_1", "How many rows?", "SELECT count(*) FROM t"),
            ],
            "dev": [("concert_singer", "How many singers?", "SELECT count(*) FROM singer")],
        },
    )
    registry = prepare(cfg, tmp_path)

    by_id = {d.db_id: d for d in registry.databases}
    assert [d.db_id for d in registry.by_role("train")] == ["pets_1"]
    assert by_id["pets_copy"].role == "excluded"
    assert by_id["pets_copy"].note == "same database as an eval DB"
    assert by_id["empty_1"].role == "excluded"
    assert "empty database" in (by_id["empty_1"].note or "")
    report = (cfg.paths.processed_dir / "reports" / "data_report.md").read_text()
    assert "**2 excluded**" in report
