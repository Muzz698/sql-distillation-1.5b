"""Run records: every pipeline run gets ``runs/<timestamp>-<kind>/`` with its resolved config,
git commit and metrics, so any number in a report can be traced back to how it was made."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from distill_sql.config import PROJECT_ROOT, AppConfig
from distill_sql.utils.io import atomic_write_text, write_json


def git_state(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {"commit": git("rev-parse", "HEAD") or None, "dirty": bool(git("status", "--porcelain"))}


class Run:
    def __init__(self, cfg: AppConfig, kind: str, params: dict[str, Any] | None = None) -> None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.dir = cfg.paths.runs_dir / f"{stamp}-{kind}"
        self.dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            self.dir / "config.yaml",
            yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
        )
        write_json(
            self.dir / "meta.json",
            {"kind": kind, "started_at": stamp, "git": git_state(), "params": params or {}},
        )

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        write_json(self.dir / "metrics.json", metrics)
