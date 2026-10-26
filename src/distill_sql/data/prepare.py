"""Turn extracted archives into processed, versioned artifacts.

Outputs under ``paths.processed_dir``:
  examples/<dataset>/<split>.jsonl   every question in a common format
  schemas/<dataset>/<db_id>.json     extracted schema per database
  registry.json                      every database with role (train/eval/excluded) + fingerprint
  manifest.json                      archive hashes and counts (what produced these files)
  reports/data_report.md             human-readable stats
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median

from tqdm import tqdm

from distill_sql import __version__
from distill_sql.config import AppConfig
from distill_sql.data.datasets import (
    ROLES,
    DatabaseEntry,
    Example,
    Registry,
    Role,
    db_path,
    load_examples,
)
from distill_sql.data.schema import DatabaseSchema, extract_schema
from distill_sql.utils.io import atomic_write_text, sha256_file, write_json, write_jsonl

log = logging.getLogger(__name__)


class SplitLeakError(RuntimeError):
    """A database is used by both a training split and an evaluation split."""


def _split_roles(cfg: AppConfig) -> list[tuple[str, str, Role]]:
    out: list[tuple[str, str, Role]] = []
    for name, ds in cfg.datasets.items():
        out += [(name, s, "train") for s in ds.train_splits]
        out += [(name, s, "eval") for s in ds.eval_splits]
    return out


def _exclusion_reason(cfg: AppConfig, dataset: str, db_id: str, n_rows: int) -> str | None:
    ds = cfg.datasets[dataset]
    if db_id in ds.exclude_dbs:
        return ds.exclude_dbs[db_id]
    if ds.exclude_empty_dbs and n_rows == 0:
        return "empty database: generated SQL cannot be execution-verified"
    return None


def prepare(cfg: AppConfig, root: Path) -> Registry:
    processed = cfg.paths.processed_dir
    examples_by_split: dict[tuple[str, str], list[Example]] = {}
    # (dataset, db_id) -> role, splits, path
    dbs: dict[tuple[str, str], tuple[Role, list[str], Path]] = {}

    for dataset, split, role in _split_roles(cfg):
        extract_dir = cfg.paths.raw_dir / cfg.datasets[dataset].source.extract_to
        examples = load_examples(extract_dir, dataset, split)
        examples_by_split[(dataset, split)] = examples
        write_jsonl(processed / "examples" / dataset / f"{split}.jsonl", examples)
        for db_id in sorted({e.db_id for e in examples}):
            key = (dataset, db_id)
            if key in dbs:
                prev_role, splits, _ = dbs[key]
                if prev_role != role:
                    raise SplitLeakError(
                        f"{dataset}/{db_id} is used by {role} split {split!r} and by "
                        f"{prev_role} split(s) {splits}"
                    )
                splits.append(split)
            else:
                dbs[key] = (role, [split], db_path(extract_dir, dataset, split, db_id))

    entries: list[DatabaseEntry] = []
    schemas: list[DatabaseSchema] = []
    failures: list[str] = []
    sc = cfg.schema_extraction
    for (dataset, db_id), (role, splits, path) in tqdm(sorted(dbs.items()), desc="schemas"):
        try:
            schema = extract_schema(path, dataset, db_id, sc.sample_rows, sc.max_value_chars)
        except Exception as exc:  # report every broken DB at once, then fail
            failures.append(f"{dataset}/{db_id} ({path}): {exc}")
            continue
        write_json(processed / "schemas" / dataset / f"{db_id}.json", schema.model_dump())
        schemas.append(schema)
        n_rows = sum(t.row_count for t in schema.tables)
        note = _exclusion_reason(cfg, dataset, db_id, n_rows) if role == "train" else None
        entries.append(
            DatabaseEntry(
                dataset=dataset,
                db_id=db_id,
                role="excluded" if note else role,
                note=note,
                splits=splits,
                path=str(path.relative_to(root)),
                fingerprint=schema.fingerprint,
                n_tables=len(schema.tables),
                n_columns=schema.n_columns,
                n_rows=n_rows,
            )
        )
    if failures:
        raise RuntimeError("schema extraction failed for:\n  " + "\n  ".join(failures))

    registry = Registry(databases=entries)
    write_json(processed / "registry.json", registry.model_dump(mode="json"))
    _write_manifest(cfg, examples_by_split, registry)
    atomic_write_text(
        processed / "reports" / "data_report.md", render_report(examples_by_split, registry)
    )
    return registry


def _write_manifest(
    cfg: AppConfig, examples: dict[tuple[str, str], list[Example]], registry: Registry
) -> None:
    archives = {}
    for name, ds in cfg.datasets.items():
        archive = cfg.paths.raw_dir / "downloads" / ds.source.filename
        archives[name] = {
            "file": ds.source.filename,
            "sha256": sha256_file(archive) if archive.exists() else None,
            "license": ds.license,
        }
    write_json(
        cfg.paths.processed_dir / "manifest.json",
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "distill_sql_version": __version__,
            "archives": archives,
            "splits": {f"{d}/{s}": len(ex) for (d, s), ex in examples.items()},
            "databases": {role: len(registry.by_role(role)) for role in ROLES},
        },
    )


def render_report(examples: dict[tuple[str, str], list[Example]], registry: Registry) -> str:
    by_split: dict[tuple[str, str], list[DatabaseEntry]] = defaultdict(list)
    for d in registry.databases:
        for s in d.splits:
            by_split[(d.dataset, s)].append(d)

    lines = [
        "# Data report",
        "",
        "| Dataset | Split | Role | Questions | DBs | Tables/DB (mean) | Columns/DB (mean) "
        "| Rows/DB (median) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for (dataset, split), exs in examples.items():
        dbs = by_split[(dataset, split)]
        role = "eval" if any(d.role == "eval" for d in dbs) else "train"
        lines.append(
            f"| {dataset} | {split} | {role} | {len(exs):,} | {len(dbs)} "
            f"| {mean(d.n_tables for d in dbs):.1f} | {mean(d.n_columns for d in dbs):.1f} "
            f"| {median(d.n_rows for d in dbs):,.0f} |"
        )
    n_train, n_eval = len(registry.by_role("train")), len(registry.by_role("eval"))
    excluded = registry.by_role("excluded")
    lines += [
        "",
        f"Unique databases: **{n_train} train**, **{n_eval} eval**, "
        f"**{len(excluded)} excluded** from training.",
        "",
    ]
    if excluded:
        lines += ["## Excluded training databases", "", "| Database | Reason |", "|---|---|"]
        lines += [f"| {d.key} | {d.note} |" for d in excluded]
        lines.append("")
    return "\n".join(lines)
