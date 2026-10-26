"""Load Spider / BIRD Mini-Dev into one common format, and the database registry.

Every question becomes an :class:`Example`; every SQLite file becomes a :class:`DatabaseEntry`
with a ``role`` of ``train``, ``eval`` or ``excluded``. Later phases only ever ask the
registry for ``train`` databases, which is what keeps evaluation databases out of training.
``excluded`` databases come from a training split but must not be trained on (see ``note``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from distill_sql.utils.io import iter_jsonl, read_json

Role = Literal["train", "eval", "excluded"]
ROLES: tuple[Role, ...] = ("train", "eval", "excluded")


class Example(BaseModel):
    uid: str  # "<dataset>/<split>/<index>", stable across runs
    dataset: str
    split: str
    db_id: str
    question: str
    sql: str
    evidence: str | None = None  # BIRD's external-knowledge hint
    difficulty: str | None = None  # BIRD: simple/moderate/challenging; Spider: added in Phase 3


class DatabaseEntry(BaseModel):
    dataset: str
    db_id: str
    role: Role
    splits: list[str]
    path: str  # relative to the project root
    fingerprint: str
    n_tables: int
    n_columns: int
    n_rows: int
    note: str | None = None  # why an `excluded` database is excluded

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.db_id}"


class Registry(BaseModel):
    databases: list[DatabaseEntry]

    def by_role(self, role: Role) -> list[DatabaseEntry]:
        return [d for d in self.databases if d.role == role]

    def get(self, dataset: str, db_id: str) -> DatabaseEntry:
        for d in self.databases:
            if d.dataset == dataset and d.db_id == db_id:
                return d
        raise KeyError(f"{dataset}/{db_id} is not in the registry")


@dataclass(frozen=True)
class SplitLayout:
    """Where one split's questions and databases live inside the extracted archive."""

    questions_file: str
    db_dir: str
    parse: Callable[[dict[str, Any]], tuple[str, str, str, str | None, str | None]]


def _parse_spider(rec: dict[str, Any]) -> tuple[str, str, str, str | None, str | None]:
    return rec["db_id"], rec["question"], rec["query"], None, None


def _parse_bird(rec: dict[str, Any]) -> tuple[str, str, str, str | None, str | None]:
    evidence = rec.get("evidence") or None
    return rec["db_id"], rec["question"], rec["SQL"], evidence, rec.get("difficulty")


# Paths are relative to paths.raw_dir / <source.extract_to>.
LAYOUTS: dict[str, dict[str, SplitLayout]] = {
    "spider": {
        "train_spider": SplitLayout(
            "spider_data/train_spider.json", "spider_data/database", _parse_spider
        ),
        "train_others": SplitLayout(
            "spider_data/train_others.json", "spider_data/database", _parse_spider
        ),
        "dev": SplitLayout("spider_data/dev.json", "spider_data/database", _parse_spider),
        "test": SplitLayout("spider_data/test.json", "spider_data/test_database", _parse_spider),
    },
    "bird_minidev": {
        "minidev_sqlite": SplitLayout(
            "minidev/MINIDEV/mini_dev_sqlite.json", "minidev/MINIDEV/dev_databases", _parse_bird
        ),
    },
}


def layout(dataset: str, split: str) -> SplitLayout:
    try:
        return LAYOUTS[dataset][split]
    except KeyError:
        raise KeyError(f"no known layout for {dataset}/{split}") from None


def db_path(extract_dir: Path, dataset: str, split: str, db_id: str) -> Path:
    return extract_dir / layout(dataset, split).db_dir / db_id / f"{db_id}.sqlite"


def load_registry(processed_dir: Path) -> Registry:
    path = processed_dir / "registry.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `make data` first.")
    return Registry.model_validate(read_json(path))


def load_processed_examples(processed_dir: Path, dataset: str, splits: list[str]) -> list[Example]:
    return [
        Example.model_validate(row)
        for split in splits
        for row in iter_jsonl(processed_dir / "examples" / dataset / f"{split}.jsonl")
    ]


def load_examples(extract_dir: Path, dataset: str, split: str) -> list[Example]:
    spec = layout(dataset, split)
    records = read_json(extract_dir / spec.questions_file)
    examples = []
    for i, rec in enumerate(records):
        db_id, question, sql, evidence, difficulty = spec.parse(rec)
        examples.append(
            Example(
                uid=f"{dataset}/{split}/{i}",
                dataset=dataset,
                split=split,
                db_id=db_id,
                question=question.strip(),
                sql=sql.strip(),
                evidence=evidence,
                difficulty=difficulty,
            )
        )
    return examples
