from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

SINGER_DDL = [
    "CREATE TABLE singer (singer_id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER)",
    "CREATE TABLE concert (concert_id INTEGER PRIMARY KEY, title TEXT)",
    """CREATE TABLE "singer in concert" (
        concert_id INTEGER, singer_id INTEGER,
        PRIMARY KEY (concert_id, singer_id),
        FOREIGN KEY (concert_id) REFERENCES concert(concert_id),
        FOREIGN KEY (singer_id) REFERENCES singer(singer_id))""",
    "INSERT INTO singer VALUES (1, 'Joe', 52), (2, 'Ann', 31), (3, 'Li', 44), (4, 'Bo', 29)",
    "INSERT INTO concert VALUES (1, 'Summer'), (2, 'Winter')",
    """INSERT INTO "singer in concert" VALUES (1, 1), (1, 2), (2, 3)""",
]

PETS_DDL = [
    "CREATE TABLE pets (pet_id INTEGER PRIMARY KEY, kind TEXT, weight REAL)",
    "INSERT INTO pets VALUES (1, 'dog', 12.5), (2, 'cat', 4.1)",
]


def make_db(path: Path, statements: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    for stmt in statements:
        conn.execute(stmt)
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def singer_db(tmp_path: Path) -> Path:
    return make_db(tmp_path / "concert_singer.sqlite", SINGER_DDL)


def make_spider_layout(root: Path, splits: dict[str, list[tuple[str, str, str]]]) -> Path:
    """Create a miniature Spider archive layout. ``splits``: split -> [(db_id, question, sql)]."""
    base = root / "spider_data"
    files = {
        "train_spider": "train_spider.json",
        "dev": "dev.json",
    }
    ddl_for = {
        "concert_singer": SINGER_DDL,
        "pets_1": PETS_DDL,
        "pets_copy": PETS_DDL,
        "empty_1": ["CREATE TABLE t (a INTEGER)"],
    }
    for split, rows in splits.items():
        (base / files[split]).parent.mkdir(parents=True, exist_ok=True)
        (base / files[split]).write_text(
            json.dumps([{"db_id": d, "question": q, "query": s} for d, q, s in rows])
        )
        for db_id in {r[0] for r in rows}:
            if (base / "database" / db_id).exists():
                continue
            make_db(base / "database" / db_id / f"{db_id}.sqlite", ddl_for[db_id])
    return root
