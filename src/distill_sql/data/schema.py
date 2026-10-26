"""Extract a typed schema (tables, columns, keys, sample rows) from a SQLite database.

The schema is used for three things:
1. Prompts: rendered as ``CREATE TABLE`` statements plus a few example rows.
2. Contamination checks: a fingerprint that is the same for two copies of one database,
   even if the copies have different names.
3. Data stats: table / column / row counts.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

_SIMPLE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class Column(BaseModel):
    name: str
    type: str
    primary_key: bool = False
    not_null: bool = False


class ForeignKey(BaseModel):
    columns: list[str]
    ref_table: str
    ref_columns: list[str]


class Table(BaseModel):
    name: str
    columns: list[Column]
    foreign_keys: list[ForeignKey] = Field(default_factory=list)
    row_count: int
    sample_rows: list[list[Any]] = Field(default_factory=list)


class DatabaseSchema(BaseModel):
    dataset: str
    db_id: str
    tables: list[Table]

    @property
    def n_columns(self) -> int:
        return sum(len(t.columns) for t in self.tables)

    @property
    def fingerprint(self) -> str:
        """Hash of normalised table and column names. Independent of db_id, order and data."""
        layout = sorted(
            (_norm(t.name), sorted(_norm(c.name) for c in t.columns)) for t in self.tables
        )
        return hashlib.sha256(json.dumps(layout).encode()).hexdigest()[:16]

    def to_ddl(self, include_samples: bool = True) -> str:
        return "\n\n".join(_table_ddl(t, include_samples) for t in self.tables)


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def quote_ident(name: str) -> str:
    if _SIMPLE_IDENT.match(name):
        return name
    return '"' + name.replace('"', '""') + '"'


def _table_ddl(table: Table, include_samples: bool) -> str:
    pk_cols = [c.name for c in table.columns if c.primary_key]
    lines = []
    for col in table.columns:
        parts = [quote_ident(col.name)]
        if col.type:
            parts.append(col.type)
        if col.primary_key and len(pk_cols) == 1:
            parts.append("PRIMARY KEY")
        elif col.not_null:
            parts.append("NOT NULL")
        lines.append("  " + " ".join(parts))
    if len(pk_cols) > 1:
        lines.append(f"  PRIMARY KEY ({', '.join(quote_ident(c) for c in pk_cols)})")
    for fk in table.foreign_keys:
        src = ", ".join(quote_ident(c) for c in fk.columns)
        dst = ", ".join(quote_ident(c) for c in fk.ref_columns)
        lines.append(f"  FOREIGN KEY ({src}) REFERENCES {quote_ident(fk.ref_table)}({dst})")
    ddl = f"CREATE TABLE {quote_ident(table.name)} (\n" + ",\n".join(lines) + "\n);"

    if include_samples and table.sample_rows:
        header = " | ".join(c.name for c in table.columns)
        rows = "\n".join(" | ".join(_cell(v) for v in row) for row in table.sample_rows)
        ddl += f"\n/* {len(table.sample_rows)} example rows:\n{header}\n{rows}\n*/"
    return ddl


def _cell(value: Any) -> str:
    return "NULL" if value is None else str(value)


def connect_readonly(path: Path) -> sqlite3.Connection:
    """Open a SQLite file so that no statement can modify it."""
    if not path.is_file():
        raise FileNotFoundError(path)
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, check_same_thread=False)
    # Some Spider databases contain invalid UTF-8; never crash on a bad byte.
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    conn.execute("PRAGMA query_only = ON")
    return conn


def _to_prompt_value(value: Any, max_chars: int) -> Any:
    if isinstance(value, bytes):
        return f"<blob {len(value)} bytes>"
    if isinstance(value, str) and len(value) > max_chars:
        return value[: max_chars - 3] + "..."
    return value


def extract_schema(
    db_path: Path,
    dataset: str,
    db_id: str,
    sample_rows: int = 3,
    max_value_chars: int = 60,
) -> DatabaseSchema:
    conn = connect_readonly(db_path)
    try:
        names = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
            )
        ]
        tables = []
        for name in names:
            q = quote_ident(name)
            # PRAGMA table_info: (cid, name, type, notnull, default, pk)
            cols = [
                Column(name=r[1], type=r[2] or "", not_null=bool(r[3]), primary_key=r[5] > 0)
                for r in conn.execute(f"PRAGMA table_info({q})")
            ]
            # PRAGMA foreign_key_list: (id, seq, table, from, to, ...); group composite keys by id
            fks: dict[int, ForeignKey] = {}
            for r in conn.execute(f"PRAGMA foreign_key_list({q})"):
                fk = fks.setdefault(r[0], ForeignKey(columns=[], ref_table=r[2], ref_columns=[]))
                fk.columns.append(r[3])
                fk.ref_columns.append(r[4] or "")
            row_count = conn.execute(f"SELECT COUNT(*) FROM {q}").fetchone()[0]
            samples = [
                [_to_prompt_value(v, max_value_chars) for v in row]
                for row in conn.execute(f"SELECT * FROM {q} LIMIT ?", (sample_rows,))
            ]
            tables.append(
                Table(
                    name=name,
                    columns=cols,
                    foreign_keys=list(fks.values()),
                    row_count=row_count,
                    sample_rows=samples,
                )
            )
    finally:
        conn.close()
    return DatabaseSchema(dataset=dataset, db_id=db_id, tables=tables)
