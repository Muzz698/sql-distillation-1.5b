"""Run SQL safely against SQLite: read-only, time-limited and row-capped.

Shared by data verification (Phase 2) and the evaluation harness (Phase 3), so "this query
is correct" means the same thing in both places.
"""

from __future__ import annotations

import hashlib
import math
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from distill_sql.data.schema import connect_readonly

Row = tuple[Any, ...]


@dataclass(frozen=True)
class ExecResult:
    ok: bool
    rows: list[Row]
    columns: list[str]
    truncated: bool  # more than max_rows rows existed
    elapsed_s: float
    error: str | None = None
    timed_out: bool = False

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def all_null(self) -> bool:
        return bool(self.rows) and all(v is None for row in self.rows for v in row)

    def result_hash(self) -> str:
        """Order-insensitive fingerprint of the result (a hash of the row multiset)."""
        rows = sorted(repr(_canon_row(r)) for r in self.rows)
        return hashlib.sha256("\n".join(rows).encode()).hexdigest()[:16]


class DatabasePool:
    """One read-only connection per database file, reused across queries."""

    def __init__(self) -> None:
        self._conns: dict[Path, sqlite3.Connection] = {}

    def get(self, path: Path) -> sqlite3.Connection:
        path = path.resolve()
        if path not in self._conns:
            self._conns[path] = connect_readonly(path)
        return self._conns[path]

    def close(self) -> None:
        for conn in self._conns.values():
            conn.close()
        self._conns.clear()


def execute(
    conn: sqlite3.Connection, sql: str, timeout_s: float = 5.0, max_rows: int = 10_000
) -> ExecResult:
    deadline = time.perf_counter() + timeout_s
    # Called every N SQLite VM instructions; a non-zero return aborts the query.
    conn.set_progress_handler(lambda: int(time.perf_counter() > deadline), 10_000)
    started = time.perf_counter()
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(max_rows + 1)
        columns = [d[0] for d in cur.description or []]
        cur.close()
    except sqlite3.Warning as exc:  # e.g. "You can only execute one statement at a time."
        return ExecResult(False, [], [], False, time.perf_counter() - started, str(exc))
    except sqlite3.Error as exc:
        timed_out = "interrupted" in str(exc).lower()
        error = f"timeout after {timeout_s}s" if timed_out else str(exc)
        return ExecResult(False, [], [], False, time.perf_counter() - started, error, timed_out)
    finally:
        conn.set_progress_handler(None, 0)
    truncated = len(rows) > max_rows
    return ExecResult(True, rows[:max_rows], columns, truncated, time.perf_counter() - started)


def _canon_value(v: Any) -> Any:
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if v.is_integer():
            return int(v)
        return round(v, 6)
    if isinstance(v, bytes):
        return v.hex()
    return v


def _canon_row(row: Row) -> Row:
    return tuple(_canon_value(v) for v in row)


def results_match(a: list[Row], b: list[Row], ordered: bool = False) -> bool:
    """Same rows (as a multiset, or as a sequence when ``ordered``), after normalising
    numbers so 3 == 3.0 and float noise below 1e-6 is ignored."""
    ca, cb = [_canon_row(r) for r in a], [_canon_row(r) for r in b]
    if ordered:
        return ca == cb
    return Counter(map(repr, ca)) == Counter(map(repr, cb))
