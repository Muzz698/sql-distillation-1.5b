from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from distill_sql.data.schema import connect_readonly, extract_schema, quote_ident
from tests.conftest import SINGER_DDL, make_db


def test_extracts_tables_columns_keys_and_rows(singer_db: Path) -> None:
    schema = extract_schema(singer_db, "spider", "concert_singer", sample_rows=3)

    assert [t.name for t in schema.tables] == ["singer", "concert", "singer in concert"]
    singer = schema.tables[0]
    assert [c.name for c in singer.columns] == ["singer_id", "name", "age"]
    assert singer.columns[0].primary_key
    assert singer.columns[1].not_null
    assert singer.row_count == 4
    assert len(singer.sample_rows) == 3  # capped at sample_rows

    link = schema.tables[2]
    assert [c.name for c in link.columns if c.primary_key] == ["concert_id", "singer_id"]
    assert {fk.ref_table for fk in link.foreign_keys} == {"concert", "singer"}
    assert schema.n_columns == 7


def test_ddl_is_valid_sqlite_and_quotes_odd_names(singer_db: Path, tmp_path: Path) -> None:
    ddl = extract_schema(singer_db, "spider", "concert_singer").to_ddl(include_samples=False)
    assert 'CREATE TABLE "singer in concert"' in ddl
    assert "PRIMARY KEY (concert_id, singer_id)" in ddl
    assert "FOREIGN KEY (singer_id) REFERENCES singer(singer_id)" in ddl
    # The rendered DDL must itself be executable: it is what the model sees as "the schema".
    conn = sqlite3.connect(tmp_path / "roundtrip.sqlite")
    conn.executescript(ddl)
    conn.close()


def test_ddl_includes_example_rows(singer_db: Path) -> None:
    ddl = extract_schema(singer_db, "spider", "concert_singer", sample_rows=2).to_ddl()
    assert "/* 2 example rows:\nsinger_id | name | age\n1 | Joe | 52\n2 | Ann | 31\n*/" in ddl


def test_long_text_and_blobs_are_shortened(tmp_path: Path) -> None:
    db = make_db(
        tmp_path / "x.sqlite",
        ["CREATE TABLE t (a TEXT, b BLOB)", "INSERT INTO t VALUES ('" + "x" * 500 + "', x'00ff')"],
    )
    row = extract_schema(db, "d", "x", max_value_chars=20).tables[0].sample_rows[0]
    assert row[0] == "x" * 17 + "..."
    assert row[1] == "<blob 2 bytes>"


def test_fingerprint_ignores_name_data_and_case(tmp_path: Path, singer_db: Path) -> None:
    copy = make_db(
        tmp_path / "renamed.sqlite",
        [s.replace("singer_id", "Singer_ID") for s in SINGER_DDL if not s.startswith("INSERT")],
    )
    a = extract_schema(singer_db, "spider", "concert_singer")
    b = extract_schema(copy, "other", "totally_different_name")
    assert a.fingerprint == b.fingerprint

    changed = make_db(tmp_path / "changed.sqlite", ["CREATE TABLE singer (singer_id INTEGER)"])
    assert extract_schema(changed, "d", "c").fingerprint != a.fingerprint


def test_readonly_connection_cannot_write(singer_db: Path) -> None:
    conn = connect_readonly(singer_db)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM singer")
    conn.close()


def test_missing_db_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        connect_readonly(tmp_path / "nope.sqlite")


@pytest.mark.parametrize(
    ("name", "quoted"),
    [("singer", "singer"), ("Order Items", '"Order Items"'), ('a"b', '"a""b"'), ("1x", '"1x"')],
)
def test_quote_ident(name: str, quoted: str) -> None:
    assert quote_ident(name) == quoted
