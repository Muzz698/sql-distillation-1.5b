"""Prompt parsing, static SQL analysis and safe execution."""

from __future__ import annotations

from pathlib import Path

import pytest

from distill_sql.data.schema import connect_readonly
from distill_sql.generation.prompts import ParseError, parse_pairs, parse_sql_list
from distill_sql.verification.executor import execute, results_match
from distill_sql.verification.sqlinfo import SqlCheckError, analyse

# ---- parsing teacher output ----------------------------------------------------


def test_parse_pairs_handles_fences_thinking_and_junk() -> None:
    text = """<think>let me plan the queries...</think>
Here you go:
```json
[{"question": "How many  singers?", "sql": "SELECT count(*) FROM singer;"},
 {"question": "", "sql": "SELECT 1"},
 {"oops": 1},
 "not an object"]
```"""
    pairs = parse_pairs(text)
    assert [(p.question, p.sql) for p in pairs] == [
        ("How many singers?", "SELECT count(*) FROM singer")
    ]


@pytest.mark.parametrize("text", ["no json here", "[1, 2", '{"question": "q", "sql": "s"}', "[]"])
def test_parse_pairs_rejects_unusable_output(text: str) -> None:
    with pytest.raises(ParseError):
        parse_pairs(text)


def test_parse_sql_list_checks_length() -> None:
    assert parse_sql_list('["SELECT 1;", ""]', 2) == ["SELECT 1", None]
    with pytest.raises(ParseError, match="expected 3"):
        parse_sql_list('["SELECT 1"]', 3)


# ---- static analysis -----------------------------------------------------------


def test_analyse_counts_features() -> None:
    info = analyse(
        "SELECT T1.name, count(*) FROM singer AS T1 JOIN concert AS T2 ON T1.id = T2.sid "
        "WHERE T1.age > (SELECT avg(age) FROM singer) GROUP BY T1.name ORDER BY 2 DESC LIMIT 3"
    )
    assert (info.n_joins, info.n_subqueries, info.n_tables) == (1, 1, 2)
    assert info.has_group_by and info.has_order_by and info.has_limit
    assert info.complexity in {"hard", "extra"}
    assert analyse("SELECT name FROM singer").complexity == "easy"


def test_canonical_form_ignores_formatting() -> None:
    a = analyse("select  name from SINGER where age>30")
    b = analyse("SELECT name\nFROM singer\nWHERE age > 30")
    assert a.canonical == b.canonical


def test_set_operation_is_detected() -> None:
    info = analyse("SELECT name FROM a UNION SELECT name FROM b")
    assert info.has_set_op and info.n_subqueries == 0


@pytest.mark.parametrize(
    ("sql", "reason"),
    [
        ("DELETE FROM singer", "not_select"),
        ("DROP TABLE singer", "not_select"),
        ("SELECT 1; SELECT 2", "multiple_statements"),
        ("SELECT FROM WHERE (", "parse_error"),
    ],
)
def test_analyse_rejects(sql: str, reason: str) -> None:
    with pytest.raises(SqlCheckError) as err:
        analyse(sql)
    assert err.value.reason == reason


# ---- execution -----------------------------------------------------------------


def test_execute_returns_rows(singer_db: Path) -> None:
    res = execute(connect_readonly(singer_db), "SELECT name FROM singer ORDER BY age")
    assert res.ok and res.rows == [("Bo",), ("Ann",), ("Li",), ("Joe",)]
    assert res.columns == ["name"]


def test_execute_reports_errors(singer_db: Path) -> None:
    res = execute(connect_readonly(singer_db), "SELECT nope FROM singer")
    assert not res.ok and "no such column" in (res.error or "")


def test_execute_times_out(singer_db: Path) -> None:
    endless = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"
    )
    res = execute(connect_readonly(singer_db), endless, timeout_s=0.2)
    assert res.timed_out and not res.ok


def test_execute_caps_rows(singer_db: Path) -> None:
    res = execute(connect_readonly(singer_db), "SELECT * FROM singer", max_rows=2)
    assert res.ok and res.truncated and res.n_rows == 2


def test_connection_is_read_only_even_through_execute(singer_db: Path) -> None:
    res = execute(connect_readonly(singer_db), "DELETE FROM singer")
    assert not res.ok


def test_results_match() -> None:
    assert results_match([(1, "a"), (2, "b")], [(2, "b"), (1, "a")])
    assert not results_match([(1, "a"), (2, "b")], [(2, "b"), (1, "a")], ordered=True)
    assert results_match([(3,)], [(3.0,)])
    assert results_match([(0.1 + 0.2,)], [(0.3,)])
    assert not results_match([(1,), (1,)], [(1,)])  # multiset, not set


def test_parse_pairs_strips_gemma_thought_blocks() -> None:
    text = (
        '<thought>Plan: [{"question": "draft", "sql": "x"}]</thought>\n'
        '[{"question": "How many pets are there?", "sql": "SELECT count(*) FROM pets"}]'
    )
    assert [p.question for p in parse_pairs(text)] == ["How many pets are there?"]
