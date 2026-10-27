"""Phase 3: prompt/answer handling, scoring rules, statistics and the official Spider scorer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from distill_sql.evaluation.harness import EvalItem, stratified_subset
from distill_sql.evaluation.metrics import (
    bootstrap_ci,
    execution_match,
    gold_is_ordered,
    mcnemar,
    percentile,
)
from distill_sql.evaluation.spider_official import SpiderScorer
from distill_sql.prompting import extract_sql, sql_task_messages
from distill_sql.verification.executor import ExecResult
from tests.conftest import make_db

# ---- prompt + answer extraction ------------------------------------------------------


def test_prompt_contains_schema_question_and_optional_hint() -> None:
    msgs = sql_task_messages("CREATE TABLE t (a INT);", "How many rows?", evidence="a is an id")
    user = msgs[-1]["content"]
    assert msgs[0]["role"] == "system"
    assert "CREATE TABLE t" in user and "Question: How many rows?" in user
    assert "Hint: a is an id" in user
    assert "Hint:" not in sql_task_messages("x", "q")[-1]["content"]


@pytest.mark.parametrize(
    ("raw", "sql"),
    [
        ("SELECT count(*) FROM singer;", "SELECT count(*) FROM singer"),
        ("```sql\nSELECT name\nFROM singer\n```\nThis lists names.", "SELECT name FROM singer"),
        ("<think>count rows</think>SQL: SELECT 1", "SELECT 1"),
        ("SELECT a FROM t WHERE b = 'x;y'; DROP TABLE t;", "SELECT a FROM t WHERE b = 'x;y'"),
        ("```sql\nSELECT 1", "SELECT 1"),  # unterminated fence (output cut off)
        ("", ""),
    ],
)
def test_extract_sql(raw: str, sql: str) -> None:
    assert extract_sql(raw) == sql


# ---- execution-match rules -----------------------------------------------------------


def _res(rows: list[tuple[object, ...]], ok: bool = True) -> ExecResult:
    return ExecResult(ok=ok, rows=rows, columns=[], truncated=False, elapsed_s=0.0)


def test_gold_is_ordered_looks_only_at_the_outer_query() -> None:
    assert gold_is_ordered("SELECT name FROM s ORDER BY age")
    assert not gold_is_ordered(
        "SELECT name FROM s WHERE id IN (SELECT id FROM t ORDER BY x LIMIT 1)"
    )
    assert not gold_is_ordered("SELECT count(*) FROM s")


def test_spider_rule_respects_order_only_when_gold_orders() -> None:
    a, b = _res([(1,), (2,)]), _res([(2,), (1,)])
    assert execution_match(a, b, "spider", ordered=False)
    assert not execution_match(a, b, "spider", ordered=True)
    assert not execution_match(_res([(1,), (1,)]), _res([(1,)]), "spider", ordered=False)


def test_bird_rule_compares_sets() -> None:
    assert execution_match(_res([(1,), (1,)]), _res([(1,)]), "bird", ordered=True)
    assert not execution_match(_res([(1,)]), _res([(2,)]), "bird", ordered=False)


def test_failed_prediction_is_never_a_match() -> None:
    assert not execution_match(_res([], ok=False), _res([]), "spider", ordered=False)


# ---- statistics ----------------------------------------------------------------------


def test_bootstrap_ci_brackets_the_mean() -> None:
    values = [True] * 70 + [False] * 30
    lo, hi = bootstrap_ci(values, n_samples=2000, seed=1)
    assert lo < 0.7 < hi
    assert lo > 0.55 and hi < 0.85  # n=100 -> about +-9 points
    assert bootstrap_ci(values, seed=1) == bootstrap_ci(values, seed=1)  # deterministic


def test_mcnemar() -> None:
    a = [True] * 10 + [False] * 5
    b = [False] * 10 + [False] * 5
    a_only, b_only, p = mcnemar(a, b)
    assert (a_only, b_only) == (10, 0)
    assert p == pytest.approx(2 / 1024)
    assert mcnemar([True, False], [True, False]) == (0, 0, 1.0)
    with pytest.raises(ValueError):
        mcnemar([True], [True, False])


def test_percentile() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert percentile([5.0], 0.95) == 5.0


# ---- subsets -------------------------------------------------------------------------


def _items(levels: dict[str, int]) -> list[EvalItem]:
    out, i = [], 0
    for lvl, k in levels.items():
        for _ in range(k):
            out.append(EvalItem(f"spider/dev/{i}", "spider", "db", f"q{i}", "SELECT 1", None, lvl))
            i += 1
    return out


def test_stratified_subset_keeps_the_difficulty_mix() -> None:
    items = _items({"easy": 248, "medium": 446, "hard": 174, "extra": 166})
    sub = stratified_subset(items, 200, seed=42)
    counts = {
        lvl: sum(it.difficulty == lvl for it in sub) for lvl in ("easy", "medium", "hard", "extra")
    }
    assert sum(counts.values()) == 200
    assert counts == {"easy": 48, "medium": 86, "hard": 34, "extra": 32}
    assert sub == stratified_subset(items, 200, seed=42)  # fixed for every model
    assert [int(it.uid.split("/")[-1]) for it in sub] == sorted(
        int(it.uid.split("/")[-1]) for it in sub
    )


def test_subset_larger_than_set_returns_everything() -> None:
    items = _items({"easy": 3})
    assert stratified_subset(items, 10, seed=0) == items


# ---- official Spider scorer ----------------------------------------------------------


@pytest.fixture
def scorer(tmp_path: Path) -> SpiderScorer:
    # The official scorer reads table names unquoted, as in Spider itself, so this fixture
    # uses plain identifiers (Spider has no table names with spaces).
    db = make_db(
        tmp_path / "concert_singer.sqlite",
        [
            "CREATE TABLE singer (singer_id INTEGER PRIMARY KEY, name TEXT, age INTEGER)",
            "CREATE TABLE concert (concert_id INTEGER PRIMARY KEY, title TEXT)",
            "CREATE TABLE singer_in_concert (concert_id INTEGER, singer_id INTEGER)",
        ],
    )
    tables = [
        {
            "db_id": "concert_singer",
            "table_names_original": ["singer", "concert", "singer_in_concert"],
            "column_names_original": [
                [-1, "*"],
                [0, "singer_id"],
                [0, "name"],
                [0, "age"],
                [1, "concert_id"],
                [1, "title"],
                [2, "concert_id"],
                [2, "singer_id"],
            ],
            "foreign_keys": [[6, 4], [7, 1]],
            "primary_keys": [1, 4],
        }
    ]
    path = tmp_path / "tables.json"
    path.write_text(json.dumps(tables))
    return SpiderScorer(path, lambda db_id: db)


def test_official_hardness(scorer: SpiderScorer) -> None:
    assert scorer.hardness("SELECT count(*) FROM singer", "concert_singer") == "easy"
    nested = "SELECT name FROM singer WHERE age > (SELECT avg(age) FROM singer)"
    assert scorer.hardness(nested, "concert_singer") in {"hard", "extra"}


def test_official_exact_match_ignores_case_and_values(scorer: SpiderScorer) -> None:
    gold = "SELECT name FROM singer WHERE age > 30"
    assert scorer.exact_match("select NAME from SINGER where age > 45", gold, "concert_singer")
    assert not scorer.exact_match("SELECT age FROM singer WHERE age > 30", gold, "concert_singer")
    assert not scorer.exact_match("this is not sql", gold, "concert_singer")


# ---- CI regression fixture -----------------------------------------------------------


def test_score_fixture(scorer: SpiderScorer, tmp_path: Path) -> None:
    from distill_sql.config import EvalRegressionConfig
    from distill_sql.evaluation.ci_check import score_fixture

    db_dir = tmp_path / "databases"
    (db_dir / "concert_singer").mkdir(parents=True)
    make_db(
        db_dir / "concert_singer" / "concert_singer.sqlite",
        [
            "CREATE TABLE singer (singer_id INTEGER PRIMARY KEY, name TEXT, age INTEGER)",
            "INSERT INTO singer VALUES (1, 'Joe', 52), (2, 'Ann', 31)",
        ],
    )
    gold = [
        {"uid": "u1", "db_id": "concert_singer", "gold_sql": "SELECT count(*) FROM singer"},
        {
            "uid": "u2",
            "db_id": "concert_singer",
            "gold_sql": "SELECT name FROM singer ORDER BY age",
        },
    ]
    preds = [
        {"uid": "u1", "raw": "```sql\nSELECT COUNT(*) FROM singer;\n```"},  # right
        {"uid": "u2", "raw": "SELECT name FROM singer ORDER BY age DESC"},  # wrong order
    ]
    for name, rows in (("gold.jsonl", gold), ("predictions.jsonl", preds)):
        (tmp_path / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    reg = EvalRegressionConfig(
        predictions_file=tmp_path / "predictions.jsonl",
        gold_file=tmp_path / "gold.jsonl",
        db_dir=db_dir,
        tables_file=tmp_path / "tables.json",  # written by the `scorer` fixture
        min_execution_accuracy=0.5,
    )
    m = score_fixture(reg, tmp_path)
    assert m == {"n": 2, "ex": 0.5, "em": 0.5, "valid_rate": 1.0}
