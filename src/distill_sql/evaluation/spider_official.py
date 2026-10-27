"""Spider's official exact-set-match (EM) and difficulty labels, via the vendored
test-suite-sql-eval code. EM compares the *structure* of two queries (clauses, columns,
operators), ignoring values and formatting; it is stricter than execution accuracy and
penalises correct-but-different SQL, which is why EX is the main metric."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from distill_sql.third_party.test_suite_sql_eval import evaluation as ev
from distill_sql.third_party.test_suite_sql_eval.process_sql import Schema, get_schema, get_sql

_EMPTY_SQL: dict[str, Any] = {
    "except": None,
    "from": {"conds": [], "table_units": []},
    "groupBy": [],
    "having": [],
    "intersect": None,
    "limit": None,
    "orderBy": [],
    "select": [False, []],
    "union": None,
    "where": [],
}


class SpiderScorer:
    def __init__(self, tables_json: Path, db_path: Callable[[str], Path]) -> None:
        self._kmaps = ev.build_foreign_key_map_from_json(str(tables_json))
        self._db_path = db_path
        self._schemas: dict[str, Any] = {}
        self._evaluator = ev.Evaluator()

    def _schema(self, db_id: str) -> Any:
        if db_id not in self._schemas:
            self._schemas[db_id] = Schema(get_schema(str(self._db_path(db_id))))
        return self._schemas[db_id]

    def hardness(self, gold_sql: str, db_id: str) -> str:
        """Official Spider difficulty: easy / medium / hard / extra."""
        return str(self._evaluator.eval_hardness(get_sql(self._schema(db_id), gold_sql)))

    def exact_match(self, pred_sql: str, gold_sql: str, db_id: str) -> bool:
        schema = self._schema(db_id)
        kmap = self._kmaps[db_id]
        g_sql = get_sql(schema, gold_sql)
        try:
            p_sql = get_sql(schema, pred_sql.replace("value", "1"))
        except Exception:  # the official script scores unparsable predictions as empty SQL
            p_sql = _EMPTY_SQL
        g_sql = ev.rebuild_sql_col(
            ev.build_valid_col_units(g_sql["from"]["table_units"], schema),
            ev.rebuild_sql_val(g_sql),
            kmap,
        )
        p_sql = ev.rebuild_sql_col(
            ev.build_valid_col_units(p_sql["from"]["table_units"], schema),
            ev.rebuild_sql_val(p_sql),
            kmap,
        )
        return bool(self._evaluator.eval_exact_match(p_sql, g_sql))
