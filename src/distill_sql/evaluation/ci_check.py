"""CI eval-regression check: re-score saved predictions on a small, committed Spider-dev subset.

No model and no GPU are needed, so it runs in seconds on GitHub Actions. It catches silent
changes to everything *after* the model: SQL extraction, execution, the comparison rules and
the official Spider scorer. If any of those change the score, the check fails.

Fixture layout (tests/fixtures/eval_subset/):
  gold.jsonl         uid, db_id, question, gold_sql, difficulty
  predictions.jsonl  uid, raw model output (as recorded by the harness)
  tables.json        Spider schema entries for the fixture DBs (needed by the EM scorer)
  databases/<db>/<db>.sqlite
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from statistics import mean
from typing import Any

from distill_sql.config import EvalRegressionConfig
from distill_sql.evaluation.harness import EvalContext, EvalItem, stratified_subset
from distill_sql.evaluation.metrics import execution_match, gold_is_ordered
from distill_sql.evaluation.spider_official import SpiderScorer
from distill_sql.prompting import extract_sql
from distill_sql.utils.io import iter_jsonl, read_json, write_json, write_jsonl
from distill_sql.verification.executor import DatabasePool, execute


def build_fixture(
    ctx: EvalContext,
    items: list[EvalItem],
    predictions_file: Path,
    out_dir: Path,
    n: int,
    max_db_kb: int = 100,
) -> None:
    preds = {r["uid"]: r for r in iter_jsonl(predictions_file)}
    small = [
        it
        for it in items
        if it.uid in preds and ctx.db_path(it.dataset, it.db_id).stat().st_size <= max_db_kb * 1024
    ]
    chosen = stratified_subset(small, n, ctx.cfg.seed)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "databases").mkdir(parents=True)
    write_jsonl(
        out_dir / "gold.jsonl",
        [
            {"uid": it.uid, "db_id": it.db_id, "question": it.question,
             "gold_sql": it.gold_sql, "difficulty": it.difficulty}
            for it in chosen
        ],
    )  # fmt: skip
    write_jsonl(
        out_dir / "predictions.jsonl",
        [{"uid": it.uid, "raw": preds[it.uid]["raw"]} for it in chosen],
    )
    db_ids = sorted({it.db_id for it in chosen})
    for db_id in db_ids:
        dest = out_dir / "databases" / db_id / f"{db_id}.sqlite"
        dest.parent.mkdir(parents=True)
        shutil.copyfile(ctx.db_path("spider", db_id), dest)
    spider = ctx.cfg.paths.raw_dir / ctx.cfg.datasets["spider"].extract_to / "spider_data"
    tables = [t for t in json.loads((spider / "tables.json").read_text()) if t["db_id"] in db_ids]
    write_json(out_dir / "tables.json", tables)


def score_fixture(reg: EvalRegressionConfig, root: Path) -> dict[str, Any]:
    def p(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    db_dir = p(reg.db_dir)
    scorer = SpiderScorer(p(reg.tables_file), lambda db: db_dir / db / f"{db}.sqlite")
    preds = {r["uid"]: r for r in iter_jsonl(p(reg.predictions_file))}
    pool = DatabasePool()
    ex, em, valid = [], [], []
    try:
        for g in iter_jsonl(p(reg.gold_file)):
            sql = extract_sql(preds[g["uid"]]["raw"])
            conn = pool.get(db_dir / g["db_id"] / f"{g['db_id']}.sqlite")
            gold = execute(conn, g["gold_sql"], timeout_s=30)
            pred = execute(conn, sql, timeout_s=30) if sql else None
            valid.append(bool(pred and pred.ok))
            ex.append(
                bool(pred and execution_match(pred, gold, "spider", gold_is_ordered(g["gold_sql"])))
            )
            em.append(scorer.exact_match(sql, g["gold_sql"], g["db_id"]))
    finally:
        pool.close()
    return {"n": len(ex), "ex": mean(ex), "em": mean(em), "valid_rate": mean(valid)}


def expected_metrics(reg: EvalRegressionConfig, root: Path) -> dict[str, Any] | None:
    path = (root / reg.gold_file).parent / "expected.json"
    return read_json(path) if path.exists() else None
