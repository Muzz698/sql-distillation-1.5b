"""Turn raw teacher output into verified training data.

Every generated pair goes through the checks below, in order. The first failing check is
its rejection reason, so the stats report adds up to 100%.

  1. bad_question         too short / too long
  2. contaminated         matches an eval question (exact, 13-gram or near-duplicate)
  3. fewshot_copy         repeats one of the few-shot exemplars shown in the prompt
  4. parse_error / multiple_statements / not_select      static checks (sqlglot)
  5. exec_error / timeout                                 runs on the real DB, read-only
  6. empty_result / all_null_result / too_many_rows       result sanity
  7. duplicate_question / duplicate_sql                   first occurrence wins, per DB

Kept pairs go to ``train/sft.jsonl``. Rejected pairs keep their SQL and reason: the ones that
ran but were wrong are raw material for DPO "rejected" answers in Phase 5.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from distill_sql.config import VerificationConfig
from distill_sql.data.contamination import EvalIndex, normalize
from distill_sql.data.datasets import Example, Registry
from distill_sql.verification.executor import DatabasePool, execute
from distill_sql.verification.sqlinfo import Complexity, SqlCheckError, analyse

REJECT_REASONS = [
    "bad_question",
    "contaminated",
    "fewshot_copy",
    "unknown_db",
    "parse_error",
    "multiple_statements",
    "not_select",
    "exec_error",
    "timeout",
    "empty_result",
    "all_null_result",
    "too_many_rows",
    "duplicate_question",
    "duplicate_sql",
]


class TrainExample(BaseModel):
    uid: str
    dataset: str
    db_id: str
    question: str
    sql: str
    teacher: str
    model: str
    focus: str
    complexity: Complexity
    n_joins: int
    n_subqueries: int
    has_group_by: bool
    has_order_by: bool
    has_set_op: bool
    n_result_rows: int
    result_hash: str
    agreement: bool | None = None  # set by the cross-check; None = not checked yet


class RejectedPair(BaseModel):
    uid: str
    dataset: str
    db_id: str
    question: str
    sql: str
    teacher: str
    reason: str
    detail: str = ""


@dataclass
class VerifyStats:
    by_teacher: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))

    def add(self, teacher: str, outcome: str) -> None:
        self.by_teacher[teacher][outcome] += 1

    def to_dict(self) -> dict[str, Any]:
        return {t: dict(c) for t, c in sorted(self.by_teacher.items())}


def verify_pairs(
    pairs: Iterable[dict[str, Any]],
    registry: Registry,
    eval_index: EvalIndex,
    fewshot: dict[str, list[Example]],
    cfg: VerificationConfig,
    root: Path,
    agreement: dict[str, bool] | None = None,
) -> tuple[list[TrainExample], list[RejectedPair], VerifyStats]:
    agreement = agreement or {}
    train_dbs = {(d.dataset, d.db_id): d for d in registry.by_role("train")}
    fewshot_q = {db: {normalize(e.question) for e in exs} for db, exs in fewshot.items()}
    seen_q: dict[str, set[str]] = defaultdict(set)
    seen_sql: dict[str, set[str]] = defaultdict(set)
    pool = DatabasePool()
    kept: list[TrainExample] = []
    rejected: list[RejectedPair] = []
    stats = VerifyStats()

    def reject(p: dict[str, Any], reason: str, detail: str = "") -> None:
        rejected.append(
            RejectedPair(
                uid=p["uid"],
                dataset=p["dataset"],
                db_id=p["db_id"],
                question=p["question"],
                sql=p["sql"],
                teacher=p["teacher"],
                reason=reason,
                detail=detail[:300],
            )
        )
        stats.add(p["teacher"], reason)

    try:
        for p in pairs:
            db_id, question, sql = p["db_id"], p["question"], p["sql"]
            n_words = len(question.split())
            if not cfg.min_question_words <= n_words <= cfg.max_question_words:
                reject(p, "bad_question", f"{n_words} words")
                continue
            if (m := eval_index.match(question)) is not None:
                reject(p, "contaminated", f"{m[0]} with {m[1].uid}")
                continue
            norm_q = normalize(question)
            if norm_q in fewshot_q.get(db_id, set()):
                reject(p, "fewshot_copy")
                continue
            entry = train_dbs.get((p["dataset"], db_id))
            if entry is None:  # only `train` DBs may produce training data
                reject(p, "unknown_db", f"{p['dataset']}/{db_id} is not a train DB")
                continue
            try:
                info = analyse(sql)
            except SqlCheckError as exc:
                reject(p, exc.reason, exc.detail)
                continue

            result = execute(pool.get(root / entry.path), sql, cfg.timeout_s, cfg.max_result_rows)
            if not result.ok:
                reject(p, "timeout" if result.timed_out else "exec_error", result.error or "")
                continue
            if result.n_rows == 0:
                reject(p, "empty_result")
                continue
            if result.all_null:
                reject(p, "all_null_result")
                continue
            if result.truncated:
                reject(p, "too_many_rows", f"> {cfg.max_result_rows}")
                continue
            if norm_q in seen_q[db_id]:
                reject(p, "duplicate_question")
                continue
            if info.canonical in seen_sql[db_id]:
                reject(p, "duplicate_sql")
                continue
            seen_q[db_id].add(norm_q)
            seen_sql[db_id].add(info.canonical)

            kept.append(
                TrainExample(
                    uid=p["uid"],
                    dataset=p["dataset"],
                    db_id=db_id,
                    question=question,
                    sql=sql,
                    teacher=p["teacher"],
                    model=p["model"],
                    focus=p["focus"],
                    complexity=info.complexity,
                    n_joins=info.n_joins,
                    n_subqueries=info.n_subqueries,
                    has_group_by=info.has_group_by,
                    has_order_by=info.has_order_by,
                    has_set_op=info.has_set_op,
                    n_result_rows=result.n_rows,
                    result_hash=result.result_hash(),
                    agreement=agreement.get(p["uid"]),
                )
            )
            stats.add(p["teacher"], "kept")
    finally:
        pool.close()
    return kept, rejected, stats
