"""Cross-teacher agreement: a second teacher answers each kept question independently.

Execution verification proves a query *runs* and returns rows, not that it answers the
question. If a different model writes its own SQL for the same question and gets the same
result, the label is very likely right. Pairs where the two disagree are not deleted (either
side may be wrong) but are marked ``agreement=False``, so training can filter or weight them.

Results go to ``<output_dir>/crosscheck/<checker>.jsonl``; the job is resumable like generation.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from distill_sql.config import VerificationConfig
from distill_sql.data.datasets import Registry
from distill_sql.generation.client import TeacherClient, TeacherError
from distill_sql.generation.generate import PromptBuilder
from distill_sql.generation.prompts import ParseError, crosscheck_messages, parse_sql_list
from distill_sql.generation.ratelimit import DailyLimitReached
from distill_sql.utils.io import iter_jsonl
from distill_sql.verification.executor import DatabasePool, execute, results_match
from distill_sql.verification.verify import TrainExample

log = logging.getLogger(__name__)


@dataclass
class CrosscheckSummary:
    checker: str
    checked: int = 0
    agreed: int = 0
    calls: int = 0
    stopped_reason: str | None = None


def crosscheck_file(output_dir: Path, checker: str) -> Path:
    return output_dir / "crosscheck" / f"{checker}.jsonl"


def load_agreement(output_dir: Path) -> dict[str, bool]:
    """uid -> True if any checker reproduced the result, False if all checkers disagreed."""
    votes: dict[str, list[bool]] = defaultdict(list)
    for path in sorted((output_dir / "crosscheck").glob("*.jsonl")):
        for row in iter_jsonl(path):
            if row.get("match") is not None:
                votes[row["uid"]].append(bool(row["match"]))
    return {uid: any(v) for uid, v in votes.items()}


def run_crosscheck(
    checker: str,
    client: TeacherClient,
    kept: list[TrainExample],
    prompts: PromptBuilder,
    registry: Registry,
    cfg: VerificationConfig,
    output_dir: Path,
    root: Path,
    max_calls: int | None = None,
) -> CrosscheckSummary:
    out = crosscheck_file(output_dir, checker)
    done = {row["uid"] for row in iter_jsonl(out)} if out.exists() else set()
    todo = [ex for ex in kept if ex.teacher != checker and ex.uid not in done]
    by_db: dict[tuple[str, str], list[TrainExample]] = defaultdict(list)
    for ex in todo:
        by_db[(ex.dataset, ex.db_id)].append(ex)

    summary = CrosscheckSummary(checker=checker)
    pool = DatabasePool()
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out.open("a", encoding="utf-8") as fh:
            for (dataset, db_id), examples in sorted(by_db.items()):
                db_file = root / registry.get(dataset, db_id).path
                schema_text = prompts.schema_text(dataset, db_id)
                for i in range(0, len(examples), cfg.crosscheck_batch_size):
                    if max_calls is not None and summary.calls >= max_calls:
                        summary.stopped_reason = f"--max-calls {max_calls} reached"
                        return summary
                    batch = examples[i : i + cfg.crosscheck_batch_size]
                    messages = crosscheck_messages(schema_text, [ex.question for ex in batch])
                    try:
                        completion = client.complete(messages, temperature=0.0, max_tokens=2048)
                        summary.calls += 1
                        sqls = parse_sql_list(completion.text, len(batch))
                    except DailyLimitReached as exc:
                        summary.stopped_reason = str(exc)
                        return summary
                    except TeacherError as exc:  # transient: leave unrecorded, retry next run
                        log.warning("%s %s batch %d: %s", checker, db_id, i, exc)
                        continue
                    except ParseError as exc:  # cached answer, would fail the same way again
                        log.warning("%s %s batch %d: %s", checker, db_id, i, exc)
                        sqls = [None] * len(batch)
                    for ex, sql in zip(batch, sqls, strict=True):
                        row: dict[str, Any] = {"uid": ex.uid, "checker": checker, "sql": sql}
                        if sql is None:
                            row["match"] = None
                        else:
                            conn = pool.get(db_file)
                            mine = execute(conn, sql, cfg.timeout_s, cfg.max_result_rows)
                            gold = execute(conn, ex.sql, cfg.timeout_s, cfg.max_result_rows)
                            row["match"] = mine.ok and results_match(gold.rows, mine.rows)
                            summary.checked += 1
                            summary.agreed += bool(row["match"])
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
    finally:
        pool.close()
    return summary
