"""Resumable synthetic-data generation job.

One *task* = one teacher call for one train database with one focus, asking for
``pairs_per_call`` question/SQL pairs. Tasks are ordered round-robin over databases (call 0
for every DB, then call 1, ...), so a run that stops at the daily limit still covers all
databases evenly.

Per teacher, two append-only files under ``generation.output_dir``:
  <teacher>.calls.jsonl   one row per finished task (status, tokens, latency, cached)
  <teacher>.pairs.jsonl   one row per generated pair (unverified)
A task that already has a row in ``calls.jsonl`` is skipped, which makes the job resumable.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from distill_sql.config import AppConfig, GenerationConfig, TeacherConfig
from distill_sql.data.datasets import Example, Registry
from distill_sql.data.schema import DatabaseSchema, column_value_hints
from distill_sql.generation.client import TeacherClient, TeacherError, estimate_tokens
from distill_sql.generation.prompts import (
    Message,
    ParseError,
    generation_messages,
    parse_pairs,
    schema_block,
)
from distill_sql.generation.ratelimit import DailyLimitReached
from distill_sql.utils.io import iter_jsonl, read_json

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Task:
    teacher: str
    dataset: str
    db_id: str
    call_index: int
    focus: str

    @property
    def task_id(self) -> str:
        return f"{self.teacher}:{self.db_id}:{self.call_index}"


@dataclass
class RunSummary:
    teacher: str
    planned: int = 0
    skipped_done: int = 0
    calls: int = 0
    cached: int = 0
    pairs: int = 0
    parse_errors: int = 0
    teacher_errors: int = 0
    tokens: int = 0
    stopped_reason: str | None = None
    errors: list[str] = field(default_factory=list)


def _stable_offset(text: str, mod: int) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest(), 16) % mod


def plan_tasks(teacher: str, registry: Registry, gen: GenerationConfig) -> list[Task]:
    dbs = sorted(registry.by_role("train"), key=lambda d: d.key)
    n_focus = len(gen.focuses)
    return [
        Task(
            teacher=teacher,
            dataset=db.dataset,
            db_id=db.db_id,
            call_index=i,
            focus=gen.focuses[(i + _stable_offset(db.key, n_focus)) % n_focus],
        )
        for i in range(gen.calls_per_db)
        for db in dbs
    ]


class PromptBuilder:
    """Builds (and memoises) the schema text and messages for each database."""

    def __init__(
        self, cfg: AppConfig, registry: Registry, fewshot: dict[str, list[Example]], root: Path
    ) -> None:
        self.cfg, self.registry, self.fewshot, self.root = cfg, registry, fewshot, root
        self._schema_text: dict[tuple[str, str], str] = {}

    def schema_text(self, dataset: str, db_id: str) -> str:
        key = (dataset, db_id)
        if key not in self._schema_text:
            entry = self.registry.get(dataset, db_id)
            schema = DatabaseSchema.model_validate(
                read_json(self.cfg.paths.processed_dir / "schemas" / dataset / f"{db_id}.json")
            )
            hints = column_value_hints(self.root / entry.path, schema)
            self._schema_text[key] = schema_block(schema.to_ddl(), hints)
        return self._schema_text[key]

    def messages(self, task: Task) -> list[Message]:
        gen = self.cfg.generation
        assert gen is not None
        return generation_messages(
            self.schema_text(task.dataset, task.db_id),
            self.fewshot.get(task.db_id, []),
            gen.pairs_per_call,
            task.focus,
            batch_no=task.call_index + 1,
        )


def _done_task_ids(calls_file: Path) -> set[str]:
    if not calls_file.exists():
        return set()
    return {row["task_id"] for row in iter_jsonl(calls_file)}


def _append(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()


def run_teacher(
    teacher: TeacherConfig,
    client: TeacherClient,
    tasks: list[Task],
    prompts: PromptBuilder,
    gen: GenerationConfig,
    max_calls: int | None = None,
) -> RunSummary:
    calls_file = gen.output_dir / f"{teacher.name}.calls.jsonl"
    pairs_file = gen.output_dir / f"{teacher.name}.pairs.jsonl"
    done = _done_task_ids(calls_file)
    summary = RunSummary(teacher=teacher.name, planned=len(tasks))

    for task in tasks:
        if task.task_id in done:
            summary.skipped_done += 1
            continue
        if max_calls is not None and summary.calls >= max_calls:
            summary.stopped_reason = f"--max-calls {max_calls} reached"
            break
        call_row: dict[str, Any] = {"task_id": task.task_id, "db_id": task.db_id}
        try:
            completion = client.complete(
                prompts.messages(task), temperature=gen.temperature, max_tokens=gen.max_tokens
            )
        except DailyLimitReached as exc:
            summary.stopped_reason = str(exc)
            break
        except TeacherError as exc:
            # Not written to calls.jsonl, so the task is retried on the next run.
            summary.teacher_errors += 1
            summary.errors.append(f"{task.task_id}: {exc}")
            log.error("%s", exc)
            if summary.teacher_errors >= 5:
                summary.stopped_reason = "5 teacher errors; check the key / model id"
                break
            continue

        summary.calls += 1
        summary.cached += completion.cached
        summary.tokens += 0 if completion.cached else completion.total_tokens
        call_row |= {
            "model": completion.model,
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
            "latency_s": completion.latency_s,
            "cached": completion.cached,
        }
        try:
            pairs = parse_pairs(completion.text)
        except ParseError as exc:
            summary.parse_errors += 1
            _append(calls_file, [call_row | {"status": "parse_error", "error": str(exc)}])
            continue

        _append(
            pairs_file,
            [
                {
                    "uid": f"{task.task_id}:{i}",
                    "task_id": task.task_id,
                    "teacher": teacher.name,
                    "model": completion.model,
                    "dataset": task.dataset,
                    "db_id": task.db_id,
                    "focus": task.focus,
                    "question": p.question,
                    "sql": p.sql,
                }
                for i, p in enumerate(pairs)
            ],
        )
        # The call row is written last: if we crash between the two writes, the task reruns
        # (served from the cache) and verification dedupes the repeated pairs.
        _append(calls_file, [call_row | {"status": "ok", "n_pairs": len(pairs)}])
        summary.pairs += len(pairs)
    return summary


def iter_generated_pairs(output_dir: Path) -> Iterator[dict[str, Any]]:
    for path in sorted(output_dir.glob("*.pairs.jsonl")):
        yield from iter_jsonl(path)


def estimate_budget(
    teacher: TeacherConfig, tasks: list[Task], prompts: PromptBuilder, gen: GenerationConfig
) -> dict[str, Any]:
    """Rough calls/tokens/days for a full pass, from a sample of real prompts."""
    sample = tasks[:: max(1, len(tasks) // 20)] or tasks
    prompt_tok = sum(estimate_tokens(prompts.messages(t)) for t in sample) / max(1, len(sample))
    out_tok = gen.pairs_per_call * 90  # ~90 tokens per question+SQL pair, observed in pilots
    per_call = prompt_tok + out_tok
    calls_per_day = teacher.rpd * 0.9
    if teacher.tpd:
        calls_per_day = min(calls_per_day, teacher.tpd * 0.9 / per_call)
    return {
        "calls": len(tasks),
        "est_tokens_per_call": round(per_call),
        "est_calls_per_day": int(calls_per_day),
        "est_days": round(len(tasks) / max(1.0, calls_per_day), 1),
        "est_raw_pairs": len(tasks) * gen.pairs_per_call,
    }
