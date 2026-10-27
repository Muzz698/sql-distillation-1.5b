"""Evaluation harness: predict → execute → score, for any model on any eval set.

Layout (committed to git, so every number in the README can be re-checked):
  results/<set>/<model>/predictions.jsonl   one row per question (raw output, SQL, timing)
  results/<set>/<model>/scores.jsonl        one row per question (ex, em, valid, difficulty)
  results/<set>/<model>/metrics.json        aggregate metrics

Prediction is resumable: questions already in predictions.jsonl are skipped, and responses
are cached, so a rate-limited teacher can be evaluated over several days.
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from distill_sql.config import (
    AppConfig,
    EvalModelConfig,
    EvalSetConfig,
    Secrets,
    TeacherConfig,
)
from distill_sql.data.datasets import Example, load_processed_examples, load_registry
from distill_sql.data.schema import DatabaseSchema
from distill_sql.evaluation.local_server import LlamaServer
from distill_sql.evaluation.metrics import (
    Comparison,
    bootstrap_ci,
    execution_match,
    gold_is_ordered,
    percentile,
)
from distill_sql.evaluation.spider_official import SpiderScorer
from distill_sql.generation.cache import ResponseCache
from distill_sql.generation.client import TeacherClient, api_key_for
from distill_sql.generation.ratelimit import DailyLimitReached, RateLimiter, day_in
from distill_sql.prompting import extract_sql, sql_task_messages
from distill_sql.utils.io import iter_jsonl, read_json, sha256_file, write_json, write_jsonl
from distill_sql.verification.executor import DatabasePool, execute

log = logging.getLogger(__name__)

DIFFICULTY_ORDER = ["easy", "medium", "hard", "extra", "simple", "moderate", "challenging"]


@dataclass(frozen=True)
class EvalItem:
    uid: str
    dataset: str
    db_id: str
    question: str
    gold_sql: str
    evidence: str | None
    difficulty: str


# ---- eval sets ----------------------------------------------------------------------


def stratified_subset(items: list[EvalItem], n: int, seed: int) -> list[EvalItem]:
    """``n`` items with the same difficulty mix as the full set (largest-remainder rounding)."""
    if n >= len(items):
        return items
    by_level: dict[str, list[EvalItem]] = defaultdict(list)
    for it in items:
        by_level[it.difficulty].append(it)
    quotas = {lvl: n * len(group) / len(items) for lvl, group in by_level.items()}
    counts = {lvl: int(q) for lvl, q in quotas.items()}
    for lvl in sorted(quotas, key=lambda k: quotas[k] - counts[k], reverse=True)[
        : n - sum(counts.values())
    ]:
        counts[lvl] += 1
    rng = random.Random(seed)
    chosen = [it for lvl in sorted(by_level) for it in rng.sample(by_level[lvl], counts[lvl])]
    return sorted(chosen, key=lambda it: int(it.uid.rsplit("/", 1)[1]))


class EvalContext:
    """Everything needed to build prompts and score answers for the configured datasets."""

    def __init__(self, cfg: AppConfig, root: Path) -> None:
        self.cfg, self.root = cfg, root
        self.registry = load_registry(cfg.paths.processed_dir)
        self._schemas: dict[tuple[str, str], str] = {}
        spider = cfg.paths.raw_dir / cfg.datasets["spider"].extract_to / "spider_data"
        self._spider: dict[str, SpiderScorer] = {
            "dev": SpiderScorer(
                spider / "tables.json", lambda db: spider / "database" / db / f"{db}.sqlite"
            ),
            "test": SpiderScorer(
                spider / "test_tables.json",
                lambda db: spider / "test_database" / db / f"{db}.sqlite",
            ),
        }

    def db_path(self, dataset: str, db_id: str) -> Path:
        return self.root / self.registry.get(dataset, db_id).path

    def schema_ddl(self, dataset: str, db_id: str) -> str:
        key = (dataset, db_id)
        if key not in self._schemas:
            path = self.cfg.paths.processed_dir / "schemas" / dataset / f"{db_id}.json"
            self._schemas[key] = DatabaseSchema.model_validate(read_json(path)).to_ddl()
        return self._schemas[key]

    def spider_scorer(self, split: str) -> SpiderScorer:
        return self._spider[split]

    def load_set(self, spec: EvalSetConfig) -> list[EvalItem]:
        examples: list[Example] = load_processed_examples(
            self.cfg.paths.processed_dir, spec.dataset, [spec.split]
        )
        items = []
        for ex in examples:
            if spec.dataset == "spider":
                difficulty = self.spider_scorer(spec.split).hardness(ex.sql, ex.db_id)
            else:
                difficulty = ex.difficulty or "unknown"
            items.append(
                EvalItem(ex.uid, ex.dataset, ex.db_id, ex.question, ex.sql, ex.evidence, difficulty)
            )
        if spec.n is not None:
            items = stratified_subset(items, spec.n, self.cfg.seed)
        return items


# ---- predictors -----------------------------------------------------------------------


@dataclass
class ModelHandle:
    client: TeacherClient
    price_in: float | None
    price_out: float | None
    server: LlamaServer | None = None


@contextmanager
def open_model(
    ctx: EvalContext, name: str, spec: EvalModelConfig, secrets: Secrets
) -> Iterator[ModelHandle]:
    cfg = ctx.cfg
    assert cfg.evaluation is not None
    cache = ResponseCache(cfg.paths.cache_dir / "responses")
    if spec.kind == "teacher":
        assert spec.teacher is not None
        teacher = cfg.teacher(spec.teacher)
        limiter = RateLimiter(
            teacher.name,
            teacher.rpm,
            teacher.tpm,
            teacher.rpd,
            teacher.tpd,
            # Shares the usage file with generation: both draw from the same daily quota.
            state_file=cfg.paths.cache_dir / "usage" / f"{teacher.name}.json",
            today=day_in(teacher.quota_timezone),
        )
        client = TeacherClient(
            teacher,
            api_key_for(teacher.provider, secrets),
            cache,
            limiter,
            expected_output_tokens=150,
        )
        yield ModelHandle(client, teacher.price_input_per_m, teacher.price_output_per_m)
        return

    assert spec.gguf is not None
    gguf = cfg.evaluation.models_dir / spec.gguf.filename
    log_file = cfg.paths.runs_dir / "llama-server" / f"{name}.log"
    with LlamaServer(gguf, cfg.evaluation.local_server, log_file) as server:
        # The model id in the request is part of the cache key: include the file hash so a
        # different GGUF can never be served answers cached for another one.
        digest = (spec.gguf.sha256 or sha256_file(gguf))[:12]
        local = TeacherConfig(
            name=f"local-{name}".lower(),
            provider="local",
            model=f"{spec.gguf.filename}@{digest}",
            rpm=10_000,
            tpm=10_000_000,
            rpd=10_000_000,
            base_url=server.base_url,
        )
        limiter = RateLimiter(
            local.name, local.rpm, local.tpm, local.rpd, None,
            state_file=cfg.paths.cache_dir / "usage" / f"{local.name}.json",
        )  # fmt: skip
        client = TeacherClient(local, "none", cache, limiter, max_retries=3)
        yield ModelHandle(client, 0.0, 0.0, server)


# ---- predict ------------------------------------------------------------------------


def results_dir(cfg: AppConfig, set_name: str, model_name: str) -> Path:
    assert cfg.evaluation is not None
    return cfg.evaluation.results_dir / set_name / model_name


def predict(
    ctx: EvalContext,
    handle: ModelHandle,
    items: list[EvalItem],
    out_dir: Path,
    max_items: int | None = None,
) -> tuple[int, str | None]:
    """Append predictions for items not done yet. Returns (n_new, stop_reason)."""
    assert ctx.cfg.evaluation is not None
    pred_file = out_dir / "predictions.jsonl"
    done = {r["uid"] for r in iter_jsonl(pred_file)} if pred_file.exists() else set()
    out_dir.mkdir(parents=True, exist_ok=True)
    n_new = 0
    with pred_file.open("a", encoding="utf-8") as fh:
        for it in items:
            if it.uid in done:
                continue
            if max_items is not None and n_new >= max_items:
                return n_new, f"--limit {max_items} reached"
            messages = sql_task_messages(
                ctx.schema_ddl(it.dataset, it.db_id), it.question, it.evidence
            )
            try:
                c = handle.client.complete(
                    messages, temperature=0.0, max_tokens=ctx.cfg.evaluation.max_new_tokens
                )
            except DailyLimitReached as exc:
                return n_new, str(exc)
            row = {
                "uid": it.uid,
                "raw": c.text,
                "sql": extract_sql(c.text),
                "latency_s": c.latency_s,
                "prompt_tokens": c.prompt_tokens,
                "completion_tokens": c.completion_tokens,
                "reasoning_tokens": c.reasoning_tokens,
                "cached": c.cached,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            n_new += 1
    return n_new, None


# ---- score --------------------------------------------------------------------------


def score(
    ctx: EvalContext,
    set_name: str,
    items: list[EvalItem],
    out_dir: Path,
    price_in: float | None,
    price_out: float | None,
    peak_rss_bytes: int | None = None,
) -> dict[str, Any]:
    cfg = ctx.cfg
    assert cfg.evaluation is not None
    spec = cfg.evaluation.sets[set_name]
    preds = {r["uid"]: r for r in iter_jsonl(out_dir / "predictions.jsonl")}
    rule: Comparison = "bird" if spec.dataset == "bird_minidev" else "spider"
    pool = DatabasePool()
    rows: list[dict[str, Any]] = []
    try:
        for it in items:
            if it.uid not in preds:
                continue
            p = preds[it.uid]
            conn = pool.get(ctx.db_path(it.dataset, it.db_id))
            t = cfg.evaluation.timeout_s
            gold = execute(conn, it.gold_sql, t, max_rows=1_000_000)
            pred = execute(conn, p["sql"], t, max_rows=1_000_000) if p["sql"] else None
            row: dict[str, Any] = {
                "uid": it.uid,
                "db_id": it.db_id,
                "difficulty": it.difficulty,
                "valid": bool(pred and pred.ok),
                "ex": bool(
                    pred and execution_match(pred, gold, rule, gold_is_ordered(it.gold_sql))
                ),
                "gold_ok": gold.ok,
                "error": (pred.error if pred else "no SQL in output"),
            }
            if spec.dataset == "spider":
                row["em"] = ctx.spider_scorer(spec.split).exact_match(
                    p["sql"], it.gold_sql, it.db_id
                )
            rows.append(row)
    finally:
        pool.close()
    write_jsonl(out_dir / "scores.jsonl", rows)

    n = len(rows)
    ex = [r["ex"] for r in rows]
    by_level: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_level[r["difficulty"]].append(r["ex"])
    done_preds = [preds[r["uid"]] for r in rows]
    live = [p for p in done_preds if not p["cached"]] or done_preds
    latencies = [p["latency_s"] for p in live]
    out_tok = [p["completion_tokens"] + p.get("reasoning_tokens", 0) for p in done_preds]
    in_tok = [p["prompt_tokens"] for p in done_preds]
    cost_per_1k = None
    if price_in is not None and price_out is not None and n:
        cost_per_1k = 1000 * (mean(in_tok) * price_in + mean(out_tok) * price_out) / 1e6
    tok_per_s = [
        (p["completion_tokens"] + p.get("reasoning_tokens", 0)) / p["latency_s"]
        for p in live
        if p["latency_s"] > 0
    ]

    def r3(x: float | None) -> float | None:
        return None if x is None else round(x, 3)

    metrics: dict[str, Any] = {
        "set": set_name,
        "kind": "local" if peak_rss_bytes is not None else "api",
        "n": n,
        "n_total": len(items),
        "complete": n == len(items),
        "ex": mean(ex) if n else None,
        "ex_ci95": bootstrap_ci(ex, cfg.evaluation.bootstrap_samples, cfg.seed) if n else None,
        "em": mean(r["em"] for r in rows) if n and "em" in rows[0] else None,
        "valid_rate": mean(r["valid"] for r in rows) if n else None,
        "by_difficulty": {
            lvl: {"n": len(v), "ex": mean(v)}
            for lvl, v in sorted(
                by_level.items(),
                key=lambda kv: DIFFICULTY_ORDER.index(kv[0]) if kv[0] in DIFFICULTY_ORDER else 99,
            )
        },
        "latency_p50_s": r3(percentile(latencies, 0.5)) if latencies else None,
        "latency_p95_s": r3(percentile(latencies, 0.95)) if latencies else None,
        # For API models this is output tokens / wall time incl. network: not model speed.
        "tokens_per_s": r3(mean(tok_per_s)) if tok_per_s else None,
        "mean_prompt_tokens": r3(mean(in_tok)) if in_tok else None,
        "mean_output_tokens": r3(mean(out_tok)) if out_tok else None,
        "cost_per_1k_usd": r3(cost_per_1k),
        "peak_memory_mb": round(peak_rss_bytes / 2**20) if peak_rss_bytes else None,
        "gold_failures": sum(not r["gold_ok"] for r in rows),
    }
    old = out_dir / "metrics.json"
    if peak_rss_bytes is None and old.exists():  # re-scoring only: keep what was measured
        prev = read_json(old)
        metrics["peak_memory_mb"] = prev.get("peak_memory_mb")
        metrics["kind"] = prev.get("kind", metrics["kind"])
    write_json(old, metrics)
    return metrics
