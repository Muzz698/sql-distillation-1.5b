"""Phase 2 orchestration: wires config, registry, teachers and verification together.

The CLI calls these functions; tests call them with fake HTTP transports.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from distill_sql.config import AppConfig, GenerationConfig, Secrets, TeacherConfig
from distill_sql.data.contamination import EvalIndex
from distill_sql.data.datasets import (
    Example,
    Registry,
    load_processed_examples,
    load_registry,
)
from distill_sql.generation.cache import ResponseCache
from distill_sql.generation.client import TeacherClient, api_key_for
from distill_sql.generation.fewshot import build_fewshot_pool, load_pool, save_pool
from distill_sql.generation.generate import (
    PromptBuilder,
    RunSummary,
    Task,
    iter_generated_pairs,
    plan_tasks,
    run_teacher,
)
from distill_sql.generation.ratelimit import RateLimiter
from distill_sql.utils.io import atomic_write_text, iter_jsonl, write_json, write_jsonl
from distill_sql.verification.crosscheck import CrosscheckSummary, load_agreement, run_crosscheck
from distill_sql.verification.report import build_stats, render_markdown
from distill_sql.verification.verify import TrainExample, verify_pairs

log = logging.getLogger(__name__)


@dataclass
class Context:
    cfg: AppConfig
    gen: GenerationConfig
    root: Path
    registry: Registry
    eval_index: EvalIndex
    fewshot: dict[str, list[Example]]
    prompts: PromptBuilder

    @property
    def sft_file(self) -> Path:
        return self.cfg.paths.processed_dir / "train" / "sft.jsonl"

    @property
    def fewshot_file(self) -> Path:
        return self.cfg.paths.processed_dir / "fewshot" / "spider_train.jsonl"


def build_context(cfg: AppConfig, root: Path) -> Context:
    if cfg.generation is None:
        raise ValueError("config has no `generation:` section")
    processed = cfg.paths.processed_dir
    registry = load_registry(processed)
    eval_examples = [
        ex
        for name, ds in cfg.datasets.items()
        for ex in load_processed_examples(processed, name, ds.eval_splits)
    ]
    eval_index = EvalIndex(eval_examples, cfg.contamination)

    fewshot_file = processed / "fewshot" / "spider_train.jsonl"
    fewshot = load_pool(fewshot_file)
    if not fewshot and cfg.generation.fewshot_per_db > 0:
        train_examples = [
            ex
            for name, ds in cfg.datasets.items()
            for ex in load_processed_examples(processed, name, ds.train_splits)
        ]
        fewshot = build_fewshot_pool(
            registry, train_examples, eval_index, cfg.generation.fewshot_per_db, cfg.seed
        )
        save_pool(fewshot, fewshot_file)
    prompts = PromptBuilder(cfg, registry, fewshot, root)
    return Context(cfg, cfg.generation, root, registry, eval_index, fewshot, prompts)


def make_client(
    ctx: Context, teacher: TeacherConfig, secrets: Secrets, http: httpx.Client | None = None
) -> TeacherClient:
    cache_dir = ctx.cfg.paths.cache_dir
    limiter = RateLimiter(
        teacher.name,
        teacher.rpm,
        teacher.tpm,
        teacher.rpd,
        teacher.tpd,
        state_file=cache_dir / "usage" / f"{teacher.name}.json",
    )
    return TeacherClient(
        teacher,
        api_key_for(teacher.provider, secrets),
        ResponseCache(cache_dir / "responses"),
        limiter,
        http=http,
        expected_output_tokens=ctx.gen.pairs_per_call * 90,
    )


def teacher_tasks(ctx: Context, teacher: TeacherConfig) -> list[Task]:
    """The teacher's slot is its position in the config, so it stays the same whichever
    subset of teachers a run uses (`--teacher`)."""
    names = [t.name for t in ctx.cfg.teachers]
    return plan_tasks(teacher.name, ctx.registry, ctx.gen, names.index(teacher.name), len(names))


def generate(
    ctx: Context,
    teachers: list[TeacherConfig],
    secrets: Secrets,
    max_calls: int | None = None,
    http: httpx.Client | None = None,
) -> list[RunSummary]:
    """Run every teacher in its own thread; each respects its own rate limits."""

    def one(teacher: TeacherConfig) -> RunSummary:
        client = make_client(ctx, teacher, secrets, http)
        return run_teacher(
            teacher, client, teacher_tasks(ctx, teacher), ctx.prompts, ctx.gen, max_calls
        )

    with ThreadPoolExecutor(max_workers=max(1, len(teachers))) as pool:
        return list(pool.map(one, teachers))


def verify(ctx: Context) -> dict[str, Any]:
    teacher_order = {t.name: i for i, t in enumerate(ctx.cfg.teachers)}
    pairs = sorted(
        iter_generated_pairs(ctx.gen.output_dir),
        key=lambda p: (teacher_order.get(p["teacher"], 99), p["uid"]),
    )
    kept, rejected, stats = verify_pairs(
        pairs,
        ctx.registry,
        ctx.eval_index,
        ctx.fewshot,
        ctx.cfg.verification,
        ctx.root,
        agreement=load_agreement(ctx.gen.output_dir),
    )
    processed = ctx.cfg.paths.processed_dir
    write_jsonl(ctx.sft_file, kept)
    write_jsonl(processed / "rejected" / "rejected.jsonl", rejected)
    summary = build_stats(ctx.cfg.teachers, ctx.gen.output_dir, kept, rejected, stats, ctx.registry)
    write_json(processed / "reports" / "generation_stats.json", summary)
    atomic_write_text(
        processed / "reports" / "generation_report.md",
        render_markdown(summary, kept, ctx.cfg.seed),
    )
    return summary


def crosscheck(
    ctx: Context,
    checker: TeacherConfig,
    secrets: Secrets,
    max_calls: int | None = None,
    http: httpx.Client | None = None,
) -> CrosscheckSummary:
    if not ctx.sft_file.exists():
        raise FileNotFoundError("No verified data yet. Run `make verify` first.")
    kept = [TrainExample.model_validate(r) for r in iter_jsonl(ctx.sft_file)]
    client = make_client(ctx, checker, secrets, http)
    return run_crosscheck(
        checker.name,
        client,
        kept,
        ctx.prompts,
        ctx.registry,
        ctx.cfg.verification,
        ctx.gen.output_dir,
        ctx.root,
        max_calls,
    )
