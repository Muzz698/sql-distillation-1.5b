"""Generation + verification statistics, as JSON (for tracking) and Markdown (for humans)."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

from distill_sql.config import TeacherConfig
from distill_sql.data.datasets import Registry
from distill_sql.utils.io import iter_jsonl
from distill_sql.verification.verify import REJECT_REASONS, RejectedPair, TrainExample, VerifyStats


def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "-"


def call_stats(output_dir: Path, teacher: str) -> dict[str, Any]:
    path = output_dir / f"{teacher}.calls.jsonl"
    rows = list(iter_jsonl(path)) if path.exists() else []
    live = [r for r in rows if not r.get("cached")]
    return {
        "calls": len(rows),
        "parse_errors": sum(r["status"] == "parse_error" for r in rows),
        "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in rows),
        "completion_tokens": sum(r.get("completion_tokens", 0) for r in rows),
        "median_latency_s": round(median(r["latency_s"] for r in live), 2) if live else None,
    }


def build_stats(
    teachers: list[TeacherConfig],
    output_dir: Path,
    kept: list[TrainExample],
    rejected: list[RejectedPair],
    stats: VerifyStats,
    registry: Registry,
) -> dict[str, Any]:
    per_teacher: dict[str, Any] = {}
    for t in teachers:
        outcomes = stats.by_teacher.get(t.name, Counter())
        raw = sum(outcomes.values())
        calls = call_stats(output_dir, t.name)
        cost = None
        if t.price_input_per_m is not None and t.price_output_per_m is not None:
            cost = (
                calls["prompt_tokens"] * t.price_input_per_m
                + calls["completion_tokens"] * t.price_output_per_m
            ) / 1e6
        mine = [k for k in kept if k.teacher == t.name]
        checked = [k for k in mine if k.agreement is not None]
        per_teacher[t.name] = {
            "model": t.model,
            **calls,
            "raw_pairs": raw,
            "kept": outcomes.get("kept", 0),
            "keep_rate": outcomes.get("kept", 0) / raw if raw else None,
            "agreement_checked": len(checked),
            "agreement_rate": sum(bool(k.agreement) for k in checked) / len(checked)
            if checked
            else None,
            "list_price_cost_usd": cost,
        }

    reasons = Counter(r.reason for r in rejected)
    kept_per_db = Counter(k.db_id for k in kept)
    train_dbs = [d.db_id for d in registry.by_role("train")]
    q_words = [len(k.question.split()) for k in kept]
    return {
        "n_raw": len(kept) + len(rejected),
        "n_kept": len(kept),
        "n_rejected": len(rejected),
        "teachers": per_teacher,
        "reject_reasons": {r: reasons.get(r, 0) for r in REJECT_REASONS},
        "coverage": {
            "train_dbs": len(train_dbs),
            "dbs_with_data": sum(1 for d in train_dbs if kept_per_db[d] > 0),
            "kept_per_db_min": min((kept_per_db[d] for d in train_dbs), default=0),
            "kept_per_db_median": median(kept_per_db[d] for d in train_dbs) if train_dbs else 0,
            "kept_per_db_max": max((kept_per_db[d] for d in train_dbs), default=0),
        },
        "complexity": dict(Counter(k.complexity for k in kept)),
        "features": {
            "join": sum(k.n_joins > 0 for k in kept),
            "subquery": sum(k.n_subqueries > 0 for k in kept),
            "group_by": sum(k.has_group_by for k in kept),
            "order_by": sum(k.has_order_by for k in kept),
            "set_op": sum(k.has_set_op for k in kept),
        },
        "question_words_median": median(q_words) if q_words else None,
    }


def render_markdown(s: dict[str, Any], kept: list[TrainExample], seed: int) -> str:
    n_raw, n_kept = s["n_raw"], s["n_kept"]
    out = [
        "# Synthetic data report",
        "",
        f"**{n_kept:,} verified pairs** kept from {n_raw:,} generated "
        f"({_pct(n_kept, n_raw)} keep rate).",
        "",
        "## By teacher",
        "",
        "| Teacher | Model | Calls | Tokens in / out | Raw pairs | Kept | Keep rate "
        "| Agreement | Cost at list price |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, t in s["teachers"].items():
        agree = (
            f"{100 * t['agreement_rate']:.0f}% of {t['agreement_checked']}"
            if t["agreement_rate"] is not None
            else "-"
        )
        cost = f"${t['list_price_cost_usd']:.3f}" if t["list_price_cost_usd"] is not None else "-"
        out.append(
            f"| {name} | `{t['model']}` | {t['calls']:,} | {t['prompt_tokens']:,} / "
            f"{t['completion_tokens']:,} | {t['raw_pairs']:,} | {t['kept']:,} "
            f"| {_pct(t['kept'], t['raw_pairs'])} | {agree} | {cost} |"
        )
    out += ["", "Actual spend: **$0** (free tiers).", "", "## Why pairs were rejected", ""]
    out += ["| Reason | Count | Share of all generated |", "|---|---:|---:|"]
    for reason, n in s["reject_reasons"].items():
        if n:
            out.append(f"| {reason} | {n:,} | {_pct(n, n_raw)} |")
    cov = s["coverage"]
    out += [
        "",
        "## Coverage",
        "",
        f"{cov['dbs_with_data']} of {cov['train_dbs']} train databases have data; kept pairs per "
        f"DB: min {cov['kept_per_db_min']}, median {cov['kept_per_db_median']}, "
        f"max {cov['kept_per_db_max']}.",
        "",
        "## What the SQL looks like",
        "",
        "| Complexity | Pairs | Share |",
        "|---|---:|---:|",
    ]
    for level in ("easy", "medium", "hard", "extra"):
        n = s["complexity"].get(level, 0)
        out.append(f"| {level} | {n:,} | {_pct(n, n_kept)} |")
    out += ["", "| Feature | Share of kept |", "|---|---:|"]
    for feat, n in s["features"].items():
        out.append(f"| {feat.replace('_', ' ')} | {_pct(n, n_kept)} |")
    out += [
        "",
        f"Median question length: {s['question_words_median']} words.",
        "",
        "## Random kept examples",
        "",
    ]
    for ex in random.Random(seed).sample(kept, min(5, len(kept))):
        out += [
            f"- **{ex.db_id}** ({ex.teacher}, {ex.complexity}): {ex.question}",
            f"  ```sql\n  {ex.sql}\n  ```",
        ]
    return "\n".join(out) + "\n"
