"""Scoring helpers: execution match, confidence intervals and a paired significance test."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Literal

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from distill_sql.verification.executor import ExecResult, Row, canon_row, results_match

Comparison = Literal["spider", "bird"]


def gold_is_ordered(gold_sql: str) -> bool:
    """True if the gold query's *outermost* SELECT has ORDER BY (then row order matters)."""
    try:
        tree = sqlglot.parse_one(gold_sql, read="sqlite")
    except SqlglotError:
        return "order by" in gold_sql.lower()
    return (
        tree is not None
        and tree.args.get("order") is not None
        and isinstance(tree, exp.Select | exp.Union | exp.Intersect | exp.Except)
    )


def execution_match(pred: ExecResult, gold: ExecResult, rule: Comparison, ordered: bool) -> bool:
    """Execution accuracy for one example.

    * spider: rows compared as a multiset, or as a list when the gold query orders them
      (the convention of Spider's official execution evaluation).
    * bird: rows compared as a set (BIRD's official evaluation: set(pred) == set(gold)).
    """
    if not pred.ok or not gold.ok:
        return False
    if rule == "bird":
        return set(map(canon_row, pred.rows)) == set(map(canon_row, gold.rows))
    return results_match(gold.rows, pred.rows, ordered=ordered)


def bootstrap_ci(
    values: Sequence[bool | float], n_samples: int = 1000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a mean (e.g. accuracy)."""
    if not values:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    xs = [float(v) for v in values]
    n = len(xs)
    means = sorted(sum(rng.choices(xs, k=n)) / n for _ in range(n_samples))
    lo = means[int(alpha / 2 * n_samples)]
    hi = means[min(n_samples - 1, int((1 - alpha / 2) * n_samples))]
    return (lo, hi)


def mcnemar(a: Sequence[bool], b: Sequence[bool]) -> tuple[int, int, float]:
    """Exact (binomial) McNemar test on paired results.

    Returns (a_only, b_only, two-sided p-value): a_only = questions A got right and B wrong.
    Only disagreements carry information about which model is better.
    """
    if len(a) != len(b):
        raise ValueError("paired results must have the same length")
    a_only = sum(x and not y for x, y in zip(a, b, strict=True))
    b_only = sum(y and not x for x, y in zip(a, b, strict=True))
    n = a_only + b_only
    if n == 0:
        return (0, 0, 1.0)
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return (a_only, b_only, min(1.0, 2 * tail))


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return math.nan
    xs = sorted(values)
    idx = (len(xs) - 1) * q
    lo, hi = math.floor(idx), math.ceil(idx)
    return xs[lo] + (xs[hi] - xs[lo]) * (idx - lo)


__all__ = [
    "Row",
    "bootstrap_ci",
    "execution_match",
    "gold_is_ordered",
    "mcnemar",
    "percentile",
]
