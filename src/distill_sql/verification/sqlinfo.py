"""Static facts about a SQL query, from its parse tree (sqlglot).

Used to (1) reject anything that is not a single read-only query, (2) deduplicate queries
that differ only in formatting, and (3) describe the dataset (how many joins, subqueries...).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

Complexity = Literal["easy", "medium", "hard", "extra"]

_READ_ONLY_ROOTS = (exp.Select, exp.Union, exp.Intersect, exp.Except)


@dataclass(frozen=True)
class SqlInfo:
    canonical: str  # normalised SQL text, for dedupe
    n_tables: int
    n_joins: int
    n_subqueries: int
    n_aggregates: int
    has_group_by: bool
    has_having: bool
    has_order_by: bool
    has_limit: bool
    has_set_op: bool

    @property
    def complexity(self) -> Complexity:
        """A rough Spider-like hardness bucket for describing training data.

        (Evaluation uses the official Spider hardness function in Phase 3.)
        """
        score = (
            self.n_joins
            + 2 * self.n_subqueries
            + 2 * self.has_set_op
            + self.has_group_by
            + self.has_having
            + (self.has_order_by and self.has_limit)
            + (self.n_aggregates > 1)
        )
        if score == 0:
            return "easy"
        if score <= 2:
            return "medium"
        if score <= 4:
            return "hard"
        return "extra"


class SqlCheckError(ValueError):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


def analyse(sql: str) -> SqlInfo:
    """Parse ``sql`` as SQLite. Raises :class:`SqlCheckError` with reason ``parse_error``,
    ``multiple_statements`` or ``not_select``."""
    try:
        trees = [t for t in sqlglot.parse(sql, read="sqlite") if t is not None]
    except SqlglotError as exc:
        raise SqlCheckError("parse_error", str(exc).splitlines()[0]) from exc
    if len(trees) != 1:
        raise SqlCheckError("multiple_statements", f"{len(trees)} statements")
    tree = trees[0]
    if not isinstance(tree, _READ_ONLY_ROOTS):
        raise SqlCheckError("not_select", type(tree).__name__)
    if any(tree.find_all(exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create)):
        raise SqlCheckError("not_select", "contains a write statement")

    selects = list(tree.find_all(exp.Select))
    return SqlInfo(
        canonical=tree.sql(dialect="sqlite", normalize=True, pretty=False),
        n_tables=len({t.name.lower() for t in tree.find_all(exp.Table)}),
        n_joins=len(list(tree.find_all(exp.Join))),
        n_subqueries=max(
            0, len(selects) - 1 - len(list(tree.find_all(exp.Union, exp.Intersect, exp.Except)))
        ),
        n_aggregates=len(list(tree.find_all(exp.AggFunc))),
        has_group_by=tree.find(exp.Group) is not None,
        has_having=tree.find(exp.Having) is not None,
        has_order_by=tree.find(exp.Order) is not None,
        has_limit=tree.find(exp.Limit) is not None,
        has_set_op=isinstance(tree, (exp.Union, exp.Intersect, exp.Except))
        or tree.find(exp.Union, exp.Intersect, exp.Except) is not None,
    )
