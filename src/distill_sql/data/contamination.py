"""Test-set contamination check.

Fails if anything from an evaluation split can reach training, at two levels:

Database level (the split itself):
  * ``split_db_overlap``      a database name is registered as both train and eval
  * ``split_schema_overlap``  a train DB has the same schema fingerprint as an eval DB
                              (catches one database copied under another name)

Row level (every JSONL file we train on or show to a teacher):
  * ``eval_db_in_train``      a training row points at an eval database
  * ``excluded_db_in_train``  a training row points at a DB excluded from training (e.g. a
                              Spider train DB that BIRD reuses for evaluation)
  * ``unknown_db_in_train``   a training row points at a DB that is not a registered train DB
  * ``exact_question``        an eval question appears verbatim (after normalisation)
  * ``ngram_overlap``         shares a word n-gram (default 13) with an eval question
  * ``near_duplicate``        word-shingle Jaccard similarity >= threshold (default 0.8)
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from distill_sql.config import ContaminationConfig
from distill_sql.data.datasets import Example, Registry
from distill_sql.utils.io import iter_jsonl

MAX_EXAMPLES_PER_RULE = 25


@dataclass(frozen=True)
class TrainRow:
    source: str  # "file:line", for error messages
    db_id: str
    question: str


@dataclass
class Finding:
    rule: str
    detail: str


@dataclass
class ContaminationReport:
    n_train_rows: int
    n_eval_questions: int
    n_train_dbs: int
    n_eval_dbs: int
    counts: Counter[str] = field(default_factory=Counter)
    examples: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    def add(self, finding: Finding) -> None:
        self.counts[finding.rule] += 1
        if len(self.examples[finding.rule]) < MAX_EXAMPLES_PER_RULE:
            self.examples[finding.rule].append(finding.detail)

    @property
    def passed(self) -> bool:
        return not self.counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "n_train_rows": self.n_train_rows,
            "n_eval_questions": self.n_eval_questions,
            "n_train_dbs": self.n_train_dbs,
            "n_eval_dbs": self.n_eval_dbs,
            "violations": dict(self.counts),
            "examples": dict(self.examples),
        }


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def shingles(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    """Like ngrams, but short texts become one shingle instead of none."""
    return ngrams(tokens, n) or ({tuple(tokens)} if tokens else set())


def load_train_rows(globs: Iterable[str], root: Path) -> list[TrainRow]:
    rows: list[TrainRow] = []
    for pattern in globs:
        for path in sorted(root.glob(pattern)):
            rel = path.relative_to(root)
            for i, rec in enumerate(iter_jsonl(path), start=1):
                if "db_id" not in rec or "question" not in rec:
                    raise ValueError(f"{rel}:{i}: training rows need 'db_id' and 'question'")
                rows.append(TrainRow(f"{rel}:{i}", str(rec["db_id"]), str(rec["question"])))
    return rows


def check(
    registry: Registry,
    eval_examples: list[Example],
    train_rows: list[TrainRow],
    cfg: ContaminationConfig,
) -> ContaminationReport:
    train_dbs = registry.by_role("train")
    eval_dbs = registry.by_role("eval")
    report = ContaminationReport(
        n_train_rows=len(train_rows),
        n_eval_questions=len(eval_examples),
        n_train_dbs=len(train_dbs),
        n_eval_dbs=len(eval_dbs),
    )

    # ---- database level ---------------------------------------------------
    eval_names = {d.db_id.lower(): d for d in eval_dbs}
    eval_prints = {d.fingerprint: d for d in eval_dbs}
    for d in train_dbs:
        if d.db_id.lower() in eval_names:
            e = eval_names[d.db_id.lower()]
            report.add(Finding("split_db_overlap", f"{d.key} is also eval DB {e.key}"))
        if d.fingerprint in eval_prints:
            e = eval_prints[d.fingerprint]
            report.add(Finding("split_schema_overlap", f"{d.key} has the same schema as {e.key}"))

    # ---- row level: databases ---------------------------------------------
    train_names = {d.db_id.lower() for d in train_dbs}
    excluded = {d.db_id.lower(): d for d in registry.by_role("excluded")}
    for row in train_rows:
        db = row.db_id.lower()
        if db in eval_names:
            report.add(Finding("eval_db_in_train", f"{row.source}: db_id={row.db_id}"))
        elif db in excluded:
            reason = excluded[db].note
            report.add(Finding("excluded_db_in_train", f"{row.source}: {row.db_id} ({reason})"))
        elif db not in train_names:
            report.add(Finding("unknown_db_in_train", f"{row.source}: db_id={row.db_id}"))

    # ---- row level: questions ---------------------------------------------
    index = EvalIndex(eval_examples, cfg)
    for row in train_rows:
        match = index.match(row.question)
        if match is not None:
            rule, ex, detail = match
            report.add(Finding(rule, f"{row.source} ~ {ex.uid}{detail}: {row.question!r}"))
    return report


class EvalIndex:
    """Fast lookup of "does this question leak an eval question?".

    Used by :func:`check` and by the generation pipeline, which drops leaking questions
    before they are ever written to training data.
    """

    def __init__(self, eval_examples: list[Example], cfg: ContaminationConfig) -> None:
        self.cfg = cfg
        self.examples = eval_examples
        self._norm: dict[str, Example] = {}
        self._ngrams: dict[tuple[str, ...], Example] = {}
        self._shingles: dict[tuple[str, ...], list[int]] = defaultdict(list)
        self._sizes: list[int] = []
        for idx, ex in enumerate(eval_examples):
            tokens = normalize(ex.question).split()
            self._norm.setdefault(" ".join(tokens), ex)
            for g in ngrams(tokens, cfg.ngram_n):
                self._ngrams.setdefault(g, ex)
            sh = shingles(tokens, cfg.near_dup_shingle_n)
            self._sizes.append(len(sh))
            for s in sh:
                self._shingles[s].append(idx)

    def match(self, question: str) -> tuple[str, Example, str] | None:
        """Return ``(rule, eval_example, detail)`` for the strongest match, or ``None``."""
        tokens = normalize(question).split()
        if (ex := self._norm.get(" ".join(tokens))) is not None:
            return "exact_question", ex, ""
        for g in ngrams(tokens, self.cfg.ngram_n):
            if (ex := self._ngrams.get(g)) is not None:
                return "ngram_overlap", ex, ""
        sh = shingles(tokens, self.cfg.near_dup_shingle_n)
        shared: Counter[int] = Counter()
        for s in sh:
            shared.update(self._shingles.get(s, ()))
        for idx, inter in shared.most_common():
            jaccard = inter / (len(sh) + self._sizes[idx] - inter)
            if jaccard >= self.cfg.near_dup_jaccard:
                return "near_duplicate", self.examples[idx], f" (J={jaccard:.2f})"
        return None
