"""Few-shot exemplars for teacher prompts: Spider *train* questions for the same database.

They show the teacher the style and difficulty we want. They are never training targets.
Any exemplar that the contamination index flags (e.g. the generic "How many books are
there?", which also appears in Spider test) is dropped before it reaches a prompt.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from distill_sql.data.contamination import EvalIndex
from distill_sql.data.datasets import Example, Registry
from distill_sql.utils.io import iter_jsonl, write_jsonl


def build_fewshot_pool(
    registry: Registry,
    train_examples: list[Example],
    eval_index: EvalIndex,
    per_db: int,
    seed: int,
) -> dict[str, list[Example]]:
    usable = {d.db_id for d in registry.by_role("train")}
    by_db: dict[str, list[Example]] = defaultdict(list)
    for ex in train_examples:
        if ex.db_id in usable and eval_index.match(ex.question) is None:
            by_db[ex.db_id].append(ex)
    rng = random.Random(seed)
    pool: dict[str, list[Example]] = {}
    for db_id in sorted(by_db):
        candidates = sorted(by_db[db_id], key=lambda e: e.uid)
        pool[db_id] = rng.sample(candidates, min(per_db, len(candidates)))
    return pool


def save_pool(pool: dict[str, list[Example]], path: Path) -> None:
    write_jsonl(path, [ex for db_id in sorted(pool) for ex in pool[db_id]])


def load_pool(path: Path) -> dict[str, list[Example]]:
    pool: dict[str, list[Example]] = defaultdict(list)
    if path.exists():
        for row in iter_jsonl(path):
            ex = Example.model_validate(row)
            pool[ex.db_id].append(ex)
    return dict(pool)
