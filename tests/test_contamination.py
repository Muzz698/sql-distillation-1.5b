from __future__ import annotations

import json
from pathlib import Path

import pytest

from distill_sql.config import ContaminationConfig
from distill_sql.data.contamination import TrainRow, check, load_train_rows, normalize
from distill_sql.data.datasets import DatabaseEntry, Example, Registry, Role

CFG = ContaminationConfig(ngram_n=8, near_dup_shingle_n=3, near_dup_jaccard=0.8)

EVAL_QUESTION = "What are the names of all singers who performed in a concert held in 2014?"


def _db(db_id: str, role: Role, fingerprint: str) -> DatabaseEntry:
    return DatabaseEntry(
        dataset="spider",
        db_id=db_id,
        role=role,
        splits=["train_spider" if role == "train" else "dev"],
        path=f"x/{db_id}.sqlite",
        fingerprint=fingerprint,
        n_tables=1,
        n_columns=1,
        n_rows=1,
    )


REGISTRY = Registry(
    databases=[
        _db("pets_1", "train", "fp-pets"),
        _db("college_1", "train", "fp-college"),
        _db("concert_singer", "eval", "fp-concert"),
        # Excluded DBs may share a schema with eval DBs; that is the reason they are excluded.
        _db("formula_1", "excluded", "fp-concert"),
    ]
)
EVAL = [
    Example(
        uid="spider/dev/0",
        dataset="spider",
        split="dev",
        db_id="concert_singer",
        question=EVAL_QUESTION,
        sql="SELECT 1",
    )
]


def _run(*rows: tuple[str, str], registry: Registry = REGISTRY) -> dict[str, int]:
    train = [TrainRow(f"t:{i}", db, q) for i, (db, q) in enumerate(rows)]
    return dict(check(registry, EVAL, train, CFG).counts)


def test_clean_data_passes() -> None:
    assert _run(("pets_1", "How many pets weigh more than 10 kilograms?")) == {}


def test_eval_db_in_training_rows_fails() -> None:
    assert _run(("concert_singer", "How many singers are there?")) == {"eval_db_in_train": 1}


def test_db_name_check_is_case_insensitive() -> None:
    assert _run(("Concert_Singer", "How many singers?")) == {"eval_db_in_train": 1}


def test_excluded_db_in_training_rows_fails() -> None:
    assert _run(("formula_1", "Which driver won in 2009?")) == {"excluded_db_in_train": 1}


def test_unknown_db_fails() -> None:
    assert _run(("made_up_db", "How many rows?")) == {"unknown_db_in_train": 1}


def test_exact_question_match_after_normalisation() -> None:
    copied = "  what are the NAMES of all singers who performed in a concert held in 2014 "
    assert _run(("pets_1", copied)) == {"exact_question": 1}


def test_ngram_overlap() -> None:
    q = "List the names of all singers who performed in a concert held in 2014 in Paris."
    assert _run(("pets_1", q)) == {"ngram_overlap": 1}


def test_near_duplicate_without_long_ngram() -> None:
    cfg = ContaminationConfig(ngram_n=13, near_dup_shingle_n=3, near_dup_jaccard=0.6)
    # One word changed in the middle: every 13-gram is broken, but most 3-word shingles survive.
    q = "What are the names of the singers who performed in a concert held in 2014?"
    train = [TrainRow("t:0", "pets_1", q)]
    assert dict(check(REGISTRY, EVAL, train, cfg).counts) == {"near_duplicate": 1}


def test_similar_template_on_other_topic_is_not_flagged() -> None:
    # Spider reuses templates across databases; shared phrasing alone is not contamination.
    assert _run(("pets_1", "What are the names of all pets that weigh more than 10?")) == {}


def test_split_level_name_and_schema_overlap() -> None:
    leaky = Registry(
        databases=[
            *REGISTRY.databases,
            _db("concert_singer", "train", "fp-other"),  # same name as an eval DB
            _db("concert_copy", "train", "fp-concert"),  # same schema as an eval DB
        ]
    )
    assert _run(registry=leaky) == {"split_db_overlap": 1, "split_schema_overlap": 1}


def test_load_train_rows_requires_fields(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "a.jsonl").write_text(json.dumps({"db_id": "pets_1"}) + "\n")
    with pytest.raises(ValueError, match="need 'db_id' and 'question'"):
        load_train_rows(["train/*.jsonl"], tmp_path)


def test_load_train_rows_recursive_glob(tmp_path: Path) -> None:
    nested = tmp_path / "train" / "teacher_a"
    nested.mkdir(parents=True)
    (nested / "x.jsonl").write_text(json.dumps({"db_id": "pets_1", "question": "q?"}) + "\n")
    rows = load_train_rows(["train/**/*.jsonl"], tmp_path)
    assert [(r.source, r.db_id) for r in rows] == [("train/teacher_a/x.jsonl:1", "pets_1")]


def test_normalize() -> None:
    assert normalize("  How MANY singers?!  are\tthere ") == "how many singers are there"
