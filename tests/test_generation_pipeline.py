"""End-to-end Phase 2 on a miniature Spider with fake teachers (no network)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from distill_sql.config import (
    PROJECT_ROOT,
    AppConfig,
    GenerationConfig,
    PathsConfig,
    Secrets,
    TeacherConfig,
    load_config,
)
from distill_sql.data.contamination import check, load_train_rows
from distill_sql.data.prepare import prepare
from distill_sql.generation import pipeline
from distill_sql.utils.io import iter_jsonl
from tests.conftest import make_spider_layout

GENERATED = [
    {"question": "How many pets are there in total?", "sql": "SELECT count(*) FROM pets"},
    {"question": "What is the weight of the heaviest pet?", "sql": "SELECT max(weight) FROM pets"},
    {
        "question": "Which pets are horses in the data?",
        "sql": "SELECT * FROM pets WHERE kind='horse'",
    },
    {"question": "List the names of every pet here.", "sql": "SELECT name FROM pets"},
    {"question": "How many singers do we have?", "sql": "SELECT count(*) FROM pets"},
    {"question": "Delete all of the pets please now", "sql": "DELETE FROM pets"},
]
CROSSCHECK_ANSWERS = ["SELECT count(*) FROM pets", "SELECT min(weight) FROM pets"]


def fake_teacher(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    prompt = body["messages"][-1]["content"]
    if "Write one SQLite query for each question" in prompt:
        n = len(re.findall(r"^\d+\. ", prompt, flags=re.MULTILINE))
        content = json.dumps(CROSSCHECK_ANSWERS[:n])
    else:
        content = "```json\n" + json.dumps(GENERATED) + "\n```"
    return httpx.Response(
        200,
        json={
            "model": body["model"],
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        },
    )


@pytest.fixture
def ctx(tmp_path: Path) -> pipeline.Context:
    base = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    paths = PathsConfig(
        data_dir=Path("data"),
        raw_dir=Path("data/raw"),
        processed_dir=Path("data/processed"),
        cache_dir=Path("data/cache"),
        runs_dir=Path("runs"),
    ).resolved(tmp_path)
    spider = base.datasets["spider"].model_copy(
        update={"train_splits": ["train_spider"], "eval_splits": ["dev"], "exclude_dbs": {}}
    )
    teachers = [
        TeacherConfig(name="teacher-a", provider="groq", model="a", rpm=999, tpm=10**6, rpd=999),
        TeacherConfig(name="teacher-b", provider="gemini", model="b", rpm=999, tpm=10**6, rpd=999),
    ]
    gen = GenerationConfig(
        output_dir=tmp_path / "data/interim/generated",
        pairs_per_call=6,
        calls_per_db=2,
        fewshot_per_db=1,
        focuses=["joins", "aggregation"],
    )
    cfg: AppConfig = base.model_copy(
        update={
            "paths": paths,
            "datasets": {"spider": spider},
            "teachers": teachers,
            "generation": gen,
        }
    )
    make_spider_layout(
        paths.raw_dir / "spider",
        {
            "train_spider": [("pets_1", "How many pets?", "SELECT count(*) FROM pets")],
            "dev": [
                ("concert_singer", "How many singers do we have?", "SELECT count(*) FROM singer")
            ],
        },
    )
    prepare(cfg, tmp_path)
    return pipeline.build_context(cfg, tmp_path)


SECRETS = Secrets(groq_api_key=SecretStr("k"), gemini_api_key=SecretStr("k"))


def http() -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(fake_teacher))


def test_generate_is_resumable(ctx: pipeline.Context) -> None:
    teacher_a = ctx.cfg.teacher("teacher-a")
    [first] = pipeline.generate(ctx, [teacher_a], SECRETS, max_calls=1, http=http())
    assert (first.calls, first.pairs, first.skipped_done) == (1, 6, 0)
    [second] = pipeline.generate(ctx, [teacher_a], SECRETS, http=http())
    assert (second.calls, second.skipped_done) == (1, 1)  # only the remaining task ran
    [third] = pipeline.generate(ctx, [teacher_a], SECRETS, http=http())
    assert (third.calls, third.skipped_done) == (0, 2)


def test_verify_crosscheck_and_contamination(ctx: pipeline.Context) -> None:
    pipeline.generate(ctx, [ctx.cfg.teacher("teacher-a")], SECRETS, http=http())
    stats = pipeline.verify(ctx)

    assert stats["n_raw"] == 12 and stats["n_kept"] == 2
    # Both calls return the same 6 pairs. Content checks run before dedupe, so each bad pair
    # is rejected for its real reason twice; the 2 good pairs are kept once, then duplicates.
    reasons = {k: v for k, v in stats["reject_reasons"].items() if v}
    assert reasons == {
        "contaminated": 2,  # copies the dev question
        "not_select": 2,  # DELETE
        "exec_error": 2,  # pets has no `name` column
        "empty_result": 2,  # no horses
        "duplicate_question": 2,
    }
    kept = [row["question"] for row in iter_jsonl(ctx.sft_file)]
    assert "How many pets are there in total?" in kept

    summary = pipeline.crosscheck(ctx, ctx.cfg.teacher("teacher-b"), SECRETS, http=http())
    assert summary.checked == 2
    pipeline.verify(ctx)
    agreement = {r["question"]: r["agreement"] for r in iter_jsonl(ctx.sft_file)}
    assert agreement["How many pets are there in total?"] is True
    assert agreement["What is the weight of the heaviest pet?"] is False

    report = (ctx.cfg.paths.processed_dir / "reports" / "generation_report.md").read_text()
    assert "verified pairs" in report and "empty_result" in report

    # The data we just produced must pass the Phase 1 contamination gate.
    rows = load_train_rows(ctx.cfg.contamination.train_globs, ctx.root)
    assert rows, "sft.jsonl and the few-shot file should both be checked"
    eval_examples = list(ctx.eval_index.examples)
    result = check(ctx.registry, eval_examples, rows, ctx.cfg.contamination)
    assert result.passed, result.to_dict()
