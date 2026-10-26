from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from distill_sql.config import Secrets, TeacherConfig
from distill_sql.generation.cache import ResponseCache
from distill_sql.generation.client import TeacherClient, TeacherError, api_key_for
from distill_sql.generation.ratelimit import DailyLimitReached, RateLimiter

TEACHER = TeacherConfig(
    name="groq-test",
    provider="groq",
    model="m",
    rpm=1000,
    tpm=1_000_000,
    rpd=1000,
    extra_body={"reasoning_effort": "low"},
)
MESSAGES = [{"role": "user", "content": "hi"}]


def ok(text: str = "SELECT 1") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "m",
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


def make(
    tmp_path: Path, handler: Callable[[httpx.Request], httpx.Response]
) -> tuple[TeacherClient, list[float]]:
    slept: list[float] = []
    limiter = RateLimiter("t", 1000, 1_000_000, 1000, None, tmp_path / "usage.json")
    client = TeacherClient(
        TEACHER,
        "key",
        ResponseCache(tmp_path / "cache"),
        limiter,
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=slept.append,
    )
    return client, slept


def test_success_is_cached_and_extra_body_is_sent(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        requests.append(req)
        return ok()

    client, _ = make(tmp_path, handler)
    first = client.complete(MESSAGES)
    second = client.complete(MESSAGES)
    assert (first.cached, second.cached) == (False, True)
    assert first.text == second.text == "SELECT 1"
    assert first.total_tokens == 15
    assert len(requests) == 1  # the second call never hit the network
    body = json.loads(requests[0].content)
    assert body["reasoning_effort"] == "low"
    assert requests[0].headers["authorization"] == "Bearer key"


def test_different_params_are_different_cache_entries(tmp_path: Path) -> None:
    calls = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return ok()

    client, _ = make(tmp_path, handler)
    client.complete(MESSAGES, temperature=0.0)
    client.complete(MESSAGES, temperature=0.7)
    assert calls == 2


def test_429_respects_retry_after(tmp_path: Path) -> None:
    responses = [httpx.Response(429, headers={"retry-after": "7"}, text="slow down"), ok()]
    client, slept = make(tmp_path, lambda req: responses.pop(0))
    assert client.complete(MESSAGES).text == "SELECT 1"
    assert 7 <= slept[0] < 8


def test_429_daily_quota_stops_the_job(tmp_path: Path) -> None:
    msg = "Rate limit reached for model on tokens per day (TPD): Limit 200000"
    client, _ = make(tmp_path, lambda req: httpx.Response(429, text=msg))
    with pytest.raises(DailyLimitReached):
        client.complete(MESSAGES)
    with pytest.raises(DailyLimitReached):  # remembered: no further requests today
        client.complete([{"role": "user", "content": "other"}])


def test_server_errors_are_retried(tmp_path: Path) -> None:
    responses = [httpx.Response(503), httpx.Response(500), ok()]
    client, slept = make(tmp_path, lambda req: responses.pop(0))
    assert client.complete(MESSAGES).text == "SELECT 1"
    assert len(slept) == 2


def test_client_errors_fail_fast(tmp_path: Path) -> None:
    calls = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, text="invalid api key")

    client, _ = make(tmp_path, handler)
    with pytest.raises(TeacherError, match="401"):
        client.complete(MESSAGES)
    assert calls == 1


@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_or_empty_key_gives_a_clear_error(value: str | None) -> None:
    secrets = Secrets(groq_api_key=SecretStr(value) if value is not None else None)
    with pytest.raises(TeacherError, match="GROQ_API_KEY is empty or not set"):
        api_key_for("groq", secrets)


def test_hidden_reasoning_tokens_count_toward_limits(tmp_path: Path) -> None:
    # Gemini bills "thinking" in total_tokens but not in completion_tokens.
    resp = httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "SELECT 1"}}],
            "usage": {"prompt_tokens": 17, "completion_tokens": 6, "total_tokens": 151},
        },
    )
    client, _ = make(tmp_path, lambda req: resp)
    completion = client.complete(MESSAGES)
    assert completion.reasoning_tokens == 128
    assert completion.total_tokens == 151
    assert client.limiter.tokens_today == 151
