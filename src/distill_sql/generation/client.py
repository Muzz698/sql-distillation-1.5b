"""One chat client for every teacher.

Groq and Gemini both expose an OpenAI-compatible ``/chat/completions`` endpoint, so a single
small client covers both, with no vendor SDKs. Every request goes through the disk cache
and the rate limiter. Retries:

* 429 (rate limited): wait for ``Retry-After`` (or back off), unless the message says the
  *daily* quota is gone, in which case :class:`DailyLimitReached` stops the job.
* 5xx and network errors: exponential backoff with jitter.
* other 4xx (bad request, bad key): fail immediately; retrying cannot help.
"""

from __future__ import annotations

import logging
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from distill_sql.config import Secrets, TeacherConfig
from distill_sql.generation.cache import ResponseCache
from distill_sql.generation.ratelimit import DailyLimitReached, RateLimiter

log = logging.getLogger(__name__)

PROVIDERS: dict[str, tuple[str, str]] = {
    # provider -> (OpenAI-compatible base URL, Secrets field holding the key)
    "groq": ("https://api.groq.com/openai/v1", "groq_api_key"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini_api_key"),
}
_DAILY_HINTS = re.compile(r"per day|\bRPD\b|\bTPD\b|daily", re.IGNORECASE)

Message = dict[str, str]


class TeacherError(RuntimeError):
    pass


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cached: bool
    # Hidden "thinking" tokens. Some providers (Gemini) bill and rate-limit them but leave
    # them out of completion_tokens, so they only show up in usage.total_tokens.
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens + self.reasoning_tokens


def estimate_tokens(messages: list[Message]) -> int:
    """Rough count (~3.5 chars/token for English + SQL); only used to pace requests."""
    return int(sum(len(m["content"]) for m in messages) / 3.5) + 8 * len(messages)


def api_key_for(provider: str, secrets: Secrets) -> str:
    field = PROVIDERS[provider][1]
    value = getattr(secrets, field)
    key = value.get_secret_value().strip() if value is not None else ""
    if not key:  # `KEY=` in .env loads as an empty string, not None
        raise TeacherError(f"{field.upper()} is empty or not set. Add it to .env (KEY=value).")
    return key


class TeacherClient:
    def __init__(
        self,
        teacher: TeacherConfig,
        api_key: str,
        cache: ResponseCache,
        limiter: RateLimiter,
        http: httpx.Client | None = None,
        max_retries: int = 6,
        expected_output_tokens: int = 1500,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.teacher = teacher
        self.cache = cache
        self.limiter = limiter
        self.base_url = PROVIDERS[teacher.provider][0]
        self.http = http or httpx.Client(timeout=httpx.Timeout(30.0, read=180.0))
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.max_retries = max_retries
        self.expected_output_tokens = expected_output_tokens
        self._sleep = sleep

    def request_payload(
        self, messages: list[Message], temperature: float, max_tokens: int
    ) -> dict[str, Any]:
        return {
            "model": self.teacher.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **self.teacher.extra_body,
        }

    def complete(
        self, messages: list[Message], temperature: float = 0.0, max_tokens: int = 2048
    ) -> Completion:
        payload = self.request_payload(messages, temperature, max_tokens)
        key = ResponseCache.key({"provider": self.teacher.provider, **payload})
        if (hit := self.cache.get(key)) is not None:
            return Completion(**hit["completion"], cached=True)

        est = estimate_tokens(messages) + min(max_tokens, self.expected_output_tokens)
        for attempt in range(1, self.max_retries + 1):
            self.limiter.acquire(est)
            started = time.perf_counter()
            try:
                resp = self.http.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=self.headers
                )
            except httpx.TransportError as exc:
                self.limiter.record(0)
                self._backoff(attempt, f"network error: {exc}")
                continue
            latency = time.perf_counter() - started

            if resp.status_code == 200:
                completion = self._parse(resp.json(), latency)
                self.limiter.record(completion.total_tokens)
                self.cache.put(
                    key,
                    {
                        "request": payload,
                        "completion": {
                            k: v for k, v in completion.__dict__.items() if k != "cached"
                        },
                    },
                )
                return completion

            self.limiter.record(0)
            body = resp.text[:500]
            if resp.status_code == 429:
                if _DAILY_HINTS.search(body):
                    self.limiter.exhaust_today()
                    raise DailyLimitReached(f"{self.teacher.name}: provider daily limit: {body}")
                self._backoff(attempt, "rate limited (429)", resp.headers.get("retry-after"))
            elif resp.status_code >= 500:
                self._backoff(attempt, f"server error {resp.status_code}")
            else:
                raise TeacherError(f"{self.teacher.name}: HTTP {resp.status_code}: {body}")
        raise TeacherError(f"{self.teacher.name}: gave up after {self.max_retries} attempts")

    def _backoff(self, attempt: int, reason: str, retry_after: str | None = None) -> None:
        if attempt == self.max_retries:
            return
        try:
            delay = float(retry_after) if retry_after else 0.0
        except ValueError:
            delay = 0.0
        delay = max(delay, min(60.0, 2.0**attempt)) + random.uniform(0, 1)
        log.warning("%s: %s; retry %d in %.1fs", self.teacher.name, reason, attempt, delay)
        self._sleep(delay)

    def _parse(self, data: dict[str, Any], latency: float) -> Completion:
        try:
            text = data["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError) as exc:
            raise TeacherError(f"{self.teacher.name}: unexpected response shape: {data}") from exc
        usage = data.get("usage") or {}
        prompt = int(usage.get("prompt_tokens", 0))
        completion = int(usage.get("completion_tokens", 0))
        total = int(usage.get("total_tokens", 0))
        return Completion(
            text=text,
            model=data.get("model", self.teacher.model),
            prompt_tokens=prompt,
            completion_tokens=completion,
            latency_s=round(latency, 3),
            cached=False,
            reasoning_tokens=max(0, total - prompt - completion),
        )

    def list_models(self) -> list[str]:
        resp = self.http.get(f"{self.base_url}/models", headers=self.headers)
        resp.raise_for_status()
        return sorted(m["id"] for m in resp.json().get("data", []))
