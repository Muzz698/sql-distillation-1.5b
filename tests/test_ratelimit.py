from __future__ import annotations

from pathlib import Path

import pytest

from distill_sql.generation.ratelimit import DailyLimitReached, RateLimiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.now += s


def _limiter(tmp_path: Path, clock: FakeClock, day: str = "2026-10-27", **kw: int) -> RateLimiter:
    limits = {"rpm": 2, "tpm": 1000, "rpd": 100, "tpd": 10_000} | kw
    return RateLimiter(
        "t",
        limits["rpm"],
        limits["tpm"],
        limits["rpd"],
        limits["tpd"],
        state_file=tmp_path / "usage.json",
        safety=1.0,
        clock=clock.clock,
        sleep=clock.sleep,
        today=lambda: day,
    )


def test_waits_when_requests_per_minute_are_used_up(tmp_path: Path) -> None:
    clock = FakeClock()
    lim = _limiter(tmp_path, clock)
    for _ in range(3):
        lim.acquire(10)
        lim.record(10)
    assert len(clock.slept) == 1
    assert clock.now >= 60.0


def test_waits_when_tokens_per_minute_are_used_up(tmp_path: Path) -> None:
    clock = FakeClock()
    lim = _limiter(tmp_path, clock, rpm=100)
    lim.acquire(800)
    lim.record(800)
    lim.acquire(300)  # 800 + 300 > 1000 -> must wait for the window to clear
    assert clock.slept


def test_record_replaces_estimate_with_actual(tmp_path: Path) -> None:
    clock = FakeClock()
    lim = _limiter(tmp_path, clock, rpm=100)
    lim.acquire(900)
    lim.record(100)  # provider says it was only 100 tokens
    lim.acquire(800)
    assert not clock.slept


def test_daily_limits_raise_and_persist(tmp_path: Path) -> None:
    clock = FakeClock()
    lim = _limiter(tmp_path, clock, rpm=100, tpm=10_000, rpd=2)
    for _ in range(2):
        lim.acquire(10)
        lim.record(10)
    with pytest.raises(DailyLimitReached):
        lim.acquire(10)
    # A new process on the same day sees the same usage...
    with pytest.raises(DailyLimitReached):
        _limiter(tmp_path, clock, rpm=100, tpm=10_000, rpd=2).acquire(10)
    # ...and the budget resets on the next day.
    _limiter(tmp_path, clock, day="2026-10-28", rpm=100, tpm=10_000, rpd=2).acquire(10)


def test_daily_token_limit(tmp_path: Path) -> None:
    lim = _limiter(tmp_path, FakeClock(), rpm=100, tpm=10_000, tpd=50)
    lim.acquire(40)
    lim.record(40)
    with pytest.raises(DailyLimitReached, match="token"):
        lim.acquire(20)


def test_request_bigger_than_tpm_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exceeds TPM"):
        _limiter(tmp_path, FakeClock()).acquire(5000)


def test_unserved_requests_do_not_use_the_daily_budget(tmp_path: Path) -> None:
    lim = _limiter(tmp_path, FakeClock(), rpm=100, tpm=10_000, rpd=2)
    for _ in range(5):  # e.g. five 503s in a row
        lim.acquire(10)
        lim.record(0, count_request=False)
    assert lim.requests_today == 0
    lim.acquire(10)  # the daily budget is untouched


def test_quota_day_follows_the_provider_timezone() -> None:
    from datetime import UTC, datetime
    from zoneinfo import ZoneInfo

    from distill_sql.generation.ratelimit import day_in

    assert day_in("UTC")() == datetime.now(UTC).date().isoformat()
    pacific = datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat()
    assert day_in("America/Los_Angeles")() == pacific
