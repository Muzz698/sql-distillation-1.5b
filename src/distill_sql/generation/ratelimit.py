"""Client-side rate limiting for free-tier APIs.

Per-minute limits (RPM, TPM) use a sliding 60-second window and simply wait. Per-day limits
(RPD, TPD) are persisted to disk, so they survive restarts. When a daily budget is used up,
:class:`DailyLimitReached` is raised and the job stops cleanly; re-running tomorrow resumes it.

We stay a little under every limit (``safety``) because providers count tokens slightly
differently from our estimate.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from distill_sql.utils.io import read_json, write_json


class DailyLimitReached(RuntimeError):  # noqa: N818 - a normal stop signal, not a bug
    pass


def _utc_day() -> str:
    return datetime.now(UTC).date().isoformat()


class RateLimiter:
    def __init__(
        self,
        name: str,
        rpm: int,
        tpm: int,
        rpd: int,
        tpd: int | None,
        state_file: Path,
        safety: float = 0.9,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        today: Callable[[], str] = _utc_day,
    ) -> None:
        self.name = name
        self.rpm = max(1, int(rpm * safety))
        self.tpm = max(1, int(tpm * safety))
        self.rpd = max(1, int(rpd * safety))
        self.tpd = int(tpd * safety) if tpd else None
        self.state_file = state_file
        self._clock, self._sleep, self._today = clock, sleep, today
        self._window: deque[tuple[float, int]] = deque()  # (timestamp, tokens)
        self._day, self.requests_today, self.tokens_today = self._load()

    # ---- daily state ------------------------------------------------------
    def _load(self) -> tuple[str, int, int]:
        today = self._today()
        if self.state_file.exists():
            s = read_json(self.state_file)
            if s.get("day") == today:
                return today, int(s["requests"]), int(s["tokens"])
        return today, 0, 0

    def _save(self) -> None:
        write_json(
            self.state_file,
            {"day": self._day, "requests": self.requests_today, "tokens": self.tokens_today},
        )

    def _roll_day(self) -> None:
        if self._today() != self._day:
            self._day, self.requests_today, self.tokens_today = self._today(), 0, 0

    # ---- public API -------------------------------------------------------
    def acquire(self, est_tokens: int) -> None:
        """Block until a request of ``est_tokens`` fits in every limit."""
        if est_tokens > self.tpm:
            raise ValueError(f"{self.name}: one request ({est_tokens} tok) exceeds TPM {self.tpm}")
        self._roll_day()
        if self.requests_today + 1 > self.rpd:
            raise DailyLimitReached(f"{self.name}: daily request limit ({self.rpd}) reached")
        if self.tpd is not None and self.tokens_today + est_tokens > self.tpd:
            raise DailyLimitReached(f"{self.name}: daily token limit ({self.tpd}) reached")
        while True:
            now = self._clock()
            while self._window and now - self._window[0][0] >= 60.0:
                self._window.popleft()
            used = sum(tok for _, tok in self._window)
            if len(self._window) < self.rpm and used + est_tokens <= self.tpm:
                self._window.append((now, est_tokens))
                return
            self._sleep(60.0 - (now - self._window[0][0]) + 0.05)

    def record(self, actual_tokens: int, count_request: bool = True) -> None:
        """Replace the latest reservation with what the provider actually reported.

        ``count_request=False`` is for requests the provider did not serve (5xx, network
        errors): they still pace the per-minute window but do not use up the daily budget.
        """
        if self._window:
            ts, _ = self._window.pop()
            self._window.append((ts, actual_tokens))
        self.requests_today += int(count_request)
        self.tokens_today += actual_tokens
        self._save()

    def exhaust_today(self) -> None:
        """The provider says the daily budget is gone (it counts more than we do): trust it."""
        self.requests_today = self.rpd
        self._save()
