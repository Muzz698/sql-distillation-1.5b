"""Content-addressed disk cache for LLM responses.

The key is a hash of everything that determines the answer (provider, model, messages,
sampling params). Re-running a job therefore never pays for the same request twice, and a
crashed job resumes for free.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from distill_sql.utils.io import read_json, write_json


class ResponseCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    @staticmethod
    def key(request: dict[str, Any]) -> str:
        blob = json.dumps(request, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        data: dict[str, Any] = read_json(path)
        return data

    def put(self, key: str, value: dict[str, Any]) -> None:
        write_json(self._path(key), value)
