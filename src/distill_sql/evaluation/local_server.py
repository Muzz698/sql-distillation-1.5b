"""Run a GGUF model behind llama.cpp's `llama-server`, which speaks the same OpenAI-compatible
API as the teachers. Evaluation (and later the FastAPI service) therefore talks to local and
hosted models through one client, with identical prompts, caching and timing.
"""

from __future__ import annotations

import logging
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from types import TracebackType

import httpx
import psutil

from distill_sql.config import LocalServerConfig

log = logging.getLogger(__name__)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class LlamaServer:
    def __init__(self, gguf: Path, cfg: LocalServerConfig, log_file: Path) -> None:
        if shutil.which(cfg.binary) is None:
            raise FileNotFoundError(
                f"`{cfg.binary}` not found. Install llama.cpp: `brew install llama.cpp`."
            )
        if not gguf.is_file():
            raise FileNotFoundError(f"{gguf} not found. Run `make models` first.")
        self.gguf, self.cfg, self.log_file = gguf, cfg, log_file
        self.port = _free_port()
        self.peak_rss_bytes = 0
        self._proc: subprocess.Popen[bytes] | None = None
        self._stop = threading.Event()
        self._sampler: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def __enter__(self) -> LlamaServer:
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.cfg.binary,
            "-m", str(self.gguf),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "-c", str(self.cfg.ctx_size),
            "-ngl", str(self.cfg.gpu_layers),
            "-np", "1",  # one request at a time: latency numbers stay clean
        ]  # fmt: skip
        with self.log_file.open("ab") as fh:
            self._proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
        self._wait_healthy()
        self._sampler = threading.Thread(target=self._sample_memory, daemon=True)
        self._sampler.start()
        log.info("llama-server ready on :%d (%s)", self.port, self.gguf.name)
        return self

    def _wait_healthy(self) -> None:
        assert self._proc is not None
        deadline = time.monotonic() + self.cfg.startup_timeout_s
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early; see {self.log_file}")
            try:
                if httpx.get(f"http://127.0.0.1:{self.port}/health", timeout=2).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        self.__exit__(None, None, None)
        raise TimeoutError(f"llama-server not healthy after {self.cfg.startup_timeout_s}s")

    def _sample_memory(self) -> None:
        assert self._proc is not None
        proc = psutil.Process(self._proc.pid)
        while not self._stop.wait(0.5):
            try:
                self.peak_rss_bytes = max(self.peak_rss_bytes, proc.memory_info().rss)
            except psutil.Error:
                return

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._proc.kill()
