"""Download dataset archives with checksum verification and extract them safely."""

from __future__ import annotations

import hashlib
import logging
import shutil
import time
import zipfile
from pathlib import Path

import httpx
from tqdm import tqdm

from distill_sql.config import SourceConfig
from distill_sql.utils.io import sha256_file

log = logging.getLogger(__name__)

_EXTRACTED_MARKER = ".extracted"


class DownloadError(RuntimeError):
    pass


def _part_path(dest: Path, url: str) -> Path:
    # One partial file per mirror: bytes from two different servers must never be mixed.
    tag = hashlib.sha1(url.encode()).hexdigest()[:8]
    return dest.with_name(f"{dest.name}.{tag}.part")


def _fetch(url: str, dest: Path, client: httpx.Client) -> None:
    """Download ``url`` to ``dest``, resuming a previous partial download when possible."""
    part = _part_path(dest, url)
    offset = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with client.stream("GET", url, headers=headers, follow_redirects=True) as resp:
        if resp.status_code == 416:  # nothing left to send: the partial file is complete
            part.replace(dest)
            return
        resp.raise_for_status()
        # Google Drive answers quota / virus-scan problems with an HTML page and status 200.
        if "text/html" in resp.headers.get("content-type", ""):
            raise DownloadError(f"{url} returned an HTML page instead of an archive")
        if resp.status_code != 206:  # server ignored the Range header: start over
            offset = 0
        length = int(resp.headers.get("content-length", 0))
        with (
            part.open("ab" if offset else "wb") as fh,
            tqdm(
                total=offset + length or None,
                initial=offset,
                unit="B",
                unit_scale=True,
                desc=dest.name,
                leave=False,
            ) as bar,
        ):
            for chunk in resp.iter_bytes(chunk_size=1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))
    part.replace(dest)


def _verify(path: Path, expected: str | None) -> None:
    actual = sha256_file(path)
    if expected is None:
        log.warning("%s has no pinned sha256; got %s. Pin it in the config.", path.name, actual)
    elif actual != expected:
        raise DownloadError(f"{path.name}: sha256 mismatch (expected {expected}, got {actual})")


def download(
    source: SourceConfig,
    dest_dir: Path,
    client: httpx.Client | None = None,
    attempts_per_mirror: int = 10,
    backoff_s: float = 3.0,
) -> Path:
    """Return the local archive path, downloading it from the first working mirror if needed.

    Network drops and stalls (common for ~1 GB files from Google Drive) are retried with
    backoff and resume from the bytes already on disk. HTTP errors, HTML pages and checksum
    mismatches move on to the next mirror instead.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / source.filename
    if dest.exists():
        _verify(dest, source.sha256)
        log.info("%s already downloaded and verified", dest.name)
        return dest

    own_client = client is None
    http = client or httpx.Client(timeout=httpx.Timeout(60.0, read=120.0))
    errors: list[str] = []
    try:
        for url in source.urls:
            log.info("downloading %s from %s", source.filename, url)
            try:
                for attempt in range(1, attempts_per_mirror + 1):
                    try:
                        _fetch(url, dest, http)
                        break
                    except httpx.TransportError as exc:  # timeout / connection drop: resume
                        if attempt == attempts_per_mirror:
                            raise
                        log.warning("attempt %d failed (%s); resuming", attempt, exc)
                        time.sleep(backoff_s * attempt)
                _verify(dest, source.sha256)
                return dest
            except (httpx.HTTPError, DownloadError) as exc:
                dest.unlink(missing_ok=True)
                _part_path(dest, url).unlink(missing_ok=True)
                errors.append(f"{url}: {exc}")
                log.warning("mirror failed: %s", errors[-1])
    finally:
        if own_client:
            http.close()
    raise DownloadError(f"all mirrors failed for {source.filename}:\n  " + "\n  ".join(errors))


def safe_extract(archive: Path, dest_dir: Path) -> Path:
    """Extract a zip into ``dest_dir``, refusing entries that escape it ("zip slip").

    Idempotent: a marker file records which archive (by hash) was extracted.
    """
    digest = sha256_file(archive)
    marker = dest_dir / _EXTRACTED_MARKER
    if marker.exists() and marker.read_text().strip() == digest:
        log.info("%s already extracted", archive.name)
        return dest_dir

    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True)
    root = dest_dir.resolve()
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = info.filename
            # Skip macOS resource forks that ship inside some dataset zips.
            if name.startswith("__MACOSX/") or Path(name).name.startswith("._"):
                continue
            target = (dest_dir / name).resolve()
            if not target.is_relative_to(root):
                raise DownloadError(f"{archive.name}: entry escapes target dir: {name!r}")
            is_symlink = (info.external_attr >> 16) & 0o170000 == 0o120000
            if is_symlink:
                raise DownloadError(f"{archive.name}: symlink entries are not allowed: {name!r}")
            zf.extract(info, dest_dir)
    marker.write_text(digest + "\n")
    return dest_dir
