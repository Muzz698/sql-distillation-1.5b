from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import httpx
import pytest

from distill_sql.config import SourceConfig
from distill_sql.data.download import DownloadError, _part_path, download, safe_extract

PAYLOAD = b"PK-fake-archive-bytes"
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


def _source(urls: list[str], sha: str | None = PAYLOAD_SHA) -> SourceConfig:
    return SourceConfig(filename="a.zip", urls=urls, sha256=sha, extract_to="a")


def _client(routes: dict[str, httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(lambda req: routes[str(req.url)]))


def test_download_verifies_checksum(tmp_path: Path) -> None:
    client = _client({"https://m1/a.zip": httpx.Response(200, content=PAYLOAD)})
    path = download(_source(["https://m1/a.zip"]), tmp_path, client)
    assert path.read_bytes() == PAYLOAD


def test_download_rejects_bad_checksum_and_leaves_no_file(tmp_path: Path) -> None:
    client = _client({"https://m1/a.zip": httpx.Response(200, content=b"tampered")})
    with pytest.raises(DownloadError, match="all mirrors failed"):
        download(_source(["https://m1/a.zip"]), tmp_path, client)
    assert not (tmp_path / "a.zip").exists()


def test_download_falls_back_to_next_mirror(tmp_path: Path) -> None:
    client = _client(
        {
            # Google Drive returns an HTML quota page with status 200 when it refuses.
            "https://m1/a.zip": httpx.Response(
                200, content=b"<html>quota</html>", headers={"content-type": "text/html"}
            ),
            "https://m2/a.zip": httpx.Response(200, content=PAYLOAD),
        }
    )
    path = download(_source(["https://m1/a.zip", "https://m2/a.zip"]), tmp_path, client)
    assert path.read_bytes() == PAYLOAD


def test_download_resumes_partial_file_with_range_request(tmp_path: Path) -> None:
    url = "https://m1/a.zip"
    _part_path(tmp_path / "a.zip", url).write_bytes(PAYLOAD[:5])
    seen_ranges: list[str | None] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen_ranges.append(req.headers.get("range"))
        return httpx.Response(206, content=PAYLOAD[5:])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    path = download(_source([url]), tmp_path, client)
    assert seen_ranges == ["bytes=5-"]
    assert path.read_bytes() == PAYLOAD


def test_download_retries_network_errors(tmp_path: Path) -> None:
    calls = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise httpx.ReadTimeout("stalled", request=req)
        return httpx.Response(200, content=PAYLOAD)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    path = download(_source(["https://m1/a.zip"]), tmp_path, client, backoff_s=0)
    assert calls == 3
    assert path.read_bytes() == PAYLOAD


def test_download_skips_existing_verified_file(tmp_path: Path) -> None:
    (tmp_path / "a.zip").write_bytes(PAYLOAD)
    client = _client({})  # any request would raise KeyError
    assert download(_source(["https://m1/a.zip"]), tmp_path, client).exists()


def _zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def test_safe_extract_skips_macos_junk_and_is_idempotent(tmp_path: Path) -> None:
    archive = _zip(
        tmp_path / "ok.zip",
        {"data/x.json": b"[]", "__MACOSX/data/._x.json": b"junk", "data/._y": b"junk"},
    )
    out = safe_extract(archive, tmp_path / "out")
    assert (out / "data" / "x.json").exists()
    assert not (out / "__MACOSX").exists()
    assert not (out / "data" / "._y").exists()

    (out / "data" / "x.json").write_text("modified")
    safe_extract(archive, out)  # same archive hash -> no re-extraction
    assert (out / "data" / "x.json").read_text() == "modified"


def test_safe_extract_blocks_zip_slip(tmp_path: Path) -> None:
    archive = _zip(tmp_path / "evil.zip", {"../../escaped.txt": b"pwned"})
    with pytest.raises(DownloadError, match="escapes"):
        safe_extract(archive, tmp_path / "out")
    assert not (tmp_path.parent / "escaped.txt").exists()
