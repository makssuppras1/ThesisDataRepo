#!/usr/bin/env python3
"""Count resolver URLs that expose a downloadable fulltext link (SSE gateway URL or direct PDF).

Modes:
  --urls + -c   Probe each resolver (network; same cookies as bulk_download).
  --artifacts-dir   Offline: scan *.pdf and *.bin (SSE) in a downloads folder.
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
import tempfile
import time
from pathlib import Path


def load_bulk_download():
    path = Path(__file__).resolve().parent / "bulk_download.py"
    spec = importlib.util.spec_from_file_location("bulk_download", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def classify_resolver_body(body: bytes, bd) -> str:
    """Return: downloadable | prohibited | no_gateway | direct_pdf | unclassified | fail"""
    if body[:8].startswith(b"%PDF"):
        return "direct_pdf"
    payload = bd.parse_sse_first_json(body)
    if payload is None:
        return "unclassified"
    ro = payload.get("resource_option")
    url = payload.get("url")
    if ro == "access_prohibited":
        return "prohibited"
    if isinstance(url, str) and url.startswith("http"):
        return "downloadable"
    return "no_gateway"


def run_artifacts(dir_path: Path) -> int:
    downloadable_ids: set[str] = set()
    prohibited_ids: set[str] = set()

    for p in dir_path.glob("*.pdf"):
        if p.is_file() and p.stat().st_size > 0:
            downloadable_ids.add(p.stem)

    bd = load_bulk_download()
    for p in dir_path.glob("*.bin"):
        if not p.is_file():
            continue
        try:
            body = p.read_bytes()
        except OSError:
            continue
        cat = classify_resolver_body(body, bd)
        tid = p.stem
        if cat == "downloadable" or cat == "direct_pdf":
            downloadable_ids.add(tid)
        elif cat == "prohibited":
            prohibited_ids.add(tid)

    suf = bd.PROHIBITED_MARK
    for p in dir_path.glob(f"*{suf}"):
        if p.is_file() and p.name.endswith(suf):
            prohibited_ids.add(p.name[: -len(suf)])

    n_dl = len(downloadable_ids)
    n_pr = len(prohibited_ids)
    print(
        f"artifacts_dir={dir_path.resolve()} "
        f"downloadable_ids_union_pdf_or_gateway_sse={n_dl} "
        f"prohibited_like_ids_tomb_or_bin={n_pr}",
        file=sys.stderr,
    )
    print(n_dl)
    return 0


def run_probe(
    urls_file: Path,
    cookie_path: Path,
    delay_ms: int,
    max_lines: int | None,
    connect_sec: int,
    read_min: int,
) -> int:
    bd = load_bulk_download()
    cookie = bd.load_cookie_header(cookie_path)
    read_sec = float(read_min * 60)
    sock_timeout = max(float(connect_sec), read_sec)

    tmp = Path(tempfile.mkdtemp(prefix="findit_probe_"))
    part = tmp / "probe.part"
    try:
        lines = urls_file.read_text(encoding="utf-8", errors="replace").splitlines()
        seen: set[str] = set()
        n_downloadable = n_prohibited = n_no_gateway = n_unclassified = n_fail = n_dup = 0
        n_unparseable = 0
        processed = 0

        for line in lines:
            if max_lines is not None and processed >= max_lines:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tid = bd.thesis_id_from_resolver_url(line)
            if tid is None:
                n_unparseable += 1
                continue
            if tid in seen:
                n_dup += 1
                continue
            seen.add(tid)

            headers = {"User-Agent": bd.UA}
            if cookie:
                headers["Cookie"] = cookie
            part.unlink(missing_ok=True)
            ok, _hdrs = bd.streaming_get(line, headers, part, tid, sock_timeout, "probe")
            if not ok:
                n_fail += 1
                processed += 1
                time.sleep((delay_ms + random.random() * 1000) / 1000.0)
                continue

            body = part.read_bytes()
            cat = classify_resolver_body(body, bd)
            if cat in ("downloadable", "direct_pdf"):
                n_downloadable += 1
            elif cat == "prohibited":
                n_prohibited += 1
            elif cat == "no_gateway":
                n_no_gateway += 1
            else:
                n_unclassified += 1
            part.unlink(missing_ok=True)
            processed += 1
            time.sleep((delay_ms + random.random() * 1000) / 1000.0)

        n_unique = len(seen)
        remainder = n_downloadable
        print(
            f"urls_file={urls_file.resolve()} unique_urls_probed={n_unique} "
            f"downloadable_link_or_pdf={n_downloadable} access_prohibited={n_prohibited} "
            f"no_gateway_url={n_no_gateway} unclassified_body={n_unclassified} "
            f"http_network_fail={n_fail} skip_duplicate_line={n_dup} skip_unparseable={n_unparseable}",
            file=sys.stderr,
        )
        print(remainder)
        return 0
    finally:
        try:
            part.unlink(missing_ok=True)
            tmp.rmdir()
        except OSError:
            pass


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts-dir", type=Path, default=None, help="Offline: count from PDF + SSE .bin files here")
    p.add_argument("--urls", type=Path, default=None, help="Deduped resolver URL list (one per line)")
    p.add_argument("-c", "--cookies", type=Path, default=None, help="Cookie file (required for --urls probe)")
    p.add_argument("--delay-ms", type=int, default=3000)
    p.add_argument("--max", type=int, default=None, help="Probe at most this many unique URLs (testing)")
    p.add_argument("--connect-timeout-sec", type=int, default=30)
    p.add_argument("--request-timeout-min", type=int, default=15)
    args = p.parse_args()

    if args.artifacts_dir is not None:
        d = args.artifacts_dir.expanduser().resolve()
        if not d.is_dir():
            print(f"error: not a directory: {d}", file=sys.stderr)
            return 2
        return run_artifacts(d)

    if args.urls is None:
        p.error("Provide --artifacts-dir DIR or --urls FILE with -c cookies.txt")
    if args.cookies is None:
        p.error("--urls probe requires -c cookies.txt")
    uf = args.urls.expanduser().resolve()
    cf = args.cookies.expanduser().resolve()
    if not uf.is_file() or not cf.is_file():
        print("error: --urls and -c must be existing files", file=sys.stderr)
        return 2
    return run_probe(uf, cf, args.delay_ms, args.max, args.connect_timeout_sec, args.request_timeout_min)


if __name__ == "__main__":
    raise SystemExit(main())
