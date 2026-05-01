#!/usr/bin/env python3
"""Fetch first-hop resolver bodies for all URLs in the deduped list, parse SSE JSON, write one file per thesis id.

Fast mode: parallel workers (urllib per thread). Tune --workers and --delay-ms if you see 429s.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def load_bulk_download():
    path = Path(__file__).resolve().parent / "bulk_download.py"
    spec = importlib.util.spec_from_file_location("bulk_download", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def classify(body: bytes, bd) -> dict:
    out: dict = {"class": "unknown"}
    if body[:8].startswith(b"%PDF"):
        out["class"] = "direct_pdf"
        return out
    payload = bd.parse_sse_first_json(body)
    if payload is None:
        out["class"] = "unparsed"
        return out
    out["resource_option"] = payload.get("resource_option")
    out["gateway_url"] = payload.get("url")
    ro = payload.get("resource_option")
    url = payload.get("url")
    if ro == "access_prohibited":
        out["class"] = "access_prohibited"
    elif ro == "download_open_access_local" and isinstance(url, str) and url.startswith("http"):
        out["class"] = "download_open_access_local"
    elif isinstance(url, str) and url.startswith("http"):
        out["class"] = "has_gateway_other_ro"
    else:
        out["class"] = "no_gateway_url"
    return out


def worker_task(
    item: tuple[str, str],
    cookie: str,
    part_dir: Path,
    out_dir: Path,
    sock_timeout: float,
    delay_ms: int,
    bd,
    keep_raw_sse: bool,
) -> tuple[str, str, bool]:
    """Returns (tid, class, ok)."""
    url, tid = item
    part = part_dir / f"{tid}.part"
    try:
        part.unlink(missing_ok=True)
        headers = {"User-Agent": bd.UA}
        if cookie:
            headers["Cookie"] = cookie
        ok, _hdrs = bd.streaming_get(url, headers, part, tid, sock_timeout, "unpack")
        if not ok:
            return tid, "fetch_fail", False
        body = part.read_bytes()
        info = classify(body, bd)
        info["thesis_id"] = tid
        info["resolver_url"] = url
        if keep_raw_sse:
            (out_dir / f"{tid}.sse.txt").write_bytes(body)
        (out_dir / f"{tid}.json").write_text(json.dumps(info, indent=0) + "\n", encoding="utf-8")
        return tid, info.get("class", "unknown"), True
    finally:
        part.unlink(missing_ok=True)
        if delay_ms > 0:
            time.sleep((delay_ms + random.random() * min(500, delay_ms)) / 1000.0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("urls_file", type=Path)
    p.add_argument("-c", "--cookies", type=Path, required=True)
    p.add_argument("-o", "--out-dir", type=Path, default=Path("resolver_unpack"))
    p.add_argument("--workers", type=int, default=6, help="Parallel resolver fetches (default 6)")
    p.add_argument("--delay-ms", type=int, default=150, help="Per-request sleep after each fetch in a worker (default 150)")
    p.add_argument("--connect-timeout-sec", type=int, default=30)
    p.add_argument("--request-timeout-min", type=int, default=2, help="Resolver first hop is small SSE; 2 min default")
    p.add_argument("--max", type=int, default=None, help="Process at most this many URLs (test)")
    p.add_argument("--keep-raw-sse", action="store_true", help="Also write {id}.sse.txt body")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip IDs that already have {id}.json in out-dir (resume)",
    )
    args = p.parse_args()

    bd = load_bulk_download()
    cookie = bd.load_cookie_header(args.cookies.expanduser().resolve())
    urls_file = args.urls_file.expanduser().resolve()
    if not urls_file.is_file():
        print(f"error: {urls_file}", file=sys.stderr)
        return 2

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    part_dir = Path(tempfile.mkdtemp(prefix="unpack_parts_"))

    read_sec = float(args.request_timeout_min * 60)
    sock_timeout = max(float(args.connect_timeout_sec), read_sec)

    items: list[tuple[str, str]] = []
    for line in urls_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tid = bd.thesis_id_from_resolver_url(line)
        if tid is None:
            continue
        items.append((line, tid))
        if args.max is not None and len(items) >= args.max:
            break

    if args.skip_existing:
        before = len(items)
        items = [(u, t) for u, t in items if not (out_dir / f"{t}.json").is_file()]
        print(f"skip_existing removed {before - len(items)} already done", file=sys.stderr)

    n_ok = n_fail = 0
    counts: dict[str, int] = {}

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = {
                ex.submit(
                    worker_task,
                    it,
                    cookie,
                    part_dir,
                    out_dir,
                    sock_timeout,
                    args.delay_ms,
                    bd,
                    args.keep_raw_sse,
                ): it[1]
                for it in items
            }
            for fut in as_completed(futs):
                try:
                    _tid, cls, ok = fut.result()
                except Exception as e:
                    print(f"error {futs[fut]}: {e}", file=sys.stderr)
                    n_fail += 1
                    continue
                if ok:
                    n_ok += 1
                    counts[cls] = counts.get(cls, 0) + 1
                else:
                    n_fail += 1
    finally:
        import shutil

        shutil.rmtree(part_dir, ignore_errors=True)

    print(
        f"done total={len(items)} json_written={n_ok} fetch_fail={n_fail} "
        f"by_class={json.dumps(counts, sort_keys=True)} out_dir={out_dir}",
        file=sys.stderr,
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
