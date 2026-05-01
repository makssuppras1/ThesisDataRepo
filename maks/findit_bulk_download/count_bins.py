#!/usr/bin/env python3
"""Offline: scan *.bin SSE JSON for resource_option (e.g. download_open_access_local vs access_prohibited)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_sse_first_json_text(text: str) -> dict | None:
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload in ("none", "[DONE]"):
            continue
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            continue
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "dir",
        type=Path,
        nargs="?",
        default=Path("downloads"),
        help="Directory containing *.bin (default: ./downloads)",
    )
    args = p.parse_args()
    d = args.dir.expanduser().resolve()
    if not d.is_dir():
        print(f"error: not a directory: {d}", file=sys.stderr)
        return 2

    n_prohibited = n_open_local = n_gateway_other_ro = n_other = 0
    for path in sorted(d.glob("*.bin")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"skip_read {path}: {e}", file=sys.stderr)
            continue
        payload = parse_sse_first_json_text(text)
        if payload is None:
            n_other += 1
            continue
        ro = payload.get("resource_option")
        url = payload.get("url")
        if ro == "access_prohibited" or not url:
            n_prohibited += 1
        elif ro == "download_open_access_local" and isinstance(url, str) and url.startswith("http"):
            n_open_local += 1
        elif isinstance(url, str) and url.startswith("http"):
            n_gateway_other_ro += 1
        else:
            n_other += 1

    total = n_prohibited + n_open_local + n_gateway_other_ro + n_other
    n_gateway = n_open_local + n_gateway_other_ro
    remainder = n_gateway + n_other
    print(
        f"dir={d} total_bin_files={total} "
        f"resource_option_download_open_access_local={n_open_local} "
        f"has_gateway_url_other_resource_option={n_gateway_other_ro} "
        f"access_prohibited_or_no_url={n_prohibited} "
        f"other_or_unparseable={n_other} "
        f"remainder_not_prohibited={remainder}",
        file=sys.stderr,
    )
    print(n_open_local)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
