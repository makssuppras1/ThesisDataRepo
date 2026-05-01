#!/usr/bin/env python3
"""Merge all FindIt resolver URL .txt files in this directory, deduped by rft_dat JSON id."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

OUTPUT_NAME = "dtu_theses_all_urls_deduped.txt"


def thesis_id_from_resolver_url(line: str) -> str | None:
    line = line.strip()
    if not line.startswith("http"):
        return None
    q = parse_qs(urlparse(line).query)
    rft_dat = q.get("rft_dat")
    if not rft_dat:
        return None
    raw = rft_dat[0]
    try:
        data = json.loads(unquote(raw))
    except json.JSONDecodeError:
        return None
    tid = data.get("id")
    return str(tid) if tid is not None else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing *.txt URL lists (default: this script's directory)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=f"Output file (default: DIR/{OUTPUT_NAME})",
    )
    args = p.parse_args()
    url_dir: Path = args.dir
    out_path = args.output or (url_dir / OUTPUT_NAME)

    id_to_url: dict[str, str] = {}
    raw_lines = 0
    skipped_bad = 0
    files_read = 0

    for path in sorted(url_dir.glob("*.txt")):
        if path.name == OUTPUT_NAME or path.resolve() == out_path.resolve():
            continue
        files_read += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            raw_lines += 1
            tid = thesis_id_from_resolver_url(line)
            if tid is None:
                skipped_bad += 1
                continue
            id_to_url[tid] = line.strip()

    out_path.write_text("\n".join(id_to_url[tid] for tid in sorted(id_to_url)) + "\n", encoding="utf-8")

    print(
        f"files_read={files_read} raw_lines={raw_lines} unique_ids={len(id_to_url)} "
        f"skipped_unparseable={skipped_bad} -> {out_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
