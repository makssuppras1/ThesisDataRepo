#!/usr/bin/env python3
"""Extract a Cookie header line from a Chrome/Safari HAR for FindIt / GetIt requests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlparse


def header_cookie(request: dict) -> str | None:
    for h in request.get("headers") or []:
        if (h.get("name") or "").lower() == "cookie":
            v = (h.get("value") or "").strip()
            return v or None
    cookies = request.get("cookies") or []
    if not cookies:
        return None
    parts = []
    for c in cookies:
        n, v = c.get("name"), c.get("value")
        if n is not None and v is not None:
            parts.append(f"{n}={v}")
    return "; ".join(parts) if parts else None


def host_matches(url: str) -> str | None:
    try:
        host = urlparse(url).hostname or ""
    except Exception:
        return None
    if host == "getit.findit.dtu.dk":
        return "getit"
    if host == "findit.dtu.dk":
        return "findit"
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("har_file", type=Path, help="Path to .har (may be misnamed e.g. .css)")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write cookie line to this file (default: print to stdout only)",
    )
    args = p.parse_args()

    path = args.har_file.expanduser().resolve()
    if not path.is_file():
        print(f"error: not a file: {path}", file=sys.stderr)
        return 2

    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"error: not valid JSON (is this really a HAR?): {e}", file=sys.stderr)
        return 2

    entries = (doc.get("log") or {}).get("entries") or []
    best_getit = ""
    best_findit = ""

    for ent in entries:
        req = ent.get("request") or {}
        url = req.get("url") or ""
        kind = host_matches(url)
        if not kind:
            continue
        c = header_cookie(req)
        if not c:
            continue
        if kind == "getit" and len(c) > len(best_getit):
            best_getit = c
        if kind == "findit" and len(c) > len(best_findit):
            best_findit = c

    # Prefer GetIt (resolver host); else catalog on findit.dtu.dk
    chosen = best_getit or best_findit
    source = "getit.findit.dtu.dk" if best_getit else ("findit.dtu.dk" if best_findit else None)

    if not chosen:
        print(
            "error: no Cookie found on findit.dtu.dk or getit.findit.dtu.dk requests.\n"
            "  Re-export HAR after opening a page that hits GetIt, or copy Cookie from DevTools.",
            file=sys.stderr,
        )
        return 1

    print(f"# host={source} chars={len(chosen)}", file=sys.stderr)
    if args.output:
        args.output.write_text(chosen + "\n", encoding="utf-8")
        print(f"wrote {args.output.resolve()}", file=sys.stderr)
    else:
        sys.stdout.write(chosen + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
