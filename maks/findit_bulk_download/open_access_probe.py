#!/usr/bin/env python3
"""Concurrently probe FindIt resolver URLs; extract open-access gateway PDF links (SSE JSON).

Uses aiohttp + asyncio. Requires valid cookies (-c). TLS verification is enabled (default).
Results are aggregated from return values (no shared mutable globals). URL work is fed
through a bounded worker pool (not one coroutine per URL).
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

_SENTINEL = object()


def load_bulk_download():
    path = Path(__file__).resolve().parent / "bulk_download.py"
    spec = importlib.util.spec_from_file_location("bulk_download", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_sse_first_json_text(text: str) -> dict[str, Any] | None:
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


def body_looks_like_html(text: str) -> bool:
    head = text[:500].lower().lstrip()
    return head.startswith("<!doctype") or head.startswith("<html")


def parse_retry_after_sec(headers: aiohttp.typedefs.LooseHeaders) -> float | None:
    v = headers.get("Retry-After")
    if not v:
        return None
    try:
        return max(0.0, float(v.strip()))
    except ValueError:
        return None


class Global429Cooldown:
    """Extend a shared cooldown deadline so workers pause together after 429s."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._lock = asyncio.Lock()
        self._until: float = 0.0

    async def cooperate(self, suggested: float) -> None:
        if not self.enabled:
            return
        sec = min(120.0, max(3.0, suggested))
        async with self._lock:
            now = time.monotonic()
            self._until = max(self._until, now + sec)
            target = self._until
        rem = target - time.monotonic()
        if rem > 0:
            await asyncio.sleep(rem)


@dataclass
class ResolveResult:
    resolver_url: str
    pdf_url: str | None = None
    resource_option: str | None = None
    error: str | None = None
    open_access: bool = False


async def resolve_one(
    session: aiohttp.ClientSession,
    resolver_url: str,
    cookie: str,
    ua: str,
    timeout_sec: float,
    retries: int,
    delay_ms: int,
    max_429_rounds: int,
    global_429: Global429Cooldown,
) -> ResolveResult:
    headers = {"User-Agent": ua}
    if cookie:
        headers["Cookie"] = cookie

    timeout = aiohttp.ClientTimeout(total=timeout_sec)
    last_err: str | None = None

    async def maybe_delay() -> None:
        if delay_ms > 0:
            await asyncio.sleep((delay_ms + random.random() * min(200, delay_ms)) / 1000.0)

    for attempt in range(max(1, retries)):
        last_err = None
        attempt_429 = 0
        while attempt_429 <= max_429_rounds:
            try:
                async with session.get(resolver_url, headers=headers, timeout=timeout) as resp:
                    if resp.status == 429:
                        if attempt_429 >= max_429_rounds:
                            await maybe_delay()
                            return ResolveResult(
                                resolver_url=resolver_url,
                                error="429_max_retries",
                            )
                        ra = parse_retry_after_sec(resp.headers)
                        backoff = min(
                            120.0,
                            (3.0 * (2**attempt_429) + random.random()) if ra is None else max(ra, 3.0),
                        )
                        if global_429.enabled:
                            await global_429.cooperate(backoff)
                        else:
                            await asyncio.sleep(backoff)
                        attempt_429 += 1
                        continue
                    if resp.status != 200:
                        last_err = f"http_{resp.status}"
                        break
                    text = await resp.text()

                payload = parse_sse_first_json_text(text)
                if payload is None:
                    if body_looks_like_html(text):
                        await maybe_delay()
                        return ResolveResult(resolver_url=resolver_url, error="http_200_html")
                    await maybe_delay()
                    return ResolveResult(resolver_url=resolver_url)

                ro = payload.get("resource_option") or ""
                raw_url = payload.get("url") or ""
                pdf_url = raw_url if isinstance(raw_url, str) else ""
                oa = ro == "download_open_access_local" and pdf_url.startswith("http")
                await maybe_delay()
                return ResolveResult(
                    resolver_url=resolver_url,
                    pdf_url=pdf_url if oa else None,
                    resource_option=str(ro) if ro else None,
                    open_access=oa,
                )
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                last_err = str(e)
                await asyncio.sleep((2**attempt) + random.random())
                break

        if last_err and attempt < retries - 1:
            await asyncio.sleep((2**attempt) + random.random())
            continue
        if last_err:
            await maybe_delay()
            return ResolveResult(resolver_url=resolver_url, error=last_err)

    await maybe_delay()
    return ResolveResult(resolver_url=resolver_url, error=last_err or "unknown")


def load_urls_from_errors_json(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(raw, list):
        raise ValueError("retry-errors JSON must be a list of objects")
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        u = item.get("url")
        if isinstance(u, str) and u.strip() and u not in seen:
            seen.add(u)
            out.append(u.strip())
    return out


def format_err_top(counter: Counter[str], n: int = 3, width: int = 72) -> str:
    if not counter:
        return ""
    parts = [f"{k}:{v}" for k, v in counter.most_common(n)]
    s = " ".join(parts)
    if len(s) > width:
        return s[: width - 3] + "..."
    return s


async def run_probe(args: argparse.Namespace) -> int:
    bd = load_bulk_download()
    cookie = bd.load_cookie_header(args.cookies.expanduser().resolve())
    ua = bd.UA

    urls: list[str]
    if args.retry_errors is not None:
        re_path = args.retry_errors.expanduser().resolve()
        if not re_path.is_file():
            print(f"error: --retry-errors file not found: {re_path}", file=sys.stderr)
            return 2
        try:
            urls = load_urls_from_errors_json(re_path)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"error: invalid retry-errors JSON: {e}", file=sys.stderr)
            return 2
        if not urls:
            print("error: no urls extracted from --retry-errors", file=sys.stderr)
            return 2
    else:
        if args.urls_file is None:
            print("error: urls_file is required unless --retry-errors is set", file=sys.stderr)
            return 2
        urls_file = args.urls_file.expanduser().resolve()
        if not urls_file.is_file():
            print(f"error: urls file not found: {urls_file}", file=sys.stderr)
            return 2
        urls = [ln.strip() for ln in urls_file.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]

    if args.max is not None:
        urls = urls[: args.max]

    total = len(urls)
    if total == 0:
        print("error: no URLs to process (empty input or --max 0)", file=sys.stderr)
        return 2

    out_json = args.output_json.expanduser().resolve()
    pdf_txt = args.pdf_urls_txt.expanduser().resolve()
    err_json = args.errors_json.expanduser().resolve()
    for p in (out_json, pdf_txt, err_json):
        p.parent.mkdir(parents=True, exist_ok=True)

    n_workers = max(1, args.concurrency)
    open_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    err_counter: Counter[str] = Counter()

    global_429 = Global429Cooldown(args.global_429_cooldown)

    t0 = time.monotonic()
    done = 0

    url_q: asyncio.Queue[str | object] = asyncio.Queue()
    for u in urls:
        await url_q.put(u)
    for _ in range(n_workers):
        await url_q.put(_SENTINEL)

    result_q: asyncio.Queue[ResolveResult] = asyncio.Queue(maxsize=n_workers * 4)

    async def worker(session: aiohttp.ClientSession) -> None:
        while True:
            item = await url_q.get()
            try:
                if item is _SENTINEL:
                    return
                assert isinstance(item, str)
                r = await resolve_one(
                    session,
                    item,
                    cookie,
                    ua,
                    float(args.timeout),
                    args.retries,
                    args.delay_ms,
                    args.max_429,
                    global_429,
                )
                await result_q.put(r)
            finally:
                url_q.task_done()

    async with aiohttp.TCPConnector(limit=n_workers, limit_per_host=n_workers) as connector:
        async with aiohttp.ClientSession(connector=connector) as session:
            workers = [asyncio.create_task(worker(session)) for _ in range(n_workers)]
            while done < total:
                r = await result_q.get()
                done += 1
                if r.error:
                    errors.append({"url": r.resolver_url, "error": r.error})
                    err_counter[r.error] += 1
                elif r.open_access and r.pdf_url:
                    open_rows.append(
                        {
                            "resolver_url": r.resolver_url,
                            "pdf_url": r.pdf_url,
                            "resource_option": r.resource_option,
                        }
                    )
                if done % 100 == 0 or done == total:
                    elapsed = time.monotonic() - t0
                    top = format_err_top(err_counter, 3)
                    extra = f" top_errors=[{top}]" if top else ""
                    print(
                        f"  {done}/{total} ({100.0 * done / total:.1f}%) open_access={len(open_rows)} "
                        f"errors={len(errors)} elapsed_s={elapsed:.1f}{extra}",
                        file=sys.stderr,
                    )
            await asyncio.gather(*workers)

    out_json.write_text(json.dumps(open_rows, indent=2) + "\n", encoding="utf-8")
    pdf_txt.write_text("\n".join(row["pdf_url"] for row in open_rows) + ("\n" if open_rows else ""), encoding="utf-8")
    err_json.write_text(json.dumps(errors, indent=2) + "\n", encoding="utf-8")

    top5 = err_counter.most_common(5)
    kinds = len(err_counter)
    summary = f"errors_total={len(errors)} error_kinds={kinds}"
    if top5:
        summary += " top5=" + ",".join(f"{k}:{c}" for k, c in top5)

    print(
        f"\ndone total={total} open_access={len(open_rows)} {summary}\n"
        f"  json: {out_json}\n"
        f"  pdf_urls_txt: {pdf_txt}\n"
        f"  errors: {err_json}",
        file=sys.stderr,
    )

    if len(errors) == 0:
        return 0
    if args.allow_partial and total > 0:
        return 0
    return 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "urls_file",
        type=Path,
        nargs="?",
        default=None,
        help="Deduped resolver URLs, one per line (omit if using only --retry-errors)",
    )
    p.add_argument("-c", "--cookies", type=Path, required=True)
    p.add_argument(
        "-o",
        "--output-json",
        type=Path,
        default=Path("open_access.json"),
        help="Write matching open-access rows here (default: ./open_access.json)",
    )
    p.add_argument(
        "--pdf-urls-txt",
        "--urls-txt",
        type=Path,
        default=None,
        help="Write one pdf_url per line (default: same stem as -o + _urls.txt)",
    )
    p.add_argument(
        "--errors-json",
        "--errors",
        type=Path,
        default=None,
        help="Write fetch errors here (default: same stem as -o + _errors.json)",
    )
    p.add_argument("--concurrency", type=int, default=8, help="Worker count and connection pool limit (default: 8)")
    p.add_argument("--timeout", type=float, default=15.0, help="Per-request total timeout seconds")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument(
        "--delay-ms",
        type=int,
        default=75,
        help="Jitter sleep after each finished request per worker (default: 75)",
    )
    p.add_argument("--max", type=int, default=None, help="Only process first N URLs (smoke test)")
    p.add_argument("--max-429", type=int, default=8, dest="max_429", help="Max extra retries for HTTP 429 per attempt")
    p.add_argument(
        "--retry-errors",
        type=Path,
        default=None,
        help="Read resolver URLs from a prior open_access *_errors.json list instead of urls_file",
    )
    p.add_argument(
        "--allow-partial",
        action="store_true",
        help="Exit 0 even if some URLs failed (outputs still written)",
    )
    p.set_defaults(global_429_cooldown=True)
    p.add_argument(
        "--no-global-429-cooldown",
        action="store_false",
        dest="global_429_cooldown",
        help="Disable shared 429 backoff (each worker backs off alone)",
    )
    args = p.parse_args()

    if args.pdf_urls_txt is None:
        args.pdf_urls_txt = args.output_json.with_name(args.output_json.stem + "_urls.txt")
    if args.errors_json is None:
        args.errors_json = args.output_json.with_name(args.output_json.stem + "_errors.json")

    if args.retry_errors is None and args.urls_file is None:
        p.error("urls_file is required unless --retry-errors is set")

    return asyncio.run(run_probe(args))


if __name__ == "__main__":
    raise SystemExit(main())
