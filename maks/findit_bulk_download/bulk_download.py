#!/usr/bin/env python3
"""Bulk-download FindIt resolver URLs (stdlib only).

First hop: GET resolver URL (often returns SSE with JSON).
Second hop: GET fulltext-gateway URL from JSON when resource_option allows.
Skips access_prohibited (writes .access_prohibited tombstone for resume).

Progress is written to OUT_DIR/bulk_download_checkpoint.json (JSON) before and after
each download attempt; on Ctrl+C the file records in_progress_* so you can see where
you stopped. Re-run the same command to continue: existing PDFs are skipped, failures retry.
Use --no-checkpoint to disable, or --checkpoint-file PATH to choose the file.

Append-only JSONL manifest (gateway oid/targetid for joins): OUT_DIR/bulk_download_manifest.jsonl
unless --no-manifest or --manifest-file PATH.

Use --skip-ok-from-manifest to skip thesis ids whose latest manifest line has outcome ok (e.g. after moving PDFs off disk). Optional --skip-ok-manifest-file PATH reads a different JSONL; default read path matches the output manifest.
"""

from __future__ import annotations

import argparse
import datetime
import http.client
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from email.utils import parsedate_to_datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qs, unquote, urlparse

MAX_429 = 10
MAX_NET = 10
UA = "FindItBulkDownload/1.0-py"
PROHIBITED_MARK = ".access_prohibited"


def thesis_id_from_resolver_url(line: str) -> str | None:
    line = line.strip()
    if not line.startswith("http"):
        return None
    q = parse_qs(urlparse(line).query)
    rft_dat = q.get("rft_dat")
    if not rft_dat:
        return None
    try:
        data = json.loads(unquote(rft_dat[0]))
    except json.JSONDecodeError:
        return None
    tid = data.get("id")
    return str(tid) if tid is not None else None


def gateway_query_ids(url: str | None) -> tuple[str | None, str | None]:
    """Parse oid and targetid from a gateway (or any) HTTP URL query string."""
    if not url or not url.startswith("http"):
        return None, None
    q = parse_qs(urlparse(url).query)
    oid = (q.get("oid") or [None])[0]
    targetid = (q.get("targetid") or [None])[0]
    return oid, targetid


def filename_from_content_disposition(cd: str) -> str | None:
    m = re.search(r"filename\*?=(?:UTF-8''|\"?)([^\";]+)", cd or "", re.I)
    if not m:
        return None
    raw = m.group(1)
    try:
        return unquote(raw)
    except Exception:
        return raw


@dataclass
class DownloadRecord:
    outcome: Literal["ok", "prohibited", "fail"]
    resolver_url: str
    rft_dat_id: str
    gateway_url: str | None
    gateway_oid: str | None
    gateway_targetid: str | None
    resource_option: str | None
    content_disposition_filename: str | None


def append_manifest_line(path: Path, rec: DownloadRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "rft_dat_id": rec.rft_dat_id,
        "resolver_url": rec.resolver_url,
        "outcome": rec.outcome,
        "gateway_url": rec.gateway_url,
        "gateway_oid": rec.gateway_oid,
        "gateway_targetid": rec.gateway_targetid,
        "resource_option": rec.resource_option,
        "content_disposition_filename": rec.content_disposition_filename,
        "saved_pdf": f"{rec.rft_dat_id}.pdf" if rec.outcome == "ok" else None,
        "ts": datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat(),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def load_manifest_latest_outcome(path: Path) -> dict[str, str]:
    """Latest outcome per rft_dat_id from JSONL (later lines override earlier ones for the same id)."""
    last: dict[str, str] = {}
    if not path.is_file():
        return last
    text = path.read_text(encoding="utf-8", errors="replace")
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            print(f"warning: manifest skip parse {path}:{line_no} invalid json", file=sys.stderr)
            continue
        if not isinstance(row, dict):
            continue
        tid = row.get("rft_dat_id")
        oc = row.get("outcome")
        if isinstance(tid, str) and tid and isinstance(oc, str):
            last[tid] = oc
    return last


def load_cookie_header(path: Path) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    stripped = [ln.strip() for ln in lines]
    netscape = any(ln.startswith("# Netscape HTTP Cookie File") for ln in stripped[:5])
    body = [ln for ln in stripped if ln and not ln.startswith("#")]
    if not body:
        return ""
    if not netscape and len(body) == 1 and "\t" not in body[0]:
        return body[0]
    by_name: dict[str, str] = {}
    for line in body:
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        domain = parts[0]
        if "dtu.dk" not in domain:
            continue
        by_name[parts[-2]] = parts[-1]
    return "; ".join(f"{k}={v}" for k, v in by_name.items())


def parse_retry_after_seconds(headers: dict[str, str]) -> int | None:
    v = headers.get("Retry-After") or headers.get("retry-after")
    if not v:
        return None
    v = v.strip()
    try:
        return int(v)
    except ValueError:
        try:
            dt = parsedate_to_datetime(v)
            return max(0, int(dt.timestamp() - time.time()))
        except (TypeError, ValueError, OSError):
            return None


def has_pdf(out_dir: Path, thesis_id: str) -> bool:
    pdf = out_dir / f"{thesis_id}.pdf"
    return pdf.is_file() and pdf.stat().st_size > 0


def has_prohibited_tombstone(out_dir: Path, thesis_id: str) -> bool:
    return (out_dir / f"{thesis_id}{PROHIBITED_MARK}").is_file()


def pick_extension(content_type: str, content_disposition: str, part: Path) -> str:
    ct = (content_type or "").lower()
    if "pdf" in ct:
        return "pdf"
    fn = None
    m = re.search(r"filename\*?=(?:UTF-8''|\"?)([^\";]+)", content_disposition or "", re.I)
    if m:
        fn = m.group(1)
    if fn and fn.lower().endswith(".pdf"):
        return "pdf"
    head = part.read_bytes()[:8]
    if head.startswith(b"%PDF"):
        return "pdf"
    return "bin"


def parse_sse_first_json(body: bytes) -> dict | None:
    text = body.decode("utf-8", errors="replace")
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


def write_bulk_checkpoint(path: Path, data: dict[str, object]) -> None:
    """Atomic JSON write so a crash mid-write does not corrupt the checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def remove_stale_sse_bin(out_dir: Path, thesis_id: str) -> None:
    binf = out_dir / f"{thesis_id}.bin"
    if not binf.is_file() or binf.stat().st_size >= 16384:
        return
    try:
        head = binf.read_bytes()[:32]
        if head.startswith(b"data:") or b"resource_option" in head[:512]:
            binf.unlink()
    except OSError:
        pass


def streaming_get(
    url: str,
    headers: dict[str, str],
    part: Path,
    thesis_id: str,
    sock_timeout: float,
    phase: str,
) -> tuple[bool, dict[str, str] | None]:
    """GET url, write 200 body to part. Returns (ok, lowercase response headers)."""
    part.parent.mkdir(parents=True, exist_ok=True)
    attempt_429 = 0
    attempt_net = 0

    while True:
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=sock_timeout) as resp:
                hdrs = {k.lower(): v for k, v in resp.headers.items()}
                if resp.status != 200:
                    part.unlink(missing_ok=True)
                    print(f"fail_http_{phase} {thesis_id} {resp.status}", file=sys.stderr)
                    return False, None
                with part.open("wb") as out:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        out.write(chunk)
                return True, hdrs

        except urllib.error.HTTPError as e:
            part.unlink(missing_ok=True)
            try:
                e.read()
            except Exception:
                pass
            eh = {k.lower(): v for k, v in e.headers.items()} if e.headers else {}
            if e.code == 429:
                if attempt_429 >= MAX_429:
                    print(f"fail_429_max_{phase} {thesis_id}", file=sys.stderr)
                    return False, None
                backoff = 3000 * (2**attempt_429) + random.random() * 1000
                ra = (parse_retry_after_seconds(eh) or 0) * 1000
                wait_ms = max(backoff, ra)
                print(
                    f"backoff_429_{phase} {thesis_id} attempt={attempt_429 + 1} wait_ms={wait_ms:.0f}",
                    file=sys.stderr,
                )
                time.sleep(wait_ms / 1000.0)
                attempt_429 += 1
                continue
            if 500 <= e.code < 600:
                if attempt_net >= MAX_NET:
                    print(f"fail_5xx_max_{phase} {thesis_id} last={e.code}", file=sys.stderr)
                    return False, None
                wait = 5_000 + random.random() * 2_000
                print(
                    f"retry_5xx_{phase} {thesis_id} code={e.code} attempt={attempt_net + 1} wait_ms={wait:.0f}",
                    file=sys.stderr,
                )
                time.sleep(wait / 1000.0)
                attempt_net += 1
                continue
            print(f"fail_http_{phase} {thesis_id} {e.code}", file=sys.stderr)
            return False, None

        except (urllib.error.URLError, TimeoutError, OSError, ConnectionResetError, http.client.IncompleteRead) as e:
            part.unlink(missing_ok=True)
            if attempt_net >= MAX_NET:
                print(f"fail_network_max_{phase} {thesis_id} {e}", file=sys.stderr)
                return False, None
            wait = 5_000 + random.random() * 2_000
            print(
                f"retry_network_{phase} {thesis_id} attempt={attempt_net + 1} wait_ms={wait:.0f} msg={e}",
                file=sys.stderr,
            )
            time.sleep(wait / 1000.0)
            attempt_net += 1


def finalize_part(
    out_dir: Path,
    thesis_id: str,
    part: Path,
    hdrs: dict[str, str],
    force: bool,
) -> tuple[Literal["ok", "fail"], str | None]:
    ctype = hdrs.get("content-type", "")
    cd = hdrs.get("content-disposition", "")
    cd_fn = filename_from_content_disposition(cd)
    ext = pick_extension(ctype, cd, part)
    dest = out_dir / f"{thesis_id}.{ext}"
    if ext != "pdf":
        part.unlink(missing_ok=True)
        print(f"fail_not_pdf {thesis_id} (got .{ext})", file=sys.stderr)
        return "fail", cd_fn
    if force and dest.exists():
        dest.unlink()
    part.replace(dest)
    (out_dir / f"{thesis_id}.bin").unlink(missing_ok=True)
    print(f"ok {thesis_id}.pdf")
    return "ok", cd_fn


def download_one(
    resolver_url: str,
    cookie: str,
    out_dir: Path,
    thesis_id: str,
    sock_timeout: float,
    force: bool,
) -> DownloadRecord:
    part = out_dir / f"{thesis_id}.part"
    part.unlink(missing_ok=True)

    base_headers: dict[str, str] = {"User-Agent": UA}
    if cookie:
        base_headers["Cookie"] = cookie

    ok, hdrs1 = streaming_get(resolver_url, base_headers, part, thesis_id, sock_timeout, "resolver")
    if not ok or hdrs1 is None:
        return DownloadRecord(
            outcome="fail",
            resolver_url=resolver_url,
            rft_dat_id=thesis_id,
            gateway_url=None,
            gateway_oid=None,
            gateway_targetid=None,
            resource_option=None,
            content_disposition_filename=None,
        )

    head = part.read_bytes()[:8]
    if head.startswith(b"%PDF"):
        oid, targetid = gateway_query_ids(resolver_url)
        fin, cd_fn = finalize_part(out_dir, thesis_id, part, hdrs1, force)
        return DownloadRecord(
            outcome=fin,
            resolver_url=resolver_url,
            rft_dat_id=thesis_id,
            gateway_url=resolver_url,
            gateway_oid=oid,
            gateway_targetid=targetid,
            resource_option=None,
            content_disposition_filename=cd_fn,
        )

    body = part.read_bytes()
    payload = parse_sse_first_json(body)
    if payload is not None:
        ro = payload.get("resource_option")
        url2 = payload.get("url")
        ro_s = str(ro) if ro is not None else None
        gw = str(url2).strip() if url2 else None
        oid, targetid = gateway_query_ids(gw) if gw else (None, None)
        if ro == "access_prohibited":
            part.unlink(missing_ok=True)
            tomb = out_dir / f"{thesis_id}{PROHIBITED_MARK}"
            tomb.write_text("access_prohibited\n", encoding="utf-8")
            (out_dir / f"{thesis_id}.bin").unlink(missing_ok=True)
            print(f"skip_prohibited {thesis_id}")
            return DownloadRecord(
                outcome="prohibited",
                resolver_url=resolver_url,
                rft_dat_id=thesis_id,
                gateway_url=gw,
                gateway_oid=oid,
                gateway_targetid=targetid,
                resource_option=ro_s,
                content_disposition_filename=None,
            )
        if not url2:
            part.unlink(missing_ok=True)
            print(f"fail_no_gateway_url {thesis_id} resource_option={ro!r}", file=sys.stderr)
            return DownloadRecord(
                outcome="fail",
                resolver_url=resolver_url,
                rft_dat_id=thesis_id,
                gateway_url=None,
                gateway_oid=None,
                gateway_targetid=None,
                resource_option=ro_s,
                content_disposition_filename=None,
            )

        h2 = dict(base_headers)
        h2["Referer"] = resolver_url
        part.unlink(missing_ok=True)
        ok2, hdrs2 = streaming_get(str(url2), h2, part, thesis_id, sock_timeout, "gateway")
        if not ok2 or hdrs2 is None:
            return DownloadRecord(
                outcome="fail",
                resolver_url=resolver_url,
                rft_dat_id=thesis_id,
                gateway_url=gw,
                gateway_oid=oid,
                gateway_targetid=targetid,
                resource_option=ro_s,
                content_disposition_filename=None,
            )
        fin, cd_fn = finalize_part(out_dir, thesis_id, part, hdrs2, force)
        return DownloadRecord(
            outcome=fin,
            resolver_url=resolver_url,
            rft_dat_id=thesis_id,
            gateway_url=gw,
            gateway_oid=oid,
            gateway_targetid=targetid,
            resource_option=ro_s,
            content_disposition_filename=cd_fn,
        )

    fin, cd_fn = finalize_part(out_dir, thesis_id, part, hdrs1, force)
    return DownloadRecord(
        outcome=fin,
        resolver_url=resolver_url,
        rft_dat_id=thesis_id,
        gateway_url=None,
        gateway_oid=None,
        gateway_targetid=None,
        resource_option=None,
        content_disposition_filename=cd_fn,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("urls_file", type=Path, help="One resolver URL per line")
    p.add_argument("-o", "--out-dir", type=Path, default=Path("findit_downloads"))
    p.add_argument("-c", "--cookies", type=Path, default=None, help="Cookie line or Netscape jar (file must exist)")
    p.add_argument("--delay-ms", type=int, default=3000)
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--connect-timeout-sec",
        type=int,
        default=30,
        help="Floor for socket timeout (urllib uses one timeout for connect+read; see --request-timeout-min)",
    )
    p.add_argument("--request-timeout-min", type=int, default=15)
    p.add_argument(
        "--checkpoint-file",
        type=Path,
        default=None,
        help="JSON progress file (default: OUT_DIR/bulk_download_checkpoint.json)",
    )
    p.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Do not read or write checkpoint files",
    )
    p.add_argument(
        "--manifest-file",
        type=Path,
        default=None,
        help="Append-only JSONL manifest (default: OUT_DIR/bulk_download_manifest.jsonl)",
    )
    p.add_argument(
        "--no-manifest",
        action="store_true",
        help="Do not append download metadata JSONL lines",
    )
    p.add_argument(
        "--skip-ok-from-manifest",
        action="store_true",
        help="Skip ids whose latest manifest line has outcome ok (after moving PDFs elsewhere)",
    )
    p.add_argument(
        "--skip-ok-manifest-file",
        type=Path,
        default=None,
        help="JSONL to read for --skip-ok-from-manifest (default: same as --manifest-file or OUT_DIR/bulk_download_manifest.jsonl)",
    )
    args = p.parse_args()

    urls_file = args.urls_file.expanduser().resolve()
    if not urls_file.is_file():
        print(f"error: urls file not found: {urls_file}", file=sys.stderr)
        return 2

    if args.cookies is not None:
        cookie_path = args.cookies.expanduser().resolve()
        if not cookie_path.is_file():
            print(
                f"error: cookie file not found: {cookie_path}\n"
                "  Create it after logging in to FindIt (one line of Cookie header, or Netscape jar).\n"
                "  See: maks/findit_bulk_download/cookies-howto.txt\n"
                "  Or omit -c to try without cookies (usually fails for fulltext).",
                file=sys.stderr,
            )
            return 2
        cookie = load_cookie_header(cookie_path)
    else:
        cookie = ""
        print("warning: no -c cookies; downloads may return login HTML.", file=sys.stderr)

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    read_sec = float(args.request_timeout_min * 60)
    sock_timeout = max(float(args.connect_timeout_sec), read_sec)

    checkpoint_path: Path | None = None
    if not args.no_checkpoint:
        checkpoint_path = (args.checkpoint_file or out_dir / "bulk_download_checkpoint.json").expanduser().resolve()

    manifest_path: Path | None = None
    if not args.no_manifest:
        manifest_path = (args.manifest_file or out_dir / "bulk_download_manifest.jsonl").expanduser().resolve()

    manifest_ok_ids: set[str] = set()
    if args.skip_ok_from_manifest:
        manifest_skip_read_path = (
            args.skip_ok_manifest_file or args.manifest_file or out_dir / "bulk_download_manifest.jsonl"
        ).expanduser().resolve()
        if not manifest_skip_read_path.is_file() or manifest_skip_read_path.stat().st_size == 0:
            print(
                f"warning: --skip-ok-from-manifest: file missing or empty, no manifest skips ({manifest_skip_read_path})",
                file=sys.stderr,
            )
        else:
            last_out = load_manifest_latest_outcome(manifest_skip_read_path)
            manifest_ok_ids = {tid for tid, oc in last_out.items() if oc == "ok"}

    def iso_now() -> str:
        return datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat()

    def base_ckpt() -> dict[str, object]:
        return {
            "urls_file": str(urls_file),
            "out_dir": str(out_dir),
            "updated": iso_now(),
        }

    lines = urls_file.read_text(encoding="utf-8", errors="replace").splitlines()
    seen: set[str] = set()
    n_ok = n_skip_dup = n_skip_exists = n_fail = 0
    n_skip_manifest_ok = 0
    n_prohibited = n_skip_prohibited_cached = n_unparseable = 0
    n_unique_ids = 0

    ckpt: dict[str, object] = {
        **base_ckpt(),
        "stopped": "running",
        "last_completed_line_no": None,
        "last_completed_thesis_id": None,
        "last_completed_outcome": None,
        "in_progress_line_no": None,
        "in_progress_thesis_id": None,
        "counts": {},
    }

    def flush_checkpoint(extra: dict[str, object] | None = None) -> None:
        if checkpoint_path is None:
            return
        payload = {**ckpt, **(extra or {}), "updated": iso_now()}
        payload["counts"] = {
            "pdf_ok": n_ok,
            "prohibited": n_prohibited,
            "failed": n_fail,
            "skip_duplicate_id": n_skip_dup,
            "skip_exists_pdf": n_skip_exists,
            "skip_manifest_ok": n_skip_manifest_ok,
            "skip_prohibited_cached": n_skip_prohibited_cached,
            "skip_unparseable": n_unparseable,
            "unique_ids_started": n_unique_ids,
        }
        write_bulk_checkpoint(checkpoint_path, payload)

    interrupted = False
    try:
        for line_no, raw in enumerate(lines, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tid = thesis_id_from_resolver_url(line)
            if tid is None:
                print(f"skip_unparseable {line[:80]!r}", file=sys.stderr)
                n_unparseable += 1
                continue
            if tid in seen:
                print(f"skip_duplicate_id {tid}")
                n_skip_dup += 1
                continue
            seen.add(tid)
            n_unique_ids += 1

            if has_prohibited_tombstone(out_dir, tid) and not args.force:
                print(f"skip_prohibited_cached {tid}")
                n_skip_prohibited_cached += 1
                continue
            if not args.force and tid in manifest_ok_ids:
                print(f"skip_manifest_ok {tid}")
                n_skip_manifest_ok += 1
                continue
            if has_pdf(out_dir, tid) and not args.force:
                print(f"skip_exists {tid}.pdf")
                n_skip_exists += 1
                continue
            if args.force:
                (out_dir / f"{tid}.pdf").unlink(missing_ok=True)
                (out_dir / f"{tid}.bin").unlink(missing_ok=True)
                (out_dir / f"{tid}{PROHIBITED_MARK}").unlink(missing_ok=True)
                (out_dir / f"{tid}.part").unlink(missing_ok=True)
            else:
                remove_stale_sse_bin(out_dir, tid)

            ckpt["in_progress_line_no"] = line_no
            ckpt["in_progress_thesis_id"] = tid
            ckpt["in_progress_resolver_url"] = line[:200] + ("…" if len(line) > 200 else "")
            flush_checkpoint({"stopped": "running"})

            try:
                rec = download_one(line, cookie, out_dir, tid, sock_timeout, args.force)
            except KeyboardInterrupt:
                interrupted = True
                ckpt["stopped"] = "interrupt"
                ckpt["note"] = (
                    "Interrupted during download_one (see in_progress_*). "
                    "Re-run the same command: existing PDFs are skipped; failed items retry."
                )
                flush_checkpoint()
                print(
                    f"\ninterrupt: checkpoint written to {checkpoint_path}\n"
                    f"  last_completed_line_no={ckpt.get('last_completed_line_no')} "
                    f"last_completed_thesis_id={ckpt.get('last_completed_thesis_id')}\n"
                    f"  in_progress_line_no={ckpt.get('in_progress_line_no')} "
                    f"in_progress_thesis_id={ckpt.get('in_progress_thesis_id')}\n"
                    f"  Re-run the same uv/python command to resume (skip_exists, manifest ok, + retries).",
                    file=sys.stderr,
                )
                raise

            ckpt["in_progress_line_no"] = None
            ckpt["in_progress_thesis_id"] = None
            ckpt["in_progress_resolver_url"] = None
            ckpt["last_completed_line_no"] = line_no
            ckpt["last_completed_thesis_id"] = tid
            outcome = rec.outcome
            ckpt["last_completed_outcome"] = outcome
            if manifest_path is not None:
                append_manifest_line(manifest_path, rec)
            if outcome == "ok":
                n_ok += 1
            elif outcome == "prohibited":
                n_prohibited += 1
            else:
                n_fail += 1

            flush_checkpoint({"stopped": "running"})

            try:
                time.sleep((args.delay_ms + random.random() * 1000) / 1000.0)
            except KeyboardInterrupt:
                interrupted = True
                ckpt["stopped"] = "interrupt"
                ckpt["note"] = (
                    "Interrupted after completing an item (during delay). "
                    "Re-run the same command: next items continue; finished PDFs are skipped."
                )
                flush_checkpoint()
                print(
                    f"\ninterrupt: checkpoint written to {checkpoint_path}\n"
                    f"  last_completed_line_no={ckpt.get('last_completed_line_no')} "
                    f"last_completed_thesis_id={ckpt.get('last_completed_thesis_id')}\n"
                    f"  Re-run the same uv/python command to resume.",
                    file=sys.stderr,
                )
                raise

    finally:
        if checkpoint_path is not None:
            if interrupted:
                ckpt["stopped"] = "interrupt"
                ckpt.setdefault(
                    "note",
                    "Interrupted — see in_progress_* if set. Re-run the same command to continue.",
                )
            else:
                exc_ty = sys.exc_info()[0]
                if exc_ty is not None:
                    ckpt["stopped"] = "error"
                    ckpt["note"] = f"Stopped on unhandled {exc_ty.__name__} — check stderr / traceback."
                else:
                    ckpt["stopped"] = "complete"
                    ckpt["note"] = "Run finished. Re-run to retry failures and fetch any new URLs."
                    ckpt["in_progress_line_no"] = None
                    ckpt["in_progress_thesis_id"] = None
                    ckpt["in_progress_resolver_url"] = None
            flush_checkpoint()

    attempted_fulltext = n_ok + n_fail + n_prohibited
    remainder_after_prohibited = n_ok + n_fail
    print(
        f"done pdf_ok={n_ok} prohibited={n_prohibited} prohibited_cached_skip={n_skip_prohibited_cached} "
        f"failed={n_fail} skip_duplicate_id={n_skip_dup} skip_exists_pdf={n_skip_exists} "
        f"skip_manifest_ok={n_skip_manifest_ok} "
        f"skip_unparseable={n_unparseable} unique_ids_in_file={n_unique_ids} "
        f"fulltext_attempts_this_run={attempted_fulltext} pdf_or_fail_after_prohibited={remainder_after_prohibited}",
        file=sys.stderr,
    )
    if checkpoint_path is not None:
        print(f"checkpoint file: {checkpoint_path}", file=sys.stderr)
    if manifest_path is not None:
        print(f"manifest file: {manifest_path}", file=sys.stderr)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
