#!/usr/bin/env python3
"""Bulk-download FindIt resolver URLs (stdlib): resolver GET (SSE/JSON), optional gateway GET, PDF save.

Checkpoint OUT_DIR/bulk_download_checkpoint.json; append-only manifest OUT_DIR/bulk_download_manifest.jsonl
unless --no-* or paths overridden.

Manifest skips (optional paths per flag; default same JSONL): --skip-ok-from-manifest,
--skip-prohibited-from-manifest, --skip-manifest-recorded (any latest outcome: ok/prohibited/fail).
--retry-prohibited-from-manifest retries only latest prohibited (exclusive with skip-prohibited / skip-manifest-recorded).

--quiet-skips, --progress-every; --config JSON (snake_case; CLI wins). See bulk_download.config.example.json & cookies-howto.txt.
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


def manifest_skip_sets(args: argparse.Namespace, out_dir: Path) -> tuple[set[str], set[str], set[str]]:
    """Load manifest JSONL once per distinct path; return (ok_ids, prohibited_ids, recorded_ids)."""
    ok_ids: set[str] = set()
    ph_ids: set[str] = set()
    rec_ids: set[str] = set()
    if not (
        args.skip_ok_from_manifest or args.skip_prohibited_from_manifest or args.skip_manifest_recorded
    ):
        return ok_ids, ph_ids, rec_ids

    def read_path(prio: Path | None) -> Path:
        return (prio or args.manifest_file or out_dir / "bulk_download_manifest.jsonl").expanduser().resolve()

    p_ok = read_path(args.skip_ok_manifest_file)
    p_ph = read_path(args.skip_prohibited_manifest_file)
    p_rec = read_path(args.skip_manifest_recorded_file)
    cache: dict[Path, dict[str, str]] = {}

    def latest(p: Path) -> dict[str, str]:
        if p not in cache:
            cache[p] = (
                load_manifest_latest_outcome(p) if p.is_file() and p.stat().st_size > 0 else {}
            )
        return cache[p]

    if args.skip_manifest_recorded:
        m = latest(p_rec)
        rec_ids = set(m)
        print(
            f"  skip-manifest-recorded: {len(rec_ids)} ids ({p_rec})"
            if m
            else f"warning: --skip-manifest-recorded: missing or empty ({p_rec})",
            file=sys.stderr,
        )

    if args.skip_ok_from_manifest and args.skip_prohibited_from_manifest and p_ok == p_ph:
        m = latest(p_ok)
        if not m:
            print(f"warning: skip-ok+skip-prohibited: missing or empty ({p_ok})", file=sys.stderr)
        else:
            ok_ids = {t for t, o in m.items() if o == "ok"}
            ph_ids = {t for t, o in m.items() if o == "prohibited"}
    else:
        if args.skip_ok_from_manifest:
            m = latest(p_ok)
            if not m:
                print(f"warning: --skip-ok-from-manifest: missing or empty ({p_ok})", file=sys.stderr)
            else:
                ok_ids = {t for t, o in m.items() if o == "ok"}
        if args.skip_prohibited_from_manifest:
            m = latest(p_ph)
            if not m:
                print(f"warning: --skip-prohibited-from-manifest: missing or empty ({p_ph})", file=sys.stderr)
            else:
                ph_ids = {t for t, o in m.items() if o == "prohibited"}

    return ok_ids, ph_ids, rec_ids


def merge_plain_cookie_lines(body: list[str]) -> str | None:
    """Merge multiple plain ``name=value`` lines into one Cookie header value."""
    pieces: list[str] = []
    for raw in body:
        ln = raw.strip()
        m = re.match(r"(?i)Cookie:\s*(.+)$", ln)
        if m:
            ln = m.group(1).strip()
        if "\t" in ln:
            return None
        if not ln or "=" not in ln:
            return None
        pieces.append(ln.rstrip().rstrip(";"))
    if len(pieces) < 2:
        return None
    return "; ".join(pieces)


def load_cookie_header(path: Path) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    stripped = [ln.strip() for ln in lines]
    netscape = any(ln.startswith("# Netscape HTTP Cookie File") for ln in stripped[:5])
    body = [ln for ln in stripped if ln and not ln.startswith("#")]
    if not body:
        return ""
    if not netscape and len(body) == 1 and "\t" not in body[0]:
        line = body[0]
        m = re.match(r"(?i)Cookie:\s*(.+)$", line)
        if m:
            return m.group(1).strip()
        return line
    by_name: dict[str, str] = {}
    for line in body:
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        domain = parts[0]
        if "dtu.dk" not in domain:
            continue
        by_name[parts[-2]] = parts[-1]
    jar = "; ".join(f"{k}={v}" for k, v in by_name.items())
    if jar:
        return jar
    merged = merge_plain_cookie_lines(body)
    if merged:
        return merged
    # Safari / DevTools request dumps: many lines, optional "Cookie: name=value; ..."
    extracted: list[str] = []
    for ln in body:
        m = re.match(r"(?i)Cookie:\s*(.+)$", ln)
        if m:
            v = m.group(1).strip()
            if v:
                extracted.append(v)
    if extracted:
        best = max(extracted, key=len)
        print(
            "warning: cookie file is not a Netscape jar or a single header line; "
            "using the longest `Cookie:` line found in the file. For GetIt PDFs, "
            "prefer a fresh Cookie header copied from a **getit.findit.dtu.dk** request "
            "after you open full text (see maks/findit_bulk_download/cookies-howto.txt).",
            file=sys.stderr,
        )
        return best
    return ""


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
    """Parse GetIt-style SSE: one or more ``data:`` JSON lines.

    Collect JSON objects from ``data:`` lines. Prefer objects that include an http(s)
    ``url`` (fulltext-gateway). If several have a URL, **prefer the last one whose
    ``resource_option`` is not ``access_prohibited``** — some streams end with a URL
    line that is still a denial stub; an earlier line may carry the real open access
    link. If every URL-bearing line is ``access_prohibited``, use the last of those.

    If no line has an http(s) URL, returns the **last** JSON object.

    Leading HTML (login/error pages) yields ``None`` so callers do not treat them as SSE.
    """
    lead = body[:8192].lstrip()
    low = lead.lower()
    if low.startswith((b"<!doctype", b"<html")) or (b"<html" in low[:4096] and not low.startswith(b"%pdf")):
        return None
    text = body.decode("utf-8", errors="replace")
    candidates: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload in ("none", "[DONE]"):
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            candidates.append(obj)
    if not candidates:
        return None
    with_url: list[dict] = []
    for obj in candidates:
        u = obj.get("url")
        if isinstance(u, str) and u.strip().lower().startswith("http"):
            with_url.append(obj)
    if with_url:
        chosen: dict | None = None
        for obj in with_url:
            if obj.get("resource_option") != "access_prohibited":
                chosen = obj
        if chosen is not None:
            return chosen
        return with_url[-1]
    return candidates[-1]


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
    prohibited_tombstones: bool,
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
            if prohibited_tombstones:
                tomb = out_dir / f"{thesis_id}{PROHIBITED_MARK}"
                tomb.write_text("access_prohibited\n", encoding="utf-8")
            (out_dir / f"{thesis_id}.bin").unlink(missing_ok=True)
            print(
                f"outcome_prohibited {thesis_id}  (resolver returned access_prohibited; a request was made)",
                file=sys.stderr,
            )
            if len(cookie) < 180:
                print(
                    "  hint: short Cookie → try longer GetIt export (cookies-howto.txt)",
                    file=sys.stderr,
                )
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


_CONFIG_PATH_KEYS = frozenset(
    {
        "urls_file",
        "out_dir",
        "cookies",
        "checkpoint_file",
        "manifest_file",
        "skip_ok_manifest_file",
        "skip_prohibited_manifest_file",
        "retry_prohibited_manifest_file",
        "skip_manifest_recorded_file",
    }
)
CONFIG_VALID_KEYS = _CONFIG_PATH_KEYS | frozenset(
    {
        "delay_ms",
        "force",
        "prohibited_tombstones",
        "connect_timeout_sec",
        "request_timeout_min",
        "no_checkpoint",
        "no_manifest",
        "skip_ok_from_manifest",
        "skip_prohibited_from_manifest",
        "retry_prohibited_from_manifest",
        "skip_manifest_recorded",
        "quiet_skips",
        "progress_every",
    }
)


def bulk_download_defaults_from_config(path: Path) -> dict[str, object]:
    """Load JSON object; keys must match argparse dest names. Ignores keys starting with ``_``."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("top-level JSON must be an object")
    out: dict[str, object] = {}
    unknown: list[str] = []
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        if k not in CONFIG_VALID_KEYS:
            unknown.append(k)
            continue
        out[k] = v
    if unknown:
        print(
            "warning: --config ignoring unknown keys: " + ", ".join(sorted(unknown)),
            file=sys.stderr,
        )
    for pk in _CONFIG_PATH_KEYS:
        if pk not in out or out[pk] is None:
            continue
        if isinstance(out[pk], str):
            out[pk] = Path(out[pk])
        elif not isinstance(out[pk], Path):
            raise ValueError(f"config key {pk!r} must be a string path or null")
    return out


def build_argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-C",
        "--config",
        type=Path,
        default=None,
        help="JSON file with default option values (snake_case keys); CLI overrides",
    )
    p.add_argument(
        "urls_file",
        type=Path,
        nargs="?",
        default=None,
        help="One resolver URL per line (optional if set in --config)",
    )
    p.add_argument("-o", "--out-dir", type=Path, default=Path("findit_downloads"))
    p.add_argument("-c", "--cookies", type=Path, default=None, help="Cookie line or Netscape jar (file must exist)")
    p.add_argument("--delay-ms", type=int, default=3000)
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--prohibited-tombstones",
        action="store_true",
        help="Write .access_prohibited tombstones and skip ids that have one (default: off — retry prohibited each run)",
    )
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
    p.add_argument(
        "--skip-prohibited-from-manifest",
        action="store_true",
        help="Skip ids whose latest manifest line has outcome prohibited (no GetIt call; omit flag to retry)",
    )
    p.add_argument(
        "--skip-prohibited-manifest-file",
        type=Path,
        default=None,
        help="JSONL to read for --skip-prohibited-from-manifest (default: same as --manifest-file or OUT_DIR/bulk_download_manifest.jsonl)",
    )
    p.add_argument(
        "--skip-manifest-recorded",
        action="store_true",
        help="Skip ids that already appear in the manifest (latest outcome ok, prohibited, or fail); no repeat GetIt",
    )
    p.add_argument(
        "--skip-manifest-recorded-file",
        type=Path,
        default=None,
        help="JSONL for --skip-manifest-recorded (default: same as --manifest-file or OUT_DIR/bulk_download_manifest.jsonl)",
    )
    p.add_argument(
        "--retry-prohibited-from-manifest",
        action="store_true",
        help="Second pass only: URL lines whose id is NOT latest-manifest prohibited are bypassed "
        "(skip_retry_manifest_filter; expected large count). Omit this flag for a normal full crawl.",
    )
    p.add_argument(
        "--retry-prohibited-manifest-file",
        type=Path,
        default=None,
        help="JSONL to read latest outcomes for --retry-prohibited-from-manifest (default: output manifest path)",
    )
    p.add_argument(
        "--quiet-skips",
        action="store_true",
        help="Do not print one line per skip_duplicate_id / skip_*_cached / skip_manifest_ok / "
        "skip_manifest_prohibited / skip_manifest_recorded / skip_exists",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=None,
        metavar="N",
        help="With --quiet-skips: print cumulative counts every N url-file lines (default: 2000)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("-C", "--config", type=Path, default=None)
    pre_opts, rest = pre.parse_known_args(argv)
    config_path_used: Path | None = None
    defaults: dict[str, object] = {}
    if pre_opts.config is not None:
        cp = pre_opts.config.expanduser().resolve()
        if not cp.is_file():
            print(f"error: --config file not found: {cp}", file=sys.stderr)
            return 2
        try:
            defaults = bulk_download_defaults_from_config(cp)
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"error: --config {cp}: {e}", file=sys.stderr)
            return 2
        config_path_used = cp
    p = build_argument_parser()
    p.set_defaults(**defaults)
    args = p.parse_args(rest)

    if args.retry_prohibited_from_manifest and args.no_manifest:
        p.error("--retry-prohibited-from-manifest needs manifest append; remove --no-manifest or omit --retry-prohibited-from-manifest")

    if args.retry_prohibited_from_manifest and args.skip_prohibited_from_manifest:
        p.error("--retry-prohibited-from-manifest and --skip-prohibited-from-manifest are mutually exclusive")

    if args.retry_prohibited_from_manifest and args.skip_manifest_recorded:
        p.error("--retry-prohibited-from-manifest and --skip-manifest-recorded are mutually exclusive")

    if args.progress_every is not None and args.progress_every < 1:
        p.error("--progress-every must be >= 1")

    if args.urls_file is None:
        print(
            "error: urls_file missing — pass the deduped URL list as the first argument or set \"urls_file\" in --config",
            file=sys.stderr,
        )
        return 2
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
        if cookie and len(cookie) < 180:
            print(
                f"warning: Cookie header is only {len(cookie)} chars — often too short for GetIt fulltext "
                "(catalog-only exports look like this). If almost every thesis shows outcome_prohibited, "
                "re-capture cookies from a getit.findit.dtu.dk request after opening full text, or use "
                "har_to_cookies.py on a HAR that includes GetIt (see maks/findit_bulk_download/cookies-howto.txt).",
                file=sys.stderr,
            )
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

    if checkpoint_path is not None and checkpoint_path.is_file():
        try:
            prev_ck = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            prev_out = prev_ck.get("out_dir")
            if isinstance(prev_out, str) and prev_out:
                if Path(prev_out).expanduser().resolve() != out_dir:
                    print(
                        "warning: checkpoint recorded a different PDF directory than this run:\n"
                        f"  was: {prev_out}\n"
                        f"  now: {out_dir}\n"
                        "  PDFs and .part files are written under 'now' only; "
                        ".access_prohibited markers only if --prohibited-tombstones. "
                        "If that is unexpected, check the command: in bash/zsh a line "
                        "continuation backslash must be the last character on that physical line "
                        "(no space after it), or flags like -o may not apply to the same command.",
                        file=sys.stderr,
                    )
        except (OSError, json.JSONDecodeError, ValueError):
            pass

    cfg_line = f"  config: {config_path_used}\n" if config_path_used is not None else ""
    print(
        "Using:\n"
        f"{cfg_line}"
        f"  PDF directory (-o): {out_dir}\n"
        f"  checkpoint: {checkpoint_path or '(disabled)'}\n"
        f"  manifest: {manifest_path or '(disabled)'}",
        file=sys.stderr,
    )

    retry_prohibited_ids: set[str] | None = None
    if args.retry_prohibited_from_manifest:
        rmanifest = (
            args.retry_prohibited_manifest_file
            or args.manifest_file
            or out_dir / "bulk_download_manifest.jsonl"
        ).expanduser().resolve()
        if not rmanifest.is_file() or rmanifest.stat().st_size == 0:
            print(
                f"error: --retry-prohibited-from-manifest: manifest missing or empty: {rmanifest}",
                file=sys.stderr,
            )
            return 2
        last_all = load_manifest_latest_outcome(rmanifest)
        retry_prohibited_ids = {tid for tid, oc in last_all.items() if oc == "prohibited"}
        print(
            f"  retry-prohibited-only: {len(retry_prohibited_ids)} ids ← {rmanifest} "
            "(other lines → skip_retry_manifest_filter)",
            file=sys.stderr,
        )
        if not retry_prohibited_ids:
            print("warning: no prohibited ids in manifest — nothing to download.", file=sys.stderr)

    manifest_ok_ids, manifest_prohibited_ids, manifest_recorded_ids = manifest_skip_sets(args, out_dir)

    def iso_now() -> str:
        return datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat()

    def base_ckpt() -> dict[str, object]:
        return {
            "urls_file": str(urls_file),
            "out_dir": str(out_dir),
            "updated": iso_now(),
        }

    lines = urls_file.read_text(encoding="utf-8", errors="replace").splitlines()
    total_lines = len(lines)
    progress_every: int | None = args.progress_every
    if args.quiet_skips and progress_every is None:
        progress_every = 2000

    seen: set[str] = set()
    n_ok = n_skip_dup = n_skip_exists = n_fail = 0
    n_skip_manifest_ok = 0
    n_skip_manifest_prohibited = 0
    n_skip_manifest_recorded = 0
    n_prohibited = n_skip_prohibited_cached = n_unparseable = 0
    n_skip_retry_manifest_filter = 0
    n_unique_ids = 0

    ckpt: dict[str, object] = {
        **base_ckpt(),
        "retry_prohibited_from_manifest": bool(args.retry_prohibited_from_manifest),
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
            "skip_manifest_prohibited": n_skip_manifest_prohibited,
            "skip_manifest_recorded": n_skip_manifest_recorded,
            "skip_prohibited_cached": n_skip_prohibited_cached,
            "skip_unparseable": n_unparseable,
            "skip_retry_manifest_filter": n_skip_retry_manifest_filter,
            "unique_ids_started": n_unique_ids,
        }
        write_bulk_checkpoint(checkpoint_path, payload)

    def print_quiet_progress() -> None:
        if not args.quiet_skips or not progress_every or line_no % progress_every != 0:
            return
        attempted = n_ok + n_fail + n_prohibited
        print(
            f"  progress urls_line={line_no}/{total_lines} pdf_ok={n_ok} prohibited={n_prohibited} "
            f"failed={n_fail} skip_dup={n_skip_dup} skip_exists={n_skip_exists} "
            f"skip_manifest_ok={n_skip_manifest_ok} skip_manifest_prohibited={n_skip_manifest_prohibited} "
            f"skip_manifest_recorded={n_skip_manifest_recorded} skip_prohibited_cached={n_skip_prohibited_cached} "
            f"skip_retry_manifest_filter={n_skip_retry_manifest_filter} "
            f"skip_unparseable={n_unparseable} unique_ids={n_unique_ids} fulltext_attempts={attempted}",
            file=sys.stderr,
        )

    interrupted = False
    try:
        for line_no, raw in enumerate(lines, start=1):
            try:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                tid = thesis_id_from_resolver_url(line)
                if tid is None:
                    print(f"skip_unparseable {line[:80]!r}", file=sys.stderr)
                    n_unparseable += 1
                    continue
                if retry_prohibited_ids is not None and tid not in retry_prohibited_ids:
                    n_skip_retry_manifest_filter += 1
                    continue
                if tid in seen:
                    if not args.quiet_skips:
                        print(f"skip_duplicate_id {tid}", file=sys.stderr)
                    n_skip_dup += 1
                    continue
                seen.add(tid)
                n_unique_ids += 1

                if args.prohibited_tombstones and has_prohibited_tombstone(out_dir, tid) and not args.force:
                    if not args.quiet_skips:
                        print(f"skip_prohibited_cached {tid}", file=sys.stderr)
                    n_skip_prohibited_cached += 1
                    continue
                if not args.force:
                    mk = None
                    if tid in manifest_recorded_ids:
                        mk = "skip_manifest_recorded"
                        n_skip_manifest_recorded += 1
                    elif tid in manifest_ok_ids:
                        mk = "skip_manifest_ok"
                        n_skip_manifest_ok += 1
                    elif tid in manifest_prohibited_ids:
                        mk = "skip_manifest_prohibited"
                        n_skip_manifest_prohibited += 1
                    if mk:
                        if not args.quiet_skips:
                            print(f"{mk} {tid}", file=sys.stderr)
                        continue
                if has_pdf(out_dir, tid) and not args.force:
                    if not args.quiet_skips:
                        print(f"skip_exists {tid}.pdf", file=sys.stderr)
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
                    rec = download_one(
                        line, cookie, out_dir, tid, sock_timeout, args.force, args.prohibited_tombstones
                    )
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
                print_quiet_progress()

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
        f"skip_manifest_ok={n_skip_manifest_ok} skip_manifest_prohibited={n_skip_manifest_prohibited} "
        f"skip_manifest_recorded={n_skip_manifest_recorded} skip_retry_manifest_filter={n_skip_retry_manifest_filter} "
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
