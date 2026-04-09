"""Stream PDFs from Google Cloud Storage through NLP extraction (no local PDF copies)."""

from __future__ import annotations

import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator, Literal

import fitz
from google.cloud import storage

from thesisdatarepo.nlp_extract import (
    FallbackPolicy,
    NlpExtractResult,
    _append_jsonl,
    _write_nlp_manifest_csv,
    _write_nlp_manifest_json,
    configure_batch_logging,
    process_pdf_nlp_reader,
)

logger = logging.getLogger(__name__)

_GS_URI_RE = re.compile(r"^gs://([^/]+)/(.+)$")

_FAILURE_END_REASONS = frozenset(
    {"download_failed", "empty_blob", "pdf_read_failed", "unexpected_error"}
)

# Transient: do not checkpoint so the next run retries (e.g. flaky network).
_CHECKPOINT_SKIP_REASONS = frozenset({"download_failed"})


def parse_gs_uri(uri: str) -> tuple[str, str]:
    """Split ``gs://bucket/path/to/object`` into ``(bucket_name, blob_path)``."""
    u = uri.strip()
    m = _GS_URI_RE.match(u)
    if not m:
        raise ValueError(f"Expected gs://bucket/object, got {uri!r}")
    return m.group(1), m.group(2)


def _normalize_prefix(prefix: str) -> str:
    p = prefix.strip().lstrip("/")
    if p and not p.endswith("/"):
        p = f"{p}/"
    return p


def _nlp_result_to_jsonable(r: NlpExtractResult) -> dict[str, object]:
    return {
        "source": r.source,
        "output_txt_path": str(r.output_txt_path) if r.output_txt_path else None,
        "pages_total": r.pages_total,
        "page_start_0based": r.page_start_0based,
        "page_end_exclusive": r.page_end_exclusive,
        "abstract_found": r.abstract_found,
        "back_matter_found": r.back_matter_found,
        "end_reason": r.end_reason,
        "notes": r.notes,
        "char_count": r.char_count,
        "skipped_empty": r.skipped_empty,
    }


def _nlp_result_from_jsonable(d: dict[str, object]) -> NlpExtractResult:
    otp = d.get("output_txt_path")
    return NlpExtractResult(
        source=str(d["source"]),
        output_txt_path=Path(str(otp)) if otp else None,
        pages_total=int(d["pages_total"]),
        page_start_0based=int(d["page_start_0based"]),
        page_end_exclusive=int(d["page_end_exclusive"]),
        abstract_found=bool(d["abstract_found"]),
        back_matter_found=bool(d["back_matter_found"]),
        end_reason=str(d["end_reason"]),
        notes=str(d.get("notes", "")),
        char_count=int(d["char_count"]),
        skipped_empty=bool(d.get("skipped_empty", False)),
    )


def _load_checkpoints(path: Path) -> dict[str, NlpExtractResult]:
    """Load checkpoint JSONL; last line per blob name wins."""
    if not path.is_file():
        return {}
    out: dict[str, NlpExtractResult] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            blob = str(obj["blob"])
            out[blob] = _nlp_result_from_jsonable(obj["result"])
    return out


def _append_checkpoint_line(
    path: Path,
    blob_name: str,
    r: NlpExtractResult,
    lock: threading.Lock,
) -> None:
    rec = {"blob": blob_name, "result": _nlp_result_to_jsonable(r)}
    payload = json.dumps(rec, ensure_ascii=False) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(payload)


def _should_write_checkpoint(r: NlpExtractResult) -> bool:
    """Persist so we skip re-download on resume. Transient failures stay retryable."""
    return r.end_reason not in _CHECKPOINT_SKIP_REASONS


def iter_pdf_blobs(
    bucket_name: str,
    prefix: str,
    *,
    max_objects: int | None = None,
) -> Iterator[storage.Blob]:
    """
    Yield PDF objects under ``prefix`` (lexicographic order by the API).

    Stops after ``max_objects`` PDFs when set.
    """
    client = storage.Client()
    pref = _normalize_prefix(prefix)
    n = 0
    for blob in client.list_blobs(bucket_name, prefix=pref):
        if not blob.name.lower().endswith(".pdf"):
            continue
        yield blob
        n += 1
        if max_objects is not None and n >= max_objects:
            return


def _process_one_gcs_blob(
    idx_1based: int,
    total: int,
    bucket_name: str,
    blob_name: str,
    output_txt_dir: Path,
    *,
    min_page_fraction_back_matter: float,
    policy: FallbackPolicy,
    fallback_fraction: float,
    skip_if_empty: bool,
) -> NlpExtractResult:
    """
    Download one object by name, run NLP extraction. Safe to call from worker threads
    (uses a fresh ``storage.Client()`` per call).
    """
    source = f"gs://{bucket_name}/{blob_name}"
    stem = Path(blob_name).stem
    dst = output_txt_dir / f"{stem}.txt"

    try:
        client = storage.Client()
        data = client.bucket(bucket_name).blob(blob_name).download_as_bytes()
    except Exception as e:
        logger.warning(
            "[%s/%s] failed (download): %s — %s",
            idx_1based,
            total,
            blob_name,
            e,
        )
        return NlpExtractResult(
            source=source,
            output_txt_path=None,
            pages_total=0,
            page_start_0based=0,
            page_end_exclusive=0,
            abstract_found=False,
            back_matter_found=False,
            end_reason="download_failed",
            notes=f"{type(e).__name__}: {e}",
            char_count=0,
            skipped_empty=False,
        )

    if len(data) == 0:
        logger.warning("[%s/%s] failed (0-byte object): %s", idx_1based, total, blob_name)
        return NlpExtractResult(
            source=source,
            output_txt_path=None,
            pages_total=0,
            page_start_0based=0,
            page_end_exclusive=0,
            abstract_found=False,
            back_matter_found=False,
            end_reason="empty_blob",
            notes="skipped_zero_byte_object",
            char_count=0,
            skipped_empty=False,
        )

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as e:
        logger.warning(
            "[%s/%s] failed (unreadable PDF): %s — %s",
            idx_1based,
            total,
            blob_name,
            e,
        )
        return NlpExtractResult(
            source=source,
            output_txt_path=None,
            pages_total=0,
            page_start_0based=0,
            page_end_exclusive=0,
            abstract_found=False,
            back_matter_found=False,
            end_reason="pdf_read_failed",
            notes=f"{type(e).__name__}: {e}",
            char_count=0,
            skipped_empty=False,
        )

    try:
        r = process_pdf_nlp_reader(
            doc,
            source,
            dst,
            min_page_fraction_back_matter=min_page_fraction_back_matter,
            policy=policy,
            fallback_fraction=fallback_fraction,
            write_txt=True,
            skip_if_empty=skip_if_empty,
        )
    except Exception as e:
        logger.exception("[%s/%s] unexpected error: %s", idx_1based, total, blob_name)
        return NlpExtractResult(
            source=source,
            output_txt_path=None,
            pages_total=0,
            page_start_0based=0,
            page_end_exclusive=0,
            abstract_found=False,
            back_matter_found=False,
            end_reason="unexpected_error",
            notes=f"{type(e).__name__}: {e}",
            char_count=0,
            skipped_empty=False,
        )
    finally:
        doc.close()

    if r.skipped_empty:
        logger.info("[%s/%s] skipped (empty text): %s", idx_1based, total, blob_name)
    else:
        logger.info(
            "[%s/%s] converted %s -> %s chars (pages [%s,%s))",
            idx_1based,
            total,
            blob_name,
            r.char_count,
            r.page_start_0based,
            r.page_end_exclusive,
        )
    return r


def process_gcs_prefix_nlp(
    bucket_name: str,
    prefix: str,
    output_txt_dir: Path,
    manifest_path: Path,
    *,
    max_objects: int | None = 10,
    max_workers: int = 6,
    manifest_format: Literal["csv", "json"] = "csv",
    jsonl_path: Path | None = None,
    clear_jsonl: bool = True,
    resume: bool = True,
    force_fresh: bool = False,
    min_page_fraction_back_matter: float = 0.45,
    policy: FallbackPolicy = FallbackPolicy.KEEP_ALL,
    fallback_fraction: float = 0.95,
    skip_if_empty: bool = True,
) -> list[NlpExtractResult]:
    """
    Download each PDF from GCS into memory, extract NLP text, write ``.txt`` locally.

    **Resume:** With ``resume=True`` (default), reads ``output_txt_dir / ".nlp_checkpoints.jsonl"``
    and **skips** blobs already recorded there (except ``download_failed``, which is not
    checkpointed and will be retried). Append new checkpoint lines as blobs finish.

    **force_fresh:** If True, deletes the checkpoint file before running and ignores prior
    progress (still respects ``clear_jsonl`` for the corpus file).

    Uses a thread pool (``max_workers``). JSONL appends are locked; when resuming with
    existing progress, ``clear_jsonl`` does not delete the JSONL file so prior lines stay.
    """
    configure_batch_logging()
    output_txt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_txt_dir / ".nlp_checkpoints.jsonl"

    if force_fresh and checkpoint_path.is_file():
        checkpoint_path.unlink()
        logger.info("force_fresh: removed checkpoint file %s", checkpoint_path)

    done_map: dict[str, NlpExtractResult] = {}
    if resume and not force_fresh and checkpoint_path.is_file():
        done_map = _load_checkpoints(checkpoint_path)
        if done_map:
            logger.info(
                "resume: loaded %s checkpoint(s) from %s",
                len(done_map),
                checkpoint_path,
            )

    if jsonl_path is not None and clear_jsonl and jsonl_path.is_file():
        if resume and not force_fresh and done_map:
            logger.info(
                "resume: keeping existing JSONL at %s (append new rows only)",
                jsonl_path,
            )
        else:
            jsonl_path.unlink()

    blobs = list(iter_pdf_blobs(bucket_name, prefix, max_objects=max_objects))
    blob_names = [b.name for b in blobs]
    total = len(blob_names)
    if total == 0:
        logger.info("GCS NLP: 0 PDF(s); nothing to do.")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if manifest_format == "csv":
            _write_nlp_manifest_csv(manifest_path, [])
        else:
            _write_nlp_manifest_json(manifest_path, [])
        return []

    pending_idx0: list[int] = []
    results: list[NlpExtractResult | None] = [None] * total
    for idx0, bn in enumerate(blob_names):
        if resume and not force_fresh and bn in done_map:
            results[idx0] = done_map[bn]
            logger.info(
                "[%s/%s] resume (checkpoint): %s",
                idx0 + 1,
                total,
                bn,
            )
        else:
            pending_idx0.append(idx0)

    pending_count = len(pending_idx0)
    workers = max(1, min(max_workers, pending_count)) if pending_count else 1
    logger.info(
        "GCS NLP: %s PDF(s) from gs://%s/%s — %s pending, %s resumed (workers=%s)",
        total,
        bucket_name,
        _normalize_prefix(prefix).rstrip("/") or "(root)",
        pending_count,
        total - pending_count,
        workers,
    )

    jsonl_lock = threading.Lock()
    checkpoint_lock = threading.Lock()

    if pending_idx0:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx0 = {
                executor.submit(
                    _process_one_gcs_blob,
                    idx0 + 1,
                    total,
                    bucket_name,
                    blob_names[idx0],
                    output_txt_dir,
                    min_page_fraction_back_matter=min_page_fraction_back_matter,
                    policy=policy,
                    fallback_fraction=fallback_fraction,
                    skip_if_empty=skip_if_empty,
                ): idx0
                for idx0 in pending_idx0
            }

            for fut in as_completed(future_to_idx0):
                idx0 = future_to_idx0[fut]
                r = fut.result()
                results[idx0] = r
                bn = blob_names[idx0]

                if _should_write_checkpoint(r):
                    _append_checkpoint_line(checkpoint_path, bn, r, checkpoint_lock)

                if jsonl_path is not None and r.output_txt_path is not None:
                    stem = Path(bn).stem
                    text_body = r.output_txt_path.read_text(encoding="utf-8")
                    meta = {
                        "source": r.source,
                        "pages_total": r.pages_total,
                        "page_start_0based": r.page_start_0based,
                        "page_end_exclusive": r.page_end_exclusive,
                        "abstract_found": r.abstract_found,
                        "back_matter_found": r.back_matter_found,
                        "end_reason": r.end_reason,
                        "char_count": r.char_count,
                        "skipped_empty": r.skipped_empty,
                        "notes": r.notes,
                    }
                    record = {"id": stem, "text": text_body, "meta": meta}
                    with jsonl_lock:
                        _append_jsonl(jsonl_path, record)

    ordered: list[NlpExtractResult] = [r for r in results if r is not None]
    assert len(ordered) == total

    failed = sum(1 for r in ordered if r.end_reason in _FAILURE_END_REASONS)
    skipped = sum(1 for r in ordered if r.skipped_empty)
    converted = total - failed - skipped

    logger.info(
        "GCS NLP done: %s converted, %s skipped (empty text), %s failed (unreadable/empty blob), %s total (workers=%s)",
        converted,
        skipped,
        failed,
        total,
        workers,
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_format == "csv":
        _write_nlp_manifest_csv(manifest_path, ordered)
    else:
        _write_nlp_manifest_json(manifest_path, ordered)
    return ordered
