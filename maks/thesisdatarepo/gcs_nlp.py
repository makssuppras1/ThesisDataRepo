"""Stream PDFs from Google Cloud Storage through NLP extraction (no local PDF copies)."""

from __future__ import annotations

import logging
import re
from io import BytesIO
from pathlib import Path
from typing import Iterator, Literal

from google.cloud import storage
from pypdf import PdfReader

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


def process_gcs_prefix_nlp(
    bucket_name: str,
    prefix: str,
    output_txt_dir: Path,
    manifest_path: Path,
    *,
    max_objects: int | None = 10,
    manifest_format: Literal["csv", "json"] = "csv",
    jsonl_path: Path | None = None,
    clear_jsonl: bool = True,
    min_page_fraction_back_matter: float = 0.45,
    policy: FallbackPolicy = FallbackPolicy.KEEP_ALL,
    fallback_fraction: float = 0.95,
    skip_if_empty: bool = True,
) -> list[NlpExtractResult]:
    """
    Download each PDF from GCS into memory, extract NLP text, write ``.txt`` locally.

    PDFs are not written to disk; only extracted ``.txt``, manifest, and optional ``jsonl`` are local.
    Uses Application Default Credentials (same as ``gcloud`` / ``gsutil`` when configured).

    Logs progress ``[i/N]`` and skips empty extractions when ``skip_if_empty`` is True.
    """
    configure_batch_logging()
    output_txt_dir.mkdir(parents=True, exist_ok=True)

    if jsonl_path is not None and clear_jsonl and jsonl_path.is_file():
        jsonl_path.unlink()

    blobs = list(iter_pdf_blobs(bucket_name, prefix, max_objects=max_objects))
    total = len(blobs)
    logger.info(
        "GCS NLP: %s PDF(s) from gs://%s/%s",
        total,
        bucket_name,
        _normalize_prefix(prefix).rstrip("/") or "(root)",
    )

    results: list[NlpExtractResult] = []
    converted = 0
    skipped = 0
    for i, blob in enumerate(blobs, start=1):
        data = blob.download_as_bytes()
        reader = PdfReader(BytesIO(data))
        source = f"gs://{bucket_name}/{blob.name}"
        stem = Path(blob.name).stem
        dst = output_txt_dir / f"{stem}.txt"
        r = process_pdf_nlp_reader(
            reader,
            source,
            dst,
            min_page_fraction_back_matter=min_page_fraction_back_matter,
            policy=policy,
            fallback_fraction=fallback_fraction,
            write_txt=True,
            skip_if_empty=skip_if_empty,
        )
        results.append(r)
        if r.skipped_empty:
            skipped += 1
            logger.info("[%s/%s] skipped (empty text): %s", i, total, blob.name)
        else:
            converted += 1
            logger.info(
                "[%s/%s] converted %s -> %s chars (pages [%s,%s))",
                i,
                total,
                blob.name,
                r.char_count,
                r.page_start_0based,
                r.page_end_exclusive,
            )

        if jsonl_path is not None and r.output_txt_path is not None:
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
            _append_jsonl(
                jsonl_path,
                {"id": stem, "text": text_body, "meta": meta},
            )

    logger.info(
        "GCS NLP done: %s converted, %s skipped (empty), %s total",
        converted,
        skipped,
        total,
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_format == "csv":
        _write_nlp_manifest_csv(manifest_path, results)
    else:
        _write_nlp_manifest_json(manifest_path, results)
    return results
