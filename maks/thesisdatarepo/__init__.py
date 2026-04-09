"""Thesis PDF utilities: extract main-body pages by heuristically skipping back matter."""

from thesisdatarepo.gcs_nlp import (
    iter_pdf_blobs,
    parse_gs_uri,
    process_gcs_prefix_nlp,
)
from thesisdatarepo.nlp_extract import (
    NlpExtractResult,
    configure_batch_logging,
    extract_nlp_page_range,
    find_abstract_start_page,
    normalize_nlp_text,
    page_looks_like_abstract_start,
    process_folder_nlp,
    process_one_pdf_nlp,
    process_pdf_nlp_reader,
)
from thesisdatarepo.pdf_context import (
    FallbackPolicy,
    ProcessResult,
    build_context_writer,
    extract_page_texts,
    find_back_matter_start_page,
    process_folder,
    process_one_pdf,
    resolve_end_exclusive,
)

__all__ = [
    "FallbackPolicy",
    "NlpExtractResult",
    "ProcessResult",
    "build_context_writer",
    "configure_batch_logging",
    "extract_nlp_page_range",
    "extract_page_texts",
    "find_abstract_start_page",
    "find_back_matter_start_page",
    "iter_pdf_blobs",
    "normalize_nlp_text",
    "page_looks_like_abstract_start",
    "parse_gs_uri",
    "process_folder",
    "process_folder_nlp",
    "process_gcs_prefix_nlp",
    "process_one_pdf",
    "process_one_pdf_nlp",
    "process_pdf_nlp_reader",
    "resolve_end_exclusive",
]
