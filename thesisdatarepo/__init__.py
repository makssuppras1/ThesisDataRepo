"""Thesis PDF utilities: extract main-body pages by heuristically skipping back matter."""

from thesisdatarepo.pdf_context import (
    FallbackPolicy,
    ProcessResult,
    build_context_writer,
    find_back_matter_start_page,
    process_folder,
    process_one_pdf,
    resolve_end_exclusive,
)

__all__ = [
    "FallbackPolicy",
    "ProcessResult",
    "build_context_writer",
    "find_back_matter_start_page",
    "process_folder",
    "process_one_pdf",
    "resolve_end_exclusive",
]
