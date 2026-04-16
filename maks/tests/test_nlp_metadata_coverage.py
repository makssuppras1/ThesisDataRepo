"""Coverage between ``maks/data/nlp_from_gcs_all`` and metadata parquet."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from thesisdatarepo.analysis.io_data import _txt_path_to_corpus_id
from thesisdatarepo.analysis.paths_util import repo_root


METADATA_PARQUET = "data/data_analysis_files/df_filtered_excl_grades_15042026.parquet"
NLP_TXT_DIR = "maks/data/nlp_from_gcs_all"


def _metadata_allowed_sets(meta_path: Path) -> tuple[set[str], set[str]]:
    df = pd.read_parquet(meta_path)
    pdf_stems: set[str] = set()
    if "pdf_file" in df.columns:
        pdf_stems.update(
            df["pdf_file"].dropna().map(lambda x: Path(str(x)).stem).tolist()
        )
    member_ids: set[str] = set()
    for col in ("member_id_ss", "primary_member_id_s"):
        if col in df.columns:
            member_ids.update(df[col].dropna().astype(str).str.strip().tolist())
    return pdf_stems, member_ids


def test_nlp_txt_referenced_in_metadata_soft_counts() -> None:
    """
    Report how many ``.txt`` files link to the metadata table via **either**
    ``Path(pdf_file).stem == file stem`` **or** ``member_id_ss`` / ``primary_member_id_s``
    equals the segment before the first ``_`` in the filename.

    The filtered parquet does not include every GCS export; a large orphan count is expected
    until metadata is expanded or the NLP folder is trimmed.
    """
    root = repo_root()
    meta_path = root / METADATA_PARQUET
    txt_dir = root / NLP_TXT_DIR
    if not meta_path.is_file() or not txt_dir.is_dir():
        pytest.skip("metadata parquet or NLP directory missing")

    pdf_stems, member_ids = _metadata_allowed_sets(meta_path)
    files = list(txt_dir.glob("*.txt"))
    matched = 0
    for f in files:
        stem = f.stem
        pre = _txt_path_to_corpus_id(stem, "member_prefix")
        if stem in pdf_stems or pre in member_ids:
            matched += 1

    assert len(files) > 0
    assert matched > 0
    # Loose sanity: current repo snapshot has ~4.2k matches out of ~9k files
    assert matched >= 3000, f"expected thousands of matches, got {matched}/{len(files)}"


@pytest.mark.xfail(
    reason=(
        "Full-folder inclusion: filtered metadata omits many GCS stems. "
        "Remove xfail when parquet lists every export or NLP folder is subset-only."
    ),
    raises=AssertionError,
    strict=False,
)
def test_every_nlp_txt_file_has_row_in_metadata() -> None:
    """Strict: each ``*.txt`` must match metadata via pdf stem OR member-id prefix."""
    root = repo_root()
    meta_path = root / METADATA_PARQUET
    txt_dir = root / NLP_TXT_DIR
    if not meta_path.is_file() or not txt_dir.is_dir():
        pytest.skip("metadata parquet or NLP directory missing")

    pdf_stems, member_ids = _metadata_allowed_sets(meta_path)
    missing: list[str] = []
    for f in sorted(txt_dir.glob("*.txt")):
        stem = f.stem
        pre = _txt_path_to_corpus_id(stem, "member_prefix")
        if stem not in pdf_stems and pre not in member_ids:
            missing.append(f.name)

    assert not missing, (
        f"{len(missing)} file(s) not found in metadata ({METADATA_PARQUET}) "
        f"by pdf_file stem or member id prefix. First few: {missing[:10]}"
    )


@pytest.mark.skipif(
    not os.environ.get("THESIS_STRICT_NLP_METADATA"),
    reason="Set THESIS_STRICT_NLP_METADATA=1 to fail CI if any NLP file lacks metadata",
)
def test_every_nlp_txt_strict_env_gate() -> None:
    """Duplicate strict check for CI when env forces failure (not xfail)."""
    root = repo_root()
    meta_path = root / METADATA_PARQUET
    txt_dir = root / NLP_TXT_DIR
    pdf_stems, member_ids = _metadata_allowed_sets(meta_path)
    missing = [
        f.name
        for f in sorted(txt_dir.glob("*.txt"))
        if f.stem not in pdf_stems
        and _txt_path_to_corpus_id(f.stem, "member_prefix") not in member_ids
    ]
    assert not missing, f"{len(missing)} orphans (see test body in test_nlp_metadata_coverage)"
