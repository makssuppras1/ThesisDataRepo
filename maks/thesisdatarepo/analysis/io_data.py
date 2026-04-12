"""Load corpus JSONL and metadata CSV; merge optional faculty table."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from thesisdatarepo.analysis.config_loader import AnalysisConfig

logger = logging.getLogger(__name__)


def load_corpus_jsonl(path: Path) -> pd.DataFrame:
    """Load JSONL written by NLP export; ``id`` = blob/file stem, ``text`` = extracted body."""
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            rows.append(
                {
                    "id": str(o["id"]),
                    "text": str(o.get("text", "") or ""),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["id", "text"])
    df = pd.DataFrame(rows)
    # Resume / re-runs may append duplicate ids; last line wins (newest extraction).
    df = df.drop_duplicates(subset=["id"], keep="last").reset_index(drop=True)
    return df


def filter_corpus_to_nlp_txt_dir(corpus: pd.DataFrame, txt_dir: Path) -> pd.DataFrame:
    """
    Keep only rows whose ``id`` matches a file ``txt_dir / f"{id}.txt"``.

    GCS NLP writes ``{blob_stem}.txt`` and JSONL uses the same stem as ``id``;
    this drops JSONL lines that have no corresponding on-disk export (or orphaned
    JSONL after partial runs).
    """
    if corpus.empty:
        return corpus
    if not txt_dir.is_dir():
        logger.warning("nlp_txt_dir is not a directory (%s) — skipping .txt filter", txt_dir)
        return corpus

    def has_file(i: str) -> bool:
        return (txt_dir / f"{i}.txt").is_file()

    mask = corpus["id"].map(has_file)
    n_before = len(corpus)
    out = corpus.loc[mask].reset_index(drop=True)
    logger.info(
        "NLP .txt cross-check (%s): kept %s / %s rows (dropped %s without matching file)",
        txt_dir,
        len(out),
        n_before,
        n_before - len(out),
    )
    return out


def load_merged_frame(cfg: AnalysisConfig) -> pd.DataFrame:
    """
    **Bucket / corpus is the source of truth:** only rows with extracted text in the
    JSONL (GCS NLP output) are analyzed. Metadata rows without a matching corpus id are
    dropped. Corpus text is stored as ``text_bucket`` so it is never confused with a
    metadata ``text`` column.
    """
    corpus = load_corpus_jsonl(cfg.corpus_jsonl)
    if cfg.nlp_txt_dir is not None:
        corpus = filter_corpus_to_nlp_txt_dir(corpus, cfg.nlp_txt_dir)
    if corpus.empty:
        logger.warning("Corpus JSONL is empty at %s", cfg.corpus_jsonl)

    meta = pd.read_csv(
        cfg.metadata_csv,
        sep=cfg.metadata_sep,
        encoding="utf-8",
        dtype=str,
        low_memory=False,
    )
    id_col = cfg.columns_id
    if id_col not in meta.columns:
        raise KeyError(
            f"Metadata column {id_col!r} not found. Columns: {list(meta.columns)}"
        )

    meta[id_col] = meta[id_col].astype(str).str.strip()
    n_meta_before = len(meta)
    meta = meta.drop_duplicates(subset=[id_col], keep="first").reset_index(drop=True)
    n_meta_deduped = n_meta_before - len(meta)

    corpus["id"] = corpus["id"].astype(str).str.strip()
    corp_ids = set(corpus["id"])
    meta_ids = set(meta[id_col])

    # Rename before merge so metadata cannot overwrite bucket text.
    corp = corpus.rename(columns={"text": "text_bucket"})

    # Left join from corpus: one row per bucket document; metadata optional per id.
    merged = corp.merge(
        meta,
        left_on="id",
        right_on=id_col,
        how="left",
        suffixes=("", "_meta_dup"),
    )
    # Join key from metadata may be NaN when a bucket id has no metadata row
    if id_col in merged.columns:
        merged[id_col] = merged[id_col].fillna(merged["id"]).astype(str)
    else:
        merged[id_col] = merged["id"].astype(str)

    orphan_meta_ids = meta_ids - corp_ids
    logger.info(
        "Bucket corpus: %s unique ids (JSONL). Metadata: %s rows (%s unique ids); "
        "metadata duplicate rows dropped: %s; metadata ids with no bucket text (excluded): %s",
        len(corpus),
        n_meta_before,
        len(meta_ids),
        n_meta_deduped,
        len(orphan_meta_ids),
    )

    if cfg.faculty_csv and cfg.faculty_csv.is_file():
        fac = pd.read_csv(
            cfg.faculty_csv,
            sep=cfg.metadata_sep,
            encoding="utf-8",
            dtype=str,
            low_memory=False,
        )
        fc = cfg.columns_faculty
        if fc not in fac.columns:
            raise KeyError(f"Faculty CSV missing column {fc!r}")
        if id_col not in fac.columns:
            raise KeyError(f"Faculty CSV missing id column {id_col!r}")
        fac[id_col] = fac[id_col].astype(str).str.strip()
        sub = fac[[id_col, fc]].drop_duplicates(subset=[id_col])
        merged = merged.drop(columns=[fc], errors="ignore")
        merged = merged.merge(sub, on=id_col, how="left")

    return merged


def column_or_empty(df: pd.DataFrame, name: str) -> pd.Series:
    if not name or name not in df.columns:
        return pd.Series([""] * len(df), index=df.index)
    return df[name].fillna("").astype(str)


def build_embedding_texts(cfg: AnalysisConfig, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return filtered dataframe and list of texts for embedding (same order)."""
    id_col = cfg.columns_id
    abstract_col = cfg.columns_abstract
    title_col = cfg.columns_title

    if cfg.embedding_source == "abstract":
        if abstract_col not in df.columns:
            raise KeyError(f"Abstract column {abstract_col!r} not in merged frame")
        texts = df[abstract_col].fillna("").astype(str).tolist()
    else:
        # Always use extracted bucket text, never a metadata column named "text".
        col = "text_bucket" if "text_bucket" in df.columns else "text"
        texts = df[col].fillna("").astype(str).tolist()

    if cfg.embedding_source == "abstract" or not cfg.embedding_chunked:
        # optional title prepend for abstract path (single string)
        if title_col and title_col in df.columns:
            titles = df[title_col].fillna("").astype(str)
            texts = [
                (f"{t}\n\n{x}" if t.strip() else x) for t, x in zip(titles, texts, strict=True)
            ]

    mask = [len(t.strip()) >= cfg.min_text_chars for t in texts]
    df2 = df.loc[mask].reset_index(drop=True)
    texts2 = [texts[i] for i, m in enumerate(mask) if m]
    logger.info(
        "After min_text_chars=%s: %s documents (dropped %s)",
        cfg.min_text_chars,
        len(df2),
        len(df) - len(df2),
    )
    return df2, texts2
