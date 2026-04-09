"""Load corpus JSONL and metadata CSV; merge optional faculty table."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from thesisdatarepo.analysis.config_loader import AnalysisConfig

logger = logging.getLogger(__name__)


def load_corpus_jsonl(path: Path) -> pd.DataFrame:
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
    return pd.DataFrame(rows)


def load_merged_frame(cfg: AnalysisConfig) -> pd.DataFrame:
    corpus = load_corpus_jsonl(cfg.corpus_jsonl)
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
    corpus["id"] = corpus["id"].astype(str).str.strip()

    merged = meta.merge(
        corpus,
        left_on=id_col,
        right_on="id",
        how="inner",
        suffixes=("", "_corpus"),
    )
    if "id" in merged.columns and id_col != "id":
        merged.drop(columns=["id"], inplace=True, errors="ignore")
    logger.info(
        "Merged metadata (%s rows) with corpus (%s rows) -> %s rows",
        len(meta),
        len(corpus),
        len(merged),
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
        texts = df["text"].fillna("").astype(str).tolist()

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
