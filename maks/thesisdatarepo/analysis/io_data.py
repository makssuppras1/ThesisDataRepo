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


def _txt_path_to_corpus_id(stem: str, mode: str) -> str:
    """
    ``full_stem``: whole filename without .txt (matches GCS blob / pdf_file stem).
    ``member_prefix``: segment before the first ``_`` — same as ``member_id_ss`` /
    ``primary_member_id_s`` when exports are named ``{id}_{title}.txt``.
    """
    mode = (mode or "full_stem").lower().strip()
    if mode == "member_prefix":
        if "_" in stem:
            return stem.split("_", 1)[0].strip()
        return stem.strip()
    if mode != "full_stem":
        raise ValueError(f"Unknown corpus_txt_id_mode: {mode!r} (use full_stem or member_prefix)")
    return stem


def load_corpus_from_txt_dir(txt_dir: Path, *, id_mode: str = "full_stem") -> pd.DataFrame:
    """One row per ``*.txt`` file; ``id`` is derived from the filename per ``id_mode``."""
    if not txt_dir.is_dir():
        raise FileNotFoundError(f"NLP text directory not found: {txt_dir}")
    rows: list[dict[str, str]] = []
    for f in sorted(txt_dir.glob("*.txt")):
        try:
            body = f.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            logger.warning("Skip unreadable %s: %s", f, e)
            continue
        cid = _txt_path_to_corpus_id(f.stem, id_mode)
        rows.append({"id": cid, "text": body})
    if not rows:
        return pd.DataFrame(columns=["id", "text"])
    n_files = len(rows)
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["id"], keep="last").reset_index(drop=True)
    logger.info(
        "Loaded %s unique id(s) from %s .txt files in %s (id_mode=%s)",
        len(df),
        n_files,
        txt_dir,
        id_mode,
    )
    return df


def load_metadata_table(cfg: AnalysisConfig) -> pd.DataFrame:
    """CSV (``metadata_sep``) or Parquet."""
    path = cfg.metadata_csv
    suf = path.suffix.lower()
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(
        path,
        sep=cfg.metadata_sep,
        encoding="utf-8",
        dtype=str,
        low_memory=False,
    )


def _stem_for_join(val: object) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val).strip()
    if not s:
        return ""
    return Path(s).stem


def metadata_join_keys(meta: pd.DataFrame, cfg: AnalysisConfig) -> pd.Series:
    """Series aligned with ``meta`` for merging to corpus ``id``."""
    cj = (cfg.columns_corpus_join or "").strip()
    if cj:
        if cj not in meta.columns:
            raise KeyError(
                f"Metadata has no corpus_join column {cj!r}. Columns: {list(meta.columns)}"
            )
        return meta[cj].map(_stem_for_join)
    col = cfg.columns_id
    if col not in meta.columns:
        raise KeyError(f"Metadata column {col!r} not found.")
    return meta[col].astype(str).str.strip()


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


def load_corpus_for_config(cfg: AnalysisConfig) -> pd.DataFrame:
    """Load corpus from JSONL or from ``nlp_txt_dir`` per ``corpus_source``."""
    src = (cfg.corpus_source or "jsonl").lower().strip()
    if src == "txt_dir":
        if cfg.nlp_txt_dir is None:
            raise ValueError("corpus_source=txt_dir requires paths.nlp_txt_dir")
        return load_corpus_from_txt_dir(
            cfg.nlp_txt_dir,
            id_mode=cfg.corpus_txt_id_mode,
        )
    if src != "jsonl":
        raise ValueError(f"Unknown corpus_source: {src!r} (use jsonl or txt_dir)")
    if cfg.corpus_jsonl is None:
        raise ValueError("corpus_source=jsonl requires paths.corpus_jsonl")
    return load_corpus_jsonl(cfg.corpus_jsonl)


def load_merged_frame(cfg: AnalysisConfig) -> pd.DataFrame:
    """
    **Corpus is the source of truth:** only rows with extracted text are analyzed.
    Corpus text is stored as ``text_bucket``. Metadata is left-joined using
    ``corpus_join`` (e.g. ``pdf_file`` stem) or ``columns.id`` when ``corpus_join`` is empty.
    """
    corpus = load_corpus_for_config(cfg)
    # JSONL mode: optional filter so rows match on-disk .txt files
    if cfg.corpus_source == "jsonl" and cfg.nlp_txt_dir is not None:
        corpus = filter_corpus_to_nlp_txt_dir(corpus, cfg.nlp_txt_dir)
    if corpus.empty:
        logger.warning("Corpus is empty after loading")

    meta = load_metadata_table(cfg)
    id_col = cfg.columns_id
    if id_col not in meta.columns:
        raise KeyError(
            f"Metadata column {id_col!r} not found. Columns: {list(meta.columns)}"
        )

    meta[id_col] = meta[id_col].astype(str).str.strip()
    n_meta_before = len(meta)
    join_keys = metadata_join_keys(meta, cfg)
    meta = meta.copy()
    meta["_join_corpus_id"] = join_keys
    meta = meta.drop_duplicates(subset=["_join_corpus_id"], keep="first").reset_index(drop=True)
    n_meta_deduped = n_meta_before - len(meta)

    corpus["id"] = corpus["id"].astype(str).str.strip()
    corp_ids = set(corpus["id"])
    meta_join_set = set(meta["_join_corpus_id"].dropna().astype(str).str.strip()) - {""}

    corp = corpus.rename(columns={"text": "text_bucket"})

    merged = corp.merge(
        meta,
        left_on="id",
        right_on="_join_corpus_id",
        how="left",
        suffixes=("", "_meta_dup"),
    )
    merged.drop(columns=["_join_corpus_id"], inplace=True, errors="ignore")

    if id_col in merged.columns:
        merged[id_col] = merged[id_col].fillna("").astype(str)
    else:
        merged[id_col] = ""

    if cfg.inner_join_metadata:
        before = len(merged)
        has_meta = merged[id_col].astype(str).str.strip().ne("")
        merged = merged.loc[has_meta].reset_index(drop=True)
        logger.info(
            "inner_join_metadata: kept %s / %s rows (dropped corpus without metadata match)",
            len(merged),
            before,
        )

    orphan_meta = meta_join_set - corp_ids
    logger.info(
        "Corpus: %s ids. Metadata: %s rows (%s unique join keys); duplicate metadata rows dropped: %s; "
        "metadata join keys with no corpus text (excluded): %s",
        len(corpus),
        n_meta_before,
        len(meta_join_set),
        n_meta_deduped,
        len(orphan_meta),
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
