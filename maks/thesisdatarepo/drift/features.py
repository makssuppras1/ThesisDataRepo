"""Per-document features for drift analysis."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from thesisdatarepo.analysis.config_loader import AnalysisConfig
from thesisdatarepo.drift.lexical_metrics import lexical_for_tokens
from thesisdatarepo.drift.preprocess import (
    simple_sentence_split,
    strip_references_section,
    word_tokens,
)
from thesisdatarepo.drift.spacy_features import augment_spacy_features


def title_body_jaccard(title: str, body: str) -> float:
    """Jaccard over word types; title proxies abstract."""
    tt = set(word_tokens(title))
    bt = set(word_tokens(body))
    if not tt and not bt:
        return 1.0
    if not tt or not bt:
        return 0.0
    union = len(tt | bt)
    return len(tt & bt) / union if union else 0.0


@dataclass(frozen=True)
class DocFeatures:
    n_chars: int
    n_words: int
    n_sentences: int
    mean_sentence_length_words: float
    title_body_jaccard: float
    mtld: float
    hapax_ratio: float
    ttr: float


def compute_doc_features(title: str, raw_body: str) -> DocFeatures:
    body = strip_references_section(raw_body)
    sents = simple_sentence_split(body)
    words = word_tokens(body)
    n_words = len(words)
    n_sents = max(len(sents), 1)
    msl = n_words / n_sents if n_sents else 0.0
    jac = title_body_jaccard(title, body)
    mtld, hapax, ttr = lexical_for_tokens(words)
    return DocFeatures(
        n_chars=len(body),
        n_words=n_words,
        n_sentences=len(sents),
        mean_sentence_length_words=float(msl),
        title_body_jaccard=float(jac),
        mtld=float(mtld),
        hapax_ratio=float(hapax),
        ttr=float(ttr),
    )


def augment_feature_table(cfg: AnalysisConfig, df: pd.DataFrame) -> pd.DataFrame:
    """Append numeric ``feat_*`` columns; expects ``text_bucket`` and title from merge."""
    text_col = "text_bucket" if "text_bucket" in df.columns else "text"
    title_col = (cfg.columns_title or "").strip()
    if not title_col or title_col not in df.columns:
        raise KeyError(
            f"Title column {title_col!r} missing for title–body features. "
            "Set [columns] title in config (e.g. MASTER THESIS TITLE)."
        )
    titles = df[title_col].fillna("").astype(str).tolist()
    bodies = df[text_col].fillna("").astype(str).tolist()
    rows = [compute_doc_features(t, b) for t, b in zip(titles, bodies, strict=True)]
    out = df.copy()
    out["feat_n_chars"] = [r.n_chars for r in rows]
    out["feat_n_words"] = [r.n_words for r in rows]
    out["feat_n_sentences"] = [r.n_sentences for r in rows]
    out["feat_mean_sentence_length_words"] = [r.mean_sentence_length_words for r in rows]
    out["feat_title_body_jaccard"] = [r.title_body_jaccard for r in rows]
    out["feat_mtld"] = [r.mtld for r in rows]
    out["feat_hapax_ratio"] = [r.hapax_ratio for r in rows]
    out["feat_ttr"] = [r.ttr for r in rows]

    spacy_cols = augment_spacy_features(bodies)
    for k, vals in spacy_cols.items():
        out[k] = vals
    return out
