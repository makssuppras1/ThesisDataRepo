"""spaCy-based stylometric features; optional if model missing."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from thesisdatarepo.drift.lexical_metrics import lexical_for_tokens
from thesisdatarepo.drift.preprocess import strip_references_section

logger = logging.getLogger(__name__)

_HEDGE = frozenset(
    {"may", "might", "could", "suggest", "appears", "seem", "seems", "appeared", "suggested"}
)
_CONNECTIVES = frozenset(
    {
        "moreover",
        "furthermore",
        "thus",
        "therefore",
        "however",
        "consequently",
    }
)

# Very long theses: cap chars to keep spaCy responsive (~150k words worst-case trim).
_MAX_CHARS = 1_200_000


def try_load_nlp(model: str = "en_core_web_sm") -> Any | None:
    try:
        import spacy

        return spacy.load(model)
    except OSError:
        logger.warning(
            "spaCy model %r not installed. Run: uv run python -m spacy download en_core_web_sm",
            model,
        )
        return None


def extract_spacy_row(nlp: Any, raw_text: str) -> dict[str, float]:
    text = strip_references_section(raw_text)
    if len(text) > _MAX_CHARS:
        text = text[:_MAX_CHARS]
    doc = nlp(text)
    toks = [t for t in doc if t.is_alpha]
    lemmas = [t.lemma_.lower() for t in toks]
    mtld, hapax, ttr = lexical_for_tokens(lemmas)

    sents = list(doc.sents)
    slens = [sum(1 for t in s if t.is_alpha and not t.is_space) for s in sents]
    if not slens:
        slens = [0]
    mean_sl = float(np.mean(slens))
    var_sl = float(np.var(slens, ddof=1)) if len(slens) > 1 else 0.0

    passive = sum(1 for t in doc if t.dep_ in ("nsubjpass", "auxpass"))
    n_alpha = len(toks)
    passive_rate = passive / n_alpha if n_alpha else 0.0

    low = [t.lower_ for t in toks]
    hedge_n = sum(1 for w in low if w in _HEDGE)
    conn_n = sum(1 for w in low if w in _CONNECTIVES)
    hedge_rate = hedge_n / n_alpha if n_alpha else 0.0
    connective_rate = conn_n / n_alpha if n_alpha else 0.0

    return {
        "feat_spacy_mtld": mtld,
        "feat_spacy_hapax_ratio": hapax,
        "feat_spacy_ttr": ttr,
        "feat_spacy_mean_sentence_length": mean_sl,
        "feat_spacy_var_sentence_length": var_sl,
        "feat_spacy_passive_rate": passive_rate,
        "feat_spacy_hedge_rate": hedge_rate,
        "feat_spacy_connective_rate": connective_rate,
    }


def augment_spacy_features(texts: list[str], *, model: str = "en_core_web_sm") -> dict[str, list[float]]:
    keys = [
        "feat_spacy_mtld",
        "feat_spacy_hapax_ratio",
        "feat_spacy_ttr",
        "feat_spacy_mean_sentence_length",
        "feat_spacy_var_sentence_length",
        "feat_spacy_passive_rate",
        "feat_spacy_hedge_rate",
        "feat_spacy_connective_rate",
    ]
    n = len(texts)
    if n == 0:
        return {k: [] for k in keys}

    nlp = try_load_nlp(model)
    if nlp is None:
        return {k: [float("nan")] * n for k in keys}

    rows = [extract_spacy_row(nlp, t) for t in texts]
    return {k: [r[k] for r in rows] for k in keys}
