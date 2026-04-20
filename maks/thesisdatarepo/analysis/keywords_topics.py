"""Per-cluster TF-IDF top terms."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


def cluster_top_terms(
    texts: list[str],
    cluster_id: np.ndarray,
    *,
    max_features: int,
    min_df: int,
    top_n: int,
) -> pd.DataFrame:
    vec = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        stop_words="english",
    )
    X = vec.fit_transform(texts)
    feat = np.array(vec.get_feature_names_out())
    rows = []
    for c in sorted(np.unique(cluster_id)):
        idx = np.where(cluster_id == c)[0]
        if len(idx) == 0:
            continue
        sub = X[idx].sum(axis=0).A1
        top = np.argsort(sub)[::-1][:top_n]
        terms = ", ".join(feat[top])
        rows.append({"cluster_id": int(c), "top_terms": terms, "n_docs": len(idx)})
    return pd.DataFrame(rows)
