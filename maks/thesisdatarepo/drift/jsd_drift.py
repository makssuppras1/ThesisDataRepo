"""Jensen–Shannon distance vs pre-baseline vocabulary by year."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy.spatial import distance


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]{3,}", text.lower(), flags=re.IGNORECASE)


def jsd_year_series(
    df: pd.DataFrame,
    *,
    year_column: str,
    text_column: str = "text_bucket",
    baseline_year_max: int = 2018,
    max_vocab: int = 6000,
    smooth: float = 1e-9,
) -> pd.DataFrame:
    """
    Baseline term distribution from documents with ``year <= baseline_year_max``;
    same vocabulary used for each calendar year's distribution. JSD uses
    ``scipy.spatial.distance.jensenshannon`` (square root of JSD when ``base=2``).
    """
    tc = text_column if text_column in df.columns else "text"
    y = pd.to_numeric(df[year_column], errors="coerce")
    texts = df[tc].fillna("").astype(str)
    d2 = pd.DataFrame({"_y": y, "_t": texts}).dropna(subset=["_y"])
    base = d2[d2["_y"] <= baseline_year_max]
    if base.empty:
        return pd.DataFrame(columns=["year", "jsd", "n_docs"])

    c = Counter()
    for t in base["_t"]:
        c.update(_tokens(t))
    vocab = [w for w, _ in c.most_common(max_vocab)]
    if not vocab:
        return pd.DataFrame(columns=["year", "jsd", "n_docs"])
    idx = {w: i for i, w in enumerate(vocab)}

    def vec_for_texts(series: pd.Series) -> np.ndarray:
        cnt = np.zeros(len(vocab), dtype=float)
        for t in series:
            for w in _tokens(t):
                i = idx.get(w)
                if i is not None:
                    cnt[i] += 1.0
        cnt = cnt + smooth
        s = cnt.sum()
        return cnt / s if s > 0 else np.ones(len(vocab)) / len(vocab)

    p0 = vec_for_texts(base["_t"])

    rows: list[dict[str, float | int]] = []
    for yr in sorted(d2["_y"].astype(int).unique()):
        sub = d2[d2["_y"].astype(int) == yr]
        if sub.empty:
            continue
        p1 = vec_for_texts(sub["_t"])
        jsd = float(distance.jensenshannon(p0, p1, base=2.0))
        rows.append({"year": int(yr), "jsd": jsd, "n_docs": int(len(sub))})
    return pd.DataFrame(rows)
