"""Distinct publishers and entropy per cluster."""

from __future__ import annotations

import pandas as pd
from scipy.stats import entropy as scipy_entropy


def interdisciplinary_table(
    df: pd.DataFrame,
    cluster_col: str,
    publisher_col: str,
) -> pd.DataFrame:
    rows = []
    for c in sorted(df[cluster_col].unique()):
        sub = df[df[cluster_col] == c]
        pubs = sub[publisher_col].fillna("unknown").astype(str)
        vc = pubs.value_counts()
        ent = float(scipy_entropy(vc.values)) if len(vc) > 0 else 0.0
        rows.append(
            {
                "cluster_id": int(c) if pd.notna(c) else c,
                "n_docs": len(sub),
                "n_distinct_publishers": int(vc.shape[0]),
                "publisher_entropy": ent,
            }
        )
    return pd.DataFrame(rows)
