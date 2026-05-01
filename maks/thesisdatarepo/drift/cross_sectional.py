"""Mann–Whitney U, Benjamini–Hochberg, Cohen's d (Road_map Phase 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def _cohens_d_post_minus_pre(pre: np.ndarray, post: np.ndarray) -> float:
    """Positive => higher values in post-LLM slice."""
    pre = pre[np.isfinite(pre)]
    post = post[np.isfinite(post)]
    if len(pre) < 2 or len(post) < 2:
        return float("nan")
    na, nb = len(pre), len(post)
    va, vb = np.var(pre, ddof=1), np.var(post, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0 or not np.isfinite(pooled):
        return float("nan")
    return float((np.mean(post) - np.mean(pre)) / pooled)


def run_cross_sectional(
    df: pd.DataFrame,
    *,
    year_column: str,
    pre_year_max: int = 2021,
    post_year_min: int = 2023,
    feature_prefix: str = "feat_",
) -> pd.DataFrame:
    """
    Pre = ``year <= pre_year_max``, post = ``year >= post_year_min``; other years dropped.
    Cohen's d: (mean_post - mean_pre) / pooled SD (positive => higher in post).
    """
    if year_column not in df.columns:
        raise KeyError(f"Year column {year_column!r} not in dataframe")
    y = pd.to_numeric(df[year_column], errors="coerce")
    mask = y.notna() & ((y <= pre_year_max) | (y >= post_year_min))
    sub = df.loc[mask].copy()
    sub["_year_num"] = pd.to_numeric(sub[year_column], errors="coerce")
    pre_m = sub["_year_num"] <= pre_year_max
    post_m = sub["_year_num"] >= post_year_min
    pre_df = sub.loc[pre_m]
    post_df = sub.loc[post_m]

    feat_cols = sorted(c for c in sub.columns if c.startswith(feature_prefix))
    if not feat_cols:
        raise ValueError(f"No numeric columns with prefix {feature_prefix!r}")

    u_stats: list[float] = []
    p_raw: list[float] = []
    cohens: list[float] = []
    for col in feat_cols:
        a = pd.to_numeric(pre_df[col], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(post_df[col], errors="coerce").to_numpy(dtype=float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) < 2 or len(b) < 2:
            u_stats.append(float("nan"))
            p_raw.append(1.0)
            cohens.append(float("nan"))
            continue
        res = stats.mannwhitneyu(a, b, alternative="two-sided")
        u_stats.append(float(res.statistic))
        p_raw.append(float(res.pvalue))
        cohens.append(_cohens_d_post_minus_pre(a, b))

    p_for_fdr = np.where(np.isfinite(p_raw), p_raw, 1.0)
    _, p_bh, _, _ = multipletests(p_for_fdr, alpha=0.05, method="fdr_bh")
    coh_arr = np.array(cohens, dtype=float)
    meaningful = (p_bh < 0.05) & (np.abs(coh_arr) > 0.3)

    return pd.DataFrame(
        {
            "feature": feat_cols,
            "u_stat": u_stats,
            "p_raw": p_raw,
            "p_bh": p_bh,
            "cohens_d": cohens,
            "significant_fdr": p_bh < 0.05,
            "meaningful_fdr_and_d": meaningful,
        }
    )
