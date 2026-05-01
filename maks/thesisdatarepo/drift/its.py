"""Interrupted time series: y ~ year + post + time_since (HC3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def fit_its(
    y: pd.Series | np.ndarray,
    year: pd.Series | np.ndarray,
    *,
    cutoff: int = 2022,
) -> dict[str, float]:
    """
    ``post = (year > cutoff)``, ``time_since = (year - cutoff) * post``.
    OLS with HC3 robust SEs; returns betas and 95% CIs for ``year``, ``post``, ``time_since``.
    """
    yv = pd.to_numeric(pd.Series(y), errors="coerce").astype(float).to_numpy()
    yr = pd.to_numeric(pd.Series(year), errors="coerce").astype(float).to_numpy()
    mask = np.isfinite(yv) & np.isfinite(yr)
    yv, yr = yv[mask], yr[mask]
    if len(yv) < 10:
        return {"n": float(len(yv)), "error": 1.0}

    post = (yr > cutoff).astype(float)
    time_since = (yr - float(cutoff)) * post
    X = pd.DataFrame({"const": 1.0, "year": yr, "post": post, "time_since": time_since})
    model = sm.OLS(yv, X).fit(cov_type="HC3")
    ci = model.conf_int(alpha=0.05)
    params = model.params
    pvals = model.pvalues

    out: dict[str, float] = {"n": float(len(yv))}
    for name in ("year", "post", "time_since"):
        out[f"beta_{name}"] = float(params[name])
        out[f"ci_low_{name}"] = float(ci.loc[name, 0])
        out[f"ci_high_{name}"] = float(ci.loc[name, 1])
        out[f"pvalue_{name}"] = float(pvals[name])
    return out


def run_its_all_features(
    df: pd.DataFrame,
    *,
    year_column: str,
    feature_prefix: str = "feat_",
    cutoff: int = 2022,
) -> pd.DataFrame:
    """One ITS row per numeric feature column with prefix."""
    yr = pd.to_numeric(df[year_column], errors="coerce")
    feat_cols = sorted(c for c in df.columns if c.startswith(feature_prefix))
    rows: list[dict[str, float | str]] = []
    for col in feat_cols:
        y = pd.to_numeric(df[col], errors="coerce")
        r = fit_its(y, yr, cutoff=cutoff)
        if r.get("error"):
            continue
        row: dict[str, float | str] = {"feature": col}
        for k, v in r.items():
            if k != "error":
                row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)
