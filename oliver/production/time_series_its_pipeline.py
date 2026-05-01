# ruff: noqa: PLR0911, PLR0912, PLR0913, PLR0915, C901 — analysis module; many branches by design
"""
Interrupted time series (ITS) pipeline for thesis writing metrics (`df_filtered_final`).

Analysis design (structural-break framing, not naïve before/after)
-------------------------------------------------------------------
We estimate piecewise-linear trends with level and slope changes at pre-specified
calendar months aligned with LLM milestones. This is **not** identification of a
causal effect of ChatGPT or reasoning models on writing: submissions are seasonal,
department composition shifts over time, grading scores may reflect pipeline changes,
and interventions are not exogenous shocks. Results are evidence of **associations
and statistical breaks** in aggregated time series under explicit linearity/autocorrelation
assumptions.

Implementation: monthly series (university-wide and optional department panels),
OLS / robust / HAC / SARIMAX-style dynamics, diagnostics, and robustness checks.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# Statsmodels is a project dependency (see pyproject.toml)
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError:
    SARIMAX = None  # type: ignore[misc, assignment]


# --- Default configuration (override via PipelineConfig) ---

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "exported_plots" / "time_series_its"
DEFAULT_CHATGPT_DATE = "2022-11-01"
DEFAULT_REASONING_DATE = "2024-07-01"

DEFAULT_METRICS: tuple[str, ...] = (
    "lexical_diversity",
    "avg_sentence_length",
    "avg_word_length",
    "total_words",
    "unique_words",
    "grading_total_score",
    "num_references",
    "equation_count",
    "num_figures",
    "num_tables",
)

NORMALIZED_METRICS: tuple[tuple[str, str, str], ...] = (
    ("refs_per_1000_words", "num_references", "total_words"),
    ("figures_per_100_pages", "num_figures", "num_tot_pages"),
    ("equations_per_1000_words", "equation_count", "total_words"),
)


@dataclass
class PipelineConfig:
    """Top-level knobs for preprocessing, models, and exports."""

    chatgpt_date: str = DEFAULT_CHATGPT_DATE
    reasoning_date: str = DEFAULT_REASONING_DATE
    extra_interventions: tuple[str, ...] = ()
    reasoning_date_grid: tuple[str, ...] = (
        "2024-03-01",
        "2024-07-01",
        "2024-11-01",
    )
    metrics: tuple[str, ...] = DEFAULT_METRICS
    include_normalized: bool = True
    rolling_window: int = 3
    seasonal: Literal["month_dummies", "fourier2"] = "month_dummies"
    hac_maxlags: int | None = None  # None -> data-driven floor(sqrt(n))
    sarimax_order: tuple[int, int, int] = (1, 0, 1)
    sarimax_seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 12)
    min_department_monthly_count: int = 2
    placebo_month_starts: tuple[str, ...] = ("2021-06-01", "2023-03-01")
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)
    year_min: int | None = None
    year_max: int | None = None
    dedupe_key: Literal["ID", "pdf_sha256"] = "ID"
    timestamp_na_share_fallback: float = 0.5  # if Timestamp NaT rate exceeds this, prefer Publication Year path
    early_adopter_depts: frozenset[str] | None = None
    late_adopter_depts: frozenset[str] | None = None
    export_plots: bool = True
    verbose: bool = True


def _log(cfg: PipelineConfig, msg: str) -> None:
    if cfg.verbose:
        print(msg)


def parse_timestamp_primary(df: pd.DataFrame) -> pd.Series:
    """Parse ``Timestamp`` to month-start datetime; NaT where invalid."""
    if "Timestamp" not in df.columns:
        return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    ts = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert(None)
    out = ts.dt.to_period("M").dt.to_timestamp(how="start")
    return out


def extract_month_from_handin(handin: pd.Series) -> pd.Series:
    """
    Month number 1–12 from ``handin_month`` strings like ``January 2025``.

    Year in the string is ignored for the calendar month (per data README).
    """
    parsed = pd.to_datetime(handin, format="%B %Y", errors="coerce")
    return parsed.dt.month.astype("float")


def build_period_datetime(
    df: pd.DataFrame,
    *,
    timestamp_na_share_fallback: float,
) -> tuple[pd.Series, dict[str, Any]]:
    """
    Canonical month-start hand-in period.

    Prefer ``Timestamp`` when it is largely populated; otherwise combine
    ``Publication Year`` + month from ``handin_month_num`` / ``handin_month``.
    """
    meta: dict[str, Any] = {"strategy": None, "timestamp_nat_rate": None}
    ts_p = parse_timestamp_primary(df)
    nat_rate = float(ts_p.isna().mean()) if len(df) else 1.0
    meta["timestamp_nat_rate"] = nat_rate

    py = pd.to_numeric(df.get("Publication Year"), errors="coerce")
    hm_num = pd.to_numeric(df.get("handin_month_num"), errors="coerce")
    if hm_num.notna().sum() < 0.5 * len(df) and "handin_month" in df.columns:
        hm_fill = extract_month_from_handin(df["handin_month"].astype(str))
        hm_num = hm_num.where(hm_num.notna(), hm_fill)

    fallback = pd.to_datetime({"year": py, "month": hm_num, "day": np.ones(len(df))}, errors="coerce")

    use_fallback = nat_rate >= timestamp_na_share_fallback
    period = ts_p.copy()
    if use_fallback:
        period = fallback.dt.to_period("M").dt.to_timestamp(how="start")
        meta["strategy"] = "publication_year_plus_month"
    else:
        fill = fallback.dt.to_period("M").dt.to_timestamp(how="start")
        period = period.fillna(fill)
        meta["strategy"] = "timestamp_fillna_fallback"

    return period, meta


def preprocess_thesis_frame(df: pd.DataFrame, cfg: PipelineConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Drop invalid periods, optionally dedupe, coerce numerics, clip years.

    Returns cleaned frame with ``period_dt`` and QC summary dict.
    """
    report: dict[str, Any] = {}
    work = df.copy()
    n0 = len(work)

    period_dt, strat = build_period_datetime(work, timestamp_na_share_fallback=cfg.timestamp_na_share_fallback)
    work["period_dt"] = period_dt
    work = work.dropna(subset=["period_dt"])
    report["rows_after_drop_invalid_period"] = len(work)

    if cfg.year_min is not None:
        work = work[work["period_dt"].dt.year >= cfg.year_min]
    if cfg.year_max is not None:
        work = work[work["period_dt"].dt.year <= cfg.year_max]

    key = cfg.dedupe_key
    if key in work.columns:
        before = len(work)
        work = work.sort_values(["period_dt", key]).drop_duplicates(subset=[key], keep="last")
        report[f"dedupe_{key}_removed"] = before - len(work)

    # Metrics used downstream
    all_numeric_cols = list(cfg.metrics)
    if cfg.include_normalized:
        all_numeric_cols.extend(["num_tot_pages", "total_words"])

    for c in all_numeric_cols:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")

    # Normalized metrics at thesis level
    tw = pd.to_numeric(work.get("total_words"), errors="coerce")
    tp = pd.to_numeric(work.get("num_tot_pages"), errors="coerce")
    ref = pd.to_numeric(work.get("num_references"), errors="coerce")
    fig = pd.to_numeric(work.get("num_figures"), errors="coerce")
    eq = pd.to_numeric(work.get("equation_count"), errors="coerce")

    work["refs_per_1000_words"] = ref / (tw / 1000.0)
    work["figures_per_100_pages"] = fig / (tp / 100.0)
    work["equations_per_1000_words"] = eq / (tw / 1000.0)

    report["rows_final"] = len(work)
    report["period_strategy"] = strat
    report["date_range"] = (
        str(work["period_dt"].min()),
        str(work["period_dt"].max()),
    )
    _log(cfg, f"[preprocess] rows {n0} -> {len(work)} | period: {report['date_range']} | strategy {strat}")

    return work.reset_index(drop=True), report


def monthly_aggregate_university(
    df: pd.DataFrame,
    metrics: Iterable[str],
    *,
    freq: str = "MS",
) -> pd.DataFrame:
    """One row per month with thesis_count, mean, median, std for each metric."""
    g = df.groupby(pd.Grouper(key="period_dt", freq=freq), observed=True)
    out: dict[str, Any] = {"thesis_count": g.size()}
    for m in metrics:
        if m not in df.columns:
            continue
        sub = g[m]
        out[f"{m}_mean"] = sub.mean()
        out[f"{m}_median"] = sub.median()
        out[f"{m}_std"] = sub.std()

    monthly = pd.DataFrame(out)
    monthly = monthly.sort_index()
    monthly.index.name = "period_dt"
    return monthly


def monthly_aggregate_by_department(
    df: pd.DataFrame,
    metrics: Iterable[str],
    *,
    freq: str = "MS",
) -> pd.DataFrame:
    """Panel: period_dt x Department_new."""
    if "Department_new" not in df.columns:
        return pd.DataFrame()
    metrics = [m for m in metrics if m in df.columns]
    rows = []
    for dept, chunk in df.groupby("Department_new", observed=False):
        muni = monthly_aggregate_university(chunk, metrics, freq=freq)
        muni = muni.reset_index()
        muni["Department_new"] = dept
        rows.append(muni)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def regularize_monthly_index(monthly: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reindex to full MS grid between min/max; expose gap info."""
    if monthly.empty:
        return monthly, {"n_gaps": 0}
    idx = monthly.index
    full = pd.date_range(idx.min(), idx.max(), freq="MS")
    aligned = monthly.reindex(full)
    gaps = int(aligned["thesis_count"].isna().sum()) if "thesis_count" in aligned.columns else 0
    info = {"n_months": len(full), "n_missing_cells": gaps}
    aligned.index.name = "period_dt"
    return aligned, info


def _month_dummies(index: pd.DatetimeIndex) -> pd.DataFrame:
    mo = index.month
    d = pd.get_dummies(mo, prefix="mo", drop_first=True, dtype=float)
    d.index = index
    return d


def _fourier_month(index: pd.DatetimeIndex, n_pairs: int = 2) -> pd.DataFrame:
    """Simple Fourier terms on month-of-year cycle (period 12)."""
    mo = index.month.values.astype(float)
    t = (mo - 1) / 12.0 * 2 * np.pi
    cols = {}
    for k in range(1, n_pairs + 1):
        cols[f"sin{k}"] = np.sin(k * t)
        cols[f"cos{k}"] = np.cos(k * t)
    out = pd.DataFrame(cols, index=index)
    return out


def intervention_indices(month_index: pd.DatetimeIndex, breaks: list[pd.Timestamp]) -> list[int]:
    """Integer position in ``month_index`` for each break (first month >= break)."""
    out: list[int] = []
    mi = pd.DatetimeIndex(month_index)
    for b in breaks:
        br = pd.Timestamp(b).normalize().to_period("M").to_timestamp()
        ix = int(mi.searchsorted(br, side="left"))
        ix = min(ix, len(mi) - 1)
        out.append(ix)
    return out


def build_its_exog(
    month_index: pd.DatetimeIndex,
    break_dates: Iterable[str | pd.Timestamp],
    *,
    seasonal: Literal["month_dummies", "fourier2"],
) -> pd.DataFrame:
    """
    Design: intercept + global time + for each break: post_k, time_post_k (segmented slope change).

    ``time_post_k`` = max(0, t - tau_k) * post_k with t = 0..T-1.
    """
    breaks = sorted({pd.Timestamp(d).normalize().to_period("M").to_timestamp() for d in break_dates})
    T = len(month_index)
    t = np.arange(T, dtype=float)
    Xlist: list[pd.Series] = [pd.Series(np.ones(T), index=month_index, name="const")]
    Xlist.append(pd.Series(t, index=month_index, name="time"))

    taus = intervention_indices(month_index, breaks)
    for k, br in enumerate(breaks):
        tau = taus[k]
        post = (t >= tau).astype(float)
        time_post = np.maximum(0.0, t - tau) * post
        Xlist.append(pd.Series(post, index=month_index, name=f"post_{k}"))
        Xlist.append(pd.Series(time_post, index=month_index, name=f"time_post_{k}"))

    X = pd.concat(Xlist, axis=1)
    if seasonal == "month_dummies":
        md = _month_dummies(month_index)
        md.index = month_index
        X = pd.concat([X, md], axis=1)
    else:
        fou = _fourier_month(month_index, n_pairs=2)
        X = pd.concat([X, fou], axis=1)

    return X.astype(float)


def _drop_na_rows(y: pd.Series, X: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    mask = y.notna() & X.notna().all(axis=1)
    return y.loc[mask], X.loc[mask]


def fit_its_ols(
    y: pd.Series,
    X: pd.DataFrame,
    *,
    cov_type: str = "nonrobust",
    cov_kwds: Mapping[str, Any] | None = None,
) -> Any:
    """OLS with optional robust/HAC covariance."""
    yy, XX = _drop_na_rows(y, X)
    if len(yy) < XX.shape[1] + 5:
        raise ValueError("Insufficient observations for ITS regression.")
    model = OLS(yy, XX)
    if cov_type == "nonrobust":
        return model.fit()
    return model.fit(cov_type=cov_type, cov_kwds=cov_kwds or {})


def fit_its_wls(
    y: pd.Series,
    X: pd.DataFrame,
    weights: pd.Series,
    *,
    cov_type: str = "nonrobust",
) -> Any:
    yy, XX = _drop_na_rows(y, X)
    ww = weights.reindex(yy.index)
    mask = ww.notna() & (ww > 0)
    yy, XX, ww = yy.loc[mask], XX.loc[mask], ww.loc[mask]
    model = WLS(yy, XX, weights=ww)
    return model.fit(cov_type=cov_type)


def fit_sarimax_or_none(
    y: pd.Series,
    X: pd.DataFrame,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> Any | None:
    """ARIMA/SARIMAX errors with exogenous ITS terms; None on failure."""
    if SARIMAX is None:
        return None
    yy, XX = _drop_na_rows(y, X)
    if len(yy) < XX.shape[1] + 10:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = SARIMAX(
                yy,
                exog=XX,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = mod.fit(disp=False, maxiter=100)
        return res
    except Exception:
        return None


def stationarity_report(series: pd.Series) -> dict[str, Any]:
    """ADF (constant+trend where needed) and KPSS (level)."""
    s = pd.to_numeric(series, errors="coerce").dropna().values
    rep: dict[str, Any] = {}
    if len(s) < 12:
        return {"note": "too_short"}
    try:
        rep["adf_stat"], rep["adf_p"], *_ = adfuller(s, autolag="AIC")
    except Exception as e:
        rep["adf_error"] = str(e)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")
        rep["kpss_stat"], rep["kpss_p"] = kpss_stat, kpss_p
    except Exception as e:
        rep["kpss_error"] = str(e)
    return rep


def residual_ljung_box(residuals: pd.Series, lags: int = 12) -> pd.DataFrame:
    clean = residuals.dropna()
    if len(clean) < lags + 2:
        return pd.DataFrame()
    try:
        return acorr_ljungbox(clean, lags=[lags], return_df=True)
    except Exception:
        return pd.DataFrame()


def save_acf_pacf_plot(series: pd.Series, path: Path, title: str) -> None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < 15:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_acf(clean, ax=axes[0], lags=min(24, len(clean) // 2))
    plot_pacf(clean, ax=axes[1], lags=min(24, len(clean) // 2))
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chow_break_f_stat(
    y: pd.Series,
    X: pd.DataFrame,
    split_index: int,
) -> tuple[float, float] | tuple[None, None]:
    """
    Chow-style F-test: pooled OLS vs two subsamples split at ``split_index`` (row position).
    """
    yy, XX = _drop_na_rows(y, X)
    if split_index <= 1 or split_index >= len(yy) - 2:
        return None, None
    r1 = OLS(yy.iloc[:split_index], XX.iloc[:split_index]).fit()
    r2 = OLS(yy.iloc[split_index:], XX.iloc[split_index:]).fit()
    pooled = OLS(yy, XX).fit()

    rss_p = pooled.ssr
    rss_u = r1.ssr + r2.ssr
    k = XX.shape[1]
    n1, n2 = split_index, len(yy) - split_index
    if rss_u <= 0:
        return None, None
    num = (rss_p - rss_u) / k
    den = rss_u / (n1 + n2 - 2 * k)
    if den <= 0:
        return None, None
    f_stat = num / den
    from scipy.stats import f as f_distribution

    p_val = float(1 - f_distribution.cdf(f_stat, k, n1 + n2 - 2 * k))
    return float(f_stat), p_val


def grid_changepoint_rss(y: pd.Series, X_base: pd.DataFrame) -> pd.DataFrame | None:
    """Exploratory: minimize RSS over single break position (simple, not PELT)."""
    yy, XXb = _drop_na_rows(y, X_base)
    rows = []
    for split in range(8, len(yy) - 8):
        rss1 = OLS(yy.iloc[:split], XXb.iloc[:split]).fit().ssr
        rss2 = OLS(yy.iloc[split:], XXb.iloc[split:]).fit().ssr
        rows.append({"split_ix": split, "split_dt": yy.index[split], "rss": rss1 + rss2})
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("rss")


def plot_metric_timeline(
    monthly: pd.DataFrame,
    metric: str,
    cfg: PipelineConfig,
    intervention_dates: list[pd.Timestamp],
    *,
    mean_or_median: Literal["mean", "median"] = "mean",
) -> None:
    col = f"{metric}_{mean_or_median}"
    if col not in monthly.columns:
        return
    y = monthly[col]
    roll = y.rolling(cfg.rolling_window, min_periods=1).mean()
    std = monthly.get(f"{metric}_std", pd.Series(index=y.index, dtype=float))

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(y.index, y.values, label=f"monthly {mean_or_median}", alpha=0.7)
    ax.plot(roll.index, roll.values, label=f"{cfg.rolling_window}-m rolling mean", color="tab:orange")
    if std is not None and std.notna().any():
        ax.fill_between(
            y.index,
            (y - std).values,
            (y + std).values,
            alpha=0.15,
            color="gray",
            label="±1 monthly std (dispersion)",
        )
    for d in intervention_dates:
        ax.axvline(pd.Timestamp(d), color="red", linestyle="--", alpha=0.8)
    ax.set_title(metric)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = cfg.output_dir / "plots" / f"ts_{metric}_{mean_or_median}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_thesis_counts(monthly: pd.DataFrame, cfg: PipelineConfig, intervention_dates: list[pd.Timestamp]) -> None:
    if "thesis_count" not in monthly.columns:
        return
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(monthly.index, monthly["thesis_count"].values, width=20, alpha=0.85)
    for d in intervention_dates:
        ax.axvline(pd.Timestamp(d), color="red", linestyle="--")
    ax.set_title("Thesis count by month")
    fig.tight_layout()
    p = cfg.output_dir / "plots" / "thesis_counts.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150)
    plt.close(fig)


def plot_department_composition(df: pd.DataFrame, cfg: PipelineConfig, top_n: int = 12) -> None:
    if "Department_new" not in df.columns:
        return
    ct = df.groupby([pd.Grouper(key="period_dt", freq="MS"), "Department_new"]).size().unstack(fill_value=0)
    # top departments by total volume
    totals = ct.sum(axis=0).sort_values(ascending=False)
    keep = list(totals.head(top_n).index)
    rest = [c for c in ct.columns if c not in keep]
    plot_df = ct[keep].copy()
    if rest:
        plot_df["Other"] = ct[rest].sum(axis=1)
    shares = plot_df.div(plot_df.sum(axis=1).replace(0, np.nan), axis=0)
    fig, ax = plt.subplots(figsize=(11, 5))
    shares.plot.area(ax=ax, linewidth=0)
    ax.set_title("Department composition (share of theses per month)")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=7)
    fig.tight_layout()
    p = cfg.output_dir / "plots" / "department_composition.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150)
    plt.close(fig)


def plot_variance_metric(df: pd.DataFrame, metric: str, cfg: PipelineConfig) -> None:
    """Dispersion of micro-level metric within each month (homogenization exploratory)."""
    if metric not in df.columns:
        return
    g = df.groupby(pd.Grouper(key="period_dt", freq="MS"), observed=True)[metric]
    iqr = g.quantile(0.75) - g.quantile(0.25)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(iqr.index, iqr.values, label="IQR")
    ax.set_title(f"Monthly IQR of {metric} (micro-level)")
    ax.legend()
    fig.tight_layout()
    p = cfg.output_dir / "plots" / f"variance_iqr_{metric}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)


def summarize_fit(
    metric: str,
    model_label: str,
    result: Any,
    *,
    primary_coef: str = "post_0",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "metric": metric,
        "model": model_label,
        "n_obs": int(getattr(result, "nobs", 0)),
        "rsquared": float(getattr(result, "rsquared", np.nan)),
    }
    pnames = list(result.params.index)
    for name in ("post_0", "time_post_0", "post_1", "time_post_1"):
        if name in pnames:
            row[f"coef_{name}"] = float(result.params[name])
            row[f"p_{name}"] = float(result.pvalues[name])
    if primary_coef in pnames:
        row["primary_coef"] = primary_coef
        tv = getattr(result, "tvalues", None)
        if tv is not None and primary_coef in tv.index:
            row["primary_abs_t"] = float(abs(tv[primary_coef]))
    return row


def filter_low_volume_departments(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Drop departments whose monthly thesis count rarely exceeds threshold."""
    if "Department_new" not in df.columns:
        return df
    gm = df.groupby(["Department_new", pd.Grouper(key="period_dt", freq="MS")]).size()
    ok = gm.groupby(level=0).max() >= cfg.min_department_monthly_count
    keep = set(ok[ok].index)
    return df[df["Department_new"].isin(keep)].reset_index(drop=True)


def placebo_run(
    monthly: pd.DataFrame,
    metric: str,
    placebo_date: str,
    cfg: PipelineConfig,
    *,
    mean_or_median: Literal["mean", "median"] = "mean",
) -> dict[str, Any]:
    col = f"{metric}_{mean_or_median}"
    if col not in monthly.columns:
        return {}
    y = monthly[col]
    breaks = [cfg.chatgpt_date, placebo_date]
    X = build_its_exog(pd.DatetimeIndex(monthly.index), breaks, seasonal=cfg.seasonal)
    try:
        fit = fit_its_ols(y, X, cov_type="HC1")
        return {
            "placebo_date": placebo_date,
            "p_post_chatgpt": float(fit.pvalues.get("post_0", np.nan)),
            "p_post_placebo_pair": float(fit.pvalues.get("post_1", np.nan)),
        }
    except Exception as e:
        return {"error": str(e)}


def pre_post_micro(
    df: pd.DataFrame,
    metric: str,
    cut: pd.Timestamp,
) -> dict[str, Any]:
    """Weak baseline: Mann–Whitney on micro rows before/after cut (non-ITS)."""
    if metric not in df.columns:
        return {}
    x = df.loc[df["period_dt"] < cut, metric].dropna()
    y = df.loc[df["period_dt"] >= cut, metric].dropna()
    if len(x) < 20 or len(y) < 20:
        return {"note": "insufficient micro sample"}
    try:
        stat, p = scipy_stats.mannwhitneyu(x, y, alternative="two-sided")
        return {"mannwhitney_u_stat": float(stat), "mannwhitney_p": float(p), "n_pre": len(x), "n_post": len(y)}
    except Exception as e:
        return {"error": str(e)}


def did_two_group_monthly(
    df: pd.DataFrame,
    metric: str,
    cfg: PipelineConfig,
    *,
    cut: pd.Timestamp | None = None,
) -> pd.DataFrame | None:
    """
    Simple DiD-style stacking: each month contributes two rows (early vs late group means).

    ``y ~ time + post + group + post:group`` with weights = group-month counts.
    Cluster-robust SE at month is not trivial in statsmodels without linearmodels; report HC1.
    """
    if cfg.early_adopter_depts is None or cfg.late_adopter_depts is None:
        return None
    cut = cut or pd.Timestamp(cfg.chatgpt_date)
    if "Department_new" not in df.columns:
        return None

    def _lab(x: str) -> int | None:
        if x in cfg.early_adopter_depts:
            return 0
        if x in cfg.late_adopter_depts:
            return 1
        return None

    df = df.copy()
    df["_g"] = df["Department_new"].map(_lab)
    df = df.dropna(subset=["_g"])
    if df.empty:
        return None

    rows = []
    for (dt, g), chunk in df.groupby([pd.Grouper(key="period_dt", freq="MS"), "_g"], observed=True):
        v = pd.to_numeric(chunk[metric], errors="coerce").dropna()
        if v.empty:
            continue
        rows.append(
            {
                "period_dt": dt,
                "group": int(g),
                "y": v.mean(),
                "w": len(v),
            }
        )
    long = pd.DataFrame(rows)
    if long.empty:
        return None

    long = long.sort_values(["period_dt", "group"])
    idx = pd.DatetimeIndex(long["period_dt"].unique()).sort_values()
    tmap = {d: i for i, d in enumerate(idx)}
    long["time"] = long["period_dt"].map(lambda d: tmap[pd.Timestamp(d)])
    long["post"] = (long["period_dt"] >= cut).astype(float)
    long["group"] = long["group"].astype(float)
    long["post_x_group"] = long["post"] * long["group"]

    y = long["y"].astype(float)
    X = sm.add_constant(long[["time", "post", "group", "post_x_group"]])
    try:
        res = WLS(y, X, weights=np.sqrt(long["w"].clip(lower=1))).fit(cov_type="HC1")
        return pd.DataFrame(
            [
                {
                    "metric": metric,
                    "coef_post_x_group": float(res.params["post_x_group"]),
                    "p_post_x_group": float(res.pvalues["post_x_group"]),
                    "n_rows": len(long),
                }
            ]
        )
    except Exception:
        return None


def forest_plot(summary_rows: list[dict[str, Any]], cfg: PipelineConfig, coef: str = "coef_post_0") -> None:
    sub = [r for r in summary_rows if coef in r and np.isfinite(r.get(coef, np.nan))]
    if not sub:
        return
    metrics = [r["metric"] for r in sub]
    est = np.array([r[coef] for r in sub])
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(sub))))
    ax.axvline(0, color="gray", lw=1)
    ax.barh(metrics, est)
    ax.set_title(f"Forest: {coef}")
    fig.tight_layout()
    p = cfg.output_dir / "plots" / f"forest_{coef}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)


def rank_metrics_by_evidence(summary_df: pd.DataFrame, *, primary_t_col: str = "primary_abs_t") -> pd.DataFrame:
    if summary_df.empty or primary_t_col not in summary_df.columns:
        return summary_df
    out = summary_df.sort_values(primary_t_col, ascending=False).copy()
    out["rank_evidence"] = np.arange(1, len(out) + 1)
    out.insert(0, "disclaimer", "Ranking is exploratory (|t| on primary intervention); not multiple-testing adjusted.")
    return out


def run_pipeline(df: pd.DataFrame, cfg: PipelineConfig | None = None) -> dict[str, Any]:
    """
    Execute full workflow: preprocess → aggregate → EDA plots → ITS → diagnostics → robustness → outputs.

    Returns a dict with keys ``qc``, ``monthly``, ``monthly_regular``, ``results_tables``,
    ``summary``, ``report_md``.
    """
    cfg = cfg or PipelineConfig()
    cfg.output_dir = Path(cfg.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = list(cfg.metrics)
    if cfg.include_normalized:
        metrics.extend([nm[0] for nm in NORMALIZED_METRICS])

    clean, qc = preprocess_thesis_frame(df, cfg)

    uni = monthly_aggregate_university(clean, metrics)
    uni_reg, gap_info = regularize_monthly_index(uni)

    dept_panel = monthly_aggregate_by_department(clean, metrics)
    if not dept_panel.empty:
        dept_panel.to_csv(cfg.output_dir / "monthly_by_department_long.csv", index=False)

    intervention_dates = sorted(
        {
            pd.Timestamp(cfg.chatgpt_date),
            pd.Timestamp(cfg.reasoning_date),
            *(pd.Timestamp(x) for x in cfg.extra_interventions),
        }
    )
    break_strings = [cfg.chatgpt_date, cfg.reasoning_date, *cfg.extra_interventions]

    # --- EDA plots ---
    if cfg.export_plots:
        for m in cfg.metrics:
            if f"{m}_mean" in uni.columns:
                plot_metric_timeline(uni, m, cfg, intervention_dates, mean_or_median="mean")
                plot_metric_timeline(uni, m, cfg, intervention_dates, mean_or_median="median")
        plot_thesis_counts(uni, cfg, intervention_dates)
        plot_department_composition(clean, cfg)
        for m in ("lexical_diversity", "avg_sentence_length"):
            if m in clean.columns:
                plot_variance_metric(clean, m, cfg)

    # --- ITS per metric (mean primary; median robustness rows) ---
    summary_rows: list[dict[str, Any]] = []
    all_tables: list[pd.DataFrame] = []

    X = build_its_exog(pd.DatetimeIndex(uni.index), break_strings, seasonal=cfg.seasonal)

    for m in metrics:
        col_mean = f"{m}_mean"
        if col_mean not in uni.columns:
            continue
        y = uni[col_mean]

        # HAC lags
        yy, XX = _drop_na_rows(y, X)
        n_hac = cfg.hac_maxlags or max(3, int(np.floor(len(yy) ** 0.25)))

        models: dict[str, Any] = {}
        try:
            models["ols"] = fit_its_ols(y, X, cov_type="nonrobust")
            models["hc1"] = fit_its_ols(y, X, cov_type="HC1")
            models["hac"] = fit_its_ols(y, X, cov_type="HAC", cov_kwds={"maxlags": n_hac})
            w = uni["thesis_count"].reindex(y.index).fillna(1.0).clip(lower=1.0)
            models["wls_sqrt_n"] = fit_its_wls(y, X, np.sqrt(w), cov_type="HC1")
            smx = fit_sarimax_or_none(y, X, cfg.sarimax_order, cfg.sarimax_seasonal_order)
            if smx is not None:
                models["sarimax"] = smx
        except Exception as e:
            _log(cfg, f"[ITS] metric {m} fit error: {e}")

        yy_fit, XX_fit = _drop_na_rows(y, X)
        for label, res in models.items():
            if hasattr(res, "params") and hasattr(res, "pvalues"):
                summ = pd.DataFrame(
                    {
                        "param": res.params.index,
                        "coef": res.params.values,
                        "std_err": res.bse.values,
                        "pvalue": res.pvalues.values,
                    }
                )
                summ.insert(0, "metric", m)
                summ.insert(1, "variant", label)
                all_tables.append(summ)
            try:
                summary_rows.append(summarize_fit(m, label, res))
            except Exception:
                pass

        # Diagnostics on OLS residuals (if present)
        if models.get("ols") is not None:
            resid = pd.Series(models["ols"].resid, index=yy_fit.index)
            lb = residual_ljung_box(resid)
            st = stationarity_report(y)
            diag_path = cfg.output_dir / "diagnostics" / m
            diag_path.mkdir(parents=True, exist_ok=True)
            save_acf_pacf_plot(resid, diag_path / "acf_pacf_resid.png", f"{m} OLS residuals")
            (lb if not lb.empty else pd.DataFrame({"note": ["ljung_box_failed"]})).to_csv(
                diag_path / "ljung_box.csv", index=False
            )
            with open(diag_path / "stationarity.json", "w", encoding="utf-8") as f:
                json.dump(st, f, indent=2)

            # Chow (first intervention only position)
            brk = pd.Timestamp(cfg.chatgpt_date).normalize().to_period("M").to_timestamp()
            try:
                mi = pd.DatetimeIndex(yy_fit.index)
                pos = int(mi.searchsorted(brk, side="left"))
                f_chow, p_chow = chow_break_f_stat(y, X, pos)
                pd.DataFrame([{"f_stat": f_chow, "p_value": p_chow, "split": str(mi[pos])}]).to_csv(
                    diag_path / "chow.csv", index=False
                )
            except Exception:
                pass

            # Changepoint grid (intercept+time only)
            tseries = np.arange(len(yy_fit), dtype=float)
            Xsimp = sm.add_constant(pd.Series(tseries, index=yy_fit.index, name="time"))
            gc = grid_changepoint_rss(y.loc[yy_fit.index], Xsimp)
            if gc is not None:
                gc.to_csv(diag_path / "changepoint_grid_rss.csv", index=False)

        # Mean vs median: HC1 on monthly median series (robustness label)
        col_med = f"{m}_median"
        if col_med in uni.columns:
            try:
                ym = uni[col_med]
                fit_med = fit_its_ols(ym, X, cov_type="HC1")
                summary_rows.append(summarize_fit(m, "hc1_median_monthly", fit_med))
            except Exception:
                pass

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(cfg.output_dir / "summary_coefficients.csv", index=False)

    hc1_summary = summary_df[summary_df["model"] == "hc1"].copy()
    ranked = rank_metrics_by_evidence(hc1_summary)
    ranked.to_csv(cfg.output_dir / "summary_ranked_exploratory.csv", index=False)

    if all_tables:
        pd.concat(all_tables, ignore_index=True).to_csv(cfg.output_dir / "results_all_coefficients_long.csv", index=False)

    # Forest plot: HC1 level-shift coefficients
    hc1_only = summary_df[summary_df["model"] == "hc1"].to_dict("records")
    forest_plot(hc1_only, cfg, coef="coef_post_0")

    # --- Robustness: low-volume departments ---
    robust_dir = cfg.output_dir / "robustness"
    robust_dir.mkdir(parents=True, exist_ok=True)
    clean_f = filter_low_volume_departments(clean, cfg)
    uni_f = monthly_aggregate_university(clean_f, metrics)
    if not uni_f.empty and len(clean_f) < len(clean):
        Xf = build_its_exog(pd.DatetimeIndex(uni_f.index), break_strings, seasonal=cfg.seasonal)
        rows_f: list[dict[str, Any]] = []
        for m in cfg.metrics:
            cm = f"{m}_mean"
            if cm not in uni_f.columns:
                continue
            try:
                fit = fit_its_ols(uni_f[cm], Xf, cov_type="HC1")
                rows_f.append(summarize_fit(m, "hc1_drop_sparse_depts", fit))
            except Exception:
                pass
        pd.DataFrame(rows_f).to_csv(robust_dir / "high_volume_departments_only.csv", index=False)

    # --- Placebo interventions ---
    placebo_rows: list[dict[str, Any]] = []
    for pl in cfg.placebo_month_starts:
        for m in cfg.metrics:
            if f"{m}_mean" not in uni.columns:
                continue
            pr = placebo_run(uni, m, pl, cfg, mean_or_median="mean")
            if pr:
                pr["metric"] = m
                placebo_rows.append(pr)
    pd.DataFrame(placebo_rows).to_csv(robust_dir / "placebo_interventions.csv", index=False)

    # --- Pre/post micro baseline ---
    pp_rows = []
    cut_cp = pd.Timestamp(cfg.chatgpt_date)
    for m in cfg.metrics:
        if m not in clean.columns:
            continue
        pp_rows.append({"metric": m, **pre_post_micro(clean, m, cut_cp)})
    pd.DataFrame(pp_rows).to_csv(robust_dir / "prepost_mannwhitney_micro.csv", index=False)

    # --- Sensitivity: reasoning date grid (second break) ---
    sens_rows: list[dict[str, Any]] = []
    for rd in cfg.reasoning_date_grid:
        br2 = [cfg.chatgpt_date, rd, *cfg.extra_interventions]
        Xs = build_its_exog(pd.DatetimeIndex(uni.index), br2, seasonal=cfg.seasonal)
        for m in cfg.metrics:
            cm = f"{m}_mean"
            if cm not in uni.columns:
                continue
            try:
                fit = fit_its_ols(uni[cm], Xs, cov_type="HC1")
                sens_rows.append(
                    {
                        "reasoning_date": rd,
                        "metric": m,
                        "coef_post_0": float(fit.params.get("post_0", np.nan)),
                        "p_post_0": float(fit.pvalues.get("post_0", np.nan)),
                        "coef_post_1": float(fit.params.get("post_1", np.nan)),
                        "p_post_1": float(fit.pvalues.get("post_1", np.nan)),
                    }
                )
            except Exception:
                continue
    pd.DataFrame(sens_rows).to_csv(robust_dir / "sensitivity_reasoning_date_grid.csv", index=False)

    # --- Optional DiD ---
    did_frames: list[pd.DataFrame] = []
    if cfg.early_adopter_depts and cfg.late_adopter_depts:
        for m in cfg.metrics:
            d = did_two_group_monthly(clean, m, cfg)
            if d is not None:
                did_frames.append(d)
    if did_frames:
        pd.concat(did_frames, ignore_index=True).to_csv(robust_dir / "did_two_group_stacked.csv", index=False)

    # --- Markdown report ---
    report_lines = [
        "# Interrupted time series — summary",
        "",
        "This is an **associational** structural-break analysis, not a causal audit of LLM effects.",
        "",
        f"- Preprocess: rows after QC = {qc.get('rows_final', '')}",
        f"- Period strategy: {qc.get('period_strategy', {})}",
        f"- Monthly index: {gap_info}",
        "",
        "## Notes",
        "- Strongest statistical evidence is labeled only in exploratory ranked table; multiple testing applies.",
        "- Grading scores may reflect changes in the grading pipeline, not only writing quality.",
        "",
    ]
    report_md = "\n".join(report_lines)
    (cfg.output_dir / "REPORT.md").write_text(report_md, encoding="utf-8")

    return {
        "qc": qc,
        "monthly": uni,
        "monthly_regular": uni_reg,
        "monthly_gap_info": gap_info,
        "department_panel": dept_panel,
        "summary": summary_df,
        "report_md": report_md,
        "output_dir": str(cfg.output_dir),
    }


def _configure_matplotlib_backend() -> None:
    import matplotlib

    matplotlib.use("Agg")


if __name__ == "__main__":
    import argparse

    _configure_matplotlib_backend()
    p = argparse.ArgumentParser(description="ITS pipeline for thesis metrics")
    p.add_argument(
        "--parquet",
        type=str,
        default="",
        help="Path to df_filtered_final parquet (optional; uses synthetic data if missing)",
    )
    args = p.parse_args()
    cfg = PipelineConfig()
    path = Path(args.parquet) if args.parquet else None
    if path and path.is_file():
        df = pd.read_parquet(path)
        _log(cfg, f"Loaded {path} shape={df.shape}")
    else:
        # Minimal synthetic sanity check without project data
        rng = np.random.default_rng(0)
        rows = []
        for i in range(240):
            ts = pd.Timestamp("2015-01-01") + pd.DateOffset(months=i)
            rows.append(
                {
                    "Timestamp": ts.isoformat(),
                    "Publication Year": ts.year,
                    "handin_month_num": ts.month,
                    "ID": i,
                    "Department_new": "DeptA" if i % 2 == 0 else "DeptB",
                    "lexical_diversity": 0.4 + 0.001 * i + rng.normal(0, 0.02),
                    "avg_sentence_length": 15.0 + rng.normal(0, 0.5),
                    "avg_word_length": 5.0 + rng.normal(0, 0.1),
                    "total_words": 8000 + rng.integers(-200, 200),
                    "unique_words": 2000 + rng.integers(-50, 50),
                    "grading_total_score": 70 + rng.normal(0, 5),
                    "num_references": 40 + rng.integers(-5, 5),
                    "equation_count": 10 + rng.integers(0, 3),
                    "num_figures": 12 + rng.integers(0, 3),
                    "num_tables": 5 + rng.integers(0, 2),
                    "num_tot_pages": 80 + rng.integers(-5, 5),
                }
            )
        df = pd.DataFrame(rows)
        _log(cfg, "Using synthetic monthly data (no --parquet).")

    out = run_pipeline(df, cfg)
    print("Done. Output:", out["output_dir"])