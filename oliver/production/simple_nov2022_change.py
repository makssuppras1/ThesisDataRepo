"""
Simple exploratory check: did monthly thesis metrics shift after Nov 2022?

This is not causal inference — only descriptive pre/post contrasts and a linear
time + post dummy model on monthly aggregates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# ---------------------------------------------------------------------------
# Config (edit here)
# ---------------------------------------------------------------------------

POST_DATE = pd.Timestamp("2022-11-01")
# Linear extrapolation of pre-cutoff drift; forecast segment starts at months >= cutoff.
FORECAST_CUTOFFS: tuple[pd.Timestamp, ...] = (
    pd.Timestamp("2022-11-01"),
    pd.Timestamp("2022-12-05"),
)
METRICS: tuple[str, ...] = (
    "lexical_diversity",
    "avg_sentence_length",
    "avg_word_length",
    "total_words",
    "grading_total_score",
)
ROLL_SHORT = 3
ROLL_LONG = 6
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "exported_plots" / "simple_nov2022"


def _month_from_handin_label(series: pd.Series) -> pd.Series:
    """
    Calendar month (1–12) from ``handin_month`` text (e.g. ``January 2025``).

    Only the **month name** matters; the year in that string is not used for dating.
    """
    s = series.astype(str).str.strip()
    m = pd.to_datetime(s, format="%B %Y", errors="coerce").dt.month
    first_token = s.str.split().str[0]
    m = m.fillna(pd.to_datetime(first_token, format="%B", errors="coerce").dt.month)
    return m.astype(float)


def _period_dt(df: pd.DataFrame) -> pd.Series:
    """
    Month-start hand-in period.

    **Year:** ``Publication Year`` only.

    **Month:** ``handin_month`` (month name from PDF extraction). If missing or
    unparsable, fall back to ``handin_month_num``.

    ``Timestamp`` is **last resort** only (library scrape/metadata; can batch badly).

    """
    py = pd.to_numeric(df.get("Publication Year"), errors="coerce")

    hm_num = pd.to_numeric(df.get("handin_month_num"), errors="coerce")
    month = pd.Series(np.nan, index=df.index, dtype=float)
    if "handin_month" in df.columns:
        month = _month_from_handin_label(df["handin_month"])
    month = month.where(month.notna(), hm_num)

    canonical = pd.to_datetime({"year": py, "month": month, "day": np.ones(len(df))}, errors="coerce")

    ts_ms = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    if "Timestamp" in df.columns:
        ts = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)
        if hasattr(ts.dt, "tz") and ts.dt.tz is not None:
            ts = ts.dt.tz_convert(None)
        ts_ms = ts.dt.to_period("M").dt.to_timestamp(how="start")

    merged = canonical.dt.to_period("M").dt.to_timestamp(how="start")
    merged = merged.fillna(ts_ms)
    return merged


def prepare(df: pd.DataFrame, *, year_max: int | None = 2025) -> pd.DataFrame:
    out = df.copy()
    out["period_dt"] = _period_dt(out)
    out = out.dropna(subset=["period_dt"])
    if year_max is not None:
        out = out[out["period_dt"].dt.year <= year_max]
    if "ID" in out.columns:
        out = out.sort_values(["period_dt", "ID"]).drop_duplicates(subset=["ID"], keep="last")
    for m in METRICS:
        if m in out.columns:
            out[m] = pd.to_numeric(out[m], errors="coerce")
    return out.reset_index(drop=True)


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(pd.Grouper(key="period_dt", freq="MS"), observed=True)
    rows = {"n_theses": g.size()}
    for m in METRICS:
        if m not in df.columns:
            continue
        rows[f"{m}_mean"] = g[m].mean()
        rows[f"{m}_median"] = g[m].median()
        rows[f"{m}_std"] = g[m].std()
    monthly = pd.DataFrame(rows).sort_index()
    monthly.index.name = "period_dt"
    monthly["Post"] = (monthly.index >= POST_DATE).astype(float)
    monthly["time"] = np.arange(len(monthly), dtype=float)
    return monthly


def compare_pre_post(monthly: pd.DataFrame, col: str) -> dict[str, Any]:
    """Pre/post on monthly observations (means series and within-month std series)."""
    pre = monthly.index < POST_DATE
    post = monthly.index >= POST_DATE
    y_mean = monthly[f"{col}_mean"]
    y_wstd = monthly[f"{col}_std"]
    pre_m, post_m = y_mean.loc[pre].dropna(), y_mean.loc[post].dropna()
    pre_s, post_s = y_wstd.loc[pre].dropna(), y_wstd.loc[post].dropna()

    out: dict[str, Any] = {
        "mean_change": float(post_m.mean() - pre_m.mean()) if len(pre_m) and len(post_m) else np.nan,
        "median_change": float(post_m.median() - pre_m.median()) if len(pre_m) and len(post_m) else np.nan,
        # Homogenization: average within-month thesis std, by period
        "within_month_std_change": float(post_s.mean() - pre_s.mean())
        if len(pre_s) and len(post_s)
        else np.nan,
    }
    if len(pre_m) > 2 and len(post_m) > 2:
        t_stat, t_p = stats.ttest_ind(pre_m, post_m, equal_var=False)
        u_stat, mw_p = stats.mannwhitneyu(pre_m, post_m, alternative="two-sided")
        out["ttest_p"] = float(t_p)
        out["mannwhitney_p"] = float(mw_p)
    else:
        out["ttest_p"] = np.nan
        out["mannwhitney_p"] = np.nan
    return out


def fit_simple_regression(monthly: pd.DataFrame, col: str) -> dict[str, Any]:
    """y_t = alpha + beta1*time + beta2*Post + eps; y = monthly mean."""
    y = monthly[f"{col}_mean"].astype(float)
    X = monthly[["time", "Post"]].astype(float)
    X = sm.add_constant(X)
    mask = y.notna() & X.notna().all(axis=1)
    y, X = y.loc[mask], X.loc[mask]
    if len(y) < 10:
        return {"beta2_Post": np.nan, "p_Post": np.nan, "rsquared": np.nan}
    res = sm.OLS(y, X).fit()
    return {
        "beta2_Post": float(res.params["Post"]),
        "p_Post": float(res.pvalues["Post"]),
        "rsquared": float(res.rsquared),
        "beta1_time": float(res.params["time"]),
    }


def _cutoff_filename_tag(cutoff: pd.Timestamp) -> str:
    return cutoff.strftime("%Y-%m-%d")


def linear_trend_forecast_series(
    monthly: pd.DataFrame,
    col: str,
    cutoff: pd.Timestamp,
) -> tuple[pd.Series, dict[str, Any]]:
    """
    OLS ``y ~ 1 + time`` on monthly means **strictly before** ``cutoff``.

    Returns predicted monthly mean for **all** rows (same ``time`` index as ``monthly``),
    plus metadata. NaN predicted where training sample too small.
    """
    y = monthly[f"{col}_mean"].astype(float)
    t = monthly["time"].astype(float)
    train = (monthly.index < cutoff) & y.notna()
    if int(train.sum()) < 4:
        return pd.Series(np.nan, index=monthly.index), {"error": "too_few_train_months"}

    yy, tt = y.loc[train], t.loc[train]
    X = sm.add_constant(tt)
    res = sm.OLS(yy, X).fit()
    X_all = sm.add_constant(t)
    pred = pd.Series(res.predict(X_all), index=monthly.index)
    meta = {
        "alpha": float(res.params.iloc[0]),
        "beta_time": float(res.params.iloc[1]),
        "train_r_squared": float(res.rsquared),
        "n_train_months": int(train.sum()),
    }
    return pred, meta


def plot_metric_with_forecast(
    monthly: pd.DataFrame,
    col: str,
    cutoff: pd.Timestamp,
    out_dir: Path,
    *,
    roll1: int = ROLL_SHORT,
    roll2: int = ROLL_LONG,
) -> None:
    """Monthly series + rollings + vertical lines + dashed linear forecast from ``cutoff`` onward."""
    mmean = monthly[f"{col}_mean"]
    pred, meta = linear_trend_forecast_series(monthly, col, cutoff)
    tag = _cutoff_filename_tag(cutoff)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(mmean.index, mmean.values, label="Monthly mean", alpha=0.85, color="C0")
    ax.plot(
        mmean.index,
        mmean.rolling(roll1, min_periods=1).mean(),
        label=f"{roll1}-m rolling mean",
        alpha=0.8,
    )
    ax.plot(
        mmean.index,
        mmean.rolling(roll2, min_periods=1).mean(),
        label=f"{roll2}-m rolling mean",
        alpha=0.8,
    )
    ax.axvline(POST_DATE, color="red", linestyle="--", alpha=0.7, label="Post marker 2022-11-01")
    ax.axvline(cutoff, color="purple", linestyle="-", linewidth=1.5, alpha=0.85, label=f"Forecast cutoff {tag}")

    fc_mask = monthly.index >= cutoff
    if fc_mask.any() and pred.notna().any():
        ax.plot(
            monthly.index[fc_mask],
            pred.loc[fc_mask].values,
            linestyle="--",
            color="purple",
            linewidth=2,
            label=f"Forecast (pre-{tag} linear trend)",
        )
        # bridge last pre point to start of forecast for visibility
        pre_mask = monthly.index < cutoff
        if pre_mask.any():
            last_pre = monthly.loc[pre_mask].index.max()
            first_fc = monthly.loc[fc_mask].index.min()
            ax.plot(
                [last_pre, first_fc],
                [mmean.loc[last_pre], pred.loc[first_fc]],
                linestyle=":",
                color="purple",
                alpha=0.6,
            )

    ax.set_title(f"{col} — forecast from {tag} (train: months < cutoff)")
    if "error" not in meta:
        ax.text(
            0.02,
            0.98,
            f"Train R²={meta['train_r_squared']:.3f}, n={meta['n_train_months']}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"metric_{col}_forecast_from_{tag}.png", dpi=140)
    plt.close(fig)


def plot_metric(
    monthly: pd.DataFrame,
    col: str,
    out_dir: Path,
    *,
    roll1: int = ROLL_SHORT,
    roll2: int = ROLL_LONG,
) -> None:
    mmean = monthly[f"{col}_mean"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(mmean.index, mmean.values, label="Monthly mean", alpha=0.8)
    ax.plot(
        mmean.index,
        mmean.rolling(roll1, min_periods=1).mean(),
        label=f"{roll1}-month rolling mean",
    )
    ax.plot(
        mmean.index,
        mmean.rolling(roll2, min_periods=1).mean(),
        label=f"{roll2}-month rolling mean",
    )
    ax.axvline(POST_DATE, color="red", linestyle="--", label="2022-11-01")
    ax.set_title(col)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"metric_{col}.png", dpi=140)
    plt.close(fig)


def plot_thesis_count(monthly: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(monthly.index, monthly["n_theses"].values, width=20, alpha=0.85)
    ax.axvline(POST_DATE, color="red", linestyle="--")
    ax.set_title("Theses per month")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "thesis_count_per_month.png", dpi=140)
    plt.close(fig)


def plot_std_over_time(monthly: pd.DataFrame, col: str, out_dir: Path) -> None:
    """Within-month std (spread of theses); lower may indicate homogenization."""
    s = monthly[f"{col}_std"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(s.index, s.values, marker=".", ms=4)
    ax.axvline(POST_DATE, color="red", linestyle="--")
    ax.set_title(f"Within-month std — {col}")
    ax.set_ylabel("std (theses in month)")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"within_month_std_{col}.png", dpi=140)
    plt.close(fig)


def run(
    df: pd.DataFrame,
    *,
    output_dir: Path | None = None,
    year_max: int | None = 2025,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build monthly panel, plots, and summary table.

    Returns ``(monthly_panel, summary_table)``.
    """
    out_dir = Path(output_dir or DEFAULT_OUTPUT)
    out_dir.mkdir(parents=True, exist_ok=True)

    prep = prepare(df, year_max=year_max)
    monthly = aggregate_monthly(prep)

    summary_rows = []
    for col in METRICS:
        if f"{col}_mean" not in monthly.columns:
            continue
        cmp = compare_pre_post(monthly, col)
        reg = fit_simple_regression(monthly, col)
        summary_rows.append(
            {
                "metric": col,
                "mean_change_monthly": cmp["mean_change"],
                "median_change_monthly": cmp["median_change"],
                "within_month_std_change": cmp["within_month_std_change"],
                "beta2_Post": reg["beta2_Post"],
                "p_value_Post": reg["p_Post"],
                "ttest_ind_monthly_means_p": cmp["ttest_p"],
                "mannwhitney_monthly_means_p": cmp["mannwhitney_p"],
                "r_squared": reg["rsquared"],
            }
        )
        plot_metric(monthly, col, out_dir / "plots")
        plot_std_over_time(monthly, col, out_dir / "plots")
        for fc_cut in FORECAST_CUTOFFS:
            plot_metric_with_forecast(monthly, col, fc_cut, out_dir / "plots")

    plot_thesis_count(monthly, out_dir / "plots")

    fc_rows: list[dict[str, Any]] = []
    for fc_cut in FORECAST_CUTOFFS:
        for col in METRICS:
            if f"{col}_mean" not in monthly.columns:
                continue
            _, meta = linear_trend_forecast_series(monthly, col, fc_cut)
            fc_rows.append({"cutoff": _cutoff_filename_tag(fc_cut), "metric": col, **meta})
    pd.DataFrame(fc_rows).to_csv(out_dir / "forecast_linear_trend_coefficients.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary_simple_nov2022.csv", index=False)
    monthly.to_csv(out_dir / "monthly_aggregate.csv")

    return monthly, summary


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--parquet", type=str, required=True)
    p.add_argument("--out", type=str, default=str(DEFAULT_OUTPUT))
    args = p.parse_args()
    d = pd.read_parquet(args.parquet)
    m, s = run(d, output_dir=Path(args.out))
    print(s.to_string())
    print("Monthly rows:", len(m))
