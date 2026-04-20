"""Topic prevalence by year: shares, slopes, emerging/declining plot."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thesisdatarepo.analysis.config_loader import AnalysisConfig

logger = logging.getLogger(__name__)


def run_evolution(cfg: AnalysisConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    labels_path = cfg.evolution_labels_csv or (cfg.output_dir / "clustering_labels.csv")
    if not labels_path.is_file():
        raise FileNotFoundError(f"Labels CSV not found: {labels_path}")

    df = pd.read_csv(labels_path)
    if "cluster_id" not in df.columns or "year" not in df.columns:
        raise KeyError("clustering_labels.csv must include cluster_id and year")

    df = df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    if cfg.evolution_min_year is not None:
        df = df[df["year"] >= cfg.evolution_min_year]

    df = df[df["cluster_id"] != -1]

    if df.empty:
        logger.warning("No rows left for evolution analysis")
        return

    counts = df.groupby(["year", "cluster_id"]).size().unstack(fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0)
    shares.to_csv(cfg.output_dir / "cluster_share_by_year.csv")

    years = shares.index.values.astype(float)
    slopes = []
    for col in shares.columns:
        y = shares[col].values.astype(float)
        if len(years) >= 2 and np.nanstd(y) > 0:
            m, b = np.polyfit(years, y, 1)
        else:
            m, b = 0.0, 0.0
        slopes.append({"cluster_id": col, "slope": m, "intercept": b})
    slope_df = pd.DataFrame(slopes).sort_values("slope", ascending=False)
    slope_df.to_csv(cfg.output_dir / "cluster_temporal_slopes.csv", index=False)

    fig_dir = cfg.output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    for col in shares.columns:
        ax.plot(shares.index, shares[col], marker="o", ms=3, label=f"cluster {col}")
    ax.set_xlabel("year")
    ax.set_ylabel("share within year")
    ax.set_title("Cluster prevalence over time (share within year)")
    n_leg = len(shares.columns)
    leg_ncol = min(5, max(1, (n_leg + 11) // 12))
    leg_fs = 6 if n_leg > 24 else 7 if n_leg > 12 else "small"
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=leg_fs,
        ncol=leg_ncol,
        frameon=True,
        title="cluster_id",
    )
    fig.tight_layout()
    out_png = fig_dir / "cluster_emerging_declining.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_png)
