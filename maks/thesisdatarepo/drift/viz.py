"""Publication-style drift figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_jsd_drift(jsd_df: pd.DataFrame, out_path: Path, *, cutoff_year: int = 2022) -> None:
    """Line plot JSD vs year with vertical line at ChatGPT-era cutoff."""
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.plot(jsd_df["year"], jsd_df["jsd"], marker="o", color="#0C447C", linewidth=1.5)
    ax.axvline(cutoff_year, color="gray", linestyle="--", linewidth=1, label="ChatGPT (Nov 2022)")
    ax.set_xlabel("Publication Year")
    ax.set_ylabel("JSD vs pre-2019 baseline")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
