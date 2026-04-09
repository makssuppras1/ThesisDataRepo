"""Matplotlib / seaborn figures for stage-2 analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def scatter_2d(
    xy: np.ndarray,
    hue: pd.Series | np.ndarray,
    title: str,
    out: Path,
    *,
    dpi: int = 150,
    s: float = 12.0,
    alpha: float = 0.7,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    u = pd.factorize(pd.Series(hue).astype(str))[0]
    ax.scatter(xy[:, 0], xy[:, 1], c=u, cmap="tab20", s=s, alpha=alpha, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def cluster_bar(counts: pd.Series, out: Path, *, dpi: int = 150) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    counts.sort_index().plot(kind="bar", ax=ax)
    ax.set_title("Documents per cluster")
    ax.set_xlabel("cluster_id")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def publisher_topic_heatmap(ct: pd.DataFrame, out: Path, *, dpi: int = 150) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(10, ct.shape[1] * 0.4), max(6, ct.shape[0] * 0.25)))
    sns.heatmap(ct, ax=ax, cmap="viridis", annot=False)
    ax.set_title("Publisher × cluster (counts)")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
