"""Matplotlib / seaborn figures for stage-2 analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgba
from matplotlib.patches import Ellipse


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


def _soft_cluster_clouds_from_covariance(
    ax,
    pts: np.ndarray,
    fc_rgb: tuple[float, float, float],
    *,
    ring_scales: tuple[float, ...] = (2.8, 2.0, 1.35),
    ring_alphas: tuple[float, ...] = (0.035, 0.065, 0.11),
    singleton_radius_frac: float = 0.035,
    xy_span: float = 1.0,
) -> None:
    """
    Draw soft-edged layers as covariance ellipses centered on the cluster mean
    (principal axes from ``np.cov``). Outer rings first, no strokes — reads as a diffuse cloud.
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 1:
        return
    mu = pts.mean(axis=0)
    if n == 1:
        r = max(xy_span * singleton_radius_frac, 1e-9)
        for scale, a in zip(ring_scales, ring_alphas, strict=True):
            d = 2.0 * r * scale
            e = Ellipse(
                xy=mu,
                width=d,
                height=d,
                angle=0.0,
                facecolor=to_rgba(fc_rgb, a),
                edgecolor=(0.0, 0.0, 0.0, 0.0),
                linewidth=0,
                zorder=1,
            )
            ax.add_patch(e)
        return

    if n == 2:
        cov = np.cov(pts.T)
        if not np.ndim(cov):
            cov = np.eye(2) * float(cov)
        else:
            cov = np.atleast_2d(cov)
        cov = cov + np.eye(2) * (np.trace(cov) * 1e-6 + 1e-8)
    else:
        cov = np.cov(pts.T)
        cov = np.atleast_2d(cov)
        cov = cov + np.eye(2) * 1e-9

    if cov.shape != (2, 2):
        return

    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, np.maximum(vals.max() * 1e-10, 1e-16))
    # eigh ascending: minor = vals[0], major = vals[1]
    lambda_major = float(vals[1])
    lambda_minor = float(vals[0])
    v_major = vecs[:, 1]
    angle_deg = float(np.degrees(np.arctan2(v_major[1], v_major[0])))

    # Draw largest scale first (soft halo), then tighter cores.
    for scale, a in zip(ring_scales, ring_alphas, strict=True):
        width = 2.0 * scale * np.sqrt(lambda_major)
        height = 2.0 * scale * np.sqrt(lambda_minor)
        e = Ellipse(
            xy=mu,
            width=width,
            height=height,
            angle=angle_deg,
            facecolor=to_rgba(fc_rgb, a),
            edgecolor=(0.0, 0.0, 0.0, 0.0),
            linewidth=0,
            zorder=1,
        )
        ax.add_patch(e)


def scatter_2d_department_with_cluster_blobs(
    xy: np.ndarray,
    department: pd.Series,
    cluster_labels: np.ndarray | pd.Series,
    title: str,
    out: Path,
    *,
    dpi: int = 150,
    s: float = 14.0,
    point_alpha: float = 0.78,
    cloud_alpha_scale: float = 1.0,
) -> None:
    """
    Points colored by department; behind them, soft clouds per cluster from the cluster
    mean and 2×2 covariance (Gaussian ellipses in several transparent layers — no crisp hull).
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    xy = np.asarray(xy, dtype=np.float64)
    dept = pd.Series(department).fillna("unknown").astype(str)
    cl = np.asarray(cluster_labels).astype(int)

    span = float(
        max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 1e-6)
    )
    fig, ax = plt.subplots(figsize=(11, 8.5))
    cmap_bg = plt.get_cmap("tab20")

    ring_scales = (2.8, 2.0, 1.35)
    base_alphas = (0.035, 0.065, 0.11)
    ring_alphas = tuple(a * cloud_alpha_scale for a in base_alphas)

    for cid in sorted(np.unique(cl)):
        mask = cl == cid
        pts = xy[mask]
        if len(pts) < 1:
            continue
        if int(cid) < 0:
            fc_base = (0.62, 0.62, 0.62)
            # quieter noise clouds
            ra = tuple(a * 0.55 for a in ring_alphas)
        else:
            rgba = cmap_bg((int(cid) % 20) / 20.0)
            fc_base = tuple(float(x) for x in rgba[:3])
            ra = ring_alphas
        _soft_cluster_clouds_from_covariance(
            ax,
            pts,
            fc_base,
            ring_scales=ring_scales,
            ring_alphas=ra,
            xy_span=span,
        )

    codes, uniques = pd.factorize(dept, sort=True)
    n_u = len(uniques)
    palette = sns.color_palette("husl", n_colors=max(n_u, 1))
    colors = [palette[int(c) % len(palette)] for c in codes]
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=colors,
        s=s,
        alpha=point_alpha,
        linewidths=0.15,
        edgecolors="white",
        zorder=3,
    )
    ax.set_title(title + " (points = department, soft clouds = cluster)")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    if n_u <= 32:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=palette[i % len(palette)],
                label=str(uniques[i])[:48],
                markersize=6,
            )
            for i in range(n_u)
        ]
        ax.legend(
            handles=handles,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=6.5,
            frameon=True,
            title="Department",
        )
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
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
