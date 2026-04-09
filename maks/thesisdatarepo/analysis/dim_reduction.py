"""UMAP, t-SNE, PCA."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)


def run_umap_2d(
    emb: np.ndarray,
    *,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
):
    import umap

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(emb)


def run_umap_10d(
    emb: np.ndarray,
    *,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
):
    import umap

    reducer = umap.UMAP(
        n_components=10,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(emb)


def run_tsne_2d(
    emb: np.ndarray,
    perplexity: float,
    random_state: int,
) -> np.ndarray:
    ts = TSNE(
        n_components=2,
        perplexity=min(perplexity, emb.shape[0] - 1),
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    return ts.fit_transform(emb)


def run_pca_2d(emb: np.ndarray, random_state: int) -> np.ndarray:
    pca = PCA(n_components=2, random_state=random_state)
    return pca.fit_transform(emb)


def save_coords(path: Path, xy: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, xy)
