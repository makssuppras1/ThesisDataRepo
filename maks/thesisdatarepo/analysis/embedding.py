"""Sentence-transformers embeddings with optional chunked weighted pooling."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

from thesisdatarepo.analysis.config_loader import AnalysisConfig

logger = logging.getLogger(__name__)


def model_slug(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name)


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    t = text.strip()
    if not t:
        return []
    step = max(1, size - overlap)
    chunks: list[str] = []
    i = 0
    while i < len(t):
        chunks.append(t[i : i + size])
        i += step
    return chunks


def chunk_weights(n: int, first: float, last: float, middle: float) -> np.ndarray:
    if n <= 0:
        return np.array([])
    if n == 1:
        return np.array([first])
    w = np.full(n, middle, dtype=np.float64)
    w[0] = first
    w[-1] = last
    return w


def embed_texts_chunked(
    encode,
    texts: list[str],
    cfg: AnalysisConfig,
    title_prefixes: list[str] | None,
) -> np.ndarray:
    """Weighted mean of chunk embeddings; optional title prepended to each chunk."""
    dim = None
    out_rows: list[np.ndarray] = []

    for idx, text in enumerate(texts):
        prefix = ""
        if title_prefixes and title_prefixes[idx].strip():
            prefix = title_prefixes[idx].strip() + "\n\n"
        body = prefix + text
        chunks = chunk_text(body, cfg.chunk_size, cfg.chunk_overlap)
        if not chunks:
            # fallback zero vector — caller should filter empties
            if dim is None:
                dim = int(encode(["x"]).shape[1])
            out_rows.append(np.zeros(dim, dtype=np.float32))
            continue
        ch_emb = encode(chunks)
        if isinstance(ch_emb, list):
            ch_emb = np.asarray(ch_emb, dtype=np.float32)
        w = chunk_weights(
            len(chunks),
            cfg.first_chunk_weight,
            cfg.last_chunk_weight,
            cfg.middle_chunk_weight,
        )
        pooled = np.average(ch_emb, axis=0, weights=w)
        if dim is None:
            dim = pooled.shape[0]
        out_rows.append(pooled.astype(np.float32))

    mat = np.vstack(out_rows)
    return normalize(mat, norm="l2", axis=1)


def embed_texts_plain(encode, texts: list[str]) -> np.ndarray:
    emb = encode(texts)
    if isinstance(emb, list):
        emb = np.asarray(emb, dtype=np.float32)
    return normalize(emb, norm="l2", axis=1)


def compute_embeddings(
    cfg: AnalysisConfig,
    texts: list[str],
    df,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(cfg.embedding_model)
    encode = model.encode

    title_prefixes = None
    if (
        cfg.embedding_chunked
        and cfg.embedding_source == "corpus"
        and cfg.columns_title
        and cfg.columns_title in df.columns
    ):
        title_prefixes = df[cfg.columns_title].fillna("").astype(str).tolist()

    if cfg.embedding_chunked and cfg.embedding_source == "corpus":
        logger.info("Embedding %s documents (chunked pooled)", len(texts))
        emb = embed_texts_chunked(encode, texts, cfg, title_prefixes)
    else:
        logger.info("Embedding %s documents (one vector each)", len(texts))
        if title_prefixes:
            texts = [
                (f"{t}\n\n{x}" if str(t).strip() else x)
                for t, x in zip(title_prefixes, texts, strict=True)
            ]
        emb = embed_texts_plain(encode, texts)

    return emb.astype(np.float32)


def cache_path(cfg: AnalysisConfig, kind: str) -> Path:
    slug = model_slug(cfg.embedding_model)
    name = f"{kind}_embeddings_{slug}.npy"
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return cfg.output_dir / name


def save_embeddings(path: Path, emb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)


def load_embeddings(path: Path) -> np.ndarray:
    return np.load(path)
