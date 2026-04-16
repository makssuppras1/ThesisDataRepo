"""Sentence-transformers embeddings with optional chunked pooling (token or char windows)."""

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


def chunk_text_by_tokens(
    tokenizer,
    text: str,
    max_tokens: int,
    overlap_ratio: float,
) -> list[str]:
    """Split on model tokenizer token ids; overlap is a fraction of ``max_tokens`` (e.g. 0.1–0.2)."""
    t = text.strip()
    if not t:
        return []
    ids = tokenizer.encode(t, add_special_tokens=False)
    if not ids:
        return []
    max_tokens = max(16, int(max_tokens))
    overlap_tok = int(round(max_tokens * float(overlap_ratio)))
    overlap_tok = max(0, min(overlap_tok, max_tokens - 1))
    step = max(1, max_tokens - overlap_tok)
    chunks: list[str] = []
    i = 0
    while i < len(ids):
        window = ids[i : i + max_tokens]
        chunk = tokenizer.decode(window, skip_special_tokens=True).strip()
        if chunk:
            chunks.append(chunk)
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


def _pool_chunks(
    ch_emb: np.ndarray,
    cfg: AnalysisConfig,
) -> np.ndarray:
    if cfg.chunk_pooling == "weighted":
        w = chunk_weights(
            ch_emb.shape[0],
            cfg.first_chunk_weight,
            cfg.last_chunk_weight,
            cfg.middle_chunk_weight,
        )
        return np.average(ch_emb, axis=0, weights=w)
    return np.mean(ch_emb, axis=0)


def embed_texts_chunked_chars(
    encode,
    texts: list[str],
    cfg: AnalysisConfig,
    title_prefixes: list[str] | None,
) -> np.ndarray:
    """Character-window chunks; pool by ``chunk_pooling`` (mean or weighted)."""
    dim = None
    out_rows: list[np.ndarray] = []

    for idx, text in enumerate(texts):
        prefix = ""
        if title_prefixes and title_prefixes[idx].strip():
            prefix = title_prefixes[idx].strip() + "\n\n"
        body = prefix + text
        chunks = chunk_text(body, cfg.chunk_size, cfg.chunk_overlap)
        if not chunks:
            if dim is None:
                dim = int(encode(["x"]).shape[1])
            out_rows.append(np.zeros(dim, dtype=np.float32))
            continue
        ch_emb = encode(chunks)
        if isinstance(ch_emb, list):
            ch_emb = np.asarray(ch_emb, dtype=np.float32)
        pooled = _pool_chunks(ch_emb, cfg)
        if dim is None:
            dim = pooled.shape[0]
        out_rows.append(pooled.astype(np.float32))

    mat = np.vstack(out_rows)
    return normalize(mat, norm="l2", axis=1)


def embed_texts_chunked_tokens(
    model,  # SentenceTransformer
    texts: list[str],
    cfg: AnalysisConfig,
    title_prefixes: list[str] | None,
) -> np.ndarray:
    """
    Token-window chunks aligned with the model tokenizer; mean or weighted pool to one vector
    per document. If ``prefer_single_embedding`` and the full text fits in ``max_seq_length``,
    one encode per thesis (no chunking).
    """
    tokenizer = model.tokenizer
    encode = model.encode
    max_seq = int(getattr(model, "max_seq_length", None) or 512)
    # Raw id windows; re-encode adds [CLS]/[SEP] — stay within model length.
    max_chunk = min(cfg.chunk_max_tokens, max(32, max_seq - 2))

    dim = None
    out_rows: list[np.ndarray] = []

    for idx, text in enumerate(texts):
        prefix = ""
        if title_prefixes and title_prefixes[idx].strip():
            prefix = title_prefixes[idx].strip() + "\n\n"
        body = prefix + text
        body = body.strip()
        if not body:
            if dim is None:
                dim = int(encode(["x"]).shape[1])
            out_rows.append(np.zeros(dim, dtype=np.float32))
            continue

        n_tok = len(tokenizer.encode(body, add_special_tokens=False))
        if cfg.prefer_single_embedding and n_tok <= max_chunk:
            vec = encode([body])
            if isinstance(vec, list):
                vec = np.asarray(vec, dtype=np.float32)
            else:
                vec = np.asarray(vec, dtype=np.float32)
            row = vec[0]
            if dim is None:
                dim = row.shape[0]
            out_rows.append(row.astype(np.float32))
            continue

        chunks = chunk_text_by_tokens(
            tokenizer,
            body,
            max_chunk,
            cfg.chunk_overlap_ratio,
        )
        if not chunks:
            if dim is None:
                dim = int(encode(["x"]).shape[1])
            out_rows.append(np.zeros(dim, dtype=np.float32))
            continue
        ch_emb = encode(chunks)
        if isinstance(ch_emb, list):
            ch_emb = np.asarray(ch_emb, dtype=np.float32)
        pooled = _pool_chunks(ch_emb, cfg)
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
        unit = cfg.chunk_unit
        if unit == "tokens":
            logger.info(
                "Embedding %s documents (tokenizer chunks: max_tokens<=%s, overlap=%.0f%%, pooling=%s)",
                len(texts),
                cfg.chunk_max_tokens,
                100.0 * cfg.chunk_overlap_ratio,
                cfg.chunk_pooling,
            )
            emb = embed_texts_chunked_tokens(model, texts, cfg, title_prefixes)
        elif unit in ("chars", "characters"):
            logger.info("Embedding %s documents (character chunks, pooled)", len(texts))
            emb = embed_texts_chunked_chars(encode, texts, cfg, title_prefixes)
        else:
            raise ValueError(
                f"Unknown embedding.chunk_unit: {unit!r} (use 'tokens' or 'chars')"
            )
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
