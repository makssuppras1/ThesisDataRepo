"""Sentence-transformers embeddings with optional chunked pooling (token or char windows)."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

from thesisdatarepo.analysis.config_loader import AnalysisConfig

logger = logging.getLogger(__name__)

_CHECKPOINT_VERSION = 1


def model_slug(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name)


def embedding_kind(cfg: AnalysisConfig) -> str:
    return "fulltext" if cfg.embedding_source == "corpus" else "abstract"


def _embedding_fingerprint(cfg: AnalysisConfig, doc_ids: list[str], kind: str) -> str:
    """Stable hash of corpus identity + embedding settings (resume must match)."""
    key = {
        "kind": kind,
        "model": cfg.embedding_model,
        "chunked": cfg.embedding_chunked,
        "chunk_unit": cfg.chunk_unit,
        "chunk_max_tokens": cfg.chunk_max_tokens,
        "chunk_overlap_ratio": cfg.chunk_overlap_ratio,
        "chunk_pooling": cfg.chunk_pooling,
        "prefer_single_embedding": cfg.prefer_single_embedding,
        "chunk_size": cfg.chunk_size,
        "chunk_overlap": cfg.chunk_overlap,
        "first_chunk_weight": cfg.first_chunk_weight,
        "last_chunk_weight": cfg.last_chunk_weight,
        "middle_chunk_weight": cfg.middle_chunk_weight,
        "embedding_source": cfg.embedding_source,
        "min_text_chars": cfg.min_text_chars,
        "n_docs": len(doc_ids),
        "doc_ids": doc_ids,
    }
    raw = json.dumps(key, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def checkpoint_paths(cfg: AnalysisConfig, kind: str) -> tuple[Path, Path]:
    slug = model_slug(cfg.embedding_model)
    base = cfg.output_dir / f"{kind}_embeddings_{slug}_checkpoint"
    return base.with_suffix(".npy"), base.with_suffix(".json")


def _atomic_numpy_save(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Path must end with .npy or np.save appends .npy and breaks replace().
    tmp = path.with_name(path.stem + "._writing" + path.suffix)
    np.save(tmp, arr)
    tmp.replace(path)


def _atomic_json_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _try_load_embedding_checkpoint(
    cfg: AnalysisConfig,
    kind: str,
    expected_fp: str,
    n_docs: int,
) -> tuple[np.ndarray, int] | None:
    """Return (partial_matrix, start_index) or None if missing/invalid."""
    npy_path, json_path = checkpoint_paths(cfg, kind)
    if not npy_path.is_file() or not json_path.is_file():
        return None
    try:
        meta = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Ignoring unreadable embedding checkpoint metadata: %s", e)
        return None
    if meta.get("version") != _CHECKPOINT_VERSION or meta.get("fingerprint") != expected_fp:
        logger.warning(
            "Embedding checkpoint does not match this run (config/corpus changed); "
            "starting fresh and removing stale checkpoint files"
        )
        clear_embedding_checkpoint(cfg, kind)
        return None
    if int(meta.get("n_docs", -1)) != n_docs:
        logger.warning(
            "Embedding checkpoint n_docs mismatch (%s vs %s); starting fresh "
            "and removing stale checkpoint files",
            meta.get("n_docs"),
            n_docs,
        )
        clear_embedding_checkpoint(cfg, kind)
        return None
    partial = np.load(npy_path)
    if partial.ndim != 2 or partial.shape[0] > n_docs:
        logger.warning(
            "Invalid checkpoint array shape %s; starting fresh and removing stale checkpoint files",
            partial.shape,
        )
        clear_embedding_checkpoint(cfg, kind)
        return None
    k = int(partial.shape[0])
    if k == 0:
        return None
    if k >= n_docs:
        logger.info(
            "Checkpoint already contains all %s rows; using it as completed embeddings", n_docs
        )
        return partial[:n_docs].astype(np.float32, copy=False), n_docs
    return partial.astype(np.float32, copy=False), k


def _write_embedding_checkpoint(
    cfg: AnalysisConfig,
    kind: str,
    fingerprint: str,
    n_docs: int,
    partial: np.ndarray,
) -> None:
    npy_path, json_path = checkpoint_paths(cfg, kind)
    _atomic_numpy_save(npy_path, partial)
    _atomic_json_save(
        json_path,
        {
            "version": _CHECKPOINT_VERSION,
            "fingerprint": fingerprint,
            "n_docs": n_docs,
            "kind": kind,
        },
    )


def clear_embedding_checkpoint(cfg: AnalysisConfig, kind: str) -> None:
    npy_path, json_path = checkpoint_paths(cfg, kind)
    for p in (npy_path, json_path):
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


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


def _embed_row_chunked_tokens(
    tokenizer,
    encode,
    max_chunk: int,
    text: str,
    title_prefix: str,
    cfg: AnalysisConfig,
    dim_hint: int | None,
) -> tuple[np.ndarray, int]:
    """Return (pooled row vector, embedding dim)."""
    prefix = ""
    if title_prefix.strip():
        prefix = title_prefix.strip() + "\n\n"
    body = (prefix + text).strip()
    if not body:
        dim = dim_hint if dim_hint is not None else int(np.asarray(encode(["x"])).shape[-1])
        return np.zeros(dim, dtype=np.float32), dim

    n_tok = len(tokenizer.encode(body, add_special_tokens=False))
    if cfg.prefer_single_embedding and n_tok <= max_chunk:
        vec = encode([body])
        vec = np.asarray(vec, dtype=np.float32)
        row = vec[0]
        dim = int(row.shape[0])
        return row.astype(np.float32), dim

    chunks = chunk_text_by_tokens(
        tokenizer,
        body,
        max_chunk,
        cfg.chunk_overlap_ratio,
    )
    if not chunks:
        dim = dim_hint if dim_hint is not None else int(np.asarray(encode(["x"])).shape[-1])
        return np.zeros(dim, dtype=np.float32), dim
    ch_emb = encode(chunks)
    ch_emb = np.asarray(ch_emb, dtype=np.float32)
    pooled = _pool_chunks(ch_emb, cfg)
    dim = int(pooled.shape[0])
    return pooled.astype(np.float32), dim


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


def _embed_row_chunked_chars(
    encode,
    text: str,
    title_prefix: str,
    cfg: AnalysisConfig,
    dim_hint: int | None,
) -> tuple[np.ndarray, int]:
    prefix = ""
    if title_prefix.strip():
        prefix = title_prefix.strip() + "\n\n"
    body = prefix + text
    chunks = chunk_text(body, cfg.chunk_size, cfg.chunk_overlap)
    if not chunks:
        dim = dim_hint if dim_hint is not None else int(encode(["x"]).shape[1])
        return np.zeros(dim, dtype=np.float32), dim
    ch_emb = encode(chunks)
    ch_emb = np.asarray(ch_emb, dtype=np.float32)
    pooled = _pool_chunks(ch_emb, cfg)
    dim = int(pooled.shape[0])
    return pooled.astype(np.float32), dim


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
    max_chunk = min(cfg.chunk_max_tokens, max(32, max_seq - 2))

    dim = None
    out_rows: list[np.ndarray] = []

    for idx, text in enumerate(texts):
        tp = title_prefixes[idx] if title_prefixes else ""
        row, d = _embed_row_chunked_tokens(
            tokenizer, encode, max_chunk, text, tp, cfg, dim
        )
        if dim is None:
            dim = d
        out_rows.append(row)

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

    kind = embedding_kind(cfg)
    n = len(texts)
    if n == 0:
        clear_embedding_checkpoint(cfg, kind)
        return np.zeros((0, 0), dtype=np.float32)

    doc_ids = df[cfg.columns_id].astype(str).str.strip().tolist()
    fingerprint = _embedding_fingerprint(cfg, doc_ids, kind)
    ck_every = cfg.embedding_checkpoint_every

    title_prefixes = None
    if (
        cfg.embedding_chunked
        and cfg.embedding_source == "corpus"
        and cfg.columns_title
        and cfg.columns_title in df.columns
    ):
        title_prefixes = df[cfg.columns_title].fillna("").astype(str).tolist()

    # Fast path: no periodic checkpoints (best throughput, esp. batched plain encode).
    if ck_every <= 0:
        model = SentenceTransformer(cfg.embedding_model)
        encode = model.encode
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
                logger.info(
                    "Embedding %s documents (character chunks, pooled)", len(texts)
                )
                emb = embed_texts_chunked_chars(encode, texts, cfg, title_prefixes)
            else:
                raise ValueError(
                    f"Unknown embedding.chunk_unit: {unit!r} (use 'tokens' or 'chars')"
                )
        else:
            logger.info("Embedding %s documents (one vector each)", len(texts))
            texts_use = texts
            if title_prefixes:
                texts_use = [
                    (f"{t}\n\n{x}" if str(t).strip() else x)
                    for t, x in zip(title_prefixes, texts, strict=True)
                ]
            emb = embed_texts_plain(encode, texts_use)
        return emb.astype(np.float32)

    loaded: tuple[np.ndarray, int] | None = _try_load_embedding_checkpoint(
        cfg, kind, fingerprint, n
    )

    start_idx = 0
    out: np.ndarray | None = None
    if loaded is not None:
        partial, start_idx = loaded
        if start_idx >= n:
            clear_embedding_checkpoint(cfg, kind)
            return partial.astype(np.float32, copy=False)
        out = np.zeros((n, partial.shape[1]), dtype=np.float32)
        out[:start_idx] = partial
        logger.info(
            "Resuming embedding from document %s / %s (checkpoint_every=%s)",
            start_idx + 1,
            n,
            ck_every,
        )

    model = SentenceTransformer(cfg.embedding_model)
    encode = model.encode

    if cfg.embedding_chunked and cfg.embedding_source == "corpus":
        unit = cfg.chunk_unit
        if unit == "tokens":
            tokenizer = model.tokenizer
            max_seq = int(getattr(model, "max_seq_length", None) or 512)
            max_chunk = min(cfg.chunk_max_tokens, max(32, max_seq - 2))
            dim_hint = None if out is None else out.shape[1]
            logger.info(
                "Embedding %s documents (tokenizer chunks: max_tokens<=%s, overlap=%.0f%%, pooling=%s)",
                n,
                cfg.chunk_max_tokens,
                100.0 * cfg.chunk_overlap_ratio,
                cfg.chunk_pooling,
            )
            for idx in range(start_idx, n):
                tp = title_prefixes[idx] if title_prefixes else ""
                row, d = _embed_row_chunked_tokens(
                    tokenizer,
                    encode,
                    max_chunk,
                    texts[idx],
                    tp,
                    cfg,
                    dim_hint,
                )
                if out is None:
                    out = np.zeros((n, d), dtype=np.float32)
                    dim_hint = d
                row_n = normalize(row.reshape(1, -1), norm="l2", axis=1).astype(
                    np.float32
                )[0]
                out[idx] = row_n
                if ck_every > 0 and (idx + 1) % ck_every == 0:
                    _write_embedding_checkpoint(cfg, kind, fingerprint, n, out[: idx + 1])
        elif unit in ("chars", "characters"):
            logger.info("Embedding %s documents (character chunks, pooled)", n)
            dim_hint = None if out is None else out.shape[1]
            for idx in range(start_idx, n):
                tp = title_prefixes[idx] if title_prefixes else ""
                row, d = _embed_row_chunked_chars(
                    encode, texts[idx], tp, cfg, dim_hint
                )
                if out is None:
                    out = np.zeros((n, d), dtype=np.float32)
                    dim_hint = d
                row_n = normalize(row.reshape(1, -1), norm="l2", axis=1).astype(
                    np.float32
                )[0]
                out[idx] = row_n
                if ck_every > 0 and (idx + 1) % ck_every == 0:
                    _write_embedding_checkpoint(cfg, kind, fingerprint, n, out[: idx + 1])
        else:
            raise ValueError(
                f"Unknown embedding.chunk_unit: {unit!r} (use 'tokens' or 'chars')"
            )
    else:
        logger.info("Embedding %s documents (one vector each)", n)
        texts_use = texts
        if title_prefixes:
            texts_use = [
                (f"{t}\n\n{x}" if str(t).strip() else x)
                for t, x in zip(title_prefixes, texts, strict=True)
            ]
        dim_hint = None if out is None else out.shape[1]
        for idx in range(start_idx, n):
            vec = encode([texts_use[idx]])
            vec = np.asarray(vec, dtype=np.float32)
            row = vec[0]
            if out is None:
                d = int(row.shape[0])
                out = np.zeros((n, d), dtype=np.float32)
                dim_hint = d
            row_n = normalize(row.reshape(1, -1), norm="l2", axis=1).astype(np.float32)[0]
            out[idx] = row_n
            if ck_every > 0 and (idx + 1) % ck_every == 0:
                _write_embedding_checkpoint(cfg, kind, fingerprint, n, out[: idx + 1])

    assert out is not None
    clear_embedding_checkpoint(cfg, kind)
    return out.astype(np.float32)


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
