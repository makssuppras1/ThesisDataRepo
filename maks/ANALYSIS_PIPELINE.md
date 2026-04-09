# Stage-2 semantic analysis (portable recipe)

This document describes the **downstream analytics** that consume thesis text produced by this repository (`corpus_*.jsonl`, per-file `.txt`, manifests). It matches the CLI in `thesisdatarepo.analysis` (install with `uv sync --group analysis`).

## Join contract: this repo → metadata

| Source | Key | Notes |
|--------|-----|--------|
| **JSONL** (`corpus_*.jsonl`) | `id` | Equals the PDF **filename stem** (local) or GCS object **stem** (e.g. `foo.pdf` → `id` = `foo`). |
| **JSONL** | `text` | Single UTF-8 string: extracted body from Abstract through content before references (not metadata abstract alone). |
| **Metadata CSV** | Configurable (e.g. `member_id_ss`) | Must match `id` from JSONL, or you supply a mapping table. |

- Metadata tables often use separator **`;`** and UTF-8; configure the path and delimiter in `analysis_config.toml`.
- This repo does **not** emit publisher/year columns; you merge them from your own CSV for clustering tables and time analysis.

## Goal

Turn a corpus of documents (here: DTU theses) into **semantic clusters**, **2D visualizations**, **publisher/faculty–topic tables**, and **per-thesis labels**. Optionally analyze **how cluster prevalence changes by year**.

## Inputs you must supply

1. **Metadata table (CSV)**  
   - Often **`;`**, UTF-8 (configurable).  
   - Minimum: **document id** (joins to JSONL `id`), **year** (for `clustering_labels.csv` and evolution), **publisher/section** (optional, for heatmaps and entropy).  
   - **Text for embedding:** either a column (e.g. abstract) **or** use **`embedding_source = "corpus"`** to take `text` from JSONL after merge.

2. **Optional: department/faculty CSV**  
   - Columns include **`Department_new`** and an id column joinable to metadata; merged into the “faculty” column for coloring and entropy.

3. **Corpus JSONL**  
   - From this repo: lines like `{"id": "...", "text": "...", "meta": {...}}`.

4. **Optional: full text from object storage**  
   - e.g. `.md` blobs keyed by thesis id — requires credentials and optional extra config (see `analysis_config.example.toml`).

## Software building blocks

- **Embeddings:** `sentence-transformers` (model name configurable, e.g. `all-mpnet-base-v2`).
- **Normalize:** L2-normalize embedding rows (`sklearn.preprocessing.normalize`).
- **Full text (chunked mode):** chunk (~1600 chars, overlap 200), embed chunks, **weighted mean** (first chunk ×2, last ×1.5), optional **title** prepended to each chunk; then L2-normalize the document vector.
- **Dimensionality:**  
  - **UMAP** → 2D for plots, **10D for clustering** (fallback: cluster in normalized embedding space if UMAP fails).  
  - **t-SNE** on **cached raw embeddings** (2D, per perplexity in config).  
  - **PCA** 2D on cached embeddings.
- **Clustering:** **HDBSCAN** on UMAP-10 (euclidean, leaf selection) by default; or **KMeans / Agglomerative** with fixed `n_clusters`.
- **Keywords:** **TF-IDF** (English stop words, `min_df=2`, `max_features=5000`) on document texts; sum TF-IDF per cluster for top terms.
- **Interdisciplinary score:** per cluster, distinct publisher count and **entropy** of the publisher distribution.
- **Plots:** matplotlib (+ seaborn for heatmaps).

## Pipeline order (CLI `thesis-cluster run`)

1. Load metadata CSV → merge JSONL on id → optional faculty CSV → filter short rows if using abstract-only mode.
2. Optional `sample_size` random subsample.
3. Embeddings: cache `abstract_embeddings_<model_slug>.npy` or `fulltext_embeddings_<model_slug>.npy` under the output directory.
4. UMAP(2), UMAP(10); free large arrays when possible.
5. Cluster on UMAP-10 (or fallback space).
6. Reload embeddings from disk for t-SNE (each perplexity) and PCA; save scatter figures.
7. Write **`clustering_labels.csv`**, **`cluster_keywords.csv`**, publisher×topic tables, interdisciplinary list.
8. Figures: UMAP/t-SNE/PCA scatters, publisher×topic heatmap, cluster size bar chart.

## Second stage (topic evolution)

- **Input:** `clustering_labels.csv` (must include **year** and **cluster_id**).
- **CLI:** `thesis-cluster evolution --config ...`
- **Logic:** drop noise cluster `-1`, aggregate counts per (cluster, year), normalize to share within year, compute change/slope → CSVs + **`cluster_emerging_declining.png`**.

## Adapting elsewhere

- Change **paths**, **bucket names**, **CSV delimiter**, and **column names** in config instead of editing code.
- **Invalidate caches** when the embedding model or corpus changes (cache filenames include a model slug).

## Commands

```bash
uv sync --group analysis
thesis-cluster run --config maks/analysis_config.toml
thesis-cluster evolution --config maks/analysis_config.toml
```

See [`maks/analysis_config.example.toml`](analysis_config.example.toml) for all options.
