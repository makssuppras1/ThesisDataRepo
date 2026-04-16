# Stage-2 semantic analysis (portable recipe)

This document describes the **downstream analytics** that consume thesis text produced by this repository (`corpus_*.jsonl`, per-file `.txt`, manifests). It matches the CLI in `thesisdatarepo.analysis` (install with `uv sync --group analysis`).

---

## Join contract: this repo → metadata

| Source | Key | Notes |
|--------|-----|--------|
| **JSONL** (`corpus_*.jsonl`) | `id` | Equals the PDF **filename stem** (local) or GCS object **stem** (e.g. `foo.pdf` → `id` = `foo`). |
| **JSONL** | `text` | Single UTF-8 string: extracted body from Abstract through content before references (not metadata abstract alone). |
| **Metadata** | CSV or **Parquet** (`metadata_csv` path) | Joined **onto** the corpus. Use **`corpus_join = "pdf_file"`** when NLP `.txt` names match **PDF/blob stems** (not `member_id_ss` alone). Set **`inner_join_metadata = true`** to drop corpus rows with no metadata row. |

**Source of truth:** the pipeline keeps **one row per corpus id** (left-join from JSONL). Duplicate JSONL lines for the same `id` use **last** line wins. In corpus mode, embeddings use **`text_bucket`** from the JSONL only, never a metadata `text` column.

**Local `.txt` cross-check:** set `paths.nlp_txt_dir` (e.g. `maks/data/nlp_from_gcs_all`) to keep only rows whose `id` has a matching file `{id}.txt` in that folder — same naming as the GCS NLP export (blob stem = JSONL `id`). Rows in JSONL without a file on disk are dropped.

**`corpus_txt_id_mode` (txt_dir only):** `full_stem` = filename without `.txt` (match `pdf_file` stem + `corpus_join = "pdf_file"`). `member_prefix` = text **before the first `_`** in the filename, matching **`member_id_ss`** / **`primary_member_id_s`** when files look like `{member_id}_{title}.txt`. Use with `corpus_join = ""` and `columns.id = "member_id_ss"`.

- Metadata tables often use separator **`;`** and UTF-8; configure paths in `analysis_config.toml`.
- Publisher/year come from metadata when ids match; ids only in the corpus still get embedded, with missing metadata fields empty where applicable.

---

## Goal

Turn a corpus of documents (here: DTU theses) into **semantic clusters**, **2D visualizations**, **publisher/faculty–topic tables**, and **per-thesis labels**. Optionally analyze **how cluster prevalence changes by year**.

---

## Inputs you must supply (or equivalent)

1. **Metadata table (CSV)**  
   - Separator: **`;`** (configurable), UTF-8.  
   - Needs at minimum:
     - **Document id** (e.g. `member_id_ss`) — stable key for joins and outputs.
     - **Text for embedding:** typically **`abstract_ts`** if you use **abstract mode**; or switch to **corpus mode** and use **`text` from JSONL** after merge (full extracted thesis body).
     - **Publisher / section** (e.g. `Publisher`) — optional but needed for publisher×cluster analysis; can be replaced or supplemented by a **department/faculty** column after a join.
     - **Year** — for `clustering_labels.csv` and downstream time analysis.
   - Rows with missing or very short text are filtered using **`min_text_chars`** (abstract mode); in **corpus** mode the same threshold applies to the merged document text.

2. **Corpus JSONL** (this repo)  
   - Lines like `{"id": "...", "text": "...", ...}` produced by the NLP pipeline. This is the usual **full-text** source when `embedding.source = "corpus"`.

3. **Optional: department/faculty**  
   - Separate CSV with **`Department_new`** (or configured faculty column) and an **id column** joinable to the metadata id.  
   - After merge, that column drives **faculty** coloring and entropy where configured.

4. **Optional: object storage**  
   - Full text may originate from GCS PDF extraction in this repo; you do **not** need a separate “download .md” step for the default path — text lands in JSONL and optional `{id}.txt` on disk. A different deployment could read markdown or blobs from storage with credentials; map that into the same **id + text** contract.

---

## Embedding model (what the vectors are)

- **Library:** [SentenceTransformers](https://www.sbert.net/) (`sentence-transformers` on PyPI).
- **Default model:** `sentence-transformers/all-mpnet-base-v2` (768-dimensional). Configurable via `embedding.model` in the analysis TOML.
- **Meaning:** Each thesis is mapped to a **dense vector** such that texts with similar *semantic* content are close in cosine distance after normalization. The model is a general-purpose English sentence encoder (MPNet backbone), not domain-finetuned on DTU theses unless you change the model name.
- **Abstract mode:** one forward pass per document (abstract string; optional title prepended if configured).
- **Corpus (full-text) mode with `chunked = true`:** default **`embedding.chunk_unit = "tokens"`** — sliding windows on the model **tokenizer**’s token ids (`chunk_max_tokens`, typically 512–768; clamped to the model’s `max_seq_length`), **`chunk_overlap_ratio`** in the 10–20% range (e.g. `0.15`), **`chunk_pooling = "mean"`** to one vector per thesis. If **`prefer_single_embedding = true`** and the full string (optional title + body) fits in one forward pass, a **single** embedding is used (preferred when feasible). Legacy **`chunk_unit = "chars"`** uses character slices (`chunk_size` / `chunk_overlap`) and optional **weighted** pooling. Result is **L2-normalized** per row.
- **Caching:** vectors are saved as `abstract_embeddings_<model_slug>.npy` or `fulltext_embeddings_<model_slug>.npy` under `output_dir` so reruns can skip recomputation when `embedding.use_cache` is true.

---

## t-SNE and PCA plots (how they relate to UMAP and clustering)

- **Clustering does not use t-SNE or PCA.** HDBSCAN (or KMeans / Agglomerative) runs on **UMAP 10D** coordinates derived from the **same L2-normalized embedding matrix** used everywhere else. If UMAP fails, clustering falls back to the **normalized high-dimensional embeddings**.
- **UMAP 2D** plots are for **visualization** (global structure in a learned 2D projection with `metric`, `n_neighbors`, `min_dist` from config).
- **PCA 2D** is a **linear** projection of the (reloaded, re-normalized) embedding matrix — cheap, interpretable as “main variance directions,” but not optimized for neighborhood preservation like UMAP.
- **t-SNE 2D** is a **nonlinear** embedding that emphasizes **local** neighborhoods; perplexity is taken from `tsne.perplexities` but **clamped** so it never exceeds what the sample size allows. Each perplexity produces `tsne2d_perp{N}_clusters.png`, optional `_publisher.png` / `_faculty.png`, and `tsne2d_perp{N}_coords.npy`.
- **Important:** After the main pass, embeddings are **reloaded from disk** for PCA/t-SNE to limit peak memory. Plots reuse **cluster labels** from the UMAP-10 step so points are colored by the same clusters.
- **Reading the figures:** **Clusters** = outcome of the clustering step. **Publisher / faculty** = metadata columns — useful to see whether semantic neighborhoods align with organizational labels, not a separate model output.

---

## Software building blocks (summary)

- **Embeddings:** `sentence-transformers` (model name configurable).
- **Normalize:** L2-normalize embedding rows (`sklearn.preprocessing.normalize`) before UMAP and again when preparing PCA/t-SNE inputs in the plotting path.
- **Full text (chunked):** default tokenizer windows + overlap ratio + **mean** pool (see `[embedding]` in TOML); optional char-based chunking + weighted pool for old configs.
- **Dimensionality:**  
  - **UMAP** → 2D for plots, **10D for clustering** (fallback: cluster in normalized embedding space if UMAP fails).  
  - **t-SNE** on **cached raw embeddings** (2D, per perplexity in config).  
  - **PCA** 2D on cached embeddings.
- **Clustering:** **HDBSCAN** on UMAP-10 by default; or **KMeans / Agglomerative** with fixed `n_clusters`.
- **Keywords:** **TF-IDF** (English stop words, `min_df`, `max_features`) on the **same per-document texts** used for embeddings; aggregate per cluster for top terms.
- **Interdisciplinary score:** per cluster, distinct publisher count and **entropy** of the publisher distribution.
- **Plots:** matplotlib (+ seaborn for heatmaps).

---

## Pipeline order (repeat in another repo)

1. **Load** metadata CSV → **merge** JSONL on id → optional faculty CSV → **filter** short texts per `min_text_chars` (and optional `nlp_txt_dir` file existence).
2. **Optional `sample_size`:** random subsample for speed.
3. **Embeddings:** compute or load cache — `abstract_embeddings_<model_slug>.npy` or `fulltext_embeddings_<model_slug>.npy`.
4. **L2-normalize** embeddings → **UMAP(2)** and **UMAP(10)**; drop large arrays when possible.
5. **Cluster** on UMAP-10 (or fallback: normalized embeddings).
6. **Write** `clustering_labels.csv`, TF-IDF / crosstabs / interdisciplinary CSVs; **UMAP** scatter figures.
7. **Reload embeddings from disk** for **PCA** and **t-SNE** (each perplexity); write PCA/t-SNE PNGs and `tsne*_coords.npy`.

To **regenerate only PCA + t-SNE** from cached `.npy` and existing `clustering_labels.csv`: `thesis-cluster tsne-plots --config …` (same config as the full run).

---

## Second stage (topic evolution / emerging–declining)

- **Input:** `clustering_labels.csv` (must include **year** and **cluster_id**).
- **CLI:** `thesis-cluster evolution --config ...`
- **Logic:** drop noise cluster `-1`, aggregate counts per (cluster, year), normalize to share within year, compute change/slope → CSVs + **`cluster_emerging_declining.png`**.

---

## What you’d adapt in another repo

- **Paths**, **bucket names**, and **CSV separators** if your data isn’t `;`-delimited or stored the same way.
- **Column mapping** in config instead of hardcoding (this codebase uses explicit TOML fields under `[columns]`).
- **Auth** for cloud storage if your corpus is only in object storage.
- **Cache filenames** (`*_embeddings_<model_slug>.npy`) so caches invalidate when the model or corpus changes.

---

## Commands

```bash
uv sync --group analysis
thesis-cluster run --config maks/analysis_config.toml
thesis-cluster tsne-plots --config maks/analysis_config.toml
thesis-cluster evolution --config maks/analysis_config.toml
```

See [`maks/analysis_config.example.toml`](analysis_config.example.toml) for all options.
