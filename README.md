# ThesisDataRepo

## Setup

From the repo root: install dependencies (including the **analysis** group), then install this package in editable mode.

```bash
cd /path/to/ThesisDataRepo
uv sync
uv sync --group analysis
uv pip install -e .
```

**Run CLIs** with `uv run …` (uses the project `.venv` automatically). Alternatively, activate the venv first: `source .venv/bin/activate`, then you can call `thesis-cluster` and `thesis-gcs-nlp` directly.

## CLI

### GCS → NLP text

Download or process PDFs from GCS and write extracted **plain text** (and related outputs) according to `maks/nlp_gcs.toml`. Use this when you need fresh `.txt` / corpus files before analysis.

```bash
uv run thesis-gcs-nlp --config maks/nlp_gcs.toml
```

### Full analysis run

Load metadata + corpus text, compute **embeddings**, **UMAP**, **cluster** documents, TF-IDF keywords, crosstabs, and write figures (UMAP, PCA, t-SNE, department “soft cloud” plots, etc.) under `output_dir` from `maks/analysis_config.toml`. **Long-running** on first run (embedding ~4k theses). With **`embedding.checkpoint_every`** ≥ 1 (default in `analysis_config.example.toml`), embedding **resumes after an interrupt** from checkpoint files next to the final `.npy`; set **`checkpoint_every = 0`** to disable checkpoints and maximize throughput.

```bash
uv run thesis-cluster run --config maks/analysis_config.toml
```

### PCA + t-SNE plots only

Rebuild **PCA and t-SNE** PNGs from **cached** embedding `.npy` and existing `clustering_labels.csv` — no re-embedding. Use after tuning plot code or `[tsne]` in config, without paying for embeddings again.

```bash
uv run thesis-cluster tsne-plots --config maks/analysis_config.toml
```

### Cluster evolution over time

Second stage: read `clustering_labels.csv`, aggregate cluster counts by **year**, and emit evolution tables / figures (e.g. emerging vs declining topics). Requires a completed `run` with year column present.

```bash
uv run thesis-cluster evolution --config maks/analysis_config.toml
```

### Tests

Run the test suite (`maks/tests`).

```bash
uv run pytest
```
