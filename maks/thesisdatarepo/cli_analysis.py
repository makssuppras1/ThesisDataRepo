"""CLI: ``thesis-cluster run|evolution --config …``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from thesisdatarepo.analysis.config_loader import load_config
from thesisdatarepo.analysis.evolution import run_evolution
from thesisdatarepo.analysis.pipeline import run_pipeline, run_tsne_plots_only


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(prog="thesis-cluster")
    sub = p.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Embeddings, UMAP, cluster, TF-IDF, figures")
    run_p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="TOML config (see maks/analysis_config.example.toml)",
    )
    run_p.add_argument(
        "--no-cache",
        action="store_true",
        help="Recompute embeddings even if cache .npy exists",
    )

    ev_p = sub.add_parser(
        "evolution",
        help="Cluster prevalence vs year from clustering_labels.csv",
    )
    ev_p.add_argument("--config", type=Path, required=True)

    ts_p = sub.add_parser(
        "tsne-plots",
        help="Regenerate PCA + t-SNE PNGs from cached embeddings and clustering_labels.csv",
    )
    ts_p.add_argument("--config", type=Path, required=True)

    args = p.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == "run":
        if args.no_cache:
            cfg.embedding_use_cache = False
        run_pipeline(cfg)
        return 0
    if args.cmd == "evolution":
        run_evolution(cfg)
        return 0
    if args.cmd == "tsne-plots":
        run_tsne_plots_only(cfg)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
