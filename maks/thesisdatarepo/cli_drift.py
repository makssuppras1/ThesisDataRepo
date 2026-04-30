"""CLI: ``thesis-drift run --config maks/analysis_config.toml``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from thesisdatarepo.analysis.config_loader import load_config
from thesisdatarepo.drift.cross_sectional import run_cross_sectional
from thesisdatarepo.drift.features import augment_feature_table
from thesisdatarepo.drift.frames import load_modeling_frame
from thesisdatarepo.drift.its import run_its_all_features
from thesisdatarepo.drift.jsd_drift import jsd_year_series
from thesisdatarepo.drift.viz import plot_jsd_drift

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(prog="thesis-drift")
    sub = p.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Load corpus+metadata, compute drift features, MWU table")
    run_p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="TOML config (same as thesis-cluster, see maks/analysis_config.toml)",
    )
    run_p.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="If >0, only process first N rows after merge (debug)",
    )

    args = p.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == "run":
        df = load_modeling_frame(cfg)
        if args.max_docs > 0:
            df = df.iloc[: args.max_docs].reset_index(drop=True)
            logger.info("Limited to max_docs=%s", args.max_docs)

        df = augment_feature_table(cfg, df)
        out_dir = cfg.output_dir / "drift"
        out_dir.mkdir(parents=True, exist_ok=True)
        feat_path = out_dir / "drift_features.parquet"
        df.to_parquet(feat_path, index=False)
        logger.info("Wrote %s (%s rows)", feat_path, len(df))

        ycol = cfg.columns_year
        sig = run_cross_sectional(df, year_column=ycol)
        sig_path = out_dir / "drift_significance.csv"
        sig.to_csv(sig_path, index=False)
        logger.info("Wrote %s", sig_path)

        jsd = jsd_year_series(df, year_column=ycol, text_column="text_bucket")
        jsd_path = out_dir / "jsd_by_year.csv"
        jsd.to_csv(jsd_path, index=False)
        logger.info("Wrote %s", jsd_path)
        if not jsd.empty:
            plot_jsd_drift(jsd, out_dir / "figures" / "jsd_drift.png")

        its_df = run_its_all_features(df, year_column=ycol, cutoff=2022)
        its_path = out_dir / "its_results.csv"
        its_df.to_csv(its_path, index=False)
        logger.info("Wrote %s", its_path)

        print(sig.to_string(index=False))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
