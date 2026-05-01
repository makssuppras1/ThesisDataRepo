"""Build a modeling-ready frame: corpus + metadata minus grading columns."""

from __future__ import annotations

import pandas as pd

from thesisdatarepo.analysis.config_loader import AnalysisConfig
from thesisdatarepo.analysis.io_data import load_merged_frame


def _drop_grading_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in df.columns if c.startswith("grading_")]
    if not drop:
        return df
    return df.drop(columns=drop, errors="ignore")


def load_modeling_frame(cfg: AnalysisConfig) -> pd.DataFrame:
    """
    Merged corpus + metadata (see ``load_merged_frame``), then remove all
    ``grading_*`` columns so they never enter drift statistics by accident.
    """
    merged = load_merged_frame(cfg)
    return _drop_grading_columns(merged)
