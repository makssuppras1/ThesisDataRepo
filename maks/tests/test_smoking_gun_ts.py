"""Tests for ``oliver/production/smoking_gun_ts.py`` (loaded by path)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
PROD = ROOT / "oliver" / "production" / "smoking_gun_ts.py"


@pytest.fixture(scope="module")
def sgt():
    assert PROD.is_file(), f"missing {PROD}"
    spec = importlib.util.spec_from_file_location("smoking_gun_ts", PROD)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _sample_df() -> pd.DataFrame:
    rows = []
    for y in (2019, 2020, 2021):
        for month in (1, 2, 3, 6, 9, 12):
            ts = pd.Timestamp(y, month, 1)
            rows.append(
                {
                    "handin_month": ts.strftime("%B %Y"),
                    "Department_new": "DeptA",
                    "num_cont_pages": 40 + y % 5 + month,
                    "lexical_diversity": 0.15 + month * 0.001,
                    "unique_words": 2000 + month * 10,
                }
            )
    return pd.DataFrame(rows)


def test_preprocess_and_aggregate(sgt):
    df = _sample_df()
    clean = sgt.preprocess_smoking_gun_frame(df, "num_cont_pages", department=None)
    assert len(clean) > 0
    tl = sgt.aggregate_to_timeline(clean, "num_cont_pages", freq="MS")
    assert len(tl) == 84
    assert tl["_hm_dt"].is_monotonic_increasing


def test_analyze_adds_columns(sgt):
    df = _sample_df()
    clean = sgt.preprocess_smoking_gun_frame(df, "num_cont_pages", department=None)
    tl = sgt.aggregate_to_timeline(clean, "num_cont_pages", freq="MS")
    out = sgt.analyze_smoking_gun_timeseries(tl, "num_cont_pages", rolling_window=3)
    for col in ("roll_mean", "ema", "pct_change_1", "anomaly_flag", "trend_smooth", "resid_decomp"):
        assert col in out.columns


def test_plot_smoking_gun_smoke(sgt, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)

    df = _sample_df()
    monkeypatch.setitem(__import__("__main__").__dict__, "IMG_EXPORT", False)
    monkeypatch.setitem(__import__("__main__").__dict__, "IMG_EXPORT_PATH", "")

    r = sgt.plot_smoking_gun_by_dept(
        df,
        0,
        None,
        1,
        show_rolling=True,
        show_ema=True,
        show_anomalies=True,
        show_forecast=False,
        return_outputs=True,
        img_export=False,
        img_export_path="",
    )
    assert r is not None
    fig, ax, work = r
    assert fig is not None
    assert work is not None
    assert "roll_mean" in work.columns
    plt.close(fig)


def test_validate_missing_column(sgt):
    df = pd.DataFrame({"handin_month": ["January 2019"], "Department_new": ["X"]})
    with pytest.raises(ValueError, match="missing columns"):
        sgt.preprocess_smoking_gun_frame(df, "num_cont_pages", department=None)
