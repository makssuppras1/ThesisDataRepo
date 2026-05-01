"""Cross-sectional Mann–Whitney + FDR."""

from __future__ import annotations

import numpy as np
import pandas as pd

from thesisdatarepo.drift.cross_sectional import run_cross_sectional


def test_run_cross_sectional_shape_and_columns() -> None:
    rng = np.random.default_rng(0)
    pre = pd.DataFrame(
        {
            "Publication Year": [2020] * 25 + [2021] * 5,
            "feat_x": rng.normal(0.0, 1.0, 30),
        }
    )
    post = pd.DataFrame(
        {
            "Publication Year": [2023] * 30,
            "feat_x": rng.normal(3.0, 1.0, 30),
        }
    )
    df = pd.concat([pre, post], ignore_index=True)
    out = run_cross_sectional(df, year_column="Publication Year")
    assert len(out) == 1
    assert out["feature"].iloc[0] == "feat_x"
    assert "p_bh" in out.columns
    assert "cohens_d" in out.columns
    assert out["cohens_d"].iloc[0] > 0
