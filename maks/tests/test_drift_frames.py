"""Drift modeling frame: grading columns stripped."""

from __future__ import annotations

import pandas as pd

from thesisdatarepo.drift import frames


def test_drop_grading_columns_removes_grading_prefix() -> None:
    df = pd.DataFrame(
        {
            "member_id_ss": ["a"],
            "grading_total_score": [80.0],
            "grading_meta_attempts": [1],
            "Department_new": ["X"],
        }
    )
    out = frames._drop_grading_columns(df)
    assert "grading_total_score" not in out.columns
    assert "grading_meta_attempts" not in out.columns
    assert "member_id_ss" in out.columns
    assert list(out.columns) == ["member_id_ss", "Department_new"]
