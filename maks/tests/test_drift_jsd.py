import pandas as pd

from thesisdatarepo.drift.jsd_drift import jsd_year_series


def test_jsd_year_series_nonempty() -> None:
    df = pd.DataFrame(
        {
            "Publication Year": [2017, 2017, 2020, 2020],
            "text_bucket": [
                "the methodology chapter describes the algorithm clearly",
                "the results section shows the methodology again",
                "the conclusion summarizes methodology and results clearly",
                "the introduction mentions methodology only briefly here",
            ],
        }
    )
    out = jsd_year_series(df, year_column="Publication Year", baseline_year_max=2018)
    assert not out.empty
    assert "jsd" in out.columns
    assert out["year"].tolist() == [2017, 2020]
