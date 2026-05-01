import numpy as np
import pandas as pd

from thesisdatarepo.drift.its import fit_its, run_its_all_features


def test_fit_its_returns_post_coefficient() -> None:
    rng = np.random.default_rng(42)
    years = np.concatenate([np.full(30, 2019), np.full(30, 2024)])
    y = np.concatenate([rng.normal(0, 0.5, 30), rng.normal(2.0, 0.5, 30)])
    r = fit_its(y, years, cutoff=2022)
    assert "error" not in r
    assert r["beta_post"] != 0.0


def test_run_its_all_features() -> None:
    df = pd.DataFrame(
        {
            "Publication Year": [2019] * 15 + [2024] * 15,
            "feat_a": np.linspace(0, 1, 30),
        }
    )
    out = run_its_all_features(df, year_column="Publication Year")
    assert len(out) == 1
    assert out["feature"].iloc[0] == "feat_a"
