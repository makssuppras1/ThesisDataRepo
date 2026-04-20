"""Distinct categorical palettes for faculty / department scatter."""

import pandas as pd

from thesisdatarepo.analysis.plotting import (
    _distinct_category_palette,
    make_department_color_map,
)


def test_distinct_palette_lengths_and_bounds():
    for n in (1, 5, 20, 21, 40, 41, 60):
        pal = _distinct_category_palette(n)
        assert len(pal) == n
        for rgb in pal:
            assert len(rgb) == 3
            assert all(0.0 <= x <= 1.0 for x in rgb)


def test_distinct_palette_no_exact_duplicates_small_n():
    pal = _distinct_category_palette(15)
    assert len(set(tuple(round(x, 5) for x in c) for c in pal)) == 15


def test_make_department_color_map_stable():
    s = pd.Series(["B", "A", "B", None, "A"])
    m1 = make_department_color_map(s)
    m2 = make_department_color_map(s)
    assert m1 == m2
    assert set(m1.keys()) == {"A", "B", "unknown"}
    assert m1["A"] == m2["A"]
