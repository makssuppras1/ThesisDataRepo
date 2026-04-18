"""Distinct categorical palettes for faculty / department scatter."""

from thesisdatarepo.analysis.plotting import _distinct_category_palette


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
