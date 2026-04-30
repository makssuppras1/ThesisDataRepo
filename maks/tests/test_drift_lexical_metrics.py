from thesisdatarepo.drift.lexical_metrics import hapax_ratio, lexical_for_tokens, mtld_forward


def test_hapax_ratio_all_unique_types() -> None:
    toks = list("abcde")
    assert hapax_ratio(toks) == 1.0


def test_hapax_ratio_repeated_type() -> None:
    assert hapax_ratio(["the", "the", "cat"]) == 1 / 3


def test_mtld_forward_positive() -> None:
    toks = ["word"] * 600
    m = mtld_forward(toks)
    assert m > 0


def test_lexical_for_tokens_short_uses_ttr() -> None:
    toks = ["a", "b", "a"]
    mt, hx, ttr = lexical_for_tokens(toks)
    assert ttr == 2 / 3
    assert mt == ttr  # short doc path
