import pytest

from eval.report import bootstrap_ci, paired_diff


def test_bootstrap_ci_brackets_the_mean_and_is_reproducible():
    values = [1.0] * 80 + [0.0] * 20
    mean, lo, hi = bootstrap_ci(values, samples=2000)
    assert mean == pytest.approx(0.8)
    assert lo < 0.8 < hi and 0.65 < lo and hi < 0.9
    assert bootstrap_ci(values, samples=2000) == (mean, lo, hi)


def test_constant_values_have_zero_width():
    assert bootstrap_ci([0.5] * 10, samples=200) == (0.5, 0.5, 0.5)


def test_paired_diff_detects_consistent_small_gain():
    a = {f"q{i}": 0.5 for i in range(100)}
    b = {f"q{i}": 0.5 + (0.1 if i % 2 else 0.0) for i in range(100)}
    d = paired_diff(a, b, samples=2000)
    assert d["diff"] == pytest.approx(0.05)
    assert d["lo"] > 0 and d["p_not_better"] == 0.0


def test_paired_diff_of_noise_straddles_zero():
    a = {f"q{i}": float(i % 2) for i in range(60)}
    b = {f"q{i}": float((i // 2) % 2) for i in range(60)}
    d = paired_diff(a, b, samples=2000)
    assert d["lo"] < 0 < d["hi"]


def test_only_shared_items_are_compared():
    d = paired_diff({"a": 1.0, "b": 0.0}, {"b": 1.0, "c": 0.0}, samples=100)
    assert d["n"] == 1 and d["diff"] == 1.0
