import pytest

from eval.calibration import cohen_kappa, sample_ids


def test_kappa_perfect_and_chance():
    assert cohen_kappa([0, 1, 2, 2], [0, 1, 2, 2]) == 1.0
    # identical marginals, no agreement beyond chance
    assert cohen_kappa([0, 1, 0, 1], [1, 0, 0, 1], labels=(0, 1)) == pytest.approx(0.0)


def test_weighted_kappa_penalizes_far_misses_more():
    human = [2, 2, 1, 0, 2, 1]
    near = [1, 2, 1, 0, 2, 1]  # one miss by 1
    far = [0, 2, 1, 0, 2, 1]  # one miss by 2
    assert cohen_kappa(human, near, weighted=True) > cohen_kappa(human, far, weighted=True)


def test_known_value():
    # 2x2 textbook example: po = 0.7, pe = 0.5 -> kappa 0.4
    a = [1] * 5 + [0] * 5
    b = [1, 1, 1, 1, 0] + [1, 1, 0, 0, 0]
    assert cohen_kappa(a, b, labels=(0, 1)) == pytest.approx(0.4)


def test_sample_is_stratified_and_reproducible():
    rows = [
        {"id": f"dev-{i:03d}", "category": c}
        for i, c in enumerate(["fact"] * 20 + ["howto"] * 15 + ["unanswerable"] * 5)
    ]
    ids = sample_ids(rows, {"fact": 4, "howto": 3, "unanswerable": 10})
    assert len(ids) == 4 + 3 + 5
    assert ids == sample_ids(rows, {"fact": 4, "howto": 3, "unanswerable": 10})
