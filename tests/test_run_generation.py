from collections import Counter
from types import SimpleNamespace

from eval.run_generation import stratified_sample


def test_stratified_sample_keeps_category_shares_and_is_seeded():
    items = [SimpleNamespace(id=f"i{n:03d}", category="a" if n < 80 else "b") for n in range(100)]
    picked = stratified_sample(items, 50)
    assert Counter(i.category for i in picked) == {"a": 40, "b": 10}
    assert [i.id for i in picked] == sorted(i.id for i in picked)
    assert picked == stratified_sample(items, 50)
