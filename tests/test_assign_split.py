import json
from collections import Counter

import pytest

from eval.assign_split import assign, main
from tests.test_golden import item


def draft(n, category="fact", notes=""):
    answerable = category not in ("unanswerable", "false_premise")
    evidence = [] if category == "unanswerable" else item()["evidence"]
    return item(
        id=f"dev-{n:03d}",
        question=f"question number {n}?",
        category=category,
        answerable=answerable,
        evidence=evidence,
        notes=notes,
    )


def test_counts_are_met_and_pairs_stay_together():
    items = [draft(i) for i in range(1, 9)] + [
        draft(20, "version", "pair:x"),
        draft(21, "version", "pair:x"),
        draft(22, "version"),
    ]
    out = assign(items, {"dev": {"fact": 3, "version": 1}, "test": {"fact": 5, "version": 2}})
    assert Counter(i["category"] for i in out["dev"]) == {"fact": 3, "version": 1}
    assert Counter(i["category"] for i in out["test"]) == {"fact": 5, "version": 2}
    [pair_split] = {s for s, rows in out.items() for i in rows if "pair:x" in i["notes"]}
    assert sum("pair:x" in i["notes"] for i in out[pair_split]) == 2


def test_assignment_is_reproducible_and_seed_dependent():
    items = [draft(i) for i in range(1, 21)]
    targets = {"dev": {"fact": 10}, "test": {"fact": 10}}
    ids = lambda out: sorted(i["id"] for i in out["dev"])  # noqa: E731
    assert ids(assign(items, targets, seed=1)) == ids(assign(items, targets, seed=1))
    assert ids(assign(items, targets, seed=1)) != ids(assign(items, targets, seed=2))


def test_count_mismatch_is_an_error():
    with pytest.raises(ValueError, match="targets"):
        assign([draft(1)], {"dev": {"fact": 1}, "test": {"fact": 1}})


def test_main_appends_with_new_ids(tmp_path):
    ds = tmp_path / "dataset"
    ds.mkdir()
    (ds / "dev.jsonl").write_text(json.dumps(draft(7)) + "\n")
    (ds / "test.jsonl").write_text("")
    drafts = tmp_path / "d.jsonl"
    drafts.write_text("".join(json.dumps(draft(100 + i)) + "\n" for i in range(3)))
    main([str(drafts), "--dev", "fact=1", "--test", "fact=2", "--dataset-dir", str(ds)])
    dev = [json.loads(line)["id"] for line in (ds / "dev.jsonl").read_text().splitlines()]
    test = [json.loads(line) for line in (ds / "test.jsonl").read_text().splitlines()]
    assert dev == ["dev-007", "dev-008"]
    assert [t["id"] for t in test] == ["test-001", "test-002"]
    assert all(t["split"] == "test" for t in test)
