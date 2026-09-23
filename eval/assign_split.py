"""Assign drafted golden-set items to dev and test, reproducibly.

Items keep their content; they get final ids and a split. The assignment is stratified by
category to the requested counts, uses a fixed seed, and keeps every version pair (items
whose notes carry the same `pair:<tag>`) in one split, since the two questions differ only
in their version and would leak across splits otherwise. Nobody picks what goes to test.

New ids continue after the highest existing id of each split.

Usage:
    uv run python -m eval.assign_split drafts/*.jsonl --dev fact=17,howto=12 --test fact=20
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from eval.golden import DATASET_DIR, Item, load_split

PAIR_RE = re.compile(r"pair:([\w.-]+)")
SEED = 20260923


def groups(items: list[dict]) -> list[list[dict]]:
    """Items that must share a split: each version pair together, everything else alone."""
    by_pair, out = defaultdict(list), []
    for item in items:
        m = PAIR_RE.search(item.get("notes", ""))
        if m:
            by_pair[m[1]].append(item)
        else:
            out.append([item])
    return out + list(by_pair.values())


def assign(items: list[dict], targets: dict[str, dict[str, int]], seed: int = SEED):
    """-> {split: [items]} meeting `targets[split][category]` counts exactly."""
    rng = random.Random(seed)
    by_cat: dict[str, list[list[dict]]] = defaultdict(list)
    for g in groups(items):
        cats = {i["category"] for i in g}
        if len(cats) != 1:
            raise ValueError(f"pair spans categories {cats}: {[i['id'] for i in g]}")
        by_cat[cats.pop()].append(g)

    out: dict[str, list[dict]] = {split: [] for split in targets}
    for cat, cat_groups in sorted(by_cat.items()):
        wanted = {split: t.get(cat, 0) for split, t in targets.items()}
        if sum(wanted.values()) != sum(len(g) for g in cat_groups):
            raise ValueError(f"{cat}: {sum(len(g) for g in cat_groups)} items, targets {wanted}")
        rng.shuffle(cat_groups)
        # Place pairs first so the singles can fill whatever room is left exactly.
        cat_groups.sort(key=len, reverse=True)
        for g in cat_groups:
            room = {s: wanted[s] - sum(i["category"] == cat for i in out[s]) for s in targets}
            fits = [s for s in targets if room[s] >= len(g)]
            if not fits:
                raise ValueError(f"{cat}: can't place group {[i['id'] for i in g]}")
            split = max(fits, key=lambda s: (room[s], s == "dev"))
            out[split].extend(g)
    return out


def next_ids(split: str, dataset_dir: Path):
    existing = [int(i.id.split("-")[1]) for i in load_split(split, dataset_dir)]
    n = max(existing, default=0)
    while True:
        n += 1
        yield f"{split}-{n:03d}"


def parse_targets(spec: str) -> dict[str, int]:
    return {k: int(v) for k, v in (part.split("=") for part in spec.split(",") if part)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("drafts", nargs="+", type=Path)
    parser.add_argument("--dev", required=True, help="category=count,... to add to dev")
    parser.add_argument("--test", required=True, help="category=count,... to add to test")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    items = [json.loads(line) for p in args.drafts for line in p.read_text().splitlines() if line]
    targets = {"dev": parse_targets(args.dev), "test": parse_targets(args.test)}
    assigned = assign(items, targets, args.seed)
    order = ["fact", "howto", "version", "multihop", "unanswerable", "false_premise"]
    for split, split_items in assigned.items():
        ids = next_ids(split, args.dataset_dir)
        split_items.sort(key=lambda i: (order.index(i["category"]), i["id"]))
        rows = []
        for item in split_items:
            item = {**item, "id": next(ids), "split": split}
            Item.model_validate(item)
            rows.append(json.dumps(item, ensure_ascii=False))
        path = args.dataset_dir / f"{split}.jsonl"
        with path.open("a") as f:
            f.write("".join(r + "\n" for r in rows))
        print(f"{split}: +{len(rows)} items -> {path}")


if __name__ == "__main__":
    main()
