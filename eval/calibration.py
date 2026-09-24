"""Calibrate the judge against human scores: sample, export for labeling, and compute kappa.

1. `sample` picks a fixed, category-stratified subset of a generation run (seeded, so the
   same items come back) and writes what a human needs to score each answer.
2. A person scores correctness 0/1/2 with the judge's rubric, without seeing the judge.
3. `agreement` compares those labels with the judge: exact agreement, Cohen's kappa, and
   quadratic-weighted kappa (the scale is ordinal: 0 vs 2 is a worse miss than 1 vs 2),
   plus the list of disagreements to read.

Usage:
    uv run python -m eval.calibration sample --run dev-baseline
    uv run python -m eval.calibration agreement --run dev-baseline --judge claude-haiku-4-5 \\
        --labels eval/calibration/dev-baseline.human.jsonl
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from eval.golden import load_split
from eval.judge import JUDGE_VERSION, RUBRIC, load_chunks
from eval.run_generation import RESULTS_DIR

ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_DIR = ROOT / "eval" / "calibration"
SEED = 20260924
PER_CATEGORY = {
    "fact": 12,
    "howto": 10,
    "version": 10,
    "multihop": 8,
    "unanswerable": 7,
    "false_premise": 3,
}  # 50, roughly the dev mix


def sample_ids(rows: list[dict], per_category=PER_CATEGORY, seed=SEED) -> list[str]:
    rng = random.Random(seed)
    picked = []
    for cat, n in per_category.items():
        ids = sorted(r["id"] for r in rows if r["category"] == cat)
        picked += rng.sample(ids, min(n, len(ids)))
    return sorted(picked)


def export(run: str, out: Path) -> list[dict]:
    rows = {
        r["id"]: r for r in map(json.loads, (RESULTS_DIR / f"{run}.jsonl").read_text().splitlines())
    }
    items = {i.id: i for i in load_split(run.split("-", 1)[0])}
    chunk_text = load_chunks({r["version"] for r in rows.values()})
    records = []
    for item_id in sample_ids(list(rows.values())):
        r, item = rows[item_id], items[item_id]
        records.append(
            {
                "id": item_id,
                "category": item.category,
                "version": item.version,
                "question": item.question,
                "reference_answer": item.reference_answer,
                "answer": r["answer"],
                "sources": [
                    {"n": n, "url": s["url"], "text": chunk_text.get(s["chunk_id"], "")}
                    for n, s in enumerate(r["sources"], start=1)
                ],
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(rec, ensure_ascii=False) + "\n" for rec in records))
    return records


def cohen_kappa(a: list[int], b: list[int], labels=(0, 1, 2), weighted: bool = False) -> float:
    """Cohen's kappa; `weighted` uses quadratic weights for ordinal labels."""
    n = len(a)
    k = len(labels)
    index = {label: i for i, label in enumerate(labels)}

    def weight(i, j):
        return ((i - j) ** 2) / ((k - 1) ** 2) if weighted else float(i != j)

    observed = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b, strict=True):
        observed[index[x]][index[y]] += 1 / n
    pa, pb = Counter(a), Counter(b)
    expected = [[pa[labels[i]] / n * pb[labels[j]] / n for j in range(k)] for i in range(k)]
    do = sum(weight(i, j) * observed[i][j] for i in range(k) for j in range(k))
    de = sum(weight(i, j) * expected[i][j] for i in range(k) for j in range(k))
    return 1.0 if de == 0 else 1 - do / de


def agreement(run: str, judge_model: str, labels_path: Path, version: str = JUDGE_VERSION) -> dict:
    human = {r["id"]: r for r in map(json.loads, labels_path.read_text().splitlines())}
    stem = f"{run}.judge-{version}-{judge_model}"
    judged = {
        r["id"]: r
        for r in map(json.loads, (RESULTS_DIR / f"{stem}.jsonl").read_text().splitlines())
    }
    ids = sorted(set(human) & set(judged))
    h = [human[i]["score"] for i in ids]
    j = [judged[i]["judge"]["correctness"] for i in ids]
    disagreements = [
        {
            "id": i,
            "category": judged[i]["category"],
            "human": human[i]["score"],
            "judge": judged[i]["judge"]["correctness"],
            "human_note": human[i].get("note", ""),
            "judge_reason": judged[i]["judge"]["correctness_reason"],
        }
        for i in ids
        if human[i]["score"] != judged[i]["judge"]["correctness"]
    ]
    return {
        "run": run,
        "judge_model": judge_model,
        "judge_version": version,
        "labels": labels_path.name,
        "n": len(ids),
        "exact_agreement": round(sum(x == y for x, y in zip(h, j, strict=True)) / len(ids), 4),
        "cohen_kappa": round(cohen_kappa(h, j), 4),
        "weighted_kappa": round(cohen_kappa(h, j, weighted=True), 4),
        "human_mean": round(sum(h) / (2 * len(h)), 4),
        "judge_mean": round(sum(j) / (2 * len(j)), 4),
        "disagreements": disagreements,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--run", required=True)
    a = sub.add_parser("agreement")
    a.add_argument("--run", required=True)
    a.add_argument("--judge", required=True)
    a.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.cmd == "sample":
        out = CALIBRATION_DIR / f"{args.run}.sample.jsonl"
        records = export(args.run, out)
        print(f"{len(records)} items -> {out}")
        print(dict(Counter(r["category"] for r in records)))
    else:
        result = agreement(args.run, args.judge, args.labels)
        labels = args.labels.name.removesuffix(".jsonl").removeprefix(f"{args.run}.")
        out = CALIBRATION_DIR / f"{args.run}.agreement-{JUDGE_VERSION}-{args.judge}.{labels}.json"
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "disagreements"}, indent=2))
        print(f"{len(result['disagreements'])} disagreements -> {out}")


RUBRIC_FOR_HUMANS = RUBRIC  # the page shows the same rubric the judge gets

if __name__ == "__main__":
    main()
