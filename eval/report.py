"""Bootstrap confidence intervals and the experiment results table.

With ~100 items a 1-2 point difference can be noise, so every number gets a 95% bootstrap
interval, and comparisons between two runs resample the *same* items for both (paired),
which is far tighter than comparing two independent intervals.

Usage:
    uv run python -m eval.report                          # rebuild experiments/results.md
    uv run python -m eval.report --compare dev-baseline dev-e4-hybrid [--kind generation]

For false_refusal lower is better, so read its P(not better) the other way round.
"""

import argparse
import json
import random
import statistics
from pathlib import Path

from eval.judge import JUDGE_VERSION
from eval.run_generation import RESULTS_DIR as GENERATION_DIR
from eval.run_retrieval import RESULTS_DIR as RETRIEVAL_DIR

ROOT = Path(__file__).resolve().parent.parent
RESULTS_MD = ROOT / "experiments" / "results.md"
SAMPLES = 10_000
SEED = 7
JUDGE_MODEL = "claude-haiku-4-5"

RETRIEVAL_METRICS = ("recall@5", "recall@10", "mrr", "ndcg@10")


def bootstrap_ci(values: list[float], samples: int = SAMPLES, seed: int = SEED):
    """Mean and 95% percentile interval of the mean."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.fmean(rng.choices(values, k=n)) for _ in range(samples))
    return statistics.fmean(values), means[int(0.025 * samples)], means[int(0.975 * samples) - 1]


def paired_diff(a: dict[str, float], b: dict[str, float], samples=SAMPLES, seed=SEED) -> dict:
    """b - a over the items both runs scored: mean, 95% interval, and P(diff <= 0)."""
    ids = sorted(set(a) & set(b))
    diffs = [b[i] - a[i] for i in ids]
    rng = random.Random(seed)
    boots = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(samples))
    return {
        "n": len(ids),
        "diff": statistics.fmean(diffs),
        "lo": boots[int(0.025 * samples)],
        "hi": boots[int(0.975 * samples) - 1],
        "p_not_better": sum(x <= 0 for x in boots) / samples,
    }


# ---------------------------------------------------------------------------- loading


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def retrieval_scores(stem: str) -> dict[str, dict[str, float]]:
    """metric -> {item id: value}."""
    rows = jsonl(RETRIEVAL_DIR / f"{stem}.jsonl")
    return {m: {r["id"]: r["metrics"][m] for r in rows} for m in RETRIEVAL_METRICS}


def generation_scores(stem: str, judge_model=JUDGE_MODEL, version=JUDGE_VERSION):
    rows = {r["id"]: r for r in jsonl(GENERATION_DIR / f"{stem}.jsonl")}
    judged = jsonl(GENERATION_DIR / f"{stem}.judge-{version}-{judge_model}.jsonl")
    scores = {
        "correctness": {j["id"]: j["judge"]["correctness"] / 2 for j in judged},
        "fully_correct": {j["id"]: float(j["judge"]["correctness"] == 2) for j in judged},
        "faithfulness": {
            j["id"]: j["judge"]["faithfulness"]
            for j in judged
            if j["judge"]["faithfulness"] is not None
        },
        "citation_hit": {
            i: float(r["citation_hit"]) for i, r in rows.items() if r["citation_hit"] is not None
        },
        "unanswerable_refused": {
            i: float(not r["found"]) for i, r in rows.items() if r["category"] == "unanswerable"
        },
        "false_refusal": {i: float(not r["found"]) for i, r in rows.items() if r["answerable"]},
    }
    return scores, rows


# ---------------------------------------------------------------------------- table


def fmt(values: dict[str, float]) -> str:
    if not values:
        return "—"
    mean, lo, hi = bootstrap_ci(list(values.values()))
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def retrieval_table(stems: list[str]) -> list[str]:
    lines = [
        "| run | n | " + " | ".join(RETRIEVAL_METRICS) + " | commit |",
        "|---|---|" + "---|" * len(RETRIEVAL_METRICS) + "---|",
    ]
    for stem in stems:
        summary = json.loads((RETRIEVAL_DIR / f"{stem}.summary.json").read_text())
        scores = retrieval_scores(stem)
        cells = [fmt(scores[m]) for m in RETRIEVAL_METRICS]
        lines.append(
            f"| {stem} | {summary['n']} | " + " | ".join(cells) + f" | {summary['commit']} |"
        )
    return lines


GENERATION_METRICS = (
    "correctness",
    "fully_correct",
    "faithfulness",
    "citation_hit",
    "unanswerable_refused",
    "false_refusal",
)


def generation_table(stems: list[str]) -> list[str]:
    lines = [
        "| run | n | " + " | ".join(GENERATION_METRICS) + " | answer $ | commit |",
        "|---|---|" + "---|" * len(GENERATION_METRICS) + "---|---|",
    ]
    for stem in stems:
        summary = json.loads((GENERATION_DIR / f"{stem}.summary.json").read_text())
        scores, _ = generation_scores(stem)
        cells = [fmt(scores[m]) for m in GENERATION_METRICS]
        lines.append(
            f"| {stem} | {summary['n']} | "
            + " | ".join(cells)
            + f" | {summary['usd']:.2f} | {summary['commit']} |"
        )
    return lines


def runs(directory: Path, split: str = "dev") -> list[str]:
    return sorted(
        p.name.removesuffix(".summary.json")
        for p in directory.glob(f"{split}-*.summary.json")
        if ".judge-" not in p.name
    )


def build_markdown() -> str:
    parts = [
        "# Experiment results",
        "",
        "Generated by `uv run python -m eval.report`; do not edit by hand. Each cell is the "
        "mean over the dev split with a 95% bootstrap interval "
        f"({SAMPLES:,} resamples). Test-split numbers are reported once per milestone.",
        "",
        "## Retrieval (no LLM)",
        "",
        "Recall@k: share of an item's evidence facts found in the top k chunks. MRR and "
        "nDCG@10 reward finding them early.",
        "",
        *retrieval_table(runs(RETRIEVAL_DIR)),
        "",
        "## Generation",
        "",
        f"Correctness (0/1/2, shown as 0-1) and faithfulness are scored by the "
        f"`{JUDGE_MODEL}` judge, rubric `{JUDGE_VERSION}`. Citation hit: the answer cites a "
        "chunk containing an evidence fact. Refusal: share of unanswerable questions "
        "declined, and share of answerable ones wrongly declined (lower is better).",
        "",
        *generation_table(runs(GENERATION_DIR)),
        "",
    ]
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--compare", nargs=2, metavar=("BASE", "NEW"))
    parser.add_argument("--kind", choices=["retrieval", "generation"], default="retrieval")
    args = parser.parse_args(argv)
    if args.compare:
        base, new = args.compare
        if args.kind == "retrieval":
            a, b = retrieval_scores(base), retrieval_scores(new)
        else:
            a, b = generation_scores(base)[0], generation_scores(new)[0]
        for metric in a:
            d = paired_diff(a[metric], b[metric])
            print(
                f"{metric:22s} {d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] "
                f"n={d['n']} P(not better)={d['p_not_better']:.3f}"
            )
        return
    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_MD.write_text(build_markdown())
    print(f"wrote {RESULTS_MD}")


if __name__ == "__main__":
    main()
