"""Generate answers for a golden-set split through the Batches API and score what needs no LLM.

For every item: retrieve its version's top chunks (query embeddings are cached), build the
same request `rag.ask` sends, and submit them all as one batch (50% off, checked against
the dollar budget first; see eval.batch). Rerunning resumes the same batch.

LLM-free metrics per run:
- refusal: unanswerable items should get the "couldn't find" answer, answerable ones
  shouldn't (false refusals); false-premise items are reported separately
- citations: whether the answer cites a chunk covering an evidence fact, and the strict
  share of cited sources that do (a cited chunk supporting the answer some other way
  counts against it, so read it as a lower bound)

Correctness and faithfulness need the judge (eval.judge).

Usage:
    uv run python -m eval.run_generation --name baseline --max-usd 2.5
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import anthropic
import psycopg

from eval.batch import collect, submit
from eval.golden import load_split
from eval.metrics_retrieval import covered
from eval.run_retrieval import git_commit, query_vectors
from ingest.embed import CACHE_DIR, Cache
from rag import generate
from rag.billing import cost, default_ledger
from rag.embedders import EMBEDDERS
from rag.retrieve import BASELINE, DEFAULT, Hit, RetrievalConfig, retrieve
from rag.settings import get_settings

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval" / "results" / "generation"


def build(conn, embedder, items, config: RetrievalConfig, cache: Cache):
    """-> [(item, hits, request params)] in item order."""
    vectors = query_vectors(embedder, [i.question for i in items], cache)
    out = []
    for item, vector in zip(items, vectors, strict=True):
        hits = retrieve(conn, item.question, vector, item.version, config, model=embedder.name)
        out.append((item, hits, generate.request(item.question, item.version, hits)))
    return out


def score_item(item, hits: list[Hit], message) -> dict:
    answer = generate.parse(message, hits, cost(generate.MODEL, message.usage, batch=True))
    facts = [[(p.url, p.quote) for p in e.passages()] for e in item.evidence]
    cited = [(s.hit.url, s.hit.text) for s in answer.sources]
    if facts and cited:
        per_source = covered(cited, facts)
        citation_hit = any(per_source)
        citation_precision = sum(bool(c) for c in per_source) / len(cited)
    else:
        citation_hit, citation_precision = (False, None) if facts else (None, None)
    return {
        "id": item.id,
        "category": item.category,
        "version": item.version,
        "answerable": item.answerable,
        "question": item.question,
        "answer": answer.text,
        "found": answer.found,
        "stop_reason": answer.stop_reason,
        "sources": [{"url": s.hit.url, "chunk_id": s.hit.chunk_id} for s in answer.sources],
        "retrieved": [h.chunk_id for h in hits],
        "citation_hit": citation_hit,
        "citation_precision": citation_precision,
        "usage": answer.usage,
        "usd": round(answer.usd, 6),
    }


def rate(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 4) if values else None


def summarize(rows: list[dict]) -> dict:
    unanswerable = [r for r in rows if r["category"] == "unanswerable"]
    answerable = [r for r in rows if r["answerable"]]
    premise = [r for r in rows if r["category"] == "false_premise"]
    return {
        "n": len(rows),
        "refusal": {
            "unanswerable_refused": rate(not r["found"] for r in unanswerable),
            "answerable_false_refusal": rate(not r["found"] for r in answerable),
            "false_premise_refused": rate(not r["found"] for r in premise),
        },
        "citations": {
            "answerable_with_citation": rate(bool(r["sources"]) for r in answerable),
            "citation_hit": rate(r["citation_hit"] for r in answerable),
            "citation_precision_strict": rate(r["citation_precision"] for r in answerable),
        },
        "stop_reasons": {
            s: sum(r["stop_reason"] == s for r in rows)
            for s in sorted({r["stop_reason"] for r in rows})
        },
        "usd": round(sum(r["usd"] for r in rows), 4),
        "tokens": {
            "input": sum(r["usage"]["input_tokens"] for r in rows),
            "output": sum(r["usage"]["output_tokens"] for r in rows),
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--name", default="baseline")
    parser.add_argument(
        "--retrieval",
        choices=["default", "baseline"],
        default="default",
        help="default: vector + rerank-3 (E5); baseline: vector top 8 (milestones 2-3)",
    )
    parser.add_argument("--poll", type=float, default=30, help="seconds between status checks")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--max-usd",
        type=float,
        required=True,
        help="refuse to submit if this batch's worst case is higher",
    )
    args = parser.parse_args(argv)

    key = get_settings().anthropic_api_key
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY is not set")
    client = anthropic.Anthropic(api_key=key, max_retries=0, timeout=120.0)
    ledger = default_ledger()
    embedder = EMBEDDERS["voyage-4"]
    items = load_split(args.split)
    stem = f"{args.split}-{args.name}"
    state_path = args.out_dir / f"{stem}.batch.json"
    raw_path = args.out_dir / f"{stem}.raw.jsonl"

    with psycopg.connect(get_settings().database_url) as conn:
        config = DEFAULT if args.retrieval == "default" else BASELINE
        built = build(conn, embedder, items, config, Cache(CACHE_DIR / "voyage-4-query.jsonl"))
    requests = [(item.id, params) for item, _, params in built]
    submit(client, ledger, requests, state_path, max_usd=args.max_usd)
    messages = collect(client, ledger, state_path, raw_path, args.poll)

    rows = [
        score_item(item, hits, messages[item.id]) for item, hits, _ in built if item.id in messages
    ]
    summary = {
        "name": args.name,
        "split": args.split,
        "config": {
            "model": generate.MODEL,
            "effort": generate.EFFORT,
            "max_tokens": generate.MAX_TOKENS,
            "retrieval": config.__dict__,
            "embedder": embedder.name,
        },
        "commit": git_commit(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "missing": [i.id for i in items if i.id not in messages],
        **summarize(rows),
    }
    with (args.out_dir / f"{stem}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (args.out_dir / f"{stem}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {k: summary[k] for k in ("n", "refusal", "citations", "stop_reasons", "usd")}, indent=2
        )
    )
    print(f"total Claude spend so far: ${ledger.spent():.4f} of ${ledger.budget_usd:.2f}")


if __name__ == "__main__":
    main()
