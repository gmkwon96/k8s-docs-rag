"""LLM judge for generated answers: correctness against the reference, faithfulness to citations.

One judge call per item, returning JSON (structured outputs) with:
- correctness 0/1/2 against the reference answer, by category-specific rubric
- the answer's factual claims, each marked supported / partial / unsupported by the cited
  sources (faithfulness = share supported, partial counting half)

The judge sees the question, the Kubernetes version, the reference answer, the generated
answer, and the text of every source the answer cited. It runs through the Batches API
behind the dollar ledger (eval.batch). The judge model and JUDGE_VERSION are recorded with
every score: change either and old scores stop being comparable.

Usage:
    uv run python -m eval.judge --run dev-baseline --model claude-haiku-4-5 --max-usd 1
"""

import argparse
import json
import statistics
import time

import anthropic

from eval.batch import collect, submit
from eval.golden import load_split
from eval.run_generation import RESULTS_DIR
from ingest.chunk import CHUNKS_DIR
from rag.billing import cost, default_ledger
from rag.settings import get_settings

JUDGE_VERSION = "v1"
MAX_TOKENS = 3000

RUBRIC = """\
Score correctness 0, 1 or 2 against the reference answer. Judge substance only: do not
reward length, confidence, formatting or extra detail, and do not penalize a correct answer
for being short.

Answerable questions (fact, howto, version, multihop):
- 2: states everything the reference answer requires, and nothing relevant is wrong.
- 1: the main point is right, but a required element is missing or a minor detail is wrong.
- 0: wrong, misses the main point, contradicts the reference, or declines to answer.
For version questions the answer must hold for the stated Kubernetes version; an answer that
is right for another version but wrong for this one scores 0.

Unanswerable questions (the Kubernetes docs do not cover them):
- 2: says the documentation doesn't cover it and invents no specifics.
- 1: says it isn't covered but still offers unsupported specifics.
- 0: answers as if the docs covered it.

False-premise questions (the question assumes something untrue):
- 2: says the premise is wrong and gives the correct facts from the reference.
- 1: gives the correct facts without saying the premise is wrong.
- 0: goes along with the premise.

Then list the answer's factual claims (skip hedges and pleasantries; merge trivially
repeated claims) and mark each against the numbered sources only, not your own knowledge:
- "supported": a source states it
- "partial": a source supports part of it
- "unsupported": no source states it
An answer saying the docs don't cover the question has no factual claims unless it adds some.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "correctness": {"type": "integer", "enum": [0, 1, 2]},
        "correctness_reason": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "support": {"type": "string", "enum": ["supported", "partial", "unsupported"]},
                },
                "required": ["claim", "support"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["correctness", "correctness_reason", "claims"],
    "additionalProperties": False,
}

CATEGORY_LABEL = {
    "fact": "answerable (fact)",
    "howto": "answerable (how-to)",
    "version": "answerable (version-dependent)",
    "multihop": "answerable (multi-hop)",
    "unanswerable": "unanswerable",
    "false_premise": "false premise",
}


def load_chunks(versions) -> dict[str, str]:
    texts = {}
    for v in versions:
        for line in (CHUNKS_DIR / f"v{v}" / "chunks.jsonl").read_text().splitlines():
            c = json.loads(line)
            texts[c["chunk_id"]] = c["text"]
    return texts


def prompt(item, row: dict, chunk_text: dict[str, str]) -> str:
    sources = (
        "\n\n".join(
            f"[{n}] {s['url']}\n{chunk_text.get(s['chunk_id'], '(missing)')}"
            for n, s in enumerate(row["sources"], start=1)
        )
        or "(the answer cited no sources)"
    )
    return (
        f"Kubernetes version: {item.version}\n"
        f"Question type: {CATEGORY_LABEL[item.category]}\n\n"
        f"<question>\n{item.question}\n</question>\n\n"
        f"<reference_answer>\n{item.reference_answer}\n</reference_answer>\n\n"
        f"<answer>\n{row['answer']}\n</answer>\n\n"
        f"<sources>\n{sources}\n</sources>"
    )


def request(model: str, item, row: dict, chunk_text: dict[str, str]) -> dict:
    params = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": "You grade answers from a Kubernetes documentation assistant.\n\n" + RUBRIC,
        "messages": [{"role": "user", "content": prompt(item, row, chunk_text)}],
        "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}},
    }
    return params


def faithfulness(claims: list[dict]) -> float | None:
    if not claims:
        return None
    weight = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}
    return sum(weight[c["support"]] for c in claims) / len(claims)


def parse(message) -> dict:
    text = next(b.text for b in message.content if b.type == "text")
    data = json.loads(text)
    return {**data, "faithfulness": faithfulness(data["claims"])}


def mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 4) if values else None


def summarize(rows: list[dict]) -> dict:
    by_cat: dict[str, list] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)

    def block(rs):
        return {
            "n": len(rs),
            "correctness": mean(r["judge"]["correctness"] / 2 for r in rs),
            "fully_correct": mean(r["judge"]["correctness"] == 2 for r in rs),
            "faithfulness": mean(r["judge"]["faithfulness"] for r in rs),
        }

    return {
        "overall": block(rows),
        "by_category": {c: block(rs) for c, rs in sorted(by_cat.items())},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", required=True, help="generation run stem, e.g. dev-baseline")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-usd", type=float, required=True)
    parser.add_argument("--poll", type=float, default=30)
    args = parser.parse_args(argv)

    rows = [
        json.loads(line) for line in (RESULTS_DIR / f"{args.run}.jsonl").read_text().splitlines()
    ]
    split = args.run.split("-", 1)[0]
    items = {i.id: i for i in load_split(split)}
    chunk_text = load_chunks({r["version"] for r in rows})
    requests = [(r["id"], request(args.model, items[r["id"]], r, chunk_text)) for r in rows]

    client = anthropic.Anthropic(
        api_key=get_settings().anthropic_api_key, max_retries=0, timeout=120
    )
    ledger = default_ledger()
    stem = f"{args.run}.judge-{JUDGE_VERSION}-{args.model}"
    state_path = RESULTS_DIR / f"{stem}.batch.json"
    submit(client, ledger, requests, state_path, max_usd=args.max_usd)
    messages = collect(client, ledger, state_path, RESULTS_DIR / f"{stem}.raw.jsonl", args.poll)

    judged = []
    for r in rows:
        if r["id"] not in messages:
            continue
        message = messages[r["id"]]
        judged.append(
            {
                "id": r["id"],
                "category": r["category"],
                "judge": parse(message),
                "judge_usd": round(cost(args.model, message.usage, batch=True), 6),
            }
        )
    summary = {
        "run": args.run,
        "judge_model": args.model,
        "judge_version": JUDGE_VERSION,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "missing": [r["id"] for r in rows if r["id"] not in messages],
        "judge_usd": round(sum(j["judge_usd"] for j in judged), 4),
        **summarize(judged),
    }
    (RESULTS_DIR / f"{stem}.jsonl").write_text("".join(json.dumps(j) + "\n" for j in judged))
    (RESULTS_DIR / f"{stem}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"total Claude spend so far: ${ledger.spent():.4f} of ${ledger.budget_usd:.2f}")


if __name__ == "__main__":
    main()
