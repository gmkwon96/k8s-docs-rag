"""Generate query rewrites or HyDE passages for a golden-set split in one batch.

Writes eval/results/rewrites/<split>-<mode>.jsonl ({id, question, text}) for
eval.run_retrieval --query-mode to use. Batched, budget-checked and resumable like every
paid run (eval.batch).

Usage:
    uv run python -m eval.make_rewrites --mode rewrite --max-usd 0.2
"""

import argparse
import json
from pathlib import Path

import anthropic

from eval.batch import collect, submit
from eval.golden import load_split
from rag import rewrite
from rag.billing import default_ledger
from rag.settings import get_settings

ROOT = Path(__file__).resolve().parent.parent
REWRITES_DIR = ROOT / "eval" / "results" / "rewrites"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", choices=sorted(rewrite.SYSTEMS), required=True)
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--max-usd", type=float, required=True)
    parser.add_argument("--poll", type=float, default=30)
    args = parser.parse_args(argv)

    items = [i for i in load_split(args.split) if i.evidence]
    requests = [(i.id, rewrite.request(args.mode, i.question, i.version)) for i in items]
    client = anthropic.Anthropic(
        api_key=get_settings().anthropic_api_key, max_retries=0, timeout=120
    )
    ledger = default_ledger()
    stem = f"{args.split}-{args.mode}"
    state = REWRITES_DIR / f"{stem}.batch.json"
    submit(client, ledger, requests, state, max_usd=args.max_usd)
    messages = collect(client, ledger, state, REWRITES_DIR / f"{stem}.raw.jsonl", args.poll)
    rows = [
        {"id": i.id, "question": i.question, "text": rewrite.text_of(messages[i.id])}
        for i in items
        if i.id in messages
    ]
    (REWRITES_DIR / f"{stem}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    print(f"{len(rows)} {args.mode} texts -> {REWRITES_DIR / f'{stem}.jsonl'}")
    print(f"total Claude spend so far: ${ledger.spent():.4f} of ${ledger.budget_usd:.2f}")


if __name__ == "__main__":
    main()
