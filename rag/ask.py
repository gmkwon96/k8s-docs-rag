"""Ask a question from the command line.

Usage:
    uv run python -m rag.ask "How do I roll back a Deployment?"
    uv run python -m rag.ask "Is SidecarContainers beta?" --version 1.35 --show-chunks
"""

import argparse
import textwrap

import psycopg

from rag.billing import make_client
from rag.pipeline import Result, ask
from rag.settings import get_settings


def render(r: Result, show_chunks: bool = False) -> str:
    lines = []
    if r.version.note:
        lines += [f"Note: {r.version.note}", ""]
    lines += [f"[Kubernetes {r.version.version}]", "", r.answer.text, ""]
    if r.answer.sources:
        lines.append("Sources:")
        for s in r.answer.sources:
            lines.append(f"  [{s.number}] {' > '.join(s.hit.heading_path)}")
            lines.append(f"      {s.hit.url}")
    if show_chunks:
        lines += ["", "Retrieved chunks:"]
        for i, h in enumerate(r.hits):
            cited = "*" if any(s.hit is h for s in r.answer.sources) else " "
            lines.append(f" {cited}{i:2d}. {h.score:.3f}  {h.url}")
            lines.append(textwrap.indent(textwrap.shorten(h.text, 200), "        "))
    u = r.answer.usage
    lines += [
        "",
        f"{u['input_tokens']:,} in / {u['output_tokens']:,} out tokens, ${r.answer.usd:.4f}, "
        f"{r.latency['total']:.1f}s (retrieve {r.latency['retrieve'] * 1000:.0f}ms, "
        f"generate {r.latency['generate']:.1f}s)",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("question")
    parser.add_argument("--version", help="Kubernetes version, e.g. 1.36 (default: latest)")
    parser.add_argument("-k", type=int, default=8, help="chunks to retrieve")
    parser.add_argument("--show-chunks", action="store_true")
    args = parser.parse_args(argv)

    claude = make_client()
    with psycopg.connect(get_settings().database_url) as conn:
        result = ask(conn, claude, args.question, version=args.version, k=args.k)
    print(render(result, args.show_chunks))
    ledger = claude.ledger
    print(f"total Claude spend so far: ${ledger.spent():.4f} of ${ledger.budget_usd:.2f}")


if __name__ == "__main__":
    main()
