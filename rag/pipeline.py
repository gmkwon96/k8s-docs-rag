"""Question -> version -> retrieval -> cited answer, with a query log."""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import psycopg

from rag.embedders import EMBEDDERS
from rag.generate import Answer, generate
from rag.query import VersionChoice, choose_version
from rag.retrieve import DEFAULT, Hit, RetrievalConfig, retrieve

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "data" / "logs" / "queries.jsonl"


@dataclass
class Result:
    question: str
    version: VersionChoice
    hits: list[Hit]
    answer: Answer
    latency: dict[str, float]  # seconds per stage


def ask(
    conn: psycopg.Connection,
    claude,
    question: str,
    *,
    version: str | None = None,
    k: int = 8,
    embedder=None,
    log_path: Path | None = LOG_PATH,
    config: RetrievalConfig | None = None,
) -> Result:
    embedder = embedder or EMBEDDERS["voyage-4"]
    choice = choose_version(question, version)
    t0 = time.monotonic()
    [query_vector], _ = embedder.embed([question], "query")
    t1 = time.monotonic()
    config = config or RetrievalConfig(**{**DEFAULT.__dict__, "k": k})
    hits = retrieve(conn, question, query_vector, choice.version, config, model=embedder.name)
    t2 = time.monotonic()
    answer = generate(claude, question, choice.version, hits, choice.note)
    t3 = time.monotonic()
    result = Result(
        question,
        choice,
        hits,
        answer,
        {"embed": t1 - t0, "retrieve": t2 - t1, "generate": t3 - t2, "total": t3 - t0},
    )
    if log_path:
        write_log(log_path, result)
    return result


def write_log(path: Path, r: Result) -> None:
    row = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "question": r.question,
        "version": asdict(r.version),
        "hits": [{"chunk_id": h.chunk_id, "score": round(h.score, 4)} for h in r.hits],
        "cited": [s.hit.chunk_id for s in r.answer.sources],
        "found": r.answer.found,
        "stop_reason": r.answer.stop_reason,
        "usage": r.answer.usage,
        "usd": round(r.answer.usd, 6),
        "latency": {k: round(v, 3) for k, v in r.latency.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
