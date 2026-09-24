import json

import pytest
from fastapi.testclient import TestClient

from api import main
from tests.test_generate import hit, response, text


class FakeEmbedder:
    name = "voyage-4"

    def embed(self, texts, input_type):
        assert input_type == "query"
        return [[0.0] * 4 for _ in texts], 0


class FakeClaude:
    def __init__(self):
        self.calls = []

    def create(self, *, purpose, **params):
        self.calls.append(params)
        return response([text("Use kubectl rollout pause.", cites=[0])]), 0.0123


@pytest.fixture
def client(monkeypatch):
    hits = [hit(0), hit(1)]
    monkeypatch.setattr(
        main, "retrieve", lambda conn, q, v, version, config, model: hits[: config.k]
    )
    main.app.dependency_overrides[main.get_conn] = lambda: None
    main.app.dependency_overrides[main.get_embedder] = FakeEmbedder
    for name in ("retrieve_cache", "ask_cache"):
        monkeypatch.setattr(main, name, main.ResponseCache())
    for name in ("retrieve_limit", "ask_limit"):
        monkeypatch.setattr(main, name, main.RateLimiter(100))
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def events(body: str) -> list[tuple[str, dict]]:
    out = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_retrieve_returns_version_and_hits(client):
    r = client.post("/retrieve", json={"question": "How do I pause a rollout?", "k": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["version"]["version"]
    assert [h["chunk_id"] for h in body["hits"]] == ["c0"]


def test_question_is_validated(client):
    assert client.post("/retrieve", json={"question": "x"}).status_code == 422
    assert client.post("/retrieve", json={"question": "valid?", "version": "v1"}).status_code == 422
    assert client.post("/retrieve", json={"question": "valid?", "k": 99}).status_code == 422


def test_ask_is_refused_when_live_answers_are_off(client):
    main.app.dependency_overrides[main.get_claude] = lambda: None
    r = client.post("/ask", json={"question": "How do I pause a rollout?"})
    assert r.status_code == 403


def test_ask_streams_version_hits_and_answer(client):
    claude = FakeClaude()
    main.app.dependency_overrides[main.get_claude] = lambda: claude
    r = client.post("/ask", json={"question": "How do I pause a rollout?", "version": "1.36"})
    assert r.headers["content-type"].startswith("text/event-stream")
    names = [name for name, _ in events(r.text)]
    assert names == ["version", "hits", "answer"]
    answer = events(r.text)[-1][1]
    assert answer["text"] == "Use kubectl rollout pause.[1]"
    assert answer["sources"][0]["chunk_id"] == "c0"
    assert answer["sources"][0]["cited_texts"] == ["quote 0"]
    assert len(claude.calls) == 1


def test_ask_reports_errors_inside_the_stream(client):
    class Broken:
        def create(self, **_):
            raise RuntimeError("budget exceeded")

    main.app.dependency_overrides[main.get_claude] = Broken
    r = client.post("/ask", json={"question": "How do I pause a rollout?"})
    name, data = events(r.text)[-1]
    assert name == "error" and data["message"] == "budget exceeded"


def test_live_answers_are_off_by_default(monkeypatch):
    monkeypatch.delenv("LIVE_ANSWERS", raising=False)
    from rag.settings import Settings

    assert Settings(_env_file=None).live_answers is False


def test_examples_and_eval_results_read_result_files(tmp_path, monkeypatch):
    gen = tmp_path / "generation"
    ret = tmp_path / "retrieval"
    gen.mkdir()
    ret.mkdir()
    row = {
        "id": "dev-001",
        "category": "fact",
        "version": "1.37",
        "question": "q",
        "answer": "a",
        "found": True,
        "sources": [],
        "usage": {},
    }
    (gen / "dev-default.jsonl").write_text(json.dumps(row) + "\n")
    (gen / "dev-default.summary.json").write_text(json.dumps({"n": 1}))
    (gen / "dev-default.judge-v2-m.summary.json").write_text(json.dumps({"correctness": 1.0}))
    (ret / "dev-baseline.summary.json").write_text(json.dumps({"n": 1}))
    monkeypatch.setattr(main, "RESULTS_DIR", tmp_path)
    c = TestClient(main.app)
    ex = c.get("/examples").json()["examples"]
    assert ex == [{k: v for k, v in row.items() if k != "usage"}]
    results = c.get("/eval/results").json()
    assert [r["run"] for r in results["generation"]] == ["dev-default"]
    assert results["generation"][0]["judge"] == {"correctness": 1.0}
    assert [r["run"] for r in results["retrieval"]] == ["dev-baseline"]


def test_repeated_ask_is_served_from_cache_without_claude(client):
    claude = FakeClaude()
    main.app.dependency_overrides[main.get_claude] = lambda: claude
    body = {"question": "How do I pause a rollout?"}
    first = events(client.post("/ask", json=body).text)
    again = events(client.post("/ask", json={"question": "  how do I PAUSE a rollout? "}).text)
    assert len(claude.calls) == 1
    assert [n for n, _ in again] == ["version", "hits", "answer"]
    assert again[-1][1]["text"] == first[-1][1]["text"]
    assert first[-1][1]["usd"] > 0 and again[-1][1]["usd"] == 0


def test_rate_limit_returns_429(client, monkeypatch):
    monkeypatch.setattr(main, "retrieve_limit", main.RateLimiter(2))
    codes = [
        client.post("/retrieve", json={"question": f"question number {n}"}).status_code
        for n in range(3)
    ]
    assert codes == [200, 200, 429]
    # a cached question is still answered: it costs nothing
    assert client.post("/retrieve", json={"question": "question number 0"}).json()["cached"]


def test_rate_limiter_window_slides():
    now = [0.0]
    limiter = main.RateLimiter(1, window=60, clock=lambda: now[0])
    limiter.check("a")
    limiter.check("b")
    with pytest.raises(main.HTTPException):
        limiter.check("a")
    now[0] = 61
    limiter.check("a")


def test_cache_evicts_least_recently_used():
    cache = main.ResponseCache(size=2)
    cache.put(("a",), 1)
    cache.put(("b",), 2)
    cache.get(("a",))
    cache.put(("c",), 3)
    assert cache.get(("b",)) is None and cache.get(("a",)) == 1
