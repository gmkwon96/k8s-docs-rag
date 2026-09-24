from types import SimpleNamespace

from rag.generate import MODEL, NOT_FOUND, document, parse, request
from rag.retrieve import Hit


def hit(i, states=()):
    return Hit(
        i,
        0.9,
        f"text {i}",
        "1.36",
        f"c{i}",
        f"https://v1-36.docs.kubernetes.io/docs/p{i}/#a",
        f"P{i}",
        [f"P{i}", "Section"],
        list(states),
    )


def text(t, cites=()):
    return SimpleNamespace(
        type="text",
        text=t,
        citations=[SimpleNamespace(document_index=i, cited_text=f"quote {i}") for i in cites]
        or None,
    )


def response(blocks, stop="end_turn"):
    u = SimpleNamespace(
        input_tokens=100, output_tokens=20, cache_read_input_tokens=0, cache_creation_input_tokens=0
    )
    return SimpleNamespace(content=blocks, stop_reason=stop, usage=u)


def test_document_carries_url_version_and_feature_state_in_context():
    d = document(hit(0, [{"feature_gate": "G", "state": "beta", "since": "1.33"}]))
    assert d["citations"] == {"enabled": True}
    assert d["title"] == "P0 > Section"
    assert "URL: https://v1-36.docs.kubernetes.io/docs/p0/#a" in d["context"]
    assert "G: beta since v1.33" in d["context"]


def test_request_shape():
    r = request("q?", "1.36", [hit(0), hit(1)], note="n")
    assert r["model"] == MODEL and r["max_tokens"] == 2000
    assert r["thinking"] == {"type": "adaptive"}
    [msg] = r["messages"]
    assert [b["type"] for b in msg["content"]] == ["document", "document", "text"]
    assert msg["content"][-1]["text"] == "Kubernetes version: 1.36\nNote: n\n\nQuestion: q?"


def test_citations_become_numbered_sources_in_order_of_use():
    hits = [hit(0), hit(1), hit(2)]
    blocks = [
        SimpleNamespace(type="thinking", thinking=""),
        text("Use rollout undo.", [2]),
        text(" It keeps history.", [0, 2]),
        text(" Done."),
    ]
    a = parse(response(blocks), hits, usd=0.01)
    assert a.text == "Use rollout undo.[1] It keeps history.[2][1] Done."
    assert [(s.number, s.hit.chunk_id) for s in a.sources] == [(1, "c2"), (2, "c0")]
    assert a.sources[0].cited_texts == ["quote 2", "quote 2"]
    assert a.found and a.usd == 0.01


def test_chunks_with_the_same_url_share_a_source_number():
    hits = [hit(0), hit(1)]
    hits[1].url = hits[0].url
    a = parse(response([text("A.", [0]), text(" B.", [1])]), hits, 0.0)
    assert a.text == "A.[1] B.[1]"
    assert [s.cited_texts for s in a.sources] == [["quote 0", "quote 1"]]


def test_not_found_answer_is_flagged():
    a = parse(response([text(f"{NOT_FOUND} The closest page is about Pods.")]), [hit(0)], 0.0)
    assert not a.found and a.sources == []


def test_refusal_and_truncation():
    assert parse(response([], stop="refusal"), [], 0.0).stop_reason == "refusal"
    a = parse(response([text("partial", [0])], stop="max_tokens"), [hit(0)], 0.0)
    assert a.text.endswith("(Answer truncated: output limit reached.)")


def test_document_title_is_never_empty():
    h = hit(0)
    h.heading_path = ["", "Synopsis"]
    assert document(h)["title"] == "Synopsis"
    h.heading_path = [""]
    assert document(h)["title"] == "/docs/p0/"
