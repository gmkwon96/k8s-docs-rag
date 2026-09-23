import json

import pytest

from eval.golden import Item, contains_quote, load
from eval.validate import Docs, validate

PAGE_TEXT = (
    "Intro.\n\n## Pod termination flow\n\nThe default terminationGracePeriodSeconds\n"
    "setting is 30 seconds.\n\n## Other\n\nMore text."
)
GATE_TEXT = "# Feature gate: MyGate\n\nIn Kubernetes v1.37, the `MyGate` feature gate is beta."


def item(**overrides):
    base = {
        "id": "dev-001",
        "split": "dev",
        "question": "What is the default grace period?",
        "version": "1.37",
        "category": "fact",
        "answerable": True,
        "reference_answer": "30 seconds.",
        "evidence": [
            {
                "url": "/docs/pods/#pod-termination-flow",
                "quote": "default terminationGracePeriodSeconds setting is 30 seconds",
            }
        ],
        "origin": "manual",
    }
    return {**base, **overrides}


@pytest.fixture
def docs(tmp_path):
    clean = tmp_path / "clean" / "v1.37"
    clean.mkdir(parents=True)
    base = "https://kubernetes.io"
    records = [
        {
            "id": "v1.37:/docs/pods/",
            "url": f"{base}/docs/pods/",
            "title": "Pods",
            "text": PAGE_TEXT,
        },
        {
            "id": "v1.37:/docs/gates/#MyGate",
            "url": f"{base}/docs/gates/#MyGate",
            "title": "MyGate",
            "text": GATE_TEXT,
        },
    ]
    (clean / "pages.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    chunks = tmp_path / "chunks" / "v1.37"
    chunks.mkdir(parents=True)
    chunk_rows = [
        {
            "record_id": "v1.37:/docs/pods/",
            "text": "Intro.\n\n## Pod termination flow\n\nThe default",
        },
        {
            "record_id": "v1.37:/docs/pods/",
            "text": "terminationGracePeriodSeconds setting is 30 seconds.",
        },
    ]
    (chunks / "chunks.jsonl").write_text("".join(json.dumps(c) + "\n" for c in chunk_rows))
    return Docs(tmp_path / "clean", tmp_path / "chunks")


def write(tmp_path, dev=(), test=()):
    d = tmp_path / "dataset"
    d.mkdir(exist_ok=True)
    (d / "dev.jsonl").write_text("".join(json.dumps(i) + "\n" for i in dev))
    (d / "test.jsonl").write_text("".join(json.dumps(i) + "\n" for i in test))
    return d


def test_quote_matching_ignores_whitespace():
    assert contains_quote("a  b\nc", "a b c")
    assert not contains_quote("a b c", "a c")


def test_schema_rules():
    Item.model_validate(item())
    with pytest.raises(ValueError, match="need at least one evidence"):
        Item.model_validate(item(evidence=[]))
    with pytest.raises(ValueError, match="answerable=false"):
        Item.model_validate(item(category="unanswerable"))
    with pytest.raises(ValueError, match="does not match split"):
        Item.model_validate(item(id="test-001"))
    Item.model_validate(item(category="unanswerable", answerable=False, evidence=[]))
    Item.model_validate(item(category="false_premise", answerable=False))


def test_valid_items_pass_and_boundary_quotes_warn(tmp_path, docs):
    gate = item(
        id="dev-002",
        question="Is MyGate beta in 1.37?",
        category="version",
        evidence=[{"url": "/docs/gates/#MyGate", "quote": "the `MyGate` feature gate is beta"}],
    )
    report = validate(write(tmp_path, dev=[item(), gate]), docs)
    assert report["errors"] == []
    # the page quote is split across the two chunks above
    assert len(report["warnings"]) == 1 and "straddles" in report["warnings"][0]
    assert report["mix"]["dev"]["fact"]["n"] == 1


def test_evidence_errors(tmp_path, docs):
    bad = [
        item(id="dev-001", evidence=[{"url": "/docs/missing/", "quote": "anything at all here"}]),
        item(
            id="dev-002",
            question="q two here?",
            evidence=[{"url": "/docs/pods/", "quote": "not in the page text"}],
        ),
        item(
            id="dev-003",
            question="q three here?",
            evidence=[{"url": "/docs/pods/#nope", "quote": "More text. More"}],
        ),
        item(
            id="dev-004",
            question="q four here?",
            evidence=[{"url": "/docs/gates/#Other", "quote": "feature gate is beta"}],
        ),
        item(id="dev-005", question="q five here?", version="1.30"),
    ]
    bad[2]["evidence"][0]["quote"] = "The default terminationGracePeriodSeconds"
    errors = validate(write(tmp_path, dev=bad), docs)["errors"]
    assert any("no page at /docs/missing/" in e for e in errors)
    assert any("dev-002: quote not found" in e for e in errors)
    assert any("#nope is not a section" in e for e in errors)
    assert any("no record for anchor #Other" in e for e in errors)
    assert not any("dev-003: quote is on" in e for e in errors)  # #nope reported instead
    assert any("version 1.30 is not indexed" in e for e in errors)


def test_quote_must_be_in_the_anchored_section(tmp_path, docs):
    wrong = item(evidence=[{"url": "/docs/pods/#other", "quote": "setting is 30 seconds"}])
    errors = validate(write(tmp_path, dev=[wrong]), docs)["errors"]
    assert errors == ["dev-001: quote is on /docs/pods/ but not in section #other"]


def test_leakage_between_dev_and_test_is_an_error(tmp_path, docs):
    t = item(id="test-001", split="test", question="  what is the DEFAULT grace period? ")
    errors = validate(write(tmp_path, dev=[item()], test=[t]), docs)["errors"]
    assert errors == ["same question in dev-001, test-001"]


def test_load_reports_line_numbers(tmp_path):
    path = tmp_path / "dev.jsonl"
    path.write_text(json.dumps(item()) + "\n" + json.dumps(item(category="nope")) + "\n")
    with pytest.raises(ValueError, match="dev.jsonl:2"):
        load(path)
