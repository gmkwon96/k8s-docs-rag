import json
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from eval.batch import collect, submit
from eval.run_generation import score_item, summarize
from rag.billing import DollarLedger, worst_case
from rag.retrieve import Hit
from rag.usage import BudgetExceeded

MODEL = "claude-sonnet-5"


def message(text="Answer.", cites=(), stop="end_turn", inp=1000, out=200):
    citations = [
        {
            "type": "char_location",
            "cited_text": f"quote {i}",
            "document_index": i,
            "document_title": "t",
            "start_char_index": 0,
            "end_char_index": 5,
        }
        for i in cites
    ] or None
    return anthropic.types.Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": [{"type": "text", "text": text, "citations": citations}],
            "stop_reason": stop,
            "stop_sequence": None,
            "usage": {"input_tokens": inp, "output_tokens": out},
        }
    )


class FakeBatches:
    def __init__(self, results, create_error=None):
        self.results_by_id, self.create_error = results, create_error
        self.created, self.polls = [], 0

    def create(self, requests):
        if self.create_error:
            raise self.create_error
        self.created.append(requests)
        return SimpleNamespace(id="batch_1")

    def retrieve(self, batch_id):
        self.polls += 1
        status = "ended" if self.polls > 1 else "in_progress"
        counts = SimpleNamespace(processing=1, succeeded=0)
        return SimpleNamespace(processing_status=status, request_counts=counts)

    def results(self, batch_id):
        for cid, msg in self.results_by_id.items():
            kind = "succeeded" if msg else "errored"
            yield SimpleNamespace(custom_id=cid, result=SimpleNamespace(type=kind, message=msg))


def client(results, input_tokens=1000, create_error=None):
    batches = FakeBatches(results, create_error)
    messages = SimpleNamespace(
        count_tokens=lambda **p: SimpleNamespace(input_tokens=input_tokens), batches=batches
    )
    return SimpleNamespace(messages=messages)


def params():
    return {"model": MODEL, "max_tokens": 2000, "messages": [{"role": "user", "content": "q"}]}


def quiet(*_):
    pass


def test_submit_reserves_worst_case_and_collect_settles(tmp_path):
    ledger = DollarLedger(tmp_path / "a.jsonl", 5.0)
    c = client({"q1": message(), "q2": message(out=100)})
    state_path, raw = tmp_path / "s.json", tmp_path / "raw.jsonl"
    state = submit(c, ledger, [("q1", params()), ("q2", params())], state_path, log=quiet)
    worst = 2 * worst_case(MODEL, 1000, 2000) * 0.5
    assert state["reserved_usd"] == pytest.approx(worst)
    assert ledger.spent() == pytest.approx(worst)  # reserved while in flight

    messages = collect(c, ledger, state_path, raw, poll_seconds=0, log=quiet)
    actual = ((1000 * 2 + 200 * 10) + (1000 * 2 + 100 * 10)) / 1e6 * 0.5
    assert ledger.spent() == pytest.approx(actual)
    assert set(messages) == {"q1", "q2"}
    assert json.loads(state_path.read_text())["status"] == "settled"


def test_rerun_resumes_instead_of_resubmitting(tmp_path):
    ledger = DollarLedger(tmp_path / "a.jsonl", 5.0)
    c = client({"q1": message()})
    state_path, raw = tmp_path / "s.json", tmp_path / "raw.jsonl"
    submit(c, ledger, [("q1", params())], state_path, log=quiet)
    submit(c, ledger, [("q1", params())], state_path, log=quiet)
    assert len(c.messages.batches.created) == 1
    collect(c, ledger, state_path, raw, poll_seconds=0, log=quiet)
    spent = ledger.spent()
    again = collect(c, ledger, state_path, raw, poll_seconds=0, log=quiet)  # offline re-parse
    assert ledger.spent() == spent and set(again) == {"q1"}


def test_over_budget_batch_is_never_submitted(tmp_path):
    ledger = DollarLedger(tmp_path / "a.jsonl", 0.01)
    c = client({})
    with pytest.raises(BudgetExceeded):
        submit(c, ledger, [("q1", params())] * 3, tmp_path / "s.json", log=quiet)
    assert c.messages.batches.created == [] and ledger.spent() == 0
    assert not (tmp_path / "s.json").exists()


def test_per_batch_cap(tmp_path):
    c = client({})
    with pytest.raises(BudgetExceeded, match="--max-usd"):
        submit(
            c,
            DollarLedger(tmp_path / "a.jsonl", 5.0),
            [("q1", params())],
            tmp_path / "s.json",
            log=quiet,
            max_usd=0.001,
        )
    assert c.messages.batches.created == []


def test_rejected_batch_releases_its_reservation(tmp_path):
    ledger = DollarLedger(tmp_path / "a.jsonl", 5.0)
    response = httpx2.Response(400, request=httpx2.Request("POST", "https://api.anthropic.com"))
    error = anthropic.BadRequestError("bad", response=response, body=None)
    c = client({}, create_error=error)
    with pytest.raises(anthropic.BadRequestError):
        submit(c, ledger, [("q1", params())], tmp_path / "s.json", log=quiet)
    assert ledger.spent() == pytest.approx(0) and not (tmp_path / "s.json").exists()


def test_interrupted_submission_blocks_a_second_one(tmp_path):
    state_path = tmp_path / "s.json"
    state_path.write_text(json.dumps({"status": "submitting"}))
    with pytest.raises(RuntimeError, match="stopped while submitting"):
        submit(
            client({}),
            DollarLedger(tmp_path / "a.jsonl", 5.0),
            [("q1", params())],
            state_path,
            log=quiet,
        )


def test_failed_requests_are_not_billed(tmp_path):
    ledger = DollarLedger(tmp_path / "a.jsonl", 5.0)
    c = client({"q1": message(), "q2": None})
    state_path, raw = tmp_path / "s.json", tmp_path / "raw.jsonl"
    submit(c, ledger, [("q1", params()), ("q2", params())], state_path, log=quiet)
    messages = collect(c, ledger, state_path, raw, poll_seconds=0, log=quiet)
    assert set(messages) == {"q1"}
    assert json.loads(state_path.read_text())["errors"] == {"q2": "errored"}


def hit(i, url, text):
    return Hit(i, 0.9, text, "1.37", f"c{i}", url, "T", ["T"], [])


def item(**kw):
    from eval.golden import Item
    from tests.test_golden import item as base

    return Item.model_validate(base(**kw))


def test_score_item_citations_and_refusal():
    good = hit(
        0,
        "https://kubernetes.io/docs/pods/#pod-termination-flow",
        "the default terminationGracePeriodSeconds setting is 30 seconds",
    )
    other = hit(1, "https://kubernetes.io/docs/other/", "unrelated")
    row = score_item(item(), [good, other], message("30 seconds.", cites=[0, 1]))
    assert row["found"] and row["citation_hit"] is True
    assert row["citation_precision"] == 0.5

    refused = score_item(
        item(
            id="dev-002",
            question="Price of EKS?",
            category="unanswerable",
            answerable=False,
            evidence=[],
        ),
        [other],
        message("I couldn't find this in the Kubernetes documentation."),
    )
    assert refused["found"] is False and refused["citation_hit"] is None
    s = summarize([row, refused])
    assert s["refusal"]["unanswerable_refused"] == 1.0
    assert s["refusal"]["answerable_false_refusal"] == 0.0
    assert s["citations"]["citation_hit"] == 1.0


def test_count_tokens_retries_rate_limits(monkeypatch):
    import eval.batch

    monkeypatch.setattr(eval.batch.time, "sleep", lambda s: None)
    response = httpx2.Response(429, request=httpx2.Request("POST", "https://api.anthropic.com"))
    calls = []

    def count(**p):
        calls.append(1)
        if len(calls) < 3:
            raise anthropic.RateLimitError("slow down", response=response, body=None)
        return SimpleNamespace(input_tokens=42)

    c = SimpleNamespace(messages=SimpleNamespace(count_tokens=count))
    assert eval.batch.count_tokens(c, params()) == 42 and len(calls) == 3
