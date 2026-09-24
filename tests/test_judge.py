import json

import anthropic

from eval.golden import Item
from eval.judge import SCHEMA, faithfulness, parse, prompt, request, summarize
from tests.test_golden import item as base_item


def message(data):
    return anthropic.types.Message.model_validate(
        {
            "id": "m",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5",
            "content": [{"type": "text", "text": json.dumps(data)}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    )


def row(sources=()):
    return {
        "id": "dev-001",
        "version": "1.37",
        "answer": "30 seconds.[1]",
        "sources": [{"url": u, "chunk_id": c} for u, c in sources],
    }


def test_prompt_numbers_cited_sources_and_states_version_and_type():
    item = Item.model_validate(base_item())
    p = prompt(
        item, row([("https://kubernetes.io/docs/pods/", "c1")]), {"c1": "The default is 30."}
    )
    assert "Kubernetes version: 1.37" in p and "Question type: answerable (fact)" in p
    assert "[1] https://kubernetes.io/docs/pods/\nThe default is 30." in p
    assert "<reference_answer>\n30 seconds.\n</reference_answer>" in p
    assert "(the answer cited no sources)" in prompt(item, row(), {})


def test_request_uses_structured_output_without_citations():
    params = request("claude-haiku-4-5", Item.model_validate(base_item()), row(), {})
    assert params["output_config"]["format"] == {"type": "json_schema", "schema": SCHEMA}
    assert "citations" not in json.dumps(params["messages"])


def test_faithfulness_weights():
    assert faithfulness([]) is None
    claims = [{"support": s} for s in ("supported", "partial", "unsupported", "supported")]
    assert faithfulness(claims) == 2.5 / 4


def test_parse_and_summarize():
    data = {
        "correctness": 2,
        "correctness_reason": "ok",
        "claims": [{"claim": "30s", "support": "supported"}],
    }
    judged = parse(message(data))
    assert judged["faithfulness"] == 1.0
    rows = [
        {"category": "fact", "judge": judged},
        {"category": "fact", "judge": {**judged, "correctness": 1, "faithfulness": None}},
        {"category": "unanswerable", "judge": {**judged, "correctness": 0, "faithfulness": None}},
    ]
    s = summarize(rows)
    assert s["overall"]["correctness"] == round((1 + 0.5 + 0) / 3, 4)
    assert s["overall"]["fully_correct"] == round(1 / 3, 4)
    assert s["by_category"]["fact"]["faithfulness"] == 1.0
