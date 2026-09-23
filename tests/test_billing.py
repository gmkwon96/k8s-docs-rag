from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from rag.billing import DollarLedger, GuardedClaude, cost, worst_case
from rag.usage import BudgetExceeded


def usage(inp=0, out=0, cw=0, cr=0):
    return SimpleNamespace(
        input_tokens=inp,
        output_tokens=out,
        cache_creation_input_tokens=cw,
        cache_read_input_tokens=cr,
    )


class FakeMessages:
    def __init__(self, input_tokens=1000, response=None, error=None):
        self.input_tokens, self.response, self.error = input_tokens, response, error
        self.created = []

    def count_tokens(self, **params):
        assert "max_tokens" not in params
        return SimpleNamespace(input_tokens=self.input_tokens)

    def create(self, **params):
        self.created.append(params)
        if self.error:
            raise self.error
        return self.response


def guarded(tmp_path, budget=1.0, **kw):
    messages = FakeMessages(**kw)
    return GuardedClaude(
        SimpleNamespace(messages=messages), DollarLedger(tmp_path / "a.jsonl", budget)
    ), messages


def test_cost_uses_sonnet_5_prices():
    # $2/M input, $10/M output, $2.50/M cache write, $0.20/M cache read
    assert cost("claude-sonnet-5", usage(1_000_000, 100_000, 200_000, 500_000)) == pytest.approx(
        2.0 + 1.0 + 0.5 + 0.1
    )


def test_worst_case_prices_input_at_cache_write_rate_and_full_max_tokens():
    assert worst_case("claude-sonnet-5", 4000, 2000) == pytest.approx(4000 * 2.5e-6 + 2000 * 10e-6)


def test_request_is_sent_and_recorded(tmp_path):
    claude, messages = guarded(tmp_path, response=SimpleNamespace(usage=usage(1000, 500)))
    _, usd = claude.create(purpose="t", model="claude-sonnet-5", max_tokens=2000, messages=[])
    assert usd == pytest.approx(1000 * 2e-6 + 500 * 10e-6)
    assert claude.ledger.spent() == pytest.approx(usd)
    assert len(messages.created) == 1


def test_request_that_could_exceed_budget_is_never_sent(tmp_path):
    claude, messages = guarded(tmp_path, budget=0.02, input_tokens=1000)
    claude.ledger.record("claude-sonnet-5", usage(5000, 0), "earlier")  # $0.01 spent
    with pytest.raises(BudgetExceeded, match="refusing request"):
        # worst case 1000*2.5e-6 + 2000*1e-5 = $0.0225 > $0.01 remaining
        claude.create(purpose="t", model="claude-sonnet-5", max_tokens=2000, messages=[])
    assert messages.created == []


def test_unknown_model_is_refused(tmp_path):
    claude, messages = guarded(tmp_path)
    with pytest.raises(BudgetExceeded, match="no price"):
        claude.create(purpose="t", model="claude-new", max_tokens=10, messages=[])
    assert messages.created == []


def test_lost_response_books_worst_case(tmp_path):
    error = anthropic.APITimeoutError(request=httpx2.Request("POST", "https://api.anthropic.com"))
    claude, _ = guarded(tmp_path, error=error, input_tokens=1000)
    with pytest.raises(anthropic.APITimeoutError):
        claude.create(purpose="t", model="claude-sonnet-5", max_tokens=2000, messages=[])
    assert claude.ledger.spent() == pytest.approx(worst_case("claude-sonnet-5", 1000, 2000))
