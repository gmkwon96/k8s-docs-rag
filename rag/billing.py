"""Dollar ledger for Claude API calls, with a hard cap checked before every request.

The worst case of a request is known before it is sent: its input tokens (counted with the
free count_tokens endpoint) priced at the highest input rate, plus max_tokens of output
(thinking included) at the output rate. A request is refused if spent + worst case would
exceed ANTHROPIC_BUDGET_USD, so the total can't pass the cap even if every call maxes out.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import anthropic

from rag.usage import USAGE_DIR, BudgetExceeded


@dataclass(frozen=True)
class Price:
    """USD per million tokens (Claude API, checked 2026-09-23 on the models overview page)."""

    input: float
    output: float
    cache_write: float  # 5-minute cache write, 1.25x input
    cache_read: float


PRICES = {
    "claude-sonnet-5": Price(input=2.00, output=10.00, cache_write=2.50, cache_read=0.20),
    "claude-haiku-4-5": Price(input=1.00, output=5.00, cache_write=1.25, cache_read=0.10),
    "claude-opus-5-5": Price(input=4.00, output=20.00, cache_write=5.00, cache_read=0.20),
}
MAX_INPUT_TOKENS = 200_000  # stay below long-context pricing tiers
BATCH_DISCOUNT = 0.5  # Message Batches API: 50% off all token usage


def cost(model: str, usage, batch: bool = False) -> float:
    p = PRICES[model]
    usd = (
        usage.input_tokens * p.input
        + (usage.cache_creation_input_tokens or 0) * p.cache_write
        + (usage.cache_read_input_tokens or 0) * p.cache_read
        + usage.output_tokens * p.output
    ) / 1e6
    return usd * BATCH_DISCOUNT if batch else usd


def worst_case(model: str, input_tokens: int, max_tokens: int) -> float:
    p = PRICES[model]
    return (input_tokens * max(p.input, p.cache_write) + max_tokens * p.output) / 1e6


@dataclass
class DollarLedger:
    path: Path
    budget_usd: float

    def spent(self) -> float:
        if not self.path.exists():
            return 0.0
        return sum(json.loads(line)["usd"] for line in self.path.read_text().splitlines())

    def check(self, model: str, input_tokens: int, max_tokens: int) -> float:
        if model not in PRICES:
            raise BudgetExceeded(f"no price for {model}; add it to rag.billing.PRICES first")
        if input_tokens > MAX_INPUT_TOKENS:
            raise BudgetExceeded(f"refusing request: {input_tokens:,} input tokens")
        worst = worst_case(model, input_tokens, max_tokens)
        self.check_amount(worst)
        return worst

    def check_amount(self, usd: float) -> None:
        spent = self.spent()
        if spent + usd > self.budget_usd:
            raise BudgetExceeded(
                f"refusing request: ${spent:.4f} spent + up to ${usd:.4f} "
                f"> budget ${self.budget_usd:.2f} (ANTHROPIC_BUDGET_USD)"
            )

    def adjust(self, usd: float, purpose: str) -> None:
        """A reservation (positive) or its release (negative), e.g. around a batch."""
        self._append(
            {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "usd": round(usd, 6), "purpose": purpose}
        )

    def _append(self, row: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def record_unconfirmed(self, model: str, usd: float, purpose: str) -> None:
        row = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": model,
            "usd": round(usd, 6),
            "purpose": f"{purpose} (unconfirmed: response lost, booked worst case)",
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def record(self, model: str, usage, purpose: str, batch: bool = False) -> float:
        usd = cost(model, usage, batch)
        row = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": model,
            "input_tokens": usage.input_tokens,
            "cache_write_tokens": usage.cache_creation_input_tokens or 0,
            "cache_read_tokens": usage.cache_read_input_tokens or 0,
            "output_tokens": usage.output_tokens,
            "usd": round(usd, 6),
            "purpose": purpose,
            **({"batch": True} if batch else {}),
        }
        self._append(row)
        return usd


def default_ledger() -> DollarLedger:
    from rag.settings import get_settings

    return DollarLedger(USAGE_DIR / "anthropic.jsonl", get_settings().anthropic_budget_usd)


class GuardedClaude:
    """messages.create behind the ledger: count, check worst case, send, record."""

    def __init__(self, client, ledger: DollarLedger):
        self.client = client
        self.ledger = ledger

    def create(self, *, purpose: str, **params):
        count_params = {k: v for k, v in params.items() if k != "max_tokens"}
        input_tokens = self.client.messages.count_tokens(**count_params).input_tokens
        worst = self.ledger.check(params["model"], input_tokens, params["max_tokens"])
        try:
            response = self.client.messages.create(**params)
        except anthropic.APITimeoutError, anthropic.APIConnectionError:
            # The server may have finished (and billed) a request whose response we lost:
            # book the worst case so the cap stays safe.
            self.ledger.record_unconfirmed(params["model"], worst, purpose)
            raise
        usd = self.ledger.record(params["model"], response.usage, purpose)
        return response, usd


def make_client() -> GuardedClaude:
    from rag.settings import get_settings

    key = get_settings().anthropic_api_key
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set (see .env.example)")
    # No automatic retries: a retried request after a lost response could bill twice.
    client = anthropic.Anthropic(api_key=key, max_retries=0, timeout=120.0)
    return GuardedClaude(client, default_ledger())
