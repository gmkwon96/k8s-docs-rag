"""Local ledger of billed API tokens, with a hard budget checked before every request.

Voyage gives each account its first 200M voyage-4 family tokens free (not for the Batch
API) but offers no API to read the remaining balance or cap spending, so we keep our own
count and refuse any request that could push it past VOYAGE_TOKEN_BUDGET.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
USAGE_DIR = ROOT / "data" / "usage"


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Ledger:
    path: Path
    budget: int

    def total(self) -> int:
        if not self.path.exists():
            return 0
        return sum(json.loads(line)["tokens"] for line in self.path.read_text().splitlines())

    def check(self, estimated: int) -> None:
        used = self.total()
        if used + estimated > self.budget:
            raise BudgetExceeded(
                f"refusing request: {used:,} tokens used + ~{estimated:,} estimated "
                f"> budget {self.budget:,} (VOYAGE_TOKEN_BUDGET)"
            )

    def record(self, model: str, tokens: int, purpose: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": model, "tokens": tokens}
        with self.path.open("a") as f:
            f.write(json.dumps({**row, "purpose": purpose}) + "\n")
