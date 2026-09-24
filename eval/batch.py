"""Run Claude requests through the Message Batches API behind the dollar ledger.

A batch can't be cancelled for free once it is processing, so the whole batch is checked
before submission: every request's input is counted (free count_tokens), its worst case
priced at max_tokens of output with the batch discount, and the sum is reserved in the
ledger. When results arrive each request's real cost is recorded and the reservation is
released, so the ledger never shows less than may have been spent.

State lives in a JSON file next to the run's outputs: rerunning after a crash resumes the
submitted batch instead of paying for a second one, and raw results are saved so parsing
can be redone offline.
"""

import json
import time
from pathlib import Path

import anthropic

from rag.billing import BATCH_DISCOUNT, DollarLedger, worst_case
from rag.usage import BudgetExceeded


def count_params(params: dict) -> dict:
    return {k: v for k, v in params.items() if k != "max_tokens"}


def count_tokens(client, params: dict, attempts: int = 6) -> int:
    """count_tokens is free but rate limited (requests per minute): back off and retry."""
    for attempt in range(attempts):
        try:
            return client.messages.count_tokens(**count_params(params)).input_tokens
        except anthropic.RateLimitError:
            if attempt == attempts - 1:
                raise
            time.sleep(min(60, 5 * 2**attempt))
    raise AssertionError("unreachable")


def estimate(client, requests: list[tuple[str, dict]]) -> tuple[float, int]:
    """Worst-case batch cost and total input tokens, from count_tokens."""
    usd, tokens = 0.0, 0
    for _, params in requests:
        n = count_tokens(client, params)
        tokens += n
        usd += worst_case(params["model"], n, params["max_tokens"]) * BATCH_DISCOUNT
    return usd, tokens


def submit(
    client,
    ledger: DollarLedger,
    requests: list[tuple[str, dict]],
    state_path: Path,
    log=print,
    max_usd: float | None = None,
) -> dict:
    """Submit once. An existing state file means the batch is already out: reuse it.

    `max_usd` caps this batch's worst case on top of the overall budget."""
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["status"] == "submitting":
            raise RuntimeError(
                f"{state_path}: a previous run stopped while submitting; check "
                "client.messages.batches.list() for that batch before submitting again"
            )
        log(f"resuming batch {state['batch_id']} ({state['status']})")
        return state
    for _, params in requests:
        ledger.check(params["model"], 0, 0)  # model has a price
    worst, tokens = estimate(client, requests)
    if max_usd is not None and worst > max_usd:
        raise BudgetExceeded(f"batch worst case ${worst:.4f} exceeds --max-usd ${max_usd:.2f}")
    ledger.check_amount(worst)
    log(
        f"{len(requests)} requests, {tokens:,} input tokens, worst case ${worst:.4f} "
        f"(spent so far ${ledger.spent():.4f} of ${ledger.budget_usd:.2f})"
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"status": "submitting"}) + "\n")
    ledger.adjust(worst, "batch reserved (worst case)")
    try:
        batch = client.messages.batches.create(
            requests=[{"custom_id": cid, "params": params} for cid, params in requests]
        )
    except anthropic.APIStatusError:
        # The API answered with an error, so no batch exists: undo the reservation.
        # (A timeout or dropped connection leaves the "submitting" state in place.)
        ledger.adjust(-worst, "batch rejected, reservation released")
        state_path.unlink()
        raise
    state = {
        "batch_id": batch.id,
        "status": "submitted",
        "reserved_usd": round(worst, 6),
        "input_tokens": tokens,
        "custom_ids": [cid for cid, _ in requests],
        "model": requests[0][1]["model"],
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    return state


def wait(client, batch_id: str, poll_seconds: float = 30, log=print):
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            return batch
        c = batch.request_counts
        log(f"  {batch.processing_status}: {c.processing} processing, {c.succeeded} done")
        time.sleep(poll_seconds)


def collect(
    client,
    ledger: DollarLedger,
    state_path: Path,
    raw_path: Path,
    poll_seconds: float = 30,
    log=print,
) -> dict[str, anthropic.types.Message]:
    """Wait for the batch, settle the ledger once, and return messages by custom_id."""
    state = json.loads(state_path.read_text())
    if state["status"] != "settled":
        wait(client, state["batch_id"], poll_seconds, log)
        rows, errors = [], {}
        for result in client.messages.batches.results(state["batch_id"]):
            if result.result.type == "succeeded":
                message = result.result.message
                ledger.record(
                    state["model"],
                    message.usage,
                    f"batch {state['batch_id']} {result.custom_id}",
                    batch=True,
                )
                rows.append({"custom_id": result.custom_id, "message": message.to_dict()})
            else:
                errors[result.custom_id] = result.result.type
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        ledger.adjust(-state["reserved_usd"], f"batch {state['batch_id']} released")
        state.update(status="settled", errors=errors, settled_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        log(f"settled: {len(rows)} succeeded, {len(errors)} not ({errors or 'none'})")
    return {
        row["custom_id"]: anthropic.types.Message.model_validate(row["message"])
        for row in map(json.loads, raw_path.read_text().splitlines())
    }
