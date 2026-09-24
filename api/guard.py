"""Per-client rate limits and a response cache for the paid endpoints.

Both are in-process: enough for one API instance (the demo), not shared across replicas.
The cache makes a repeated question free: /retrieve skips Voyage, /ask skips Voyage and
Claude and replays the stored events.
"""

import re
import threading
import time
from collections import OrderedDict, defaultdict, deque

from fastapi import HTTPException, Request


class RateLimiter:
    """At most `limit` requests per client in any `window` seconds (sliding window)."""

    def __init__(self, limit: int, window: float = 60.0, clock=time.monotonic):
        self.limit, self.window, self.clock = limit, window, clock
        self.hits: dict[str, deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, client: str) -> None:
        now = self.clock()
        with self.lock:
            q = self.hits[client]
            while q and q[0] <= now - self.window:
                q.popleft()
            if len(q) >= self.limit:
                retry = int(q[0] + self.window - now) + 1
                raise HTTPException(
                    429, "Too many requests; try again shortly.", {"Retry-After": str(retry)}
                )
            q.append(now)


def client_id(request: Request) -> str:
    # Behind one trusted proxy the client is the first X-Forwarded-For entry.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class ResponseCache:
    """LRU of responses keyed by the normalized question and its parameters."""

    def __init__(self, size: int = 512):
        self.size = size
        self.items: OrderedDict[tuple, object] = OrderedDict()
        self.lock = threading.Lock()

    @staticmethod
    def key(*parts) -> tuple:
        question, *rest = parts
        return (re.sub(r"\s+", " ", question.strip().lower()), *rest)

    def get(self, key: tuple):
        with self.lock:
            if key not in self.items:
                return None
            self.items.move_to_end(key)
            return self.items[key]

    def put(self, key: tuple, value) -> None:
        with self.lock:
            self.items[key] = value
            self.items.move_to_end(key)
            while len(self.items) > self.size:
                self.items.popitem(last=False)
