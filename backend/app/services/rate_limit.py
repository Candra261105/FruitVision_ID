from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable
from threading import Lock
import time


class SlidingWindowRateLimiter:
    def __init__(
        self,
        max_requests: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_requests = max(1, max_requests)
        self.window_seconds = window_seconds
        self.clock = clock
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str) -> bool:
        now = self.clock()
        with self._lock:
            recent = self._requests[key]
            while recent and now - recent[0] >= self.window_seconds:
                recent.popleft()
            if len(recent) >= self.max_requests:
                return False
            recent.append(now)
            return True
