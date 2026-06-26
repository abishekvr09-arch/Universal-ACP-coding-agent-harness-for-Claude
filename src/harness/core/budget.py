"""Iteration budget — bounds how many model turns a run may take.

Thread-safe (the loop runs tools on a pool; a budget may be shared across a
sub-agent tree). A "code-only turn" — one where the model called only
execute-tagged tools and produced no text — is refunded, so pure execution
churn doesn't burn the budget. When the budget is exhausted, the loop gets
exactly ONE grace call to deliver a closing answer.
"""

from __future__ import annotations

import threading


class IterationBudget:
    def __init__(self, max_iterations: int = 90) -> None:
        self._max = max_iterations
        self._used = 0
        self._grace_spent = False
        self._lock = threading.Lock()

    def consume(self) -> None:
        with self._lock:
            self._used += 1

    def refund(self) -> None:
        with self._lock:
            if self._used > 0:
                self._used -= 1

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    @property
    def exhausted(self) -> bool:
        with self._lock:
            return self._used >= self._max

    def take_grace(self) -> bool:
        """Return True exactly once after exhaustion, for the closing turn."""
        with self._lock:
            if self._grace_spent:
                return False
            self._grace_spent = True
            return True
