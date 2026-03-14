from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import threading
from typing import Any


@dataclass
class DecisionTraceStore:
    max_items: int = 500

    def __post_init__(self) -> None:
        self._lock = threading.RLock()
        self._store: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def put(self, run_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            if run_id in self._store:
                self._store.pop(run_id, None)
            self._store[run_id] = payload
            while len(self._store) > self.max_items:
                self._store.popitem(last=False)

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            payload = self._store.get(run_id)
            if payload is None:
                return None
            return dict(payload)
