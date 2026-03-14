from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Optional


@dataclass
class ProviderKeyHealth:
    key_id: str
    available: bool
    backoff_until: float
    consecutive_failures: int
    last_error: str | None


class ProviderPool:
    """Shared provider key pool with backoff/circuit-breaker semantics.

    Keys are identified by a stable key_id (index label), not raw value.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._providers: dict[str, list[str]] = {}
        self._cursor: dict[str, int] = {}
        self._backoff_until: dict[tuple[str, str], float] = {}
        self._failures: dict[tuple[str, str], int] = {}
        self._last_error: dict[tuple[str, str], str] = {}

    def configure(self, provider: str, keys: list[str]) -> None:
        with self._lock:
            self._providers[provider] = [k for k in keys if k.strip()]
            self._cursor.setdefault(provider, 0)

    def configured(self, provider: str) -> bool:
        with self._lock:
            return len(self._providers.get(provider, [])) > 0

    def key_count(self, provider: str) -> int:
        with self._lock:
            return len(self._providers.get(provider, []))

    def available_key_count(self, provider: str) -> int:
        with self._lock:
            keys = self._providers.get(provider, [])
            now = time.time()
            return sum(1 for i in range(len(keys)) if now >= self._backoff_until.get((provider, self._key_id(i)), 0.0))

    def lease(self, provider: str) -> tuple[Optional[str], Optional[str]]:
        """Return (key, key_id)."""
        with self._lock:
            keys = self._providers.get(provider, [])
            if not keys:
                return None, None
            now = time.time()
            n = len(keys)
            start = self._cursor.get(provider, 0) % n
            for offset in range(n):
                idx = (start + offset) % n
                key_id = self._key_id(idx)
                if now >= self._backoff_until.get((provider, key_id), 0.0):
                    self._cursor[provider] = (idx + 1) % n
                    return keys[idx], key_id
            # All in backoff: return next anyway to prevent deadlock.
            idx = start
            self._cursor[provider] = (idx + 1) % n
            return keys[idx], self._key_id(idx)

    def report_success(self, provider: str, key_id: str) -> None:
        with self._lock:
            slot = (provider, key_id)
            self._failures[slot] = 0
            self._last_error.pop(slot, None)
            self._backoff_until.pop(slot, None)

    def report_failure(self, provider: str, key_id: str, status_code: int, error: str | None) -> None:
        with self._lock:
            slot = (provider, key_id)
            fails = self._failures.get(slot, 0) + 1
            self._failures[slot] = fails
            if error:
                self._last_error[slot] = error

            backoff = self._compute_backoff_seconds(status_code=status_code, error=error, failures=fails)
            if backoff > 0.0:
                self._backoff_until[slot] = max(self._backoff_until.get(slot, 0.0), time.time() + backoff)

    def provider_health(self, provider: str) -> dict[str, object]:
        with self._lock:
            keys = self._providers.get(provider, [])
            now = time.time()
            health: list[ProviderKeyHealth] = []
            for idx, _ in enumerate(keys):
                key_id = self._key_id(idx)
                slot = (provider, key_id)
                until = self._backoff_until.get(slot, 0.0)
                health.append(
                    ProviderKeyHealth(
                        key_id=key_id,
                        available=now >= until,
                        backoff_until=until,
                        consecutive_failures=self._failures.get(slot, 0),
                        last_error=self._last_error.get(slot),
                    )
                )
            return {
                "provider": provider,
                "configured": len(keys) > 0,
                "key_count": len(keys),
                "available_key_count": sum(1 for h in health if h.available),
                "keys": [
                    {
                        "key_id": h.key_id,
                        "available": h.available,
                        "backoff_until": h.backoff_until,
                        "consecutive_failures": h.consecutive_failures,
                        "last_error": h.last_error,
                    }
                    for h in health
                ],
            }

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {provider: self.provider_health(provider) for provider in sorted(self._providers.keys())}

    def _compute_backoff_seconds(self, *, status_code: int, error: str | None, failures: int) -> float:
        msg = (error or "").lower()
        if status_code in {401, 403}:
            return 300.0
        if status_code == 429:
            if "quota" in msg or "limit" in msg:
                return 300.0
            return min(120.0, 15.0 * failures)
        if status_code in {500, 502, 503, 504}:
            return min(60.0, 4.0 * failures)
        if status_code == 0 and ("timeout" in msg or "transport" in msg):
            return min(30.0, 3.0 * failures)
        return 0.0

    def _key_id(self, idx: int) -> str:
        return f"k{idx}"
