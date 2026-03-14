from __future__ import annotations

from chat_api.services.provider_pool import ProviderPool


def test_provider_pool_rotates_keys_deterministically() -> None:
    pool = ProviderPool()
    pool.configure("gemini", ["kA", "kB", "kC"])
    leased = [pool.lease("gemini")[1] for _ in range(5)]
    assert leased == ["k0", "k1", "k2", "k0", "k1"]


def test_provider_pool_skips_backoff_key_when_available() -> None:
    pool = ProviderPool()
    pool.configure("openai", ["x", "y"])
    key, key_id = pool.lease("openai")
    assert key_id == "k0"
    pool.report_failure("openai", key_id, status_code=429, error="rate limit")
    # Next lease should avoid k0 while in backoff and use k1.
    _, next_key_id = pool.lease("openai")
    assert next_key_id == "k1"
    health = pool.provider_health("openai")
    assert health["configured"] is True
    assert health["key_count"] == 2
    assert health["available_key_count"] <= 1
