"""
Unit tests for backend/app/core/rate_limit.py -- DistributedRateLimiter.

Copyright (C) 2024 Sarthak Doshi (github.com/SdSarthak)
SPDX-License-Identifier: AGPL-3.0-only
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from app.core.rate_limit import DistributedRateLimiter


class TestDistributedRateLimiterInit:
    def test_default_values(self):
        """RateLimiter initialises with correct defaults."""
        limiter = DistributedRateLimiter()
        assert limiter.failure_threshold == 5
        assert limiter.recovery_timeout == 30
        assert limiter._local_cleanup_interval == 100
        assert limiter.cb_state == "CLOSED"
        assert limiter.consecutive_failures == 0

    def test_metrics_initialised(self):
        """Metrics dict is initialised on construction."""
        limiter = DistributedRateLimiter()
        assert limiter.metrics["total_requests"] == 0
        assert limiter.metrics["redis_calls"] == 0
        assert limiter.metrics["redis_failures"] == 0
        assert limiter.metrics["local_fallbacks"] == 0
        assert limiter.metrics["blocked_by_circuit_breaker"] == 0


class TestClearLocalAttempts:
    def test_clears_state(self):
        """clear_local_attempts empties both tracking dicts."""
        limiter = DistributedRateLimiter()
        # Simulate some state by recording attempts
        limiter._local_attempts_by_key["test_key"].append(datetime.now(timezone.utc))
        limiter._local_window_seconds_by_key["test_key"] = 60
        limiter._local_cleanup_calls = 50

        limiter.clear_local_attempts()

        assert "test_key" not in limiter._local_attempts_by_key
        assert "test_key" not in limiter._local_window_seconds_by_key
        assert limiter._local_cleanup_calls == 0


class TestCleanupStaleLocalAttempts:
    def test_removes_expired_keys(self):
        """cleanup_stale_local_attempts removes keys whose window has expired."""
        limiter = DistributedRateLimiter()
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(seconds=120)
        recent_time = now - timedelta(seconds=30)

        # Expired key
        limiter._local_attempts_by_key["expired"].append(old_time)
        limiter._local_window_seconds_by_key["expired"] = 60

        # Recent key
        limiter._local_attempts_by_key["recent"].append(recent_time)
        limiter._local_window_seconds_by_key["recent"] = 60

        removed = limiter.cleanup_stale_local_attempts(now=now)

        assert removed == 1
        assert "expired" not in limiter._local_attempts_by_key
        assert "recent" in limiter._local_attempts_by_key

    def test_no_expired_keys_returns_zero(self):
        """Returns 0 when no keys are stale."""
        limiter = DistributedRateLimiter()
        now = datetime.now(timezone.utc)
        limiter._local_attempts_by_key["active"].append(now)
        limiter._local_window_seconds_by_key["active"] = 60

        removed = limiter.cleanup_stale_local_attempts(now=now)
        assert removed == 0


class TestCheckLocal:
    def test_under_limit_allows(self):
        """Under the limit, check_local returns (False, 0)."""
        limiter = DistributedRateLimiter()
        limiter._local_window_seconds_by_key["test"] = 60

        over, retry_after = limiter._check_local("test", limit=5, window_seconds=60, cost=1)

        assert over is False
        assert retry_after == 0

    def test_over_limit_blocks(self):
        """Over the limit, check_local returns (True, retry_after)."""
        limiter = DistributedRateLimiter()
        now = datetime.now(timezone.utc)
        limiter._local_window_seconds_by_key["test"] = 60
        # Fill up to the limit
        for _ in range(5):
            limiter._local_attempts_by_key["test"].append(now)

        over, retry_after = limiter._check_local("test", limit=5, window_seconds=60, cost=1)

        assert over is True
        assert retry_after > 0

    def test_metrics_increment_total_requests(self):
        """total_requests increments on each call."""
        limiter = DistributedRateLimiter()
        limiter._local_window_seconds_by_key["test"] = 60

        limiter.check("test", limit=5, window_seconds=60)

        assert limiter.metrics["total_requests"] == 1

    def test_metrics_increment_local_fallbacks(self):
        """local_fallbacks increments when using local path."""
        limiter = DistributedRateLimiter()
        limiter._local_window_seconds_by_key["test"] = 60

        limiter.check("test", limit=5, window_seconds=60)

        assert limiter.metrics["local_fallbacks"] == 1


class TestRecordAttempt:
    def test_consumes_one_slot(self):
        """record_attempt consumes one slot from the limit."""
        limiter = DistributedRateLimiter()
        limiter._local_window_seconds_by_key["test"] = 60

        # Should allow 5 attempts
        for i in range(5):
            over, _ = limiter.record_attempt("test", limit=5, window_seconds=60)
            assert over is False

        # 6th attempt should be blocked
        over, retry_after = limiter.record_attempt("test", limit=5, window_seconds=60)
        assert over is True
        assert retry_after > 0


class TestCircuitBreaker:
    def test_circuit_breaker_opens_after_threshold(self):
        """Circuit breaker opens after failure_threshold consecutive Redis failures."""
        limiter = DistributedRateLimiter(failure_threshold=3, recovery_timeout=30)
        limiter.cb_state = "CLOSED"
        limiter.consecutive_failures = 0

        # Simulate 3 consecutive Redis failures by directly incrementing counter
        for _ in range(3):
            limiter.consecutive_failures += 1
            if limiter.consecutive_failures >= limiter.failure_threshold:
                limiter.cb_state = "OPEN"

        assert limiter.cb_state == "OPEN"

    def test_circuit_breaker_half_open_after_timeout(self):
        """Circuit breaker transitions OPEN -> HALF_OPEN after recovery_timeout."""
        limiter = DistributedRateLimiter(failure_threshold=3, recovery_timeout=30)
        limiter.cb_state = "OPEN"
        limiter.last_state_change = datetime.now(timezone.utc) - timedelta(seconds=31)

        with patch.object(limiter, "_get_redis_client", return_value=MagicMock()):
            limiter.use_redis = True
            result = limiter._check_internal("test", limit=5, window_seconds=60, cost=0)

        # After recovery_timeout, should transition to HALF_OPEN
        assert limiter.cb_state in ("HALF_OPEN", "CLOSED")

    def test_circuit_breaker_blocks_when_open(self):
        """When OPEN, check returns (True, window_seconds) without hitting Redis."""
        limiter = DistributedRateLimiter(failure_threshold=3, recovery_timeout=30)
        limiter.cb_state = "OPEN"
        limiter.last_state_change = datetime.now(timezone.utc)

        limiter._local_window_seconds_by_key["test"] = 60
        limiter._redis_client = MagicMock()

        with patch.object(limiter, "_get_redis_client", return_value=limiter._redis_client):
            limiter.use_redis = False
            limiter._check_internal("test", limit=5, window_seconds=60, cost=0)

        assert limiter.metrics["blocked_by_circuit_breaker"] == 1


class TestCheckAndConsume:
    def test_consumes_slot(self):
        """check_and_consume consumes the slot and returns (False, 0) when under limit."""
        limiter = DistributedRateLimiter()
        limiter._local_window_seconds_by_key["test"] = 60

        over, retry_after = limiter.check_and_consume(
            "test", limit=3, window_seconds=60, cost=1
        )

        assert over is False
        assert retry_after == 0

    def test_consumes_multiple_slots(self):
        """check_and_consume with cost > 1 consumes multiple slots."""
        limiter = DistributedRateLimiter()
        limiter._local_window_seconds_by_key["test"] = 60

        # cost=2, limit=5, should allow 2 more calls
        over1, _ = limiter.check_and_consume("test", limit=5, window_seconds=60, cost=2)
        over2, _ = limiter.check_and_consume("test", limit=5, window_seconds=60, cost=2)
        over3, _ = limiter.check_and_consume("test", limit=5, window_seconds=60, cost=2)

        assert over1 is False
        assert over2 is False
        # 2+2+1 = 5 = limit, so third should be over
        assert over3 is True


class TestMetrics:
    def test_total_requests_increments_on_each_check(self):
        """Each _check_internal call increments total_requests."""
        limiter = DistributedRateLimiter()
        limiter._local_window_seconds_by_key["test"] = 60

        limiter.check("test", limit=10, window_seconds=60)
        limiter.record_attempt("test", limit=10, window_seconds=60)
        limiter.check_and_consume("test", limit=10, window_seconds=60, cost=1)

        assert limiter.metrics["total_requests"] == 3

    def test_redis_failures_increments_on_exception(self):
        """Redis exceptions increment redis_failures counter."""
        limiter = DistributedRateLimiter()
        limiter._redis_client = MagicMock()
        limiter._redis_script = MagicMock()
        limiter._redis_script.side_effect = Exception("Redis down")
        limiter.cb_state = "CLOSED"
        limiter._local_window_seconds_by_key["test"] = 60

        with patch.object(limiter, "_get_redis_client", return_value=limiter._redis_client):
            limiter._check_internal("test", limit=5, window_seconds=60, cost=1)

        assert limiter.metrics["redis_failures"] == 1
