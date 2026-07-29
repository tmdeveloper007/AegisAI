"""Tests for telemetry.py instrumentation decorators."""

import asyncio
import time
from unittest.mock import patch, MagicMock

import pytest

from app.core.telemetry import instrument_guard, instrument_rag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DummyResult(dict):
    """Result object that quacks like a guard dict."""
    def __init__(self, decision: str = "allow"):
        super().__init__()
        self._decision = decision

    def get(self, key: str, default=None):
        if key == "decision":
            return self._decision
        return super().get(key, default)


# ---------------------------------------------------------------------------
# instrument_guard tests
# ---------------------------------------------------------------------------

class TestInstrumentGuardSync:
    """instrument_guard applied to a synchronous function."""

    def test_decorator_returns_correct_result(self):
        @instrument_guard
        def sync_guard(prompt: str) -> dict:
            return {"decision": "block", "sanitized": prompt}

        result = sync_guard("test prompt")
        assert result == {"decision": "block", "sanitized": "test prompt"}

    def test_decorator_increments_guard_scan_total_counter(self):
        """Guard scan counter should be incremented after each call."""
        inc_calls = []

        mock_child = MagicMock()
        mock_child.inc = MagicMock(side_effect=lambda v=1: inc_calls.append(v))
        mock_counter = MagicMock()
        mock_counter.labels = MagicMock(return_value=mock_child)

        with patch("app.core.telemetry.GUARD_SCAN_TOTAL", mock_counter), \
             patch("app.core.telemetry.GUARD_INFERENCE_LATENCY.labels", MagicMock()):
            @instrument_guard
            def guard_fn(p: str) -> dict:
                return {"decision": "allow"}

            guard_fn("hello")

        assert len(inc_calls) == 1

    def test_decorator_records_latency(self):
        """Latency histogram should be observed after each call."""
        observed_values = []

        mock_child = MagicMock()
        mock_child.observe = MagicMock(side_effect=lambda v: observed_values.append(v))
        mock_histogram = MagicMock()
        mock_histogram.labels = MagicMock(return_value=mock_child)

        with patch("app.core.telemetry.GUARD_INFERENCE_LATENCY", mock_histogram), \
             patch("app.core.telemetry.GUARD_SCAN_TOTAL.labels", MagicMock()):
            @instrument_guard
            def guard_fn(p: str) -> dict:
                return {"decision": "block"}

            guard_fn("slow prompt")

        assert len(observed_values) == 1
        assert observed_values[0] >= 0  # positive duration


class TestInstrumentGuardAsync:
    """instrument_guard applied to an asynchronous function."""

    @pytest.mark.asyncio
    async def test_decorator_returns_correct_result_async(self):
        @instrument_guard
        async def async_guard(prompt: str) -> dict:
            return {"decision": "sanitize", "clean": prompt}

        result = await async_guard("test")
        assert result == {"decision": "sanitize", "clean": "test"}

    @pytest.mark.asyncio
    async def test_decorator_increments_counter_async(self):
        """Counter should be incremented for async guard calls."""
        inc_calls = []

        mock_child = MagicMock()
        mock_child.inc = MagicMock(side_effect=lambda v=1: inc_calls.append(v))
        mock_counter = MagicMock()
        mock_counter.labels = MagicMock(return_value=mock_child)

        with patch("app.core.telemetry.GUARD_SCAN_TOTAL", mock_counter), \
             patch("app.core.telemetry.GUARD_INFERENCE_LATENCY.labels", MagicMock()):
            @instrument_guard
            async def async_guard(p: str) -> dict:
                return {"decision": "block"}

            await async_guard("hello")

        assert len(inc_calls) == 1


# ---------------------------------------------------------------------------
# instrument_rag tests
# ---------------------------------------------------------------------------

class TestInstrumentRagSync:
    """instrument_rag applied to a synchronous function."""

    def test_decorator_returns_correct_result(self):
        @instrument_rag
        def rag_retrieve(query: str) -> list:
            return [{"page_content": f"doc for {query}"}]

        result = rag_retrieve("compliance")
        assert result == [{"page_content": "doc for compliance"}]

    def test_decorator_observes_rag_latency(self):
        """RAG retrieval latency histogram should be observed."""
        observed_values = []

        mock_histogram = MagicMock()
        mock_histogram.observe = MagicMock(side_effect=lambda v: observed_values.append(v))

        with patch("app.core.telemetry.RAG_RETRIEVAL_LATENCY", mock_histogram), \
             patch("app.core.telemetry.RAG_QUERY_TOTAL.labels", MagicMock()):
            @instrument_rag
            def rag_fn(q: str) -> list:
                return []

            rag_fn("test")

        assert len(observed_values) == 1
        assert observed_values[0] >= 0


class TestInstrumentRagAsync:
    """instrument_rag applied to an asynchronous function."""

    @pytest.mark.asyncio
    async def test_decorator_returns_correct_result_async(self):
        @instrument_rag
        async def async_rag(query: str) -> list:
            return [{"content": f"async result for {query}"}]

        result = await async_rag("test")
        assert result == [{"content": "async result for test"}]

    @pytest.mark.asyncio
    async def test_decorator_observes_latency_async(self):
        """Latency should be recorded for async RAG calls."""
        observed_values = []

        mock_histogram = MagicMock()
        mock_histogram.observe = MagicMock(side_effect=lambda v: observed_values.append(v))

        with patch("app.core.telemetry.RAG_RETRIEVAL_LATENCY", mock_histogram), \
             patch("app.core.telemetry.RAG_QUERY_TOTAL.labels", MagicMock()):
            @instrument_rag
            async def async_rag(q: str) -> list:
                return []

            await async_rag("test")

        assert len(observed_values) == 1
        assert observed_values[0] >= 0
