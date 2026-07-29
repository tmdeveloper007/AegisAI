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
        counter_labels_seen = []

        def capture_labels(labels, value):
            counter_labels_seen.append(labels)

        with patch(
            "app.core.telemetry.GUARD_SCAN_TOTAL.labels",
            return_value=MagicMock(inc=capture_labels),
        ), patch(
            "app.core.telemetry.GUARD_INFERENCE_LATENCY.labels",
            return_value=MagicMock(observe=lambda v: None),
        ):
            @instrument_guard
            def guard_fn(p: str) -> dict:
                return {"decision": "allow"}

            guard_fn("hello")

        assert len(counter_labels_seen) == 1
        assert counter_labels_seen[0].get("decision") == "allow"

    def test_decorator_records_latency(self):
        """Latency histogram should be observed after each call."""
        latency_values = []

        def capture_duration(labels, value):
            latency_values.append(value)

        with patch(
            "app.core.telemetry.GUARD_SCAN_TOTAL.labels",
            return_value=MagicMock(inc=lambda: None),
        ), patch(
            "app.core.telemetry.GUARD_INFERENCE_LATENCY.labels",
            return_value=MagicMock(observe=capture_duration),
        ):
            @instrument_guard
            def guard_fn(p: str) -> dict:
                return {"decision": "block"}

            guard_fn("slow prompt")

        assert len(latency_values) == 1
        assert latency_values[0] >= 0  # positive duration


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
        counter_labels_seen = []

        def capture_labels(labels, value):
            counter_labels_seen.append(labels)

        with patch(
            "app.core.telemetry.GUARD_SCAN_TOTAL.labels",
            return_value=MagicMock(inc=capture_labels),
        ), patch(
            "app.core.telemetry.GUARD_INFERENCE_LATENCY.labels",
            return_value=MagicMock(observe=lambda v: None),
        ):
            @instrument_guard
            async def async_guard(p: str) -> dict:
                return {"decision": "block"}

            await async_guard("hello")

        assert len(counter_labels_seen) == 1
        assert counter_labels_seen[0].get("decision") == "block"


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
        latency_values = []

        with patch(
            "app.core.telemetry.RAG_QUERY_TOTAL.labels",
            return_value=MagicMock(inc=lambda: None),
        ), patch(
            "app.core.telemetry.RAG_RETRIEVAL_LATENCY.observe",
            latency_values.append,
        ):
            @instrument_rag
            def rag_fn(q: str) -> list:
                return []

            rag_fn("test")

        assert len(latency_values) == 1
        assert latency_values[0] >= 0


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
        latency_values = []

        with patch(
            "app.core.telemetry.RAG_QUERY_TOTAL.labels",
            return_value=MagicMock(inc=lambda: None),
        ), patch(
            "app.core.telemetry.RAG_RETRIEVAL_LATENCY.observe",
            latency_values.append,
        ):
            @instrument_rag
            async def async_rag(q: str) -> list:
                return []

            await async_rag("test")

        assert len(latency_values) == 1
        assert latency_values[0] >= 0
