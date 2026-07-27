"""
Unit tests for instrument_guard and instrument_rag decorators in telemetry.py.

Verifies that both sync and async variants record the correct Prometheus
histogram and counter labels, and fall back to "unknown" when the result
lacks the expected key.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from app.core.telemetry import instrument_guard, instrument_rag


# ---------------------------------------------------------------------------
# instrument_guard — sync variant
# ---------------------------------------------------------------------------

def test_instrument_guard_sync_records_correct_decision(mock_guard_metrics):
    """Sync guard decorated with a dict result records the decision label."""
    mock_latency, mock_counter, mock_guard_metrics_getter = mock_guard_metrics

    @instrument_guard
    def sync_guard(input_text: str) -> dict:
        return {"decision": "safe", "score": 0.9}

    result = sync_guard("hello")

    assert result["decision"] == "safe"
    mock_latency.labels.assert_called_once_with(decision="safe")
    mock_latency.labels.return_value.observe.assert_called_once()
    mock_counter.labels.assert_called_once_with(decision="safe")
    mock_counter.labels.return_value.inc.assert_called_once()


def test_instrument_guard_sync_unknown_decision(mock_guard_metrics):
    """Sync guard returning a dict without a 'decision' key falls back to 'unknown'."""
    mock_latency, mock_counter, _ = mock_guard_metrics

    @instrument_guard
    def sync_guard(input_text: str) -> dict:
        return {"score": 0.9}  # no 'decision' key

    result = sync_guard("hello")

    assert "score" in result
    mock_latency.labels.assert_called_with(decision="unknown")
    mock_counter.labels.assert_called_with(decision="unknown")


def test_instrument_guard_sync_non_dict_result(mock_guard_metrics):
    """Sync guard returning a non-dict falls back to 'unknown' label."""
    mock_latency, mock_counter, _ = mock_guard_metrics

    @instrument_guard
    def sync_guard(input_text: str) -> str:
        return "plain string"

    result = sync_guard("hello")

    assert result == "plain string"
    mock_latency.labels.assert_called_with(decision="unknown")
    mock_counter.labels.assert_called_with(decision="unknown")


# ---------------------------------------------------------------------------
# instrument_guard — async variant
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_instrument_guard_async_records_correct_decision(mock_guard_metrics):
    """Async guard decorated with a dict result records the decision label."""
    mock_latency, mock_counter, _ = mock_guard_metrics

    @instrument_guard
    async def async_guard(input_text: str) -> dict:
        return {"decision": "blocked", "reason": "injection detected"}

    result = await async_guard("prompt injection")

    assert result["decision"] == "blocked"
    mock_latency.labels.assert_called_with(decision="blocked")
    mock_latency.labels.return_value.observe.assert_called()
    mock_counter.labels.assert_called_with(decision="blocked")
    mock_counter.labels.return_value.inc.assert_called()


@pytest.mark.asyncio
async def test_instrument_guard_async_unknown_decision(mock_guard_metrics):
    """Async guard returning a dict without 'decision' falls back to 'unknown'."""
    mock_latency, mock_counter, _ = mock_guard_metrics

    @instrument_guard
    async def async_guard(input_text: str) -> dict:
        return {"foo": "bar"}

    result = await async_guard("hello")

    assert "foo" in result
    mock_latency.labels.assert_called_with(decision="unknown")
    mock_counter.labels.assert_called_with(decision="unknown")


# ---------------------------------------------------------------------------
# instrument_rag — sync variant
# ---------------------------------------------------------------------------

def test_instrument_rag_sync_records_latency_and_counter(mock_rag_metrics):
    """Sync RAG decorated function records retrieval latency and success counter."""
    mock_latency, mock_counter, _ = mock_rag_metrics

    @instrument_rag
    def sync_rag(query: str) -> list:
        return [{"chunk": "result", "score": 0.95}]

    result = sync_rag("what is GDPR?")

    assert len(result) == 1
    mock_latency.observe.assert_called_once()
    mock_counter.labels.assert_called_once_with(status="success")
    mock_counter.labels.return_value.inc.assert_called_once()


# ---------------------------------------------------------------------------
# instrument_rag — async variant
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_instrument_rag_async_records_latency_and_counter(mock_rag_metrics):
    """Async RAG decorated function records retrieval latency and success counter."""
    mock_latency, mock_counter, _ = mock_rag_metrics

    @instrument_rag
    async def async_rag(query: str) -> list:
        return [{"chunk": "async result", "score": 0.88}]

    result = await async_rag("what is AI Act?")

    assert len(result) == 1
    mock_latency.observe.assert_called()
    mock_counter.labels.assert_called_with(status="success")
    mock_counter.labels.return_value.inc.assert_called()


# ---------------------------------------------------------------------------
# Fixtures — patch Prometheus metrics at import time
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_guard_metrics():
    """Patch GUARD_INFERENCE_LATENCY and GUARD_SCAN_TOTAL for each test."""
    mock_latency = MagicMock()
    mock_counter = MagicMock()
    mock_latency.labels.return_value = MagicMock()
    mock_counter.labels.return_value = MagicMock()

    with patch("app.core.telemetry.GUARD_INFERENCE_LATENCY", mock_latency), \
         patch("app.core.telemetry.GUARD_SCAN_TOTAL", mock_counter):
        yield mock_latency, mock_counter, None


@pytest.fixture(autouse=True)
def mock_rag_metrics():
    """Patch RAG_RETRIEVAL_LATENCY and RAG_QUERY_TOTAL for each test."""
    mock_latency = MagicMock()
    mock_counter = MagicMock()
    mock_latency.labels.return_value = MagicMock()
    mock_counter.labels.return_value = MagicMock()

    with patch("app.core.telemetry.RAG_RETRIEVAL_LATENCY", mock_latency), \
         patch("app.core.telemetry.RAG_QUERY_TOTAL", mock_counter):
        yield mock_latency, mock_counter, None
