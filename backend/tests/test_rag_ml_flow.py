import importlib
import sys
from unittest.mock import MagicMock, patch


def _reload_mlflow():
    """Reload ml_flow module with a fake mlflow injected."""
    fake_mlflow = MagicMock()
    with patch.dict(sys.modules, {"mlflow": fake_mlflow}):
        ml_flow = importlib.import_module("app.modules.rag.ml_flow")
        ml_flow = importlib.reload(ml_flow)
    return ml_flow, fake_mlflow


def _mock_run_ctx():
    """Return a mock run context manager."""
    mock_run = MagicMock()
    mock_run.__enter__.return_value = mock_run
    mock_run.__exit__.return_value = None
    return mock_run


def test_log_query_records_rag_metrics():
    ml_flow, fake_mlflow = _reload_mlflow()
    mock_run = _mock_run_ctx()

    with (
        patch.object(ml_flow.settings, "MLFLOW_TRACKING_URI", ""),
        patch.object(fake_mlflow, "start_run", return_value=mock_run),
        patch.object(fake_mlflow, "log_param") as mock_log_param,
        patch.object(fake_mlflow, "log_metric") as mock_log_metric,
        patch.object(fake_mlflow, "log_text") as mock_log_text,
    ):
        ml_flow.log_query(
            question="What does the EU AI Act require?",
            answer="Maintain technical documentation.",
            sources=["eu_ai_act.pdf", "iso_42001.pdf"],
            latency_ms=125.5,
        )

    mock_log_param.assert_called_once_with(
        "question",
        "What does the EU AI Act require?",
    )
    mock_log_metric.assert_any_call("answer_length", 33)
    mock_log_metric.assert_any_call("source_count", 2)
    mock_log_metric.assert_any_call("response_latency_ms", 125.5)
    mock_log_metric.assert_any_call("cache_hit", 0.0)
    mock_log_metric.assert_any_call("cache_exact_hit", 0.0)
    mock_log_metric.assert_any_call("cache_semantic_hit", 0.0)
    mock_log_text.assert_called_once_with("Maintain technical documentation.", "answer.txt")


def test_log_query_exact_cache_hit():
    ml_flow, fake_mlflow = _reload_mlflow()
    mock_run = _mock_run_ctx()

    with (
        patch.object(ml_flow.settings, "MLFLOW_TRACKING_URI", ""),
        patch.object(fake_mlflow, "start_run", return_value=mock_run),
        patch.object(fake_mlflow, "log_metric") as mock_log_metric,
        patch.object(fake_mlflow, "log_text"),
    ):
        ml_flow.log_query(
            question="Test question",
            answer="Cached answer.",
            sources=["doc.pdf"],
            latency_ms=10.0,
            cache_hit=True,
            cache_type="exact",
        )

    mock_log_metric.assert_any_call("cache_hit", 1.0)
    mock_log_metric.assert_any_call("cache_exact_hit", 1.0)
    mock_log_metric.assert_any_call("cache_semantic_hit", 0.0)


def test_log_query_semantic_cache_hit():
    ml_flow, fake_mlflow = _reload_mlflow()
    mock_run = _mock_run_ctx()

    with (
        patch.object(ml_flow.settings, "MLFLOW_TRACKING_URI", ""),
        patch.object(fake_mlflow, "start_run", return_value=mock_run),
        patch.object(fake_mlflow, "log_metric") as mock_log_metric,
        patch.object(fake_mlflow, "log_text"),
    ):
        ml_flow.log_query(
            question="Test question",
            answer="Cached answer.",
            sources=["doc.pdf"],
            latency_ms=10.0,
            cache_hit=True,
            cache_type="semantic",
        )

    mock_log_metric.assert_any_call("cache_hit", 1.0)
    mock_log_metric.assert_any_call("cache_exact_hit", 0.0)
    mock_log_metric.assert_any_call("cache_semantic_hit", 1.0)


def test_log_query_empty_sources():
    ml_flow, fake_mlflow = _reload_mlflow()
    mock_run = _mock_run_ctx()

    with (
        patch.object(ml_flow.settings, "MLFLOW_TRACKING_URI", ""),
        patch.object(fake_mlflow, "start_run", return_value=mock_run),
        patch.object(fake_mlflow, "log_metric") as mock_log_metric,
        patch.object(fake_mlflow, "log_text"),
    ):
        ml_flow.log_query(
            question="Test question",
            answer="Answer.",
            sources=[],
            latency_ms=50.0,
        )

    mock_log_metric.assert_any_call("source_count", 0)


def test_log_query_mlflow_failure_is_silent():
    """MLflow errors should be caught and logged as warnings, not raised."""
    ml_flow, fake_mlflow = _reload_mlflow()
    mock_run = _mock_run_ctx()

    fake_mlflow.start_run.side_effect = RuntimeError("MLflow connection failed")

    with (
        patch.object(ml_flow.settings, "MLFLOW_TRACKING_URI", ""),
        patch.object(fake_mlflow, "start_run", side_effect=RuntimeError("MLflow connection failed")),
    ):
        # Should not raise
        ml_flow.log_query(
            question="Test question",
            answer="Answer.",
            sources=["doc.pdf"],
            latency_ms=50.0,
        )


def test_log_query_question_truncation():
    """Questions longer than 500 chars should be truncated before logging."""
    ml_flow, fake_mlflow = _reload_mlflow()
    mock_run = _mock_run_ctx()

    long_question = "A" * 600

    with (
        patch.object(ml_flow.settings, "MLFLOW_TRACKING_URI", ""),
        patch.object(fake_mlflow, "start_run", return_value=mock_run),
        patch.object(fake_mlflow, "log_param") as mock_log_param,
        patch.object(fake_mlflow, "log_text"),
    ):
        ml_flow.log_query(
            question=long_question,
            answer="Short answer.",
            sources=[],
            latency_ms=10.0,
        )

    logged_question = mock_log_param.call_args[0][1]
    assert len(logged_question) == 500, "Question should be truncated to 500 chars"
    assert logged_question == "A" * 500
