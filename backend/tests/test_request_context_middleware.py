"""
Unit tests for RequestContextMiddleware in middleware.py.

Verifies: request-ID passthrough, UUID minting for invalid/missing IDs,
X-Request-ID response header injection, status-code tracking, and
context-variable cleanup in the finally block.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock
import uuid

import pytest

from app.core.middleware import (
    RequestContextMiddleware,
    _resolve_request_id,
    request_id_ctx,
    user_id_ctx,
)


# ---------------------------------------------------------------------------
# Tests for _resolve_request_id
# ---------------------------------------------------------------------------

def test_resolve_request_id_passthrough_valid_header():
    """A valid inbound X-Request-ID is returned unchanged."""
    headers = MagicMock()
    headers.get.return_value = "abc123-xyz789"
    result = _resolve_request_id(headers)
    assert result == "abc123-xyz789"
    headers.get.assert_called_once_with("x-request-id")


def test_resolve_request_id_mints_uuid_for_missing_header():
    """When X-Request-ID is absent, a fresh UUID is minted."""
    headers = MagicMock()
    headers.get.return_value = None
    result = _resolve_request_id(headers)
    # Verify it's a valid UUID hex string (32 hex chars, no dashes)
    assert len(result) == 32
    assert all(c in "0123456789abcdef" for c in result)


def test_resolve_request_id_mints_uuid_for_invalid_header_too_long():
    """Headers exceeding 128 characters are rejected and a UUID is minted."""
    headers = MagicMock()
    headers.get.return_value = "a" * 200
    result = _resolve_request_id(headers)
    assert len(result) == 32


def test_resolve_request_id_mints_uuid_for_invalid_header_newline():
    """Headers with newlines are rejected and a UUID is minted."""
    headers = MagicMock()
    headers.get.return_value = "abc\ndef"
    result = _resolve_request_id(headers)
    assert len(result) == 32


def test_resolve_request_id_mints_uuid_for_invalid_header_special_chars():
    """Headers with characters outside [A-Za-z0-9._-] are rejected."""
    headers = MagicMock()
    headers.get.return_value = "abc@def!ghi"
    result = _resolve_request_id(headers)
    assert len(result) == 32


# ---------------------------------------------------------------------------
# Tests for RequestContextMiddleware.__call__
# ---------------------------------------------------------------------------

def _make_scope(path: str = "/api/v1/test", method: str = "GET") -> dict:
    """Construct a minimal ASGI scope dict for testing."""
    return {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": b"",
        "headers": [],
        "root_path": "",
        "server": ("testserver", 80),
    }


async def test_middleware_sets_request_id_and_user_id_resets_to_none():
    """Middleware sets request_id_ctx to resolved ID and resets user_id to None."""
    middleware_scope = _make_scope()
    middleware = RequestContextMiddleware(MagicMock())

    sent_messages: list[dict] = []

    async def send(message: dict) -> None:
        if message["type"] == "http.response.start":
            sent_messages.append(message)

    inner_app = AsyncMock()
    await middleware(middleware_scope, AsyncMock(), send)
    inner_app.assert_awaited_once()

    # request_id_ctx should be set to a 32-char UUID hex
    rid = request_id_ctx.get()
    assert rid is not None
    assert len(rid) == 32


async def test_middleware_passes_valid_request_id_through():
    """Middleware passes through a valid inbound X-Request-ID."""
    middleware_scope = _make_scope()
    middleware_scope["headers"] = [(b"x-request-id", b"my-valid-id-123")]
    middleware = RequestContextMiddleware(MagicMock())

    sent_messages: list[dict] = []

    async def send(message: dict) -> None:
        sent_messages.append(message)

    inner_app = AsyncMock()
    await middleware(middleware_scope, AsyncMock(), send)

    # The X-Request-ID in the response should be "my-valid-id-123"
    start_msg = next(m for m in sent_messages if m["type"] == "http.response.start")
    assert dict(start_msg.get("headers", []))[b"x-request-id"] == b"my-valid-id-123"


async def test_middleware_injects_x_request_id_response_header():
    """The X-Request-ID header is set in the HTTP response start."""
    middleware_scope = _make_scope()
    middleware = RequestContextMiddleware(MagicMock())

    sent_messages: list[dict] = []

    async def send(message: dict) -> None:
        sent_messages.append(message)

    inner_app = AsyncMock()
    await middleware(middleware_scope, AsyncMock(), send)

    start_msg = next(m for m in sent_messages if m["type"] == "http.response.start")
    rid_header = dict(start_msg.get("headers", [])).get(b"x-request-id")
    assert rid_header is not None
    assert len(rid_header.decode()) == 32  # UUID hex


async def test_middleware_resets_context_vars_on_exception():
    """Context vars are reset even when the inner app raises an exception."""
    middleware_scope = _make_scope()
    middleware = RequestContextMiddleware(MagicMock())

    async def send(message: dict) -> None:
        pass

    inner_app = AsyncMock(side_effect=ValueError("boom"))

    # Set context vars before the request to confirm they get reset
    request_id_ctx.set("before-rid")
    user_id_ctx.set(999)

    with pytest.raises(ValueError, match="boom"):
        await middleware(middleware_scope, AsyncMock(), send)

    # After the middleware finishes (even with exception), context vars should be None
    assert request_id_ctx.get() is None
    assert user_id_ctx.get() is None


async def test_middleware_resets_context_vars_on_normal_completion():
    """Context vars are reset after a successful request."""
    middleware_scope = _make_scope()
    middleware = RequestContextMiddleware(MagicMock())

    async def send(message: dict) -> None:
        pass

    inner_app = AsyncMock()

    await middleware(middleware_scope, AsyncMock(), send)

    # After completion, context vars should be None (reset by finally block)
    assert request_id_ctx.get() is None
    assert user_id_ctx.get() is None


async def test_middleware_passes_through_non_http_scopes():
    """Non-HTTP ASGI scopes (e.g. websocket) are passed through without modification."""
    middleware_scope = {"type": "websocket", "path": "/ws", "headers": []}
    middleware = RequestContextMiddleware(MagicMock())

    inner_app = AsyncMock()
    await middleware(middleware_scope, AsyncMock(), AsyncMock())
    inner_app.assert_awaited_once()
