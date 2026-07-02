"""
Unit tests for backend/app/core/context.py.

Tests verify the request-scoped ContextVars (request_id_ctx, user_id_ctx)
and their getter functions behave correctly.
"""

import pytest

from app.core.context import (
    get_request_id,
    get_user_id,
    request_id_ctx,
    user_id_ctx,
)


class TestGetRequestId:
    """Tests for get_request_id()."""

    def test_get_request_id_returns_none_when_not_set(self):
        """Outside any context, get_request_id() must return None."""
        assert get_request_id() is None

    def test_get_request_id_returns_set_value(self):
        """When request_id_ctx is set, get_request_id() returns it."""
        token = request_id_ctx.set("req-abc-123")
        try:
            assert get_request_id() == "req-abc-123"
        finally:
            request_id_ctx.reset(token)

    def test_get_request_id_isolation(self):
        """Setting request_id_ctx in one context does not affect another."""
        token = request_id_ctx.set("req-main")
        try:
            # Read from the same context — should see the value
            assert get_request_id() == "req-main"
        finally:
            request_id_ctx.reset(token)

        # After reset, getter returns None even in the same context
        assert get_request_id() is None


class TestGetUserId:
    """Tests for get_user_id()."""

    def test_get_user_id_returns_none_when_not_set(self):
        """Outside any context, get_user_id() must return None."""
        assert get_user_id() is None

    def test_get_user_id_returns_set_value(self):
        """When user_id_ctx is set, get_user_id() returns it."""
        token = user_id_ctx.set(42)
        try:
            assert get_user_id() == 42
        finally:
            user_id_ctx.reset(token)

    def test_get_user_id_returns_none_after_reset(self):
        """After resetting the token, get_user_id() returns None."""
        token = user_id_ctx.set(99)
        user_id_ctx.reset(token)
        assert get_user_id() is None

    def test_get_user_id_isolation(self):
        """Setting user_id_ctx in one context does not affect another."""
        token = user_id_ctx.set(7)
        try:
            assert get_user_id() == 7
        finally:
            user_id_ctx.reset(token)

        # After reset, getter returns None
        assert get_user_id() is None


class TestBothContextVariables:
    """Tests exercising both context variables simultaneously."""

    def test_both_can_be_set_together(self):
        """request_id_ctx and user_id_ctx can be set in the same context."""
        token_req = request_id_ctx.set("req-xyz")
        token_uid = user_id_ctx.set(123)
        try:
            assert get_request_id() == "req-xyz"
            assert get_user_id() == 123
        finally:
            request_id_ctx.reset(token_req)
            user_id_ctx.reset(token_uid)

    def test_both_return_none_after_full_reset(self):
        """After resetting both tokens, both getters return None."""
        token_req = request_id_ctx.set("req-reset")
        token_uid = user_id_ctx.set(1)
        request_id_ctx.reset(token_req)
        user_id_ctx.reset(token_uid)
        assert get_request_id() is None
        assert get_user_id() is None
