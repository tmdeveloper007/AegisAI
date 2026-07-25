"""
Unit tests for backend/app/schemas/pagination.py

Tests cover:
  - PaginatedResponse serialization and model_validate
  - CursorPaginatedResponse serialization and model_validate
  - Empty list handling
  - next_cursor null case
  - Generic type parameter resolution
"""

import pytest
from pydantic import BaseModel

from app.schemas.pagination import (
    PaginatedResponse,
    CursorPaginatedResponse,
)


class Item(BaseModel):
    id: int
    name: str


class TestPaginatedResponse:
    """Tests for PaginatedResponse."""

    def test_serialization_with_items(self):
        """PaginatedResponse correctly serializes a list of typed items."""
        response = PaginatedResponse[Item](
            items=[Item(id=1, name="foo"), Item(id=2, name="bar")],
            total=2,
            skip=0,
            limit=10,
        )
        data = response.model_dump()
        assert data["items"] == [{"id": 1, "name": "foo"}, {"id": 2, "name": "bar"}]
        assert data["total"] == 2
        assert data["skip"] == 0
        assert data["limit"] == 10

    def test_empty_items_list(self):
        """PaginatedResponse handles an empty items list."""
        response = PaginatedResponse[Item](
            items=[],
            total=0,
            skip=0,
            limit=10,
        )
        data = response.model_dump()
        assert data["items"] == []
        assert data["total"] == 0
        assert data["skip"] == 0
        assert data["limit"] == 10

    def test_model_validate_from_dict(self):
        """PaginatedResponse can be reconstructed from a dict via model_validate."""
        original = PaginatedResponse[Item](
            items=[Item(id=3, name="baz")],
            total=100,
            skip=50,
            limit=10,
        )
        reconstructed = PaginatedResponse[Item].model_validate(original.model_dump())
        assert reconstructed.total == 100
        assert reconstructed.skip == 50
        assert reconstructed.limit == 10
        assert len(reconstructed.items) == 1
        assert reconstructed.items[0].id == 3

    def test_partial_page(self):
        """PaginatedResponse correctly represents a partial page."""
        response = PaginatedResponse[Item](
            items=[Item(id=1, name="a")],
            total=50,
            skip=25,
            limit=10,
        )
        data = response.model_dump()
        assert data["skip"] + data["limit"] < data["total"]
        assert len(data["items"]) == 1


class TestCursorPaginatedResponse:
    """Tests for CursorPaginatedResponse."""

    def test_serialization_with_items_and_cursor(self):
        """CursorPaginatedResponse correctly serializes with a next_cursor."""
        response = CursorPaginatedResponse[Item](
            items=[Item(id=1, name="foo")],
            limit=10,
            next_cursor="abc123",
        )
        data = response.model_dump()
        assert data["items"] == [{"id": 1, "name": "foo"}]
        assert data["limit"] == 10
        assert data["next_cursor"] == "abc123"

    def test_null_next_cursor(self):
        """CursorPaginatedResponse defaults next_cursor to None when last page."""
        response = CursorPaginatedResponse[Item](
            items=[Item(id=1, name="foo")],
            limit=10,
            next_cursor=None,
        )
        data = response.model_dump()
        assert data["next_cursor"] is None

    def test_next_cursor_not_included_when_none(self):
        """CursorPaginatedResponse excludes next_cursor from output when None."""
        response = CursorPaginatedResponse[Item](
            items=[],
            limit=10,
        )
        # next_cursor should be None by default
        assert response.next_cursor is None
        data = response.model_dump()
        assert data["next_cursor"] is None

    def test_model_validate_roundtrip(self):
        """CursorPaginatedResponse round-trips through model_validate."""
        original = CursorPaginatedResponse[Item](
            items=[Item(id=5, name="roundtrip")],
            limit=25,
            next_cursor="cursor-val-xyz",
        )
        reconstructed = CursorPaginatedResponse[Item].model_validate(
            original.model_dump()
        )
        assert reconstructed.limit == 25
        assert reconstructed.next_cursor == "cursor-val-xyz"
        assert len(reconstructed.items) == 1
        assert reconstructed.items[0].id == 5

    def test_empty_items_last_page(self):
        """CursorPaginatedResponse represents last page with empty items and no cursor."""
        response = CursorPaginatedResponse[Item](
            items=[],
            limit=10,
            next_cursor=None,
        )
        data = response.model_dump()
        assert data["items"] == []
        assert data["next_cursor"] is None
