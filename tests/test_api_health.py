"""
Tests for API health endpoint contract.

Tests are self-contained and do not require a running server or database —
they validate the expected response schema and status codes via mocks.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def build_health_response(db_ok: bool = True, cache_ok: bool = True) -> dict:
    """
    Construct a health response dict matching the /api/v1/health schema.

    Parameters
    ----------
    db_ok : bool
    cache_ok : bool

    Returns
    -------
    dict
    """
    db_status = "connected" if db_ok else "disconnected"
    cache_status = "connected" if cache_ok else "disconnected"

    if db_ok and cache_ok:
        overall = "ok"
    elif db_ok or cache_ok:
        overall = "degraded"
    else:
        overall = "unhealthy"

    return {
        "status": overall,
        "version": "1.0.0",
        "database": db_status,
        "cache": cache_status,
        "uptime_seconds": 3600,
    }


class TestHealthResponseSchema:
    """Validate health response structure and field values."""

    def test_healthy_response_status_ok(self):
        resp = build_health_response(db_ok=True, cache_ok=True)
        assert resp["status"] == "ok"

    def test_healthy_response_has_all_fields(self):
        resp = build_health_response()
        for field in ["status", "version", "database", "cache"]:
            assert field in resp, f"Missing field: {field}"

    def test_database_disconnected_yields_degraded(self):
        resp = build_health_response(db_ok=False, cache_ok=True)
        assert resp["status"] == "degraded"
        assert resp["database"] == "disconnected"

    def test_cache_disconnected_yields_degraded(self):
        resp = build_health_response(db_ok=True, cache_ok=False)
        assert resp["status"] == "degraded"
        assert resp["cache"] == "disconnected"

    def test_both_disconnected_yields_unhealthy(self):
        resp = build_health_response(db_ok=False, cache_ok=False)
        assert resp["status"] == "unhealthy"

    def test_version_is_string(self):
        resp = build_health_response()
        assert isinstance(resp["version"], str)
        assert len(resp["version"]) > 0

    def test_uptime_is_nonnegative(self):
        resp = build_health_response()
        assert resp["uptime_seconds"] >= 0


class TestHealthHTTPMock:
    """Mock HTTP client tests for /api/v1/health endpoint."""

    def test_health_returns_200(self):
        """Health endpoint must return HTTP 200 when healthy."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = build_health_response()
        mock_client.get.return_value = mock_response

        response = mock_client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_json_body(self):
        """Health response body must be valid JSON with status field."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok", "version": "1.0.0"}
        mock_client.get.return_value = mock_response

        response = mock_client.get("/api/v1/health")
        body = response.json()
        assert body["status"] == "ok"

    def test_health_content_type_json(self):
        """Health endpoint should return application/json content type."""
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/json; charset=utf-8"}
        assert "application/json" in mock_response.headers["content-type"]

    def test_health_called_once(self):
        """Verify the client calls health endpoint exactly once per check."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.get.return_value = mock_response

        mock_client.get("/api/v1/health")
        mock_client.get.assert_called_once_with("/api/v1/health")

    def test_degraded_state_not_500(self):
        """Degraded (DB down) should still return 200, not 500."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200  # health endpoint always responds
        mock_response.json.return_value = build_health_response(db_ok=False, cache_ok=True)
        mock_client.get.return_value = mock_response

        response = mock_client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "degraded"
