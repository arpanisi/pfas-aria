"""
API integration tests.
Catches route 404s, CORS misconfig, and response contract issues
before they hit Render.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

FRONTEND_ORIGIN = "https://pfas-aria-frontend.onrender.com"


# ── CORS ──────────────────────────────────────────────────────────────────────

class TestCORS:
    def test_cors_allowed_for_frontend(self):
        response = client.options(
            "/pipeline/runs",
            headers={
                "Origin": FRONTEND_ORIGIN,
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code != 404
        assert "access-control-allow-origin" in response.headers

    def test_cors_blocked_for_unknown_origin(self):
        response = client.get(
            "/pipeline/runs",
            headers={"Origin": "https://evil.com"},
        )
        assert "access-control-allow-origin" not in response.headers


# ── Routes exist (not 404) ────────────────────────────────────────────────────

class TestRoutesExist:
    """
    These don't test auth or logic — just that routes are registered.
    401/403 is fine. 404 means the route is missing or prefix is wrong.
    """

    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_pipeline_runs_exists(self):
        response = client.get("/pipeline/runs")
        assert response.status_code != 404

    def test_pipeline_upload_exists(self):
        response = client.post("/pipeline/upload")
        assert response.status_code != 404

    def test_corpus_stats_exists(self):
        response = client.get("/corpus/stats")
        assert response.status_code != 404

    def test_corpus_delete_exists(self):
        response = client.delete("/corpus/some-id")
        assert response.status_code != 404

    def test_results_exists(self):
        response = client.get("/results")
        assert response.status_code != 404


# ── Response contracts ────────────────────────────────────────────────────────

class TestHealthContract:
    def test_health_returns_expected_fields(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
