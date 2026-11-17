"""
FastAPI TestClient integration tests.

Tests:
1. GET /health returns healthy status.
2. POST /api/query with mocked LLM returns structured telemetry.
3. Verifies that mock pipeline runs fully offline.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app, _pipeline, get_pipeline
from app.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_pipeline(monkeypatch):
    """Replace the global pipeline with a mock-LLM variant for all tests."""
    import app.main as main_module
    main_module._pipeline = Pipeline(use_mock_llm=True)
    yield
    main_module._pipeline = None


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["tenants_count"] >= 2

    def test_health_json_shape(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "status" in data
        assert "tenants_count" in data


class TestQueryEndpoint:
    def test_query_basic(self, client):
        resp = client.post(
            "/api/query",
            json={
                "question": "Show all students in CSE",
                "tenant_id": 1,
                "mode": "ast_guardcache",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "generated_sql" in data
        assert "verified_sql" in data
        assert "execution_rows" in data
        assert "latency_ms" in data
        assert "cache_hit" in data
        assert "mode" in data
        assert data["mode"] == "ast_guardcache"

    def test_query_returns_rows(self, client):
        resp = client.post(
            "/api/query",
            json={
                "question": "Show all students in CSE",
                "tenant_id": 1,
                "mode": "ast_guardcache",
            },
        )
        data = resp.json()
        # With mock LLM and tenant 1, CSE should return Alice and Bob
        assert isinstance(data["execution_rows"], list)
        assert len(data["execution_rows"]) >= 1

    def test_query_tenant_isolation(self, client):
        """Rows returned should only belong to the requested tenant."""
        resp = client.post(
            "/api/query",
            json={
                "question": "Count total students",
                "tenant_id": 1,
                "mode": "rag_ast",
            },
        )
        data = resp.json()
        assert data["error"] is None
        # Tenant 1 has 5 students, the database 8: the count must be scoped
        assert data["execution_rows"] == [{"total_students": 5}]

    def test_query_zero_shot_mode(self, client):
        resp = client.post(
            "/api/query",
            json={
                "question": "Show all students",
                "tenant_id": 1,
                "mode": "zero_shot",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "zero_shot"

    def test_query_invalid_mode(self, client):
        resp = client.post(
            "/api/query",
            json={
                "question": "test",
                "tenant_id": 1,
                "mode": "invalid_mode",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["error"] is not None

    def test_cache_hit_on_repeat(self, client):
        """Second identical query should be a cache hit in ast_guardcache mode."""
        payload = {
            "question": "Show pending billing invoices",
            "tenant_id": 1,
            "mode": "ast_guardcache",
        }
        # First call — cache miss
        resp1 = client.post("/api/query", json=payload)
        data1 = resp1.json()
        assert data1["cache_hit"] is False

        # Second call — should be a cache hit
        resp2 = client.post("/api/query", json=payload)
        data2 = resp2.json()
        assert data2["cache_hit"] is True


class TestDashboard:
    def test_dashboard_serves_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "AST-GuardCache" in resp.text


class TestProductionMode:
    """APP_ENV=production disables unverified modes and the benchmark endpoint."""

    def test_unverified_mode_refused(self, client, monkeypatch):
        from app.config import settings
        monkeypatch.setattr(settings, "APP_ENV", "production")
        r = client.post("/api/query", json={"question": "List students", "tenant_id": 1, "mode": "zero_shot"})
        assert r.status_code == 403

    def test_benchmark_endpoint_refused(self, client, monkeypatch):
        from app.config import settings
        monkeypatch.setattr(settings, "APP_ENV", "production")
        assert client.post("/api/benchmark").status_code == 403
