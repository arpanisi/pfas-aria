"""
Opt-in deployment E2E test for the authenticated user pipeline.

Run manually from repo root:
    RUN_DEPLOYMENT_E2E=1 .venv/bin/pytest tests/integration/test_deployment_user_flow.py -q

This test uses fake Clerk subjects through FastAPI dependency overrides. It does
not create real Clerk users, but it exercises the same tenant-scoped API paths
used by real users: dataset upload, corpus upload, screening, grounding, and
run visibility.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from pymongo import MongoClient

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from src.api import auth  # noqa: E402
from src.api.main import app  # noqa: E402
from src.api.tenant import tenant_storage_key  # noqa: E402
from src.services.pipeline_service import raw_upload_path  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DEPLOYMENT_E2E") != "1",
    reason="Set RUN_DEPLOYMENT_E2E=1 to run the deployment user-flow test.",
)

ROOT = Path(__file__).resolve().parents[2]
DATASET = (
    ROOT
    / "data/raw/user_b5f8dbd7bd0fd8e6d970fbe5/Copy of Unified Experimental Data Spreadsheet 2.xlsx"
)
PDF = (
    ROOT
    / "data/corpus/user_b5f8dbd7bd0fd8e6d970fbe5/"
    "Development and Application of Different Non_thermal Plasma Reactors for the Removal of Perfluorosurfactants in Water_A Comparative Study.pdf"
)
USER_A = "deployment_e2e_user_a"
USER_B = "deployment_e2e_user_b"


def _preflight_mongo() -> None:
    """Fail early with the DNS/connectivity error users would otherwise see later."""
    mongo_url = os.getenv("MONGO_URL", "mongodb://localhost:27017")
    client: MongoClient[Any] = MongoClient(mongo_url, serverSelectionTimeoutMS=8000)
    try:
        client.admin.command("ping")
    finally:
        client.close()


def _override_user(subject_holder: dict[str, str]):
    async def _fake_user() -> dict[str, str]:
        return {"sub": subject_holder["sub"]}

    return _fake_user


def _post_file(client: TestClient, url: str, path: Path, mime: str):
    with path.open("rb") as fh:
        return client.post(url, files={"file": (path.name, fh, mime)})


def _cleanup_raw_uploads(filename: str) -> None:
    for user_sub in (USER_A, USER_B):
        path = raw_upload_path(filename, user_sub=user_sub)
        if path.exists():
            path.unlink()
        tenant_dir = path.parent
        if tenant_dir.exists() and not any(tenant_dir.iterdir()):
            tenant_dir.rmdir()


def test_two_dummy_users_full_pipeline_and_tenant_isolation(monkeypatch: pytest.MonkeyPatch):
    if not DATASET.exists():
        pytest.skip(f"Deployment E2E dataset not found: {DATASET}")
    if not PDF.exists():
        pytest.skip(f"Deployment E2E PDF not found: {PDF}")

    _preflight_mongo()

    async def _identity_enrichment(bundles):
        return bundles

    import src.api.routes.pipeline as pipeline_routes

    # Keep this test focused on pipeline/data plumbing, not external LLM availability.
    monkeypatch.setattr(
        pipeline_routes, "enrich_bundles_with_rationale", _identity_enrichment
    )
    monkeypatch.setattr(
        pipeline_routes,
        "aggregate_system_summary",
        lambda rationales: "Deployment E2E summary.",
    )
    monkeypatch.setattr(
        pipeline_routes,
        "generate_next_steps",
        lambda **_: "Review the selected hypothesis and literature evidence.",
    )
    monkeypatch.setattr(
        pipeline_routes,
        "generate_display_title",
        lambda *_args, **_kwargs: "Deployment E2E Grounding",
    )

    subject = {"sub": USER_A}
    app.dependency_overrides[auth.verify_clerk_token] = _override_user(subject)

    uploaded_filename = DATASET.name
    run_id: str | None = None

    with TestClient(app) as client:
        try:
            for user_sub in (USER_A, USER_B):
                subject["sub"] = user_sub
                client.delete("/pipeline/runs/screening/all")
                client.delete("/corpus")

            subject["sub"] = USER_A

            upload = _post_file(
                client,
                "/pipeline/upload",
                DATASET,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            assert upload.status_code == 200, upload.text
            uploaded_filename = upload.json()["filename"]
            assert upload.json()["n_rows"] == 678

            paper = _post_file(client, "/corpus/upload", PDF, "application/pdf")
            assert paper.status_code == 200, paper.text
            assert paper.json()["n_chunks"] > 0

            stats = client.get("/corpus/stats")
            assert stats.status_code == 200, stats.text
            assert stats.json()["n_papers"] == 1
            assert stats.json()["n_chunks_total"] > 0

            run = client.post(
                "/pipeline/automated-screening-iteration",
                json={
                    "filename": uploaded_filename,
                    "run_name": "Deployment E2E",
                    "regime_id": 1,
                },
            )
            assert run.status_code == 200, run.text
            run_id = run.json()["run_id"]
            assert run_id

            screening = client.post(
                "/pipeline/screening-stats",
                json={
                    "filename": uploaded_filename,
                    "regime_id": 1,
                    "run_name": "Deployment E2E",
                    "run_id": run_id,
                },
            )
            assert screening.status_code == 200, screening.text
            assert len(screening.json()["bundles"]) > 0

            grounded = client.post(
                "/pipeline/screening-grounded",
                json={
                    "filename": uploaded_filename,
                    "regime_id": 1,
                    "run_name": "Deployment E2E",
                    "run_id": run_id,
                },
            )
            assert grounded.status_code == 200, grounded.text
            grounded_payload = grounded.json()
            assert len(grounded_payload["bundles"]) > 0
            citations = [
                c
                for bundle in grounded_payload["bundles"]
                for c in bundle.get("citations", [])
            ]
            assert citations, "Grounding returned no literature citations."
            assert any(c["source"] == "corpus" for c in citations), citations

            subject["sub"] = USER_B
            assert client.get("/corpus/stats").json()["n_papers"] == 0
            runs_b = client.get("/pipeline/runs")
            assert runs_b.status_code == 200, runs_b.text
            assert all(r["run_id"] != run_id for r in runs_b.json())
            assert client.get(f"/pipeline/status/{run_id}").status_code == 404

        finally:
            for user_sub in (USER_A, USER_B):
                subject["sub"] = user_sub
                client.delete("/pipeline/runs/screening/all")
                client.delete("/corpus")
            _cleanup_raw_uploads(uploaded_filename)
            for user_sub in (USER_A, USER_B):
                tenant_dir = ROOT / "data/corpus" / tenant_storage_key(user_sub)
                if tenant_dir.exists() and not any(tenant_dir.iterdir()):
                    tenant_dir.rmdir()
            app.dependency_overrides.pop(auth.verify_clerk_token, None)
