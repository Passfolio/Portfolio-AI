from unittest.mock import patch

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import verify_internal_key
from app.core.config import Settings

_test_app = FastAPI()


@_test_app.get("/test-auth", dependencies=[Depends(verify_internal_key)])
def _test_route():
    return {"ok": True}


def test_settings_has_passfolio_internal_api_key_field():
    s = Settings(postgres_user="u", postgres_password="p", passfolio_internal_api_key="")
    assert s.passfolio_internal_api_key == ""


def test_settings_passfolio_internal_api_key_accepts_value():
    s = Settings(
        postgres_user="u",
        postgres_password="p",
        passfolio_internal_api_key="secret-key",
    )
    assert s.passfolio_internal_api_key == "secret-key"


def test_valid_key_returns_200():
    with patch("app.api.dependencies.get_settings") as mock_cfg:
        mock_cfg.return_value.passfolio_internal_api_key = "test-key-123"
        client = TestClient(_test_app)
        response = client.get("/test-auth", headers={"X-INTERNAL-API-KEY": "test-key-123"})
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_missing_header_returns_401():
    with patch("app.api.dependencies.get_settings") as mock_cfg:
        mock_cfg.return_value.passfolio_internal_api_key = "test-key-123"
        client = TestClient(_test_app)
        response = client.get("/test-auth")
    assert response.status_code == 401


def test_wrong_key_returns_401():
    with patch("app.api.dependencies.get_settings") as mock_cfg:
        mock_cfg.return_value.passfolio_internal_api_key = "test-key-123"
        client = TestClient(_test_app)
        response = client.get("/test-auth", headers={"X-INTERNAL-API-KEY": "wrong-key"})
    assert response.status_code == 401


def test_empty_configured_key_rejects_any_header():
    with patch("app.api.dependencies.get_settings") as mock_cfg:
        mock_cfg.return_value.passfolio_internal_api_key = ""
        client = TestClient(_test_app)
        response = client.get("/test-auth", headers={"X-INTERNAL-API-KEY": "anything"})
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Portfolio router integration tests
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock

from app.main import app as main_app


def test_portfolio_from_pdf_requires_key():
    with patch("app.api.dependencies.get_settings") as mock_cfg:
        mock_cfg.return_value.passfolio_internal_api_key = "prod-key"
        client = TestClient(main_app)
        response = client.post(
            "/api/v1/portfolio/from-pdf",
            json={"pdf_url": "s3://bucket/file.pdf", "user_id": 1},
        )
    assert response.status_code == 401


def test_portfolio_from_pdf_passes_with_key():
    from app.jobs.store import JobStatus
    with patch("app.api.dependencies.get_settings") as mock_cfg, \
         patch("app.api.v1.portfolio.create_job") as mock_job, \
         patch("app.api.v1.portfolio.run_portfolio_from_pdf_task"):
        mock_cfg.return_value.passfolio_internal_api_key = "prod-key"
        mock_job.return_value = MagicMock(job_id="00000000-0000-0000-0000-000000000001", status=JobStatus.PENDING)
        client = TestClient(main_app)
        response = client.post(
            "/api/v1/portfolio/from-pdf",
            json={"pdf_url": "s3://bucket/file.pdf", "user_id": 1},
            headers={"X-INTERNAL-API-KEY": "prod-key"},
        )
    assert response.status_code != 401


# ---------------------------------------------------------------------------
# Cover-letter router integration tests
# ---------------------------------------------------------------------------

def test_cover_letter_from_pdf_requires_key():
    with patch("app.api.dependencies.get_settings") as mock_cfg:
        mock_cfg.return_value.passfolio_internal_api_key = "prod-key"
        client = TestClient(main_app)
        response = client.post(
            "/api/v1/cover-letter/from-pdf",
            json={"pdf_url": "s3://bucket/file.pdf", "user_id": 1},
        )
    assert response.status_code == 401


def test_cover_letter_from_pdf_passes_with_key():
    from app.jobs.store import JobStatus
    with patch("app.api.dependencies.get_settings") as mock_cfg, \
         patch("app.api.v1.cover_letter.create_job") as mock_job, \
         patch("app.api.v1.cover_letter.run_cover_letter_from_pdf_task"):
        mock_cfg.return_value.passfolio_internal_api_key = "prod-key"
        mock_job.return_value = MagicMock(job_id="00000000-0000-0000-0000-000000000002", status=JobStatus.PENDING)
        client = TestClient(main_app)
        response = client.post(
            "/api/v1/cover-letter/from-pdf",
            json={"pdf_url": "s3://bucket/file.pdf", "user_id": 1},
            headers={"X-INTERNAL-API-KEY": "prod-key"},
        )
    assert response.status_code != 401
