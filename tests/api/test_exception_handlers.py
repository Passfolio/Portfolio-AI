from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


def test_http_exception_returns_message_not_detail():
    """HTTPException 발생 시 {"message": "..."} 형식으로 반환되고 "detail" 키가 없어야 한다."""
    with TestClient(app) as client:
        response = client.get("/api/v1/jobs/nonexistent-job-id")
    assert response.status_code == 404
    body = response.json()
    assert "message" in body
    assert "detail" not in body


def test_http_exception_message_content():
    """HTTPException의 detail 값이 message 필드로 그대로 전달되어야 한다."""
    with TestClient(app) as client:
        response = client.get("/api/v1/jobs/some-random-id")
    assert response.status_code == 404
    assert "some-random-id" in response.json()["message"]


def test_global_exception_handler_returns_500_message():
    """처리되지 않은 예외 발생 시 500 + {"message": "서버 내부 처리 중 오류가 발생했습니다."} 반환."""
    with patch("app.api.v1.jobs.get_job", side_effect=RuntimeError("DB down")):
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/jobs/some-id")
    assert response.status_code == 500
    body = response.json()
    assert body == {"message": "서버 내부 처리 중 오류가 발생했습니다."}
