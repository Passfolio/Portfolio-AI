from unittest.mock import AsyncMock, MagicMock, patch

from app.core.config import Settings


def test_settings_has_be_base_url():
    s = Settings(
        postgres_user="u",
        postgres_password="p",
        be_base_url="http://be-service:8080",
    )
    assert s.be_base_url == "http://be-service:8080"


def test_settings_be_base_url_default():
    s = Settings(
        postgres_user="u",
        postgres_password="p",
    )
    assert s.be_base_url == "http://localhost:8080"


async def test_notify_be_done():
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("app.services.webhook.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        from app.services.webhook import notify_be
        await notify_be("job-abc", output_pdf_url="https://s3.example.com/f.pdf")

    mock_client.post.assert_called_once()
    payload = mock_client.post.call_args.kwargs["json"]
    assert payload["status"] == "DONE"
    assert payload["ai_job_id"] == "job-abc"
    assert payload["output_pdf_url"] == "https://s3.example.com/f.pdf"
    assert payload["error_message"] is None
    mock_resp.raise_for_status.assert_called_once()


async def test_notify_be_error():
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("app.services.webhook.httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        from app.services.webhook import notify_be
        await notify_be("job-xyz", error_message="파싱 오류")

    payload = mock_client.post.call_args.kwargs["json"]
    assert payload["status"] == "ERROR"
    assert payload["ai_job_id"] == "job-xyz"
    assert payload["output_pdf_url"] is None
    assert payload["error_message"] == "파싱 오류"
    mock_resp.raise_for_status.assert_called_once()


async def test_notify_be_posts_to_correct_url():
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("app.services.webhook.httpx.AsyncClient") as mock_cls, \
         patch("app.services.webhook.get_settings") as mock_settings:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_settings.return_value.be_base_url = "http://be-host:9090"

        from app.services.webhook import notify_be
        await notify_be("job-1", output_pdf_url="https://s3/f.pdf")

    call_url = mock_client.post.call_args.args[0]
    assert call_url == "http://be-host:9090/api/v1/ai/jobs/complete"


async def test_notify_be_sends_internal_api_key_header():
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("app.services.webhook.httpx.AsyncClient") as mock_cls, \
         patch("app.services.webhook.get_settings") as mock_settings:
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_settings.return_value.be_base_url = "http://be-host:9090"
        mock_settings.return_value.passfolio_internal_api_key = "webhook-secret"

        from app.services.webhook import notify_be
        await notify_be("job-2", output_pdf_url="https://s3/out.pdf")

    call_kwargs = mock_client.post.call_args.kwargs
    assert "headers" in call_kwargs, "POST call must include a 'headers' kwarg"
    assert call_kwargs["headers"]["X-INTERNAL-API-KEY"] == "webhook-secret"
