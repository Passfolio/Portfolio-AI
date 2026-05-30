import inspect
from unittest.mock import AsyncMock, patch


def test_run_job_pipeline_is_coroutine():
    from app.services.pdf_pipeline import run_job_pipeline
    assert inspect.iscoroutinefunction(run_job_pipeline)


async def test_run_job_pipeline_done_calls_webhook():
    fake_result = {"sections": [{"text": "ok"}], "outputPdfS3Url": "https://s3/out.pdf"}

    with patch("app.services.pdf_pipeline.update_job") as mock_update, \
         patch("app.services.pdf_pipeline.notify_be", new_callable=AsyncMock) as mock_webhook:

        from app.services.pdf_pipeline import run_job_pipeline
        from app.jobs.store import JobStatus

        await run_job_pipeline("job-1", lambda: fake_result, tag="TEST")

        mock_webhook.assert_awaited_once_with(
            ai_job_id="job-1",
            output_pdf_url="https://s3/out.pdf",
            error_message=None,
        )
        mock_update.assert_any_call("job-1", JobStatus.DONE, result=fake_result)


async def test_run_job_pipeline_error_calls_webhook():
    with patch("app.services.pdf_pipeline.update_job") as mock_update, \
         patch("app.services.pdf_pipeline.notify_be", new_callable=AsyncMock) as mock_webhook:

        from app.services.pdf_pipeline import run_job_pipeline
        from app.jobs.store import JobStatus

        await run_job_pipeline("job-2", lambda: (_ for _ in ()).throw(ValueError("파싱 실패")), tag="TEST")

        mock_webhook.assert_awaited_once_with(
            ai_job_id="job-2",
            output_pdf_url=None,
            error_message="파싱 실패",
        )
        mock_update.assert_any_call("job-2", JobStatus.ERROR, message="파싱 실패")


async def test_run_job_pipeline_webhook_failure_does_not_propagate():
    """webhook 전송 실패가 Job 상태에 영향을 주지 않는다."""
    fake_result = {"sections": [], "outputPdfS3Url": None}

    with patch("app.services.pdf_pipeline.update_job"), \
         patch("app.services.pdf_pipeline.notify_be", new_callable=AsyncMock) as mock_webhook:

        mock_webhook.side_effect = Exception("BE 서버 다운")

        from app.services.pdf_pipeline import run_job_pipeline

        # 예외가 전파되지 않아야 한다
        await run_job_pipeline("job-3", lambda: fake_result, tag="TEST")
