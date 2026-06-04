from __future__ import annotations

import logging

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)


async def notify_be(
    ai_job_id: str,
    output_pdf_url: str | None = None,
    error_message: str | None = None,
) -> None:
    settings = get_settings()
    status = "DONE" if output_pdf_url else "ERROR"
    payload = {
        "ai_job_id":      ai_job_id,
        "status":         status,
        "output_pdf_url": output_pdf_url,
        "error_message":  error_message,
    }
    be_url = f"{settings.be_base_url}/api/v1/ai/jobs/complete"
    headers = {"X-INTERNAL-API-KEY": settings.passfolio_internal_api_key}
    logger.info("[Webhook] POST %s | status=%s | job=%s", be_url, status, ai_job_id)
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(be_url, json=payload, headers=headers)
        resp.raise_for_status()
    logger.info("[Webhook] 전송 완료: %s", ai_job_id)


async def notify_be_roadmap(
    ai_job_id: str,
    status: str,
    result: list | dict | None = None,
    error_message: str | None = None,
) -> None:
    """로드맵 전용 완료 콜백. PDF가 없는 로드맵은 평가 결과(result)를 BE에 직접 실어 보낸다.
    (포폴용 notify_be는 result를 싣지 않으므로 별도 엔드포인트로 분리.)"""
    settings = get_settings()
    payload = {
        "ai_job_id":     ai_job_id,
        "status":        status,
        "result":        result,
        "error_message": error_message,
    }
    be_url = f"{settings.be_base_url}/api/v1/ai/roadmap/complete"
    headers = {"X-INTERNAL-API-KEY": settings.passfolio_internal_api_key}
    logger.info("[Webhook] POST %s | status=%s | job=%s", be_url, status, ai_job_id)
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(be_url, json=payload, headers=headers)
        resp.raise_for_status()
    logger.info("[Webhook] 로드맵 전송 완료: %s", ai_job_id)
