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
    status = "DONE" if output_pdf_url else "ERROR"
    payload = {
        "ai_job_id":      ai_job_id,
        "status":         status,
        "output_pdf_url": output_pdf_url,
        "error_message":  error_message,
    }
    be_url = f"{get_settings().be_base_url}/api/v1/ai/jobs/complete"
    logger.info("[Webhook] POST %s | status=%s | job=%s", be_url, status, ai_job_id)
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(be_url, json=payload)
        resp.raise_for_status()
    logger.info("[Webhook] 전송 완료: %s", ai_job_id)
