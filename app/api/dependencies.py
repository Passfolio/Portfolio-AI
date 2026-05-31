from __future__ import annotations

from fastapi import Header, HTTPException

from app.core.config import get_settings


async def verify_internal_key(
    x_internal_api_key: str | None = Header(default=None),
) -> None:
    """BE → FastAPI 요청의 X-INTERNAL-API-KEY 헤더 검증.
    Header(default=None)으로 헤더 누락 시 422 대신 401을 직접 반환.
    """
    expected = get_settings().passfolio_internal_api_key
    if x_internal_api_key is None or x_internal_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
