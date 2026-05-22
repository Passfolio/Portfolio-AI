from __future__ import annotations

from fastapi import Header, HTTPException

from app.core.config import get_settings


async def verify_api_key(
    x_api_key: str = Header(..., alias="X-API-Key"),
) -> None:
    if x_api_key != get_settings().api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
