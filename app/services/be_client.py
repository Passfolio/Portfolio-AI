"""BE HTTP 클라이언트 — ATK 세션 관리 및 아티클 저장."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta

import httpx

from app.core.config import get_settings

_logger = logging.getLogger(__name__)

_ATK_REFRESH_INTERVAL = timedelta(minutes=25)


class BeClient:
    def __init__(self) -> None:
        self._atk: str | None = None
        self._rtk: str | None = None
        self._atk_issued_at: datetime | None = None
        self._lock = asyncio.Lock()

    # ── 인증 ─────────────────────────────────────────────────────

    async def get_atk(self) -> str:
        async with self._lock:
            if self._atk is None:
                await self._login()
            elif self._needs_refresh():
                await self._refresh()
            return self._atk  # type: ignore[return-value]

    def _needs_refresh(self) -> bool:
        if self._atk_issued_at is None:
            return True
        return datetime.now() - self._atk_issued_at >= _ATK_REFRESH_INTERVAL

    async def _login(self) -> None:
        cfg = get_settings()
        url = f"{cfg.be_base_url}/api/v1/system/auth/login"
        _logger.info("BE 서비스 계정 로그인 시도")
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                url,
                json={
                    "username": cfg.service_account_email,
                    "password": cfg.service_account_password,
                    "rememberMe": False,
                },
            )
        r.raise_for_status()
        self._atk = r.cookies["ATK"]
        self._rtk = r.cookies.get("RTK")
        self._atk_issued_at = datetime.now()
        _logger.info("BE 로그인 성공, ATK 발급 완료")

    async def _refresh(self) -> None:
        cfg = get_settings()
        url = f"{cfg.be_base_url}/api/v1/auth/refresh"
        _logger.info("ATK 갱신 시도")
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.post(url, cookies={"RTK": self._rtk or ""})
            r.raise_for_status()
            self._atk = r.cookies["ATK"]
            self._atk_issued_at = datetime.now()
            _logger.info("ATK 갱신 완료")
        except Exception as exc:
            _logger.warning("ATK 갱신 실패, 재로그인 시도: %s", exc)
            await self._login()

    # ── 아티클 저장 ───────────────────────────────────────────────

    async def post_article(
        self,
        title: str,
        contents: str,
        file_urls: list[str] | None = None,
    ) -> dict:
        """POST /api/v1/articles. 401 응답 시 ATK 갱신 후 1회 재시도."""
        cfg = get_settings()
        url = f"{cfg.be_base_url}/api/v1/articles"
        body = {"title": title, "contents": contents, "fileUrls": file_urls or []}

        for attempt in range(2):
            atk = await self.get_atk()
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(url, json=body, cookies={"ATK": atk})

            if r.status_code == 201:
                return r.json()

            if r.status_code == 401 and attempt == 0:
                _logger.warning("아티클 저장 401, ATK 갱신 후 재시도")
                async with self._lock:
                    self._atk = None  # 강제 재로그인
                continue

            r.raise_for_status()

        raise RuntimeError("아티클 저장 실패")


be_client = BeClient()
