from fastapi import FastAPI

from app.api.v1.analyze import router as analyze_router
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title="Passfolio AI Server",
    description="포트폴리오 PDF ↔ GitHub 소스코드 교차 검증 AI 서버",
    version="0.1.0",
    debug=settings.app_debug,
)

app.include_router(analyze_router, prefix="/api/v1", tags=["analysis"])


@app.get("/health", tags=["system"])
async def health_check() -> dict:
    return {"status": "ok", "env": settings.app_env}
