from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.api.v1.cover_letter import router as cover_letter_router
from app.api.v1.jobs import router as jobs_router
from app.api.v1.portfolio import router as portfolio_router

app = FastAPI(title="Passfolio AI", version="1.0.0")

app.include_router(cover_letter_router, prefix="/api/v1")
app.include_router(portfolio_router,    prefix="/api/v1")
app.include_router(jobs_router,         prefix="/api/v1")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"message": "서버 내부 처리 중 오류가 발생했습니다."},
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
