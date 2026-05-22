from __future__ import annotations

from fastapi import FastAPI

from app.api.v1.cover_letter import router as cover_letter_router
from app.api.v1.jobs import router as jobs_router
from app.api.v1.portfolio import router as portfolio_router

app = FastAPI(title="Passfolio AI", version="1.0.0")

app.include_router(cover_letter_router, prefix="/api/v1")
app.include_router(portfolio_router,    prefix="/api/v1")
app.include_router(jobs_router,         prefix="/api/v1")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
