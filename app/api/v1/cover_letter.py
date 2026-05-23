from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.jobs.store import create_job, get_job
from app.schemas.cover_letter import (
    CoverLetterImproveRequest,
    CoverLetterImproveResponse,
    FromPortfolioRequest,
    ToCoverLetterRequest,
)
from app.schemas.job import JobStatusResponse
from app.services.cover_letter import (
    rag_cover_letter,
    run_cover_letter_to_portfolio_task,
    run_portfolio_to_cover_letter_task,
)

router = APIRouter(prefix="/cover-letter", tags=["cover-letter"])


@router.post("/improve", response_model=CoverLetterImproveResponse)
async def cover_letter_improve(req: CoverLetterImproveRequest) -> CoverLetterImproveResponse:
    result = await rag_cover_letter(
        query=req.text,
        char_limit=req.char_limit,
        section=req.section,
    )
    return CoverLetterImproveResponse(**result)


@router.post("/from-portfolio", response_model=JobStatusResponse)
async def cover_letter_from_portfolio(
    req: FromPortfolioRequest,
    background_tasks: BackgroundTasks,
) -> JobStatusResponse:
    job = create_job()
    background_tasks.add_task(
        run_portfolio_to_cover_letter_task,
        job_id=str(job.job_id),
        pdf_path=req.pdf_path,
    )
    return JobStatusResponse(job_id=str(job.job_id), status=job.status)


@router.post("/to-portfolio", response_model=JobStatusResponse)
async def cover_letter_to_portfolio(
    req: ToCoverLetterRequest,
    background_tasks: BackgroundTasks,
) -> JobStatusResponse:
    job = create_job()
    background_tasks.add_task(
        run_cover_letter_to_portfolio_task,
        job_id=str(job.job_id),
        pdf_path=req.pdf_path,
    )
    return JobStatusResponse(job_id=str(job.job_id), status=job.status)
