from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends

from app.api.dependencies import verify_internal_key
from app.jobs.store import create_job
from app.schemas.cover_letter import (
    FromCoverLetterRequest,
    FromPortfolioRequest,
)
from app.services._rag_utils import map_career_input
from app.schemas.job import JobStatusResponse
from app.services.cover_letter import (
    run_cover_letter_from_pdf_task,
    run_cover_letter_to_portfolio_task,
    run_portfolio_to_cover_letter_task,
)

router = APIRouter(
    prefix="/cover-letter",
    tags=["cover-letter"],
    dependencies=[Depends(verify_internal_key)],
)


@router.post("/from-portfolio", response_model=JobStatusResponse)
async def cover_letter_from_portfolio(
    req: FromPortfolioRequest,
    background_tasks: BackgroundTasks,
) -> JobStatusResponse:
    job = create_job()
    background_tasks.add_task(
        run_portfolio_to_cover_letter_task,
        job_id=str(job.job_id),
        pdf_s3_url=req.pdf_url,
        user_id=req.user_id,
        job=req.job_position or None,
        career=map_career_input(req.career) if req.career else None,
    )
    return JobStatusResponse(job_id=str(job.job_id), status=job.status)


@router.post("/from-pdf", response_model=JobStatusResponse)
async def cover_letter_from_pdf(
    req: FromCoverLetterRequest,
    background_tasks: BackgroundTasks,
) -> JobStatusResponse:
    job = create_job()
    background_tasks.add_task(
        run_cover_letter_from_pdf_task,
        job_id=str(job.job_id),
        pdf_s3_url=req.pdf_url,
        user_id=req.user_id,
        job=req.job_position or None,
        career=map_career_input(req.career) if req.career else None,
    )
    return JobStatusResponse(job_id=str(job.job_id), status=job.status)
