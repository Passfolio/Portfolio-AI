from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks

from app.jobs.store import create_job
from app.schemas.cover_letter import FromCoverLetterRequest
from app.schemas.job import JobStatusResponse
from app.schemas.portfolio import FromPdfRequest
from app.services.cover_letter import run_cover_letter_to_portfolio_task
from app.services.portfolio import run_portfolio_from_pdf_task

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.post("/from-pdf", response_model=JobStatusResponse)
async def portfolio_from_pdf(
    req: FromPdfRequest,
    background_tasks: BackgroundTasks,
) -> JobStatusResponse:
    job = create_job()
    background_tasks.add_task(
        run_portfolio_from_pdf_task,
        job_id=str(job.job_id),
        pdf_s3_url=req.pdf_url,
        user_id=req.user_id,
    )
    return JobStatusResponse(job_id=str(job.job_id), status=job.status)


@router.post("/from-cover-letter", response_model=JobStatusResponse)
async def portfolio_from_cover_letter(
    req: FromCoverLetterRequest,
    background_tasks: BackgroundTasks,
) -> JobStatusResponse:
    job = create_job()
    background_tasks.add_task(
        run_cover_letter_to_portfolio_task,
        job_id=str(job.job_id),
        pdf_s3_url=req.pdf_url,
        user_id=req.user_id,
    )
    return JobStatusResponse(job_id=str(job.job_id), status=job.status)
