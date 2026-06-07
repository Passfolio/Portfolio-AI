from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends

from app.api.dependencies import verify_internal_key
from app.jobs.store import create_job
from app.schemas.cover_letter import FromCoverLetterRequest
from app.schemas.job import JobStatusResponse
from app.schemas.portfolio import FromPdfRequest
from app.services._rag_utils import map_career_input
from app.services.cover_letter import run_cover_letter_to_portfolio_task
from app.services.portfolio import run_portfolio_from_pdf_task

router = APIRouter(
    prefix="/portfolio",
    tags=["portfolio"],
    dependencies=[Depends(verify_internal_key)],
)


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
        job=req.job_position or None,
        career=map_career_input(req.career) if req.career else None,
        code_analysis_urls=req.code_analysis_urls,
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
        job=req.job_position or None,
        career=map_career_input(req.career) if req.career else None,
        code_analysis_urls=req.code_analysis_urls,
    )
    return JobStatusResponse(job_id=str(job.job_id), status=job.status)
