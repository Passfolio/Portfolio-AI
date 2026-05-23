from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks

from app.jobs.store import create_job
from app.schemas.job import JobStatusResponse
from app.schemas.portfolio import FromPdfRequest, PortfolioImproveRequest, PortfolioImproveResponse
from app.services.portfolio import rag_portfolio, run_portfolio_from_pdf_task

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.post("/improve", response_model=PortfolioImproveResponse)
async def portfolio_improve(req: PortfolioImproveRequest) -> PortfolioImproveResponse:
    result = await rag_portfolio(query=req.query, img_context=req.img_context)
    return PortfolioImproveResponse(**result)


@router.post("/from-pdf", response_model=JobStatusResponse)
async def portfolio_from_pdf(
    req: FromPdfRequest,
    background_tasks: BackgroundTasks,
) -> JobStatusResponse:
    job = create_job()
    background_tasks.add_task(
        run_portfolio_from_pdf_task,
        job_id=str(job.job_id),
        pdf_path=req.pdf_path,
    )
    return JobStatusResponse(job_id=str(job.job_id), status=job.status)
