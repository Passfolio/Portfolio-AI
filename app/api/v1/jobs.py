from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.jobs.store import get_job
from app.schemas.job import JobStatusResponse

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str) -> JobStatusResponse:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job_id '{job_id}' 를 찾을 수 없습니다.")
    return JobStatusResponse(
        job_id=str(job.job_id),
        status=job.status,
        result=job.result,
        message=job.message,
    )
