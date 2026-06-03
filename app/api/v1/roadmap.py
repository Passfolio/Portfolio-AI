from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.api.dependencies import verify_internal_key
from app.jobs.store import JobStatus, create_job, get_job
from app.schemas.job import JobStatusResponse
from app.schemas.roadmap import AssessRequest
from app.services.roadmap_assess import run_assess_task

router = APIRouter(
    prefix="/roadmap",
    tags=["roadmap"],
    dependencies=[Depends(verify_internal_key)],
)


@router.post("/assess", response_model=JobStatusResponse)
async def assess(
    req: AssessRequest,
    background_tasks: BackgroundTasks,
) -> JobStatusResponse:
    if not req.code_analysis_urls:
        raise HTTPException(status_code=400, detail="code_analysis_urls가 비어 있습니다.")
    if len(req.code_analysis_urls) > 10:
        raise HTTPException(status_code=400, detail="code_analysis_urls는 최대 10개입니다.")

    job = create_job()
    background_tasks.add_task(
        run_assess_task,
        job_id=str(job.job_id),
        ai_job_id=req.ai_job_id,
        code_analysis_urls=req.code_analysis_urls,
        merge=req.merge,
    )
    return JobStatusResponse(job_id=str(job.job_id), status=job.status)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_assess_status(job_id: str) -> JobStatusResponse:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job_id '{job_id}'를 찾을 수 없습니다.")
    return JobStatusResponse(
        job_id=str(job.job_id),
        status=job.status,
        result=job.result,
        message=job.message,
    )


@router.get("/jobs/{job_id}/stream")
async def stream_assess(job_id: str) -> EventSourceResponse:
    async def generator():
        if get_job(job_id) is None:
            yield {"event": "error", "data": f"job_id '{job_id}'를 찾을 수 없습니다."}
            return

        while True:
            job = get_job(job_id)
            if job.status == JobStatus.DONE:
                yield {"event": "done", "data": json.dumps(job.result, ensure_ascii=False)}
                break
            if job.status == JobStatus.ERROR:
                yield {"event": "error", "data": job.message or "error"}
                break
            yield {"event": "status", "data": job.status.value}
            await asyncio.sleep(1)

    return EventSourceResponse(generator())
