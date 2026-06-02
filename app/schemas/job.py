from __future__ import annotations

from pydantic import BaseModel

from app.jobs.store import JobStatus


class JobStatusResponse(BaseModel):
    job_id:  str
    status:  JobStatus
    result:  dict | None = None
    message: str | None  = None
