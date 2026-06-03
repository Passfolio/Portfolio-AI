from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from app.jobs.store import JobStatus


class JobStatusResponse(BaseModel):
    job_id:  str
    status:  JobStatus
    result:  Any = None
    message: str | None  = None
