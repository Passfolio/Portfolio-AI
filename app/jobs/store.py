from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID, uuid4


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE    = "done"
    ERROR   = "error"


@dataclass
class JobState:
    job_id:  UUID      = field(default_factory=uuid4)
    status:  JobStatus = JobStatus.PENDING
    result:  dict | None = None
    message: str | None  = None


_store: dict[str, JobState] = {}


def create_job() -> JobState:
    job = JobState()
    _store[str(job.job_id)] = job
    return job


def get_job(job_id: str) -> JobState | None:
    return _store.get(job_id)


def update_job(
    job_id: str,
    status: JobStatus,
    result: dict | None = None,
    message: str | None = None,
) -> None:
    if job := _store.get(job_id):
        job.status  = status
        job.result  = result
        job.message = message
