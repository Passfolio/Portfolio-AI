from __future__ import annotations

from pydantic import BaseModel


class AssessRequest(BaseModel):
    ai_job_id: str
    code_analysis_urls: list[str]
    merge: bool = False
