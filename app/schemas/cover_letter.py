from __future__ import annotations

from pydantic import BaseModel


class FromPortfolioRequest(BaseModel):
    pdf_url:           str
    user_id:           int | None = None
    job_position:      str = ""
    career:            str = ""
    code_analysis_url: str | None = None


class FromCoverLetterRequest(BaseModel):
    pdf_url:           str
    user_id:           int | None = None
    job_position:      str = ""
    career:            str = ""
    code_analysis_url: str | None = None
