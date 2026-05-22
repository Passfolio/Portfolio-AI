from __future__ import annotations

from pydantic import BaseModel


class CoverLetterImproveRequest(BaseModel):
    section:      str
    text:         str
    job_position: str = ""
    career:       str = ""
    char_limit:   int | None = None


class CoverLetterImproveResponse(BaseModel):
    improved:  str
    reasoning: str
    changes:   list[str]


class FromPortfolioRequest(BaseModel):
    pdf_path:     str
    job_position: str = ""
    career:       str = ""


class ToCoverLetterRequest(BaseModel):
    pdf_path:     str
    job_position: str = ""
    career:       str = ""
