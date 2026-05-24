from __future__ import annotations

from pydantic import BaseModel


class FromPortfolioRequest(BaseModel):
    pdfS3Url:     str
    userId:       int | None = None
    job_position: str = ""
    career:       str = ""


class ToCoverLetterRequest(BaseModel):
    pdfS3Url:     str
    userId:       int | None = None
    job_position: str = ""
    career:       str = ""


class FromCoverLetterRequest(BaseModel):
    pdfS3Url: str
    userId:   int | None = None
