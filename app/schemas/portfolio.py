from __future__ import annotations

from pydantic import BaseModel


class FromPdfRequest(BaseModel):
    pdfS3Url: str
    userId: int | None = None
