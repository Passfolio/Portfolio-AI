from __future__ import annotations

from pydantic import BaseModel


class PortfolioImproveRequest(BaseModel):
    query:       str
    img_context: str = ""


class PortfolioImproveResponse(BaseModel):
    improved:  str
    reasoning: str
    changes:   list[str]


class FromPdfRequest(BaseModel):
    pdf_path: str
