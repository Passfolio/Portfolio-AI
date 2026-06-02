from __future__ import annotations

from pydantic import BaseModel


class FromPdfRequest(BaseModel):
    pdf_url: str
    user_id: int | None = None
