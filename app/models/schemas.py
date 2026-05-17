from pydantic import BaseModel


class ImprovedSectionOut(BaseModel):
    section_title: str
    section_type: str
    order: int
    before: str
    after: str
    status: str


class AnalyzeRequest(BaseModel):
    pdf_source: str
    tech_data: dict | None = None


class AnalyzeResponse(BaseModel):
    pdf_id: str
    sections: list[ImprovedSectionOut]
