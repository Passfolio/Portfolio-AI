from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    pdf_id: str
    github_code: str
    

class SectionResult(BaseModel):
    original: str
    enhanced: str
    evidence_code: str
    confidence: float


class AnalyzeResponse(BaseModel):
    pdf_id: str
    sections: list[SectionResult]
    roadmap: list[str]
