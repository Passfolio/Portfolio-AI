from dataclasses import dataclass

from pydantic import BaseModel


@dataclass
class ImprovedSection:
    section_title: str
    section_type: str
    order: int
    before: str
    after: str
    status: str  # "improved" | "no_context"


@dataclass
class AnalysisResult:
    pdf_id: str
    sections: list[ImprovedSection]


class GeminiOutput(BaseModel):
    after: str
