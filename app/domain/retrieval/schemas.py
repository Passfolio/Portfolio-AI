import uuid
from dataclasses import dataclass

from app.domain.pdf.schemas import PdfSection


@dataclass
class EmbeddedSection:
    section: PdfSection
    embedding: list[float]
    chunk_id: uuid.UUID


@dataclass
class SearchResult:
    chunk_id: uuid.UUID
    pdf_id: str
    section_title: str
    content: str
    section_type: str
    score: float
    page: int
    order: int
    parent_title: str | None
