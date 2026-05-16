import uuid
from dataclasses import dataclass

from app.domain.pdf.schemas import PdfSection


@dataclass
class EmbeddedSection:
    section: PdfSection
    embedding: list[float]
    chunk_id: uuid.UUID
