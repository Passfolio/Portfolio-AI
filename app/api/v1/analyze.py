import dataclasses
import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.domain.analysis.generator import grounded_generate
from app.domain.pdf.loader import load_mock_json
from app.domain.retrieval.embedder import embed
from app.domain.retrieval.searcher import hybrid_search
from app.domain.retrieval.vector_store import upsert
from app.models.schemas import AnalyzeRequest, AnalyzeResponse, ImprovedSectionOut

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: AnalyzeRequest,
    session: AsyncSession = Depends(get_db),
) -> AnalyzeResponse:
    pdf_id = str(uuid.uuid4())

    sections = await load_mock_json(request.pdf_source)
    embeddings = await embed([s.plain_text for s in sections])
    await upsert(pdf_id, sections, embeddings, session)

    improved = []
    for section in sections:
        context = await hybrid_search(section.plain_text, pdf_id, session)
        result = await grounded_generate(section, context, request.tech_data)
        improved.append(ImprovedSectionOut(**dataclasses.asdict(result)))

    return AnalyzeResponse(pdf_id=pdf_id, sections=improved)
