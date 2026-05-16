from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.pdf.schemas import PdfSection
from app.models.portfolio_chunk import PortfolioChunk


async def upsert(
    pdf_id: str,
    sections: list[PdfSection],
    embeddings: list[list[float]],
    session: AsyncSession,
) -> list[PortfolioChunk]:
    """섹션 + 임베딩 → portfolio_chunks 테이블에 저장."""
    chunks = [
        PortfolioChunk(
            pdf_id=pdf_id,
            section_title=section.section_title,
            content=section.plain_text,
            embedding=embedding,
            section_type=section.section_type,
            page=section.page,
            order=section.order,
            parent_title=section.parent_title,
        )
        for section, embedding in zip(sections, embeddings)
    ]
    session.add_all(chunks)
    await session.commit()
    return chunks
