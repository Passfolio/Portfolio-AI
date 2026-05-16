import uuid

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.retrieval.embedder import embed
from app.domain.retrieval.schemas import SearchResult

_RRF_K = 60


def _row_to_result(row, score: float) -> SearchResult:
    return SearchResult(
        chunk_id=row.id,
        pdf_id=row.pdf_id,
        section_title=row.section_title,
        content=row.content,
        section_type=row.section_type,
        score=score,
        page=row.page,
        order=row.order,
        parent_title=row.parent_title,
    )


async def vector_search(
    embedding: list[float],
    pdf_id: str,
    session: AsyncSession,
    top_k: int = 20,
) -> list[SearchResult]:
    """pgvector 코사인 유사도 검색."""
    rows = await session.execute(
        text(
            "SELECT *, 1 - (embedding <=> CAST(:vec AS vector)) AS score "
            "FROM portfolio_chunks "
            "WHERE pdf_id = :pdf_id "
            "ORDER BY embedding <=> CAST(:vec AS vector) "
            "LIMIT :top_k"
        ),
        {"vec": str(embedding), "pdf_id": pdf_id, "top_k": top_k},
    )
    return [_row_to_result(r, r.score) for r in rows.fetchall()]


async def bm25_search(
    query_text: str,
    pdf_id: str,
    session: AsyncSession,
    top_k: int = 20,
) -> list[SearchResult]:
    """PostgreSQL Full-Text 검색 (한국어 포함 simple 딕셔너리 사용)."""
    rows = await session.execute(
        text(
            "SELECT *, ts_rank_cd("
            "  to_tsvector('simple', content),"
            "  plainto_tsquery('simple', :query)"
            ") AS score "
            "FROM portfolio_chunks "
            "WHERE pdf_id = :pdf_id "
            "  AND to_tsvector('simple', content) @@ plainto_tsquery('simple', :query) "
            "ORDER BY score DESC "
            "LIMIT :top_k"
        ),
        {"query": query_text, "pdf_id": pdf_id, "top_k": top_k},
    )
    return [_row_to_result(r, r.score) for r in rows.fetchall()]


def rrf_merge(
    vector_results: list[SearchResult],
    bm25_results: list[SearchResult],
    top_k: int = 5,
) -> list[SearchResult]:
    """Reciprocal Rank Fusion으로 두 결과 리스트를 결합."""
    scores: dict[uuid.UUID, float] = {}
    sources: dict[uuid.UUID, SearchResult] = {}

    for rank, result in enumerate(vector_results):
        scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
        sources[result.chunk_id] = result

    for rank, result in enumerate(bm25_results):
        scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
        sources[result.chunk_id] = result

    ranked = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)[:top_k]
    return [
        SearchResult(**{**sources[cid].__dict__, "score": scores[cid]})
        for cid in ranked
    ]


async def hybrid_search(
    query_text: str,
    pdf_id: str,
    session: AsyncSession,
    top_k: int = 5,
) -> list[SearchResult]:
    """vector + BM25 하이브리드 검색 → RRF 결합."""
    embedding = (await embed([query_text]))[0]
    vec_results = await vector_search(embedding, pdf_id, session, top_k=top_k * 4)
    bm25_results = await bm25_search(query_text, pdf_id, session, top_k=top_k * 4)
    return rrf_merge(vec_results, bm25_results, top_k=top_k)
