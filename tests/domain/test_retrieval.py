from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import text

from app.db.session import make_session_factory
from app.domain.retrieval.bm25_store import gin_index_exists
from app.domain.retrieval.embedder import embed
from app.domain.retrieval.searcher import bm25_search, hybrid_search, rrf_merge, vector_search
from app.domain.retrieval.schemas import SearchResult
from app.domain.retrieval.vector_store import upsert

_PDF_ID = "test-retrieval-001"
_DIMS = 1536


def _fake_embedding(dim: int = _DIMS) -> list[float]:
    return [0.1] * dim


def _mock_openai_response(n: int) -> MagicMock:
    resp = MagicMock()
    resp.data = [MagicMock(embedding=_fake_embedding()) for _ in range(n)]
    return resp


# ── embedder ──────────────────────────────────────────────────────────────────

async def test_embed_returns_correct_shape():
    texts = ["안녕하세요", "Python FastAPI", "pgvector 연동", "임베딩 테스트", "검색"]
    with patch("app.domain.retrieval.embedder.AsyncOpenAI") as MockClient:
        MockClient.return_value.embeddings.create = AsyncMock(
            return_value=_mock_openai_response(len(texts))
        )
        result = await embed(texts)
    assert len(result) == len(texts)


async def test_embed_each_vector_has_1536_dims():
    texts = ["FastAPI", "PostgreSQL"]
    with patch("app.domain.retrieval.embedder.AsyncOpenAI") as MockClient:
        MockClient.return_value.embeddings.create = AsyncMock(
            return_value=_mock_openai_response(len(texts))
        )
        result = await embed(texts)
    assert all(len(vec) == _DIMS for vec in result)


# ── vector_store ──────────────────────────────────────────────────────────────

async def test_upsert_stores_chunks(test_settings, mock_sections):
    SessionLocal = make_session_factory(
        test_settings.database_url, connect_args={"ssl": False}
    )
    embeddings = [_fake_embedding() for _ in mock_sections]
    async with SessionLocal() as session:
        chunks = await upsert(_PDF_ID, mock_sections, embeddings, session)
    assert len(chunks) == len(mock_sections)
    async with SessionLocal() as session:
        await session.execute(
            text("DELETE FROM portfolio_chunks WHERE pdf_id = :pid"), {"pid": _PDF_ID}
        )
        await session.commit()


async def test_upsert_chunk_has_embedding(test_settings, mock_sections):
    SessionLocal = make_session_factory(
        test_settings.database_url, connect_args={"ssl": False}
    )
    embeddings = [_fake_embedding() for _ in mock_sections]
    async with SessionLocal() as session:
        chunks = await upsert(_PDF_ID, mock_sections, embeddings, session)
    assert all(chunk.embedding is not None for chunk in chunks)
    assert all(len(chunk.embedding) == _DIMS for chunk in chunks)
    async with SessionLocal() as session:
        await session.execute(
            text("DELETE FROM portfolio_chunks WHERE pdf_id = :pid"), {"pid": _PDF_ID}
        )
        await session.commit()


async def test_upsert_chunk_content_matches_plain_text(test_settings, mock_sections):
    SessionLocal = make_session_factory(
        test_settings.database_url, connect_args={"ssl": False}
    )
    embeddings = [_fake_embedding() for _ in mock_sections]
    async with SessionLocal() as session:
        chunks = await upsert(_PDF_ID, mock_sections, embeddings, session)
    for chunk, section in zip(chunks, mock_sections):
        assert chunk.content == section.plain_text
    async with SessionLocal() as session:
        await session.execute(
            text("DELETE FROM portfolio_chunks WHERE pdf_id = :pid"), {"pid": _PDF_ID}
        )
        await session.commit()


# ── searcher ─────────────────────────────────────────────────────────────────

_SEARCH_PDF_ID = "test-searcher-001"


@pytest.fixture
async def stored_chunks(test_settings, mock_sections):
    SessionLocal = make_session_factory(
        test_settings.database_url, connect_args={"ssl": False}
    )
    embeddings = [_fake_embedding() for _ in mock_sections]
    async with SessionLocal() as session:
        await upsert(_SEARCH_PDF_ID, mock_sections, embeddings, session)
    yield SessionLocal
    async with SessionLocal() as session:
        await session.execute(
            text("DELETE FROM portfolio_chunks WHERE pdf_id = :pid"),
            {"pid": _SEARCH_PDF_ID},
        )
        await session.commit()


async def test_vector_search_returns_results(stored_chunks):
    async with stored_chunks() as session:
        results = await vector_search(_fake_embedding(), _SEARCH_PDF_ID, session, top_k=3)
    assert len(results) > 0
    assert all(isinstance(r, SearchResult) for r in results)


async def test_bm25_search_returns_results(stored_chunks):
    async with stored_chunks() as session:
        results = await bm25_search("Python FastAPI", _SEARCH_PDF_ID, session, top_k=3)
    assert isinstance(results, list)


async def test_rrf_merge_combines_and_deduplicates():
    import uuid
    def _sr(chunk_id, score):
        return SearchResult(
            chunk_id=chunk_id, pdf_id="p", section_title="t",
            content="c", section_type="general", score=score,
            page=1, order=0, parent_title=None,
        )
    uid_a, uid_b, uid_c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    vec = [_sr(uid_a, 0.9), _sr(uid_b, 0.8), _sr(uid_c, 0.7)]
    bm25 = [_sr(uid_b, 0.95), _sr(uid_a, 0.85)]
    merged = rrf_merge(vec, bm25, top_k=3)
    ids = [r.chunk_id for r in merged]
    assert len(ids) == len(set(ids)), "중복 chunk_id 없어야 함"
    assert len(merged) <= 3


async def test_hybrid_search_returns_results(stored_chunks):
    with patch("app.domain.retrieval.searcher.embed") as mock_embed:
        mock_embed.return_value = [_fake_embedding()]
        async with stored_chunks() as session:
            results = await hybrid_search("Python FastAPI", _SEARCH_PDF_ID, session, top_k=3)
    assert isinstance(results, list)
    assert all(isinstance(r, SearchResult) for r in results)


# ── bm25_store ────────────────────────────────────────────────────────────────

async def test_gin_index_exists(test_settings):
    SessionLocal = make_session_factory(
        test_settings.database_url, connect_args={"ssl": False}
    )
    async with SessionLocal() as session:
        exists = await gin_index_exists(session)
    assert exists is True
