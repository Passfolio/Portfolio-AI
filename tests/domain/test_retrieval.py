from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import text

from app.db.session import make_session_factory
from app.domain.retrieval.bm25_store import gin_index_exists
from app.domain.retrieval.embedder import embed
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


# ── bm25_store ────────────────────────────────────────────────────────────────

async def test_gin_index_exists(test_settings):
    SessionLocal = make_session_factory(
        test_settings.database_url, connect_args={"ssl": False}
    )
    async with SessionLocal() as session:
        exists = await gin_index_exists(session)
    assert exists is True
