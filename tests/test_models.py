import uuid

import pytest
from sqlalchemy import text

from app.db.session import make_session_factory
from app.models.portfolio_chunk import PortfolioChunk


@pytest.mark.asyncio
async def test_portfolio_chunk_table_exists(test_settings):
    """portfolio_chunks 테이블이 DB에 존재하는지 확인 (alembic upgrade head 필요)."""
    SessionLocal = make_session_factory(
        test_settings.database_url, connect_args={"ssl": False}
    )
    async with SessionLocal() as session:
        result = await session.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name = 'portfolio_chunks'"
            )
        )
        row = result.fetchone()
    assert row is not None, "portfolio_chunks table not found — run: alembic upgrade head"


@pytest.mark.asyncio
async def test_portfolio_chunk_insert_and_retrieve(test_settings):
    """PortfolioChunk 행 삽입 후 조회 (embedding=None 허용)."""
    SessionLocal = make_session_factory(
        test_settings.database_url, connect_args={"ssl": False}
    )
    chunk_id = uuid.uuid4()
    async with SessionLocal() as session:
        chunk = PortfolioChunk(
            id=chunk_id,
            pdf_id="test-pdf-001",
            section_title="프로젝트 경험",
            content="FastAPI 기반 백엔드 서버를 개발했습니다.",
            embedding=None,
            section_type="experience",
            page=1,
            order=0,
            parent_title=None,
        )
        session.add(chunk)
        await session.commit()

    async with SessionLocal() as session:
        result = await session.execute(
            text("SELECT pdf_id, section_title, section_type FROM portfolio_chunks WHERE id = :id"),
            {"id": str(chunk_id)},
        )
        row = result.fetchone()

    assert row is not None
    assert row[0] == "test-pdf-001"
    assert row[1] == "프로젝트 경험"
    assert row[2] == "experience"

    async with SessionLocal() as session:
        await session.execute(
            text("DELETE FROM portfolio_chunks WHERE id = :id"), {"id": str(chunk_id)}
        )
        await session.commit()


@pytest.mark.asyncio
async def test_portfolio_chunk_embedding_dimension(test_settings):
    """Vector(1536) 임베딩 저장 및 조회 정확도 검증."""
    SessionLocal = make_session_factory(
        test_settings.database_url, connect_args={"ssl": False}
    )
    embedding = [0.1] * 1536
    chunk_id = uuid.uuid4()

    async with SessionLocal() as session:
        chunk = PortfolioChunk(
            id=chunk_id,
            pdf_id="test-pdf-embed",
            section_title="기술 스택",
            content="Python, FastAPI, PostgreSQL을 사용합니다.",
            embedding=embedding,
            section_type="skill",
            page=2,
            order=1,
            parent_title=None,
        )
        session.add(chunk)
        await session.commit()

    async with SessionLocal() as session:
        result = await session.execute(
            text(
                "SELECT array_length(embedding::real[], 1) FROM portfolio_chunks WHERE id = :id"
            ),
            {"id": str(chunk_id)},
        )
        row = result.fetchone()

    assert row is not None
    assert row[0] == 1536

    async with SessionLocal() as session:
        await session.execute(
            text("DELETE FROM portfolio_chunks WHERE id = :id"), {"id": str(chunk_id)}
        )
        await session.commit()
