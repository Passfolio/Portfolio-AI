import pytest
from sqlalchemy import text


@pytest.mark.asyncio
async def test_pgvector_extension_is_installed(test_settings):
    """Docker 컨테이너의 pgvector 익스텐션이 정상 설치되었는지 확인."""
    from app.db.session import make_session_factory

    SessionLocal = make_session_factory(test_settings.database_url, connect_args={"ssl": False})
    async with SessionLocal() as session:
        result = await session.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        )
        row = result.fetchone()
    assert row is not None, "pgvector extension not found — run: docker compose up -d"
    assert row[0] == "vector"


@pytest.mark.asyncio
async def test_vector_type_usable(test_settings):
    """VECTOR 타입으로 임시 테이블을 생성/삭제하여 pgvector 동작 검증."""
    from app.db.session import make_session_factory

    SessionLocal = make_session_factory(test_settings.database_url, connect_args={"ssl": False})
    async with SessionLocal() as session:
        await session.execute(
            text("CREATE TEMP TABLE _vec_test (id SERIAL PRIMARY KEY, emb VECTOR(3));")
        )
        await session.execute(
            text("INSERT INTO _vec_test (emb) VALUES ('[0.1, 0.2, 0.3]');")
        )
        result = await session.execute(text("SELECT emb FROM _vec_test LIMIT 1;"))
        row = result.fetchone()
    assert row is not None
