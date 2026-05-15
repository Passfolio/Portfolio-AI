from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings


def make_session_factory(
    database_url: str,
    connect_args: dict | None = None,
) -> async_sessionmaker[AsyncSession]:
    engine = create_async_engine(database_url, echo=False, connect_args=connect_args or {})
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


_SessionLocal = make_session_factory(get_settings().database_url)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with _SessionLocal() as session:
        yield session
