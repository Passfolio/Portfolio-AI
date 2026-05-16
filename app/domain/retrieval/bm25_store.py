from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def gin_index_exists(session: AsyncSession) -> bool:
    """portfolio_chunks.content GIN 인덱스 존재 여부 확인."""
    result = await session.execute(
        text(
            "SELECT 1 FROM pg_indexes "
            "WHERE tablename = 'portfolio_chunks' "
            "AND indexdef LIKE '%to_tsvector%'"
        )
    )
    return result.fetchone() is not None
