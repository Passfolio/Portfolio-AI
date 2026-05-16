from openai import AsyncOpenAI

from app.core.config import get_settings


async def embed(texts: list[str]) -> list[list[float]]:
    """plain_text 리스트 → 1536차원 벡터 리스트."""
    client = AsyncOpenAI(api_key=get_settings().openai_api_key)
    resp = await client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
        dimensions=1536,
    )
    return [item.embedding for item in resp.data]
