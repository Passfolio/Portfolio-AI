import dataclasses
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.db.session import get_db
from app.domain.analysis.schemas import ImprovedSection
from app.main import app

_MOCK_PDF_SOURCE = "tests/assets/mock_data.json"
_FAKE_IMPROVED = ImprovedSection(
    section_title="프로젝트 경험",
    section_type="project",
    order=0,
    before="## 원본",
    after="## 개선된 내용",
    status="improved",
)


async def _mock_db():
    yield MagicMock()


@pytest.fixture
def override_db():
    app.dependency_overrides[get_db] = _mock_db
    yield
    app.dependency_overrides.clear()


async def test_analyze_returns_200_with_sections(override_db):
    with (
        patch("app.api.v1.analyze.embed", new_callable=AsyncMock) as mock_embed,
        patch("app.api.v1.analyze.upsert", new_callable=AsyncMock),
        patch("app.api.v1.analyze.hybrid_search", new_callable=AsyncMock) as mock_search,
        patch("app.api.v1.analyze.grounded_generate", new_callable=AsyncMock) as mock_gen,
    ):
        mock_embed.return_value = [[0.1] * 1536]
        mock_search.return_value = []
        mock_gen.return_value = _FAKE_IMPROVED

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/analyze",
                json={"pdf_source": _MOCK_PDF_SOURCE},
            )

    assert response.status_code == 200
    data = response.json()
    assert "pdf_id" in data
    assert isinstance(data["pdf_id"], str)
    assert "sections" in data
    assert len(data["sections"]) > 0


async def test_analyze_section_has_required_fields(override_db):
    with (
        patch("app.api.v1.analyze.embed", new_callable=AsyncMock) as mock_embed,
        patch("app.api.v1.analyze.upsert", new_callable=AsyncMock),
        patch("app.api.v1.analyze.hybrid_search", new_callable=AsyncMock) as mock_search,
        patch("app.api.v1.analyze.grounded_generate", new_callable=AsyncMock) as mock_gen,
    ):
        mock_embed.return_value = [[0.1] * 1536]
        mock_search.return_value = []
        mock_gen.return_value = _FAKE_IMPROVED

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/analyze",
                json={"pdf_source": _MOCK_PDF_SOURCE},
            )

    section = response.json()["sections"][0]
    for field in ("section_title", "section_type", "order", "before", "after", "status"):
        assert field in section, f"'{field}' 필드 누락"
