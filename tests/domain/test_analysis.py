import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.domain.analysis.generator import grounded_generate
from app.domain.analysis.prompts import SYSTEM_PROMPT
from app.domain.analysis.schemas import ImprovedSection
from app.domain.pdf.schemas import PdfSection
from app.domain.retrieval.schemas import SearchResult


def _make_section() -> PdfSection:
    return PdfSection(
        section_title="프로젝트 경험",
        section_type="project",
        plain_text="Python과 FastAPI를 사용하여 REST API 개발",
        markdown="## 프로젝트 경험\n\nPython과 FastAPI를 사용하여 REST API 개발",
        page=1,
        order=0,
        parent_title=None,
        token_count=20,
    )


def _make_search_result() -> SearchResult:
    return SearchResult(
        chunk_id=uuid.uuid4(),
        pdf_id="test-pdf-001",
        section_title="기술 스택",
        content="FastAPI 기반 비동기 REST API 서버 구축 경험",
        section_type="skill",
        score=0.9,
        page=2,
        order=1,
        parent_title=None,
    )


async def test_no_context_refuses_generation():
    section = _make_section()

    result = await grounded_generate(section, context=[])

    assert isinstance(result, ImprovedSection)
    assert result.status == "no_context"
    assert result.after == section.markdown
    assert result.before == section.markdown


async def test_with_context_returns_before_after():
    section = _make_section()
    context = [_make_search_result()]
    mock_response = MagicMock(text='{"after": "## 프로젝트 경험\\n\\nPython과 FastAPI로 고성능 REST API 설계 및 구현"}')

    with patch(
        "app.domain.analysis.generator.get_gemini_model"
    ) as mock_get_model:
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_get_model.return_value = mock_model

        result = await grounded_generate(section, context=context)

    assert isinstance(result, ImprovedSection)
    assert result.status == "improved"
    assert result.before == section.markdown
    assert result.after == "## 프로젝트 경험\n\nPython과 FastAPI로 고성능 REST API 설계 및 구현"


async def test_no_hallucinated_skills():
    assert "절대 생성하지 마세요" in SYSTEM_PROMPT
