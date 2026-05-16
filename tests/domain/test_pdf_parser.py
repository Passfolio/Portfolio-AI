import pytest

from app.domain.pdf.loader import CATEGORY_TYPE_MAP, load_mock_json
from app.domain.pdf.parser import _extract_sections_from_docling_dict, _split_into_chunks, _strip_markdown
from app.domain.pdf.schemas import PdfSection

_MOCK_DOCLING_DICT = {
    "texts": [
        {"label": "section_header", "text": "프로젝트 경험", "prov": [{"page_no": 1}]},
        {"label": "text", "text": "FastAPI로 AI 서버 구축했습니다.", "prov": [{"page_no": 1}]},
        {"label": "list_item", "text": "pgvector 연동", "prov": [{"page_no": 1}]},
        {"label": "section_header", "text": "보유 기술", "prov": [{"page_no": 2}]},
        {"label": "text", "text": "Python, FastAPI, PostgreSQL을 사용합니다.", "prov": [{"page_no": 2}]},
    ]
}

_MOCK_JSON_PATH = "tests/assets/mock_data.json"


async def test_load_mock_json_returns_sections():
    sections = await load_mock_json(_MOCK_JSON_PATH)
    assert len(sections) > 0
    assert all(isinstance(s, PdfSection) for s in sections)


async def test_plain_text_has_no_markdown_markers(mock_sections: list[PdfSection]):
    for section in mock_sections:
        assert "#" not in section.plain_text, f"# found in plain_text of '{section.section_title}'"
        assert "**" not in section.plain_text, f"** found in plain_text of '{section.section_title}'"
        assert section.plain_text == section.plain_text.strip()


async def test_section_metadata_populated(mock_sections: list[PdfSection]):
    for i, section in enumerate(mock_sections):
        assert section.order == i, f"Expected order {i}, got {section.order}"
        assert section.token_count > 0, f"token_count must be > 0 for '{section.section_title}'"
        assert section.page >= 1


async def test_section_type_classified(mock_sections: list[PdfSection]):
    category_to_section = {s.section_title: s for s in mock_sections}

    skill_section = next(
        (s for s in mock_sections if s.section_type == "skill"), None
    )
    assert skill_section is not None, "직무역량 카테고리가 skill 타입으로 분류되어야 함"

    project_section = next(
        (s for s in mock_sections if s.section_type == "project"), None
    )
    assert project_section is not None, "프로젝트 카테고리가 project 타입으로 분류되어야 함"

    experience_section = next(
        (s for s in mock_sections if s.section_type == "experience"), None
    )
    assert experience_section is not None, "경력 카테고리가 experience 타입으로 분류되어야 함"

    education_section = next(
        (s for s in mock_sections if s.section_type == "education"), None
    )
    assert education_section is not None, "학력 카테고리가 education 타입으로 분류되어야 함"


async def test_markdown_contains_key_points_and_achievements(mock_sections: list[PdfSection]):
    section_with_achievements = next(
        (s for s in mock_sections if "주요 성과" in s.markdown), None
    )
    assert section_with_achievements is not None, "achievements가 있는 섹션의 markdown에 '주요 성과'가 포함되어야 함"

    section_with_kp = next(
        (s for s in mock_sections if "핵심 포인트" in s.markdown), None
    )
    assert section_with_kp is not None, "key_points가 있는 섹션의 markdown에 '핵심 포인트'가 포함되어야 함"


async def test_chunking_splits_large_section():
    # 문단당 약 60 토큰 × 10개 = 약 600 토큰 → 512 초과로 분할됨
    long_para = (
        "이 프로젝트는 FastAPI와 PostgreSQL을 기반으로 한 AI 포트폴리오 분석 시스템입니다. "
        "PDF 문서를 파싱하여 섹션별로 청킹하고 OpenAI 임베딩으로 벡터화하여 저장합니다. "
        "pgvector를 활용한 코사인 유사도 검색과 BM25 키워드 검색을 결합한 하이브리드 검색을 구현하였습니다."
    )
    paragraphs = "\n\n".join([long_para] * 10)

    chunks = _split_into_chunks(
        title="긴 섹션",
        markdown=paragraphs,
        section_type="general",
        page=1,
        start_order=0,
        parent_title="긴 섹션",
    )

    assert len(chunks) > 1, "512 토큰 초과 섹션은 복수 청크로 분할되어야 함"
    for chunk in chunks:
        assert chunk.parent_title == "긴 섹션"
        assert chunk.token_count > 0


async def test_strip_markdown_removes_markers():
    md = "## 제목\n\n**굵은** 텍스트와 `코드` 그리고 [링크](http://example.com)"
    plain = _strip_markdown(md)
    assert "#" not in plain
    assert "**" not in plain
    assert "`" not in plain
    assert "링크" in plain
    assert "굵은" in plain


def test_docling_dict_returns_sections():
    sections = _extract_sections_from_docling_dict(_MOCK_DOCLING_DICT)
    assert len(sections) == 2
    assert all(isinstance(s, PdfSection) for s in sections)
    assert sections[0].section_title == "프로젝트 경험"
    assert sections[1].section_title == "보유 기술"


def test_docling_dict_page_numbers():
    sections = _extract_sections_from_docling_dict(_MOCK_DOCLING_DICT)
    assert sections[0].page == 1
    assert sections[1].page == 2


def test_docling_dict_list_item_in_markdown():
    sections = _extract_sections_from_docling_dict(_MOCK_DOCLING_DICT)
    project_section = sections[0]
    assert "- pgvector 연동" in project_section.markdown


def test_docling_dict_plain_text_no_markdown():
    sections = _extract_sections_from_docling_dict(_MOCK_DOCLING_DICT)
    for section in sections:
        assert "#" not in section.plain_text
        assert "**" not in section.plain_text
