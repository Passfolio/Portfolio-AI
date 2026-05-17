from pathlib import Path

import pytest

from app.domain.analysis.schemas import ImprovedSection
from app.domain.generation.renderer import _build_html, render
from app.domain.generation.uploader import upload


def _make_section(title: str, order: int, after: str = "## Content") -> ImprovedSection:
    return ImprovedSection(
        section_title=title,
        section_type="general",
        order=order,
        before="## Before",
        after=after,
        status="improved",
    )


def test_render_returns_pdf_bytes():
    sections = [_make_section("섹션 A", 0), _make_section("섹션 B", 1)]

    result = render(sections)

    assert isinstance(result, bytes)
    assert result.startswith(b"%PDF")


def test_render_preserves_section_order():
    section_first = _make_section("첫 번째", order=0, after="## 첫 번째 섹션")
    section_second = _make_section("두 번째", order=1, after="## 두 번째 섹션")

    html = _build_html([section_second, section_first])

    pos_first = html.find("첫 번째 섹션")
    pos_second = html.find("두 번째 섹션")
    assert pos_first != -1 and pos_second != -1
    assert pos_first < pos_second


def test_uploader_saves_file_locally(tmp_path):
    pdf_bytes = b"%PDF-1.4 test content"
    key = str(tmp_path / "output" / "test.pdf")

    result = upload(pdf_bytes, key)

    assert Path(result).exists()
    assert Path(result).read_bytes() == pdf_bytes
