import html
import os
import platform
import re

import markdown as md
from fpdf import FPDF

from app.domain.analysis.schemas import ImprovedSection

_TAG_RE = re.compile(r"<[^>]+>")
_FONT_NAME = "body"


def _find_unicode_font() -> str | None:
    """플랫폼별 유니코드(CJK 포함) TTF 폰트 경로 반환."""
    candidates = []
    if platform.system() == "Windows":
        candidates = [
            r"C:\Windows\Fonts\malgun.ttf",
            r"C:\Windows\Fonts\gulim.ttc",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ]
    return next((p for p in candidates if os.path.exists(p)), None)


def _strip_tags(text: str) -> str:
    return html.unescape(_TAG_RE.sub("", text))


def _build_html(sections: list[ImprovedSection]) -> str:
    """순서 보존 HTML — 테스트 및 내부 검증용."""
    ordered = sorted(sections, key=lambda s: s.order)
    parts = ["<html><body>"]
    for section in ordered:
        parts.append(md.markdown(section.after, extensions=["tables"]))
    parts.append("</body></html>")
    return "\n".join(parts)


def render(sections: list[ImprovedSection]) -> bytes:
    """list[ImprovedSection] → PDF 바이너리."""
    ordered = sorted(sections, key=lambda s: s.order)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    font_path = _find_unicode_font()
    if font_path:
        pdf.add_font(_FONT_NAME, fname=font_path)
    else:
        pdf.add_font(_FONT_NAME, fname="Helvetica")

    pdf.add_page()

    for section in ordered:
        html_content = md.markdown(section.after, extensions=["tables"])
        plain = _strip_tags(html_content).strip()
        if plain:
            pdf.set_font(_FONT_NAME, size=13)
            pdf.multi_cell(pdf.epw, 8, section.section_title)
            pdf.set_font(_FONT_NAME, size=11)
            pdf.multi_cell(pdf.epw, 6, plain)
            pdf.ln(4)

    return bytes(pdf.output())
