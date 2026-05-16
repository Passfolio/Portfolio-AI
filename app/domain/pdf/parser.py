import re
from io import BytesIO

import tiktoken

from app.domain.pdf.schemas import PdfSection

_enc = tiktoken.get_encoding("cl100k_base")
_MAX_TOKENS = 512
_OVERLAP_TOKENS = 50

SECTION_TYPE_KEYWORDS: dict[str, list[str]] = {
    "project": ["프로젝트", "project", "개발"],
    "skill": ["기술", "스택", "skill", "stack", "언어", "프레임워크"],
    "experience": ["경력", "경험", "experience", "work", "인턴"],
    "education": ["학력", "교육", "education", "대학"],
}


def _infer_section_type(title: str) -> str:
    title_lower = title.lower()
    for type_name, keywords in SECTION_TYPE_KEYWORDS.items():
        if any(kw in title_lower for kw in keywords):
            return type_name
    return "general"


def _strip_markdown(text: str) -> str:
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text, flags=re.DOTALL)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    return text.strip()


def _token_count(text: str) -> int:
    return len(_enc.encode(text))


def _split_into_chunks(
    title: str,
    markdown: str,
    section_type: str,
    page: int,
    start_order: int,
    parent_title: str | None,
) -> list[PdfSection]:
    paragraphs = [p.strip() for p in markdown.split("\n\n") if p.strip()]
    chunks: list[PdfSection] = []
    current_paras: list[str] = []
    current_tokens = 0
    order = start_order

    def flush(paras: list[str]) -> None:
        nonlocal order
        chunk_md = "\n\n".join(paras)
        chunk_plain = _strip_markdown(chunk_md)
        chunks.append(
            PdfSection(
                section_title=title,
                section_type=section_type,
                plain_text=chunk_plain,
                markdown=chunk_md,
                page=page,
                order=order,
                parent_title=parent_title,
                token_count=_token_count(chunk_plain),
            )
        )
        order += 1

    for para in paragraphs:
        para_tokens = _token_count(para)
        if current_tokens + para_tokens > _MAX_TOKENS and current_paras:
            flush(current_paras)
            overlap: list[str] = []
            overlap_tokens = 0
            for p in reversed(current_paras):
                t = _token_count(p)
                if overlap_tokens + t <= _OVERLAP_TOKENS:
                    overlap.insert(0, p)
                    overlap_tokens += t
                else:
                    break
            current_paras = overlap + [para]
            current_tokens = overlap_tokens + para_tokens
        else:
            current_paras.append(para)
            current_tokens += para_tokens

    if current_paras:
        flush(current_paras)

    return chunks


def _parse_sections_from_markdown(markdown: str) -> list[PdfSection]:
    lines = markdown.splitlines()
    sections: list[PdfSection] = []
    current_title: str | None = None
    current_lines: list[str] = []
    order = 0

    def flush_section(title: str, body_lines: list[str]) -> None:
        nonlocal order
        body_md = "\n".join(body_lines).strip()
        if not body_md:
            return
        section_md = f"## {title}\n\n{body_md}"
        plain = _strip_markdown(body_md)
        token_count = _token_count(plain)
        if token_count <= _MAX_TOKENS:
            sections.append(
                PdfSection(
                    section_title=title,
                    section_type=_infer_section_type(title),
                    plain_text=plain,
                    markdown=section_md,
                    page=1,
                    order=order,
                    parent_title=None,
                    token_count=token_count,
                )
            )
            order += 1
        else:
            chunks = _split_into_chunks(
                title=title,
                markdown=body_md,
                section_type=_infer_section_type(title),
                page=1,
                start_order=order,
                parent_title=title,
            )
            sections.extend(chunks)
            order += len(chunks)

    for line in lines:
        if re.match(r"^#{1,2}\s+", line):
            if current_title is not None:
                flush_section(current_title, current_lines)
            current_title = re.sub(r"^#{1,2}\s+", "", line).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_title is not None:
        flush_section(current_title, current_lines)

    return sections


def parse_pdf(pdf_bytes: bytes) -> list[PdfSection]:
    """Docling으로 PDF bytes 파싱 → list[PdfSection].

    현재 개발 단계에서는 load_mock_json() 사용.
    BE 완료 후 이 함수로 교체.
    """
    from docling.datamodel.base_models import DocumentStream
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    buf = BytesIO(pdf_bytes)
    stream = DocumentStream(name="portfolio.pdf", stream=buf)
    result = converter.convert(stream)

    if not result.status.success:
        raise RuntimeError(f"Docling 변환 실패: {result.errors}")

    markdown = result.document.export_to_markdown()
    return _parse_sections_from_markdown(markdown)
