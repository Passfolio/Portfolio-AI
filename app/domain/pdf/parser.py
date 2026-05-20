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


def _extract_sections_from_docling_dict(doc: dict) -> list[PdfSection]:
    texts = doc.get("texts", [])
    sections: list[PdfSection] = []
    current_title: str | None = None
    current_parts: list[str] = []
    current_page: int = 1
    order = 0

    def flush() -> None:
        nonlocal order
        if not current_title or not current_parts:
            return
        body_md = "\n\n".join(current_parts)
        section_md = f"## {current_title}\n\n{body_md}"
        plain = _strip_markdown(body_md)
        tok = _token_count(plain)
        if tok <= _MAX_TOKENS:
            sections.append(
                PdfSection(
                    section_title=current_title,
                    section_type=_infer_section_type(current_title),
                    plain_text=plain,
                    markdown=section_md,
                    page=current_page,
                    order=order,
                    parent_title=None,
                    token_count=tok,
                )
            )
            order += 1
        else:
            chunks = _split_into_chunks(
                title=current_title,
                markdown=body_md,
                section_type=_infer_section_type(current_title),
                page=current_page,
                start_order=order,
                parent_title=current_title,
            )
            sections.extend(chunks)
            order += len(chunks)

    for item in texts:
        label = item.get("label", "text")
        text = item.get("text", "").strip()
        if not text:
            continue
        prov = item.get("prov", [])
        page_no = prov[0].get("page_no", current_page) if prov else current_page

        if label in ("section_header", "title"):
            flush()
            current_title = text
            current_parts = []
            current_page = page_no
        elif label == "list_item":
            current_parts.append(f"- {text}")
        else:
            current_parts.append(text)

    flush()
    return sections


def parse_pdf(pdf_bytes: bytes) -> list[PdfSection]:
    """Docling으로 PDF bytes 파싱 → list[PdfSection]."""
    from docling.datamodel.base_models import ConversionStatus, DocumentStream
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    buf = BytesIO(pdf_bytes)
    stream = DocumentStream(name="portfolio.pdf", stream=buf)
    result = converter.convert(stream)

    if result.status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
        raise RuntimeError(f"Docling 변환 실패: {result.errors}")

    doc_dict = result.document.export_to_dict()
    return _extract_sections_from_docling_dict(doc_dict)
