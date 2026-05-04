from chunkers import cover_letter, resume

Literal = str  # "resume" | "cover_letter"


def chunk(text: str, source: str, doc_type: Literal) -> list[dict]:
    """
    doc_type에 따라 청킹 방식 라우팅.

    Args:
        text: docling으로 추출한 raw 텍스트
        source: 파일명 등 출처 식별자
        doc_type: "resume" | "cover_letter"

    Returns:
        통합 청크 리스트. 스키마: {source, doc_type, section, sub_section, text, char_count}
    """
    if doc_type == "resume":
        return resume.chunk(text, source)
    else:
        return _normalize_cover_letter(cover_letter.chunk(text, source))


def _normalize_cover_letter(chunks: list[dict]) -> list[dict]:
    return [
        {
            "source": c["source"],
            "doc_type": "cover_letter",
            "section": c["main_section"],
            "sub_section": c["sub_section"],
            "text": c["text"],
            "char_count": c["char_count"],
        }
        for c in chunks
    ]
