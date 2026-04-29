from chunkers import cover_letter, resume


def chunk(text: str, source: str) -> list[dict]:
    """
    이력서+자소서 혼합 문서를 통합 청킹.

    - 이력서 섹션(인적사항, 학력 등): 엔티티 단위 청킹
    - 자기소개서 섹션: 대제목/소제목 단위 청킹
    - 이력서 섹션이 전혀 없으면 순수 자소서로 fallback

    Returns:
        통합 청크 리스트. 스키마: {source, doc_type, section, sub_section, text, char_count}
    """
    cleaned = resume.clean(text)
    sections = resume.split_sections(cleaned)

    has_resume_sections = any(
        k not in ("기타_상단", "자기소개서") for k in sections
    )
    if not has_resume_sections:
        return _normalize_cover_letter(cover_letter.chunk(text, source))

    final = []
    for entity, content in sections.items():
        content = content.strip()
        if not content:
            continue

        if entity == "자기소개서":
            cl_chunks = cover_letter.chunk(content, source)
            final.extend(_normalize_cover_letter(cl_chunks))
        else:
            if len(content) > resume.MAX_CHARS:
                for i in range(0, len(content), resume.MAX_CHARS):
                    piece = content[i:i + resume.MAX_CHARS]
                    final.append(_resume_chunk(source, entity, piece))
            else:
                final.append(_resume_chunk(source, entity, content))

    return final


def _resume_chunk(source: str, entity: str, text: str) -> dict:
    return {
        "source": source,
        "doc_type": "resume",
        "section": entity,
        "sub_section": entity,
        "text": text,
        "char_count": len(text),
    }


def _normalize_cover_letter(chunks: list[dict]) -> list[dict]:
    """cover_letter.chunk() 출력을 통합 스키마로 변환."""
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
