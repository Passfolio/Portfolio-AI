import re
from rapidfuzz import process

MAX_CHARS = 3000

_SECTION_MAPPING = {
    "인적사항": ["인적사항", "인적사", "기본정보", "프로필", "personalinfo"],
    "학력사항": ["학력사항", "학력", "education", "출신학교", "학력및전공"],
    "병역사항": ["병역", "병역사항", "군필여부", "military", "병역관계"],
    "경력사항": ["경력", "경력사항", "업무경험", "experience", "실무경력"],
    "수상및활동": ["수상및경력", "수상", "대외활동", "프로젝트", "수상내역", "주요활동"],
    "자격및어학": ["외국어", "어학", "자격", "자격증", "면허", "어학능력", "license", "language"],
    "자기소개서": ["자소서", "자기소개서", "coverletter", "자기소개"],
}

_ALL_ALIASES: list[str] = []
_ALIAS_TO_STANDARD: dict[str, str] = {}
for _std, _aliases in _SECTION_MAPPING.items():
    for _a in _aliases:
        _ALL_ALIASES.append(_a)
        _ALIAS_TO_STANDARD[_a] = _std


def clean(text: str) -> str:
    text = re.sub(r'[\-\|=]{3,}', '', text)
    lines = text.split('\n')
    result = []
    for line in lines:
        words = line.split('|')
        seen, deduped = set(), []
        for w in words:
            cw = w.strip()
            if cw and cw not in seen:
                deduped.append(cw)
                seen.add(cw)
        if deduped:
            result.append(" | ".join(deduped))
    return "\n".join(result)


def split_sections(text: str) -> dict[str, str]:
    """텍스트를 이력서 엔티티별로 분리해 {entity: content} 반환."""
    buckets: dict[str, list[str]] = {}
    current = "기타_상단"

    for line in text.split('\n'):
        if not line.strip():
            continue
        pure = re.sub(r'[\#\|\-\s□■○●·]+', '', line).strip()
        if not pure:
            continue
        candidate = pure[:10].lower()

        matched_alias = None
        for alias in _ALL_ALIASES:
            if candidate.startswith(alias):
                matched_alias = alias
                break
        if not matched_alias and len(candidate) >= 2:
            result = process.extractOne(candidate, _ALL_ALIASES, score_cutoff=80)
            if result:
                matched_alias = result[0]

        if matched_alias:
            current = _ALIAS_TO_STANDARD[matched_alias]
            chars = list(matched_alias)
            pattern = r'[\s\|\-\#]*'.join(chars)
            line = re.sub(pattern, '', line, count=1, flags=re.IGNORECASE).strip()
            line = re.sub(r'^[\#\|\-\s]+', '', line).strip()

        buckets.setdefault(current, [])
        if line.strip():
            buckets[current].append(line.strip())

    return {entity: "\n".join(lines) for entity, lines in buckets.items()}


def chunk(text: str, source: str) -> list[dict]:
    """이력서 텍스트를 엔티티 단위로 청킹."""
    sections = split_sections(clean(text))
    final = []

    for entity, content in sections.items():
        content = content.strip()
        if not content:
            continue
        if len(content) > MAX_CHARS:
            for i in range(0, len(content), MAX_CHARS):
                piece = content[i:i + MAX_CHARS]
                final.append({
                    "source": source,
                    "doc_type": "resume",
                    "section": entity,
                    "sub_section": entity,
                    "text": piece,
                    "char_count": len(piece),
                })
        else:
            final.append({
                "source": source,
                "doc_type": "resume",
                "section": entity,
                "sub_section": entity,
                "text": content,
                "char_count": len(content),
            })

    return final
