import re

RE_MAIN_SECTION = re.compile(r'^(■\s*.+|\d+[\.\)]\s+.+)$', re.MULTILINE)
RE_SUB_SECTION = re.compile(r'^\[(.+?)\]|^"(.+)"$', re.MULTILINE)

MAX_CHARS = 1500


def clean_text(text: str) -> str:
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _split_by_paragraph(chunk: dict) -> list[dict]:
    paragraphs = [p.strip() for p in chunk["text"].split("\n\n") if p.strip()]
    if len(paragraphs) <= 1:
        return [chunk]

    result = []
    buffer = []
    buffer_len = 0

    for para in paragraphs:
        if buffer_len + len(para) > MAX_CHARS and buffer:
            result.append({**chunk, "text": "\n\n".join(buffer), "char_count": buffer_len})
            buffer = [para]
            buffer_len = len(para)
        else:
            buffer.append(para)
            buffer_len += len(para)

    if buffer:
        result.append({**chunk, "text": "\n\n".join(buffer), "char_count": buffer_len})

    if len(result) > 1:
        for i, r in enumerate(result):
            r["sub_index"] = i
    return result


def chunk(text: str, source: str) -> list[dict]:
    """
    자소서 텍스트를 [소제목] 단위로 청킹.

    Args:
        text: docling으로 추출한 raw 텍스트
        source: 출처 식별자 (파일명 등)

    Returns:
        청크 리스트. 각 청크: {source, main_section, sub_section, text, char_count}
    """
    text = clean_text(text)
    lines = text.splitlines()

    chunks = []
    current_main = "헤더"
    current_sub = None
    buffer = []

    def flush(main, sub, buf):
        content = "\n".join(buf).strip()
        if content:
            chunks.append({
                "source": source,
                "doc_type": "cover_letter",
                "main_section": main,
                "sub_section": sub or "본문",
                "text": content,
                "char_count": len(content),
            })

    for line in lines:
        if RE_MAIN_SECTION.match(line.strip()):
            flush(current_main, current_sub, buffer)
            current_main = line.strip()
            current_sub = None
            buffer = []
        elif RE_SUB_SECTION.match(line.strip()):
            flush(current_main, current_sub, buffer)
            m = RE_SUB_SECTION.match(line.strip())
            current_sub = m.group(1) or m.group(2)
            buffer = [line]
        else:
            buffer.append(line)

    flush(current_main, current_sub, buffer)

    final = []
    for c in chunks:
        if c["char_count"] > MAX_CHARS:
            final.extend(_split_by_paragraph(c))
        else:
            final.append(c)

    return final
