import re
from rapidfuzz import process
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

_pipeline = PdfPipelineOptions()
_pipeline.do_table_structure = True
_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=_pipeline)
    }
)

_SECTION_MAPPING = {
    "인적사항": ["인적사항", "인적사", "기본정보", "프로필", "personalinfo"],
    "학력사항": ["학력사항", "학력", "education", "출신학교", "학력및전공"],
    "병역사항": ["병역", "병역사항", "군필여부", "military", "병역관계"],
    "경력사항": ["경력", "경력사항", "업무경험", "experience", "실무경력"],
    "수상및활동": ["수상및경력", "수상", "대외활동", "수상내역", "주요활동", "활동사항", "활동"],
    "자격및어학": ["외국어", "어학", "자격", "자격증", "면허", "어학능력", "license", "language"],
}

# 이 패턴이 나오면 자소서 시작으로 판단 → 이력서 청킹 중단
_STOP_PATTERNS = [
    re.compile(r'자기소개서|자소서|coverletter', re.IGNORECASE),
    re.compile(r'^\d+[\.\)]\s+.{10,}'),  # "1. SA(Solutions Architect)..." 같은 번호 질문
]

_ALL_ALIASES: list[str] = []
_ALIAS_TO_STANDARD: dict[str, str] = {}
for _std, _aliases in _SECTION_MAPPING.items():
    for _a in _aliases:
        _ALL_ALIASES.append(_a)
        _ALIAS_TO_STANDARD[_a] = _std

MAX_CHARS = 3000


def _is_stop(text: str) -> bool:
    for pat in _STOP_PATTERNS:
        if pat.search(text):
            return True
    return False


def _detect_section(text: str) -> str | None:
    """텍스트에서 표준 섹션명 반환. 없으면 None."""
    pure = re.sub(r'[\#\|\-\s□■○●·\(\)]+', '', text).strip()
    if not pure:
        return None
    candidate = pure[:10].lower()

    for alias in _ALL_ALIASES:
        if candidate.startswith(alias):
            return _ALIAS_TO_STANDARD[alias]

    if len(candidate) >= 2:
        result = process.extractOne(candidate, _ALL_ALIASES, score_cutoff=80)
        if result:
            return _ALIAS_TO_STANDARD[result[0]]

    return None


def _is_form_table(rows: list[list[str]]) -> bool:
    """
    인적사항처럼 (라벨, 값, 라벨, 값...) 형태의 form 테이블 감지.
    첫 행 셀이 4개 이상이고, 짝수 위치 셀이 숫자/날짜 없이 짧으면 form으로 판단.
    """
    if not rows or len(rows[0]) < 4:
        return False
    first = rows[0]
    labels = first[::2]  # 0, 2, 4 ... 위치
    return all(len(l) <= 10 and not re.search(r'\d{4,}', l) for l in labels)


def _form_to_lines(rows: list[list[str]]) -> str:
    """Form 테이블: 각 행을 라벨: 값 쌍으로 변환."""
    lines = []
    for row in rows:
        pairs = []
        i = 0
        while i < len(row):
            label = row[i]
            value = row[i + 1] if i + 1 < len(row) else ""
            if label and value:
                pairs.append(f"{label}: {value}")
            elif label:
                pairs.append(label)
            i += 2
        if pairs:
            lines.append(" | ".join(pairs))
    return "\n".join(lines)


def _table_to_lines(item) -> tuple[str | None, str]:
    """
    TableItem → (감지된_섹션명 | None, 컴팩트_텍스트).

    처리 패턴:
      1. 첫 행 단일 셀 = 섹션 헤더  → 나머지를 컬럼 헤더 + 데이터로 처리
      2. Form 테이블 (인적사항 등)   → 라벨: 값 쌍으로 변환
      3. 일반 컬럼 테이블            → 헤더: 값 형식, 중복 컬럼 제거
    """
    grid = item.data.grid
    if not grid:
        return None, ""

    rows: list[list[str]] = [
        [c.text.strip() for c in row if c.text.strip()]
        for row in grid
    ]
    rows = [r for r in rows if r]

    if not rows:
        return None, ""

    # 섹션 헤더 감지 (첫 행 셀 1~2개)
    detected_section = None
    data_start = 0
    if len(rows[0]) <= 2:
        detected_section = _detect_section(" ".join(rows[0]))
        if detected_section:
            data_start = 1

    data_rows = rows[data_start:]
    if not data_rows:
        return detected_section, ""

    # Form 테이블 감지 (인적사항 등)
    if _is_form_table(data_rows):
        return detected_section, _form_to_lines(data_rows)

    # 일반 컬럼 테이블
    col_headers = data_rows[0]
    content_rows = data_rows[1:]

    if not content_rows:
        return detected_section, " | ".join(col_headers)

    lines = []
    for row in content_rows:
        pairs = []
        seen_headers: set[str] = set()  # 중복 컬럼 헤더 제거
        for i, val in enumerate(row):
            h = col_headers[i] if i < len(col_headers) else ""
            # 같은 헤더가 이미 나왔으면 건너뜀 (좌우 분할 테이블 합침 현상 대응)
            if h and h in seen_headers:
                continue
            if h:
                seen_headers.add(h)
            if h and h != val:
                pairs.append(f"{h}: {val}")
            elif val:
                pairs.append(val)
        if pairs:
            lines.append(" | ".join(pairs))

    return detected_section, "\n".join(lines)


def _make_chunk(source: str, section: str, text: str) -> dict:
    text = text.strip()
    return {
        "source": source,
        "doc_type": "resume",
        "section": section,
        "sub_section": section,
        "text": text,
        "char_count": len(text),
    }


def chunk(pdf_path: str, source: str) -> list[dict]:
    """
    이력서 PDF를 TableFormer로 변환 후 섹션별 청킹.
    - TextItem으로 섹션 헤더 감지
    - TableItem 첫 행에서도 섹션 헤더 감지 (한국 이력서 패턴)
    - 셀 데이터는 '헤더: 값' 컴팩트 라인으로 변환 (마크다운 패딩 제거)
    - 자소서 시작 감지 시 즉시 중단
    """
    result = _converter.convert(pdf_path)
    doc = result.document

    chunks: list[dict] = []
    current_section = "기타_상단"
    pending_texts: list[str] = []

    def flush_pending():
        nonlocal pending_texts
        content = "\n".join(pending_texts).strip()
        pending_texts = []
        if content:
            chunks.append(_make_chunk(source, current_section, content))

    for item, _ in doc.iterate_items():
        item_class = type(item).__name__

        if item_class == "TableItem":
            flush_pending()
            table_section, table_text = _table_to_lines(item)

            # 테이블 첫 행에서 섹션 감지
            if table_section:
                current_section = table_section

            if table_text:
                chunks.append(_make_chunk(source, current_section, table_text))
            continue

        text = getattr(item, "text", "").strip()
        if not text:
            continue

        # 자소서 시작 감지 → 중단
        if _is_stop(text):
            flush_pending()
            break

        # 섹션 헤더 감지
        detected = _detect_section(text)
        if detected:
            flush_pending()
            current_section = detected
            continue

        pending_texts.append(text)

    flush_pending()
    return chunks
