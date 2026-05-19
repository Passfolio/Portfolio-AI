"""
resume_chunker.py
────────────────────────────────────────────────────────────
Gemini 1-pass: 이력서 PDF → 섹션 단위 청킹 + 메타 추출 동시

이전 방식 (단일 청크 반환 → main.py에서 Gemini 호출) 대비:
  - 청커 내부에서 완결 — main.py 2차 추출 불필요
"""

from __future__ import annotations

import json
import os
import re
import time

from pydantic import BaseModel
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from google import genai as _genai
from google.genai import types as _types


def _get_gemini_client() -> _genai.Client:
    project = os.getenv("GCP_PROJECT_ID")
    if project:
        return _genai.Client(vertexai=True, project=project, location=os.getenv("GCP_LOCATION", "us-central1"))
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GCP_PROJECT_ID 또는 GEMINI_API_KEY 환경변수를 설정하세요.")
    return _genai.Client(api_key=api_key)


MIN_CHUNK_CHARS = 30
_LLM_MODEL      = "gemini-3-flash-preview"
_LLM_RETRIES    = 3

_pipeline = PdfPipelineOptions()
_pipeline.do_table_structure = True
_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=_pipeline)
    }
)

_STOP_PATTERNS = [
    re.compile(r'자기소개서|자소서|coverletter', re.IGNORECASE),
    re.compile(r'^\d+[\.\)]\s+.{10,}', re.MULTILINE),
]


# ═══════════════════════════════════════════════════════════════
# Pydantic 스키마
# ═══════════════════════════════════════════════════════════════

class _Section(BaseModel):
    section:    str
    start_line: int
    end_line:   int
    facts:      list[str]
    skills:     list[str]
    period:     str


class _ResumeExtraction(BaseModel):
    sections: list[_Section]


# ═══════════════════════════════════════════════════════════════
# 유틸리티
# ═══════════════════════════════════════════════════════════════

def _cut_at_cover_letter(markdown: str) -> str:
    for pat in _STOP_PATTERNS:
        m = pat.search(markdown)
        if m:
            markdown = markdown[:m.start()]
    return markdown.strip()


def _number_lines(text: str) -> tuple[str, list[str]]:
    lines = text.split("\n")
    numbered = "\n".join(f"{i + 1:04d} | {line}" for i, line in enumerate(lines))
    return numbered, lines


def _slice_lines(lines: list[str], start: int, end: int) -> str:
    s = max(0, start - 1)
    e = min(len(lines), end)
    return "\n".join(lines[s:e]).strip()


_RE_BLANK = re.compile(r"\n{3,}")
_RE_CTRL  = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _clean(text: str) -> str:
    import html
    text = _RE_CTRL.sub("", text)
    text = _RE_BLANK.sub("\n\n", text)
    return html.unescape(text).strip()


# ═══════════════════════════════════════════════════════════════
# Gemini 1-pass 섹션 분리 + 메타 추출
# ═══════════════════════════════════════════════════════════════

def _gemini_split(markdown: str, source: str) -> list[dict]:
    client = _get_gemini_client()
    numbered, raw_lines = _number_lines(markdown)
    total = len(raw_lines)

    prompt = (
        "다음은 이력서 마크다운입니다. 각 줄 앞에 줄 번호가 붙어 있습니다.\n"
        "섹션 단위로 분리하고, 각 섹션의 정보도 함께 추출해주세요.\n\n"
        "[section 선택 기준] — 아래 중 가장 적합한 이름 사용\n"
        "  인적사항 / 학력사항 / 경력사항 / 수상및활동 / 자격및어학 / 병역사항 / 기타\n\n"
        "[분리 기준]\n"
        "- 이력서의 각 섹션(헤더)을 독립적인 단위로 분리\n"
        f"- start_line과 end_line은 실제 줄 번호(1~{total})여야 하며 누락 없이 커버\n"
        "- 모든 줄은 정확히 하나의 섹션에 속해야 합니다\n\n"
        "[facts 작성 기준]\n"
        "- 각 항목을 '기간 / 기관 / 내용' 형태로 정리 (해당 정보가 있을 경우)\n"
        "- 사실 그대로 추출, 해석이나 요약 금지\n\n"
        "[skills 작성 기준]\n"
        "- 기술 스택, 자격증, 어학 점수 등 역량 관련 항목만\n"
        "- 없으면 빈 배열 []\n\n"
        "[period 작성 기준]\n"
        "- 해당 섹션의 전체 기간 범위 (예: '2019.03 ~ 2023.02')\n"
        "- 없으면 빈 문자열 ''\n\n"
        f"이력서:\n{numbered}"
    )

    for attempt in range(_LLM_RETRIES):
        try:
            response = client.models.generate_content(
                model=_LLM_MODEL,
                contents=prompt,
                config=_types.GenerateContentConfig(
                    system_instruction=(
                        "당신은 이력서 파싱 전문 AI입니다. "
                        "줄 번호가 붙은 이력서를 읽고 섹션 경계(start_line, end_line)와 "
                        "정보(facts, skills, period)를 정확히 추출합니다. "
                        "해석이나 평가 없이 원문에 충실하게 추출하세요. "
                        "텍스트를 복사하지 말고 줄 번호와 정제된 값만 반환하세요."
                    ),
                    response_mime_type="application/json",
                    response_schema=_ResumeExtraction,
                ),
            )
            data = json.loads(response.text)
            break
        except Exception as e:
            err = str(e)
            if attempt < _LLM_RETRIES - 1:
                wait = 30 if "429" in err else 5
                print(f"  [WARN] LLM 호출 실패 ({err[:50]}). {wait}초 후 재시도...")
                time.sleep(wait)
            else:
                raise

    chunks: list[dict] = []
    for item in data["sections"]:
        start = int(item["start_line"])
        end   = int(item["end_line"])
        raw   = _slice_lines(raw_lines, start, end)
        text  = _clean(raw)

        if len(text) < MIN_CHUNK_CHARS:
            continue

        facts  = item.get("facts", [])
        skills = item.get("skills", [])

        chunks.append({
            "source":      source,
            "doc_type":    "resume",
            "section":     item["section"],
            "sub_section": item["section"],
            "facts":       facts,
            "skills":      skills,
            "period":      item.get("period", ""),
            "text":        text,
            "char_count":  len(text),
        })

    return chunks


# ═══════════════════════════════════════════════════════════════
# 공개 API
# ═══════════════════════════════════════════════════════════════

def chunk(pdf_path: str, source: str) -> list[dict]:
    """이력서 PDF를 Gemini 1-pass로 섹션 단위 청킹 + 메타 추출."""
    result   = _converter.convert(pdf_path)
    markdown = result.document.export_to_markdown()
    markdown = _cut_at_cover_letter(markdown)

    if not markdown:
        return []

    return _gemini_split(markdown, source)
