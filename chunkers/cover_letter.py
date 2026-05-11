"""
cover_letter_chunker.py
────────────────────────────────────────────────────────────
Gemini 1-pass: 자소서 텍스트 → 문항 단위 청킹 + 메타 추출 동시

이전 방식 (규칙 기반 청킹 → 청크마다 Gemini N회 호출) 대비:
  - 특수기호(■ ● [ ]) 패턴에 의존하지 않아 자소서 형식 무관
  - Gemini 호출 N회 → 1회로 감소
"""

from __future__ import annotations

import json
import os
import re
import time

from pydantic import BaseModel
from google import genai as _genai
from google.genai import types as _types


MIN_CHUNK_CHARS = 50
_LLM_MODEL      = "gemini-3-flash-preview"
_LLM_RETRIES    = 3

CATEGORY_DEFINITIONS = {
    "지원동기":     "해당 기업 또는 직무에 지원하게 된 이유, 관심 계기",
    "입사포부":     "입사 후 이루고 싶은 목표, 성장 계획, 기여 방향",
    "직무역량":     "직무 수행에 필요한 전문 기술, 지식, 자격",
    "문제해결경험":  "문제를 발견하고 분석하여 해결한 구체적 경험",
    "협업태도":     "팀원과의 소통, 갈등 조율, 협력 방식에 관한 경험",
    "리더십":       "팀을 이끌거나 주도적으로 역할을 맡은 경험",
    "자기소개":     "성격, 장단점, 가치관 등 자신에 대한 소개",
    "성장과정":     "성장 배경, 가치관 형성에 영향을 준 사건이나 인물",
    "취미":         "여가 활동, 관심사",
    "프로젝트경험":  "참여한 프로젝트의 역할, 과정, 결과",
    "경력":         "인턴, 직장 경험, 학력, 자격증, 수상 이력",
    "사회이슈":     "사회적 현상이나 이슈에 대한 본인의 견해",
    "기타":         "위 카테고리에 해당하지 않는 내용",
}

_CATEGORY_GUIDE = "\n".join(f"  - {k}: {v}" for k, v in CATEGORY_DEFINITIONS.items())


# ═══════════════════════════════════════════════════════════════
# Pydantic 스키마
# ═══════════════════════════════════════════════════════════════

class _Meta(BaseModel):
    category:     str
    key_points:   list[str]
    achievements: list[str]
    keywords:     list[str]


class _Section(BaseModel):
    title:      str
    start_line: int
    end_line:   int
    meta:       _Meta


class _SectionList(BaseModel):
    sections: list[_Section]


# ═══════════════════════════════════════════════════════════════
# 유틸리티
# ═══════════════════════════════════════════════════════════════

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
# Gemini 1-pass 분리 + 메타 추출
# ═══════════════════════════════════════════════════════════════

def _gemini_split(text: str, source: str) -> list[dict]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")

    client = _genai.Client(api_key=api_key)
    numbered, raw_lines = _number_lines(text)
    total = len(raw_lines)

    prompt = (
        "다음은 자기소개서 텍스트입니다. 각 줄 앞에 줄 번호가 붙어 있습니다.\n"
        "문항(질문) 단위로 분리하고, 각 문항의 메타데이터도 함께 추출해주세요.\n\n"
        "[분리 기준]\n"
        "- 각 자소서 문항(질문)은 독립적인 섹션으로 분리\n"
        "- 질문 제목과 답변을 같은 섹션에 포함\n"
        f"- start_line과 end_line은 실제 줄 번호(1~{total})여야 하며 누락 없이 커버\n"
        "- 모든 줄은 정확히 하나의 섹션에 속해야 합니다\n\n"
        "[category 선택 기준] — 반드시 아래 중 하나만 선택\n"
        f"{_CATEGORY_GUIDE}\n\n"
        "[key_points 작성 기준]\n"
        "- 지원자의 구체적 행동과 결과 중심으로 2~3문장\n"
        "- 단순 요약이 아닌 무엇을 했고 어떤 성과가 있었는지 서술\n\n"
        "[achievements 작성 기준]\n"
        "- 수치(%, 배수, 개수 등)가 포함된 정량적 성과\n"
        "- 수치 없어도 명확한 결과(출시, 수상, 채택 등)면 포함\n"
        "- 없으면 빈 배열 []\n\n"
        "[keywords 작성 기준]\n"
        "- 기술 스택, 직무 역량, 도메인 키워드 (중복 없이, 핵심만)\n\n"
        f"자기소개서:\n{numbered}"
    )

    for attempt in range(_LLM_RETRIES):
        try:
            response = client.models.generate_content(
                model=_LLM_MODEL,
                contents=prompt,
                config=_types.GenerateContentConfig(
                    system_instruction=(
                        "당신은 채용 전문가이자 자기소개서 분석 AI입니다. "
                        "줄 번호가 붙은 자소서를 읽고 문항 경계(start_line, end_line)와 "
                        "메타데이터(category, key_points, achievements, keywords)를 정확히 추출합니다. "
                        "텍스트를 복사하지 말고 줄 번호와 정제된 메타값만 반환하세요."
                    ),
                    response_mime_type="application/json",
                    response_schema=_SectionList,
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
        text_clean = _clean(raw)

        if len(text_clean) < MIN_CHUNK_CHARS:
            continue

        meta     = item.get("meta", {})
        keywords = meta.get("keywords", [])

        chunks.append({
            "source":       source,
            "doc_type":     "cover_letter",
            "section":      item["title"],
            "sub_section":  item["title"],
            "category":     meta.get("category", "기타"),
            "key_points":   meta.get("key_points", []),
            "achievements": meta.get("achievements", []),
            "keywords":     keywords,
            "keywords_str": ", ".join(keywords),
            "text":         text_clean,
            "char_count":   len(text_clean),
        })

    return chunks


# ═══════════════════════════════════════════════════════════════
# 공개 API
# ═══════════════════════════════════════════════════════════════

def chunk(text: str, source: str) -> list[dict]:
    """자소서 텍스트를 Gemini 1-pass로 문항 단위 청킹 + 메타 추출."""
    return _gemini_split(text, source)
