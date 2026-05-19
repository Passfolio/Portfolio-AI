"""
cover_letter_chunker.py
────────────────────────────────────────────────────────────
Gemini 1-pass: 자소서 텍스트 → 문항 단위 청킹 + 메타 추출 동시

개선 사항:
  1. 경계 검증 + 재시도 루프 (누락/중복 줄 탐지 → DB 무결성 보장)
  2. 형식 자동 감지 힌트 주입 (특수기호 / 번호 / 표 / 자유서술)
  3. category 다중화 (주 카테고리 + 보조 카테고리)
  4. key_points STAR 구조 유도 (Situation / Action / Result)
"""

from __future__ import annotations

import html
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Iterator

from pydantic import BaseModel
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


# ═══════════════════════════════════════════════════════════════
# 상수
# ═══════════════════════════════════════════════════════════════

MIN_CHUNK_CHARS  = 50
_LLM_MODEL       = "gemini-3-flash-preview"
_LLM_RETRIES     = 3
_VALIDATE_RETRIES = 2          # 경계 검증 실패 시 재시도 횟수

CATEGORY_DEFINITIONS: dict[str, str] = {
    "지원동기":    "해당 기업 또는 직무에 지원하게 된 이유, 관심 계기",
    "입사포부":    "입사 후 이루고 싶은 목표, 성장 계획, 기여 방향",
    "직무역량":    "직무 수행에 필요한 전문 기술, 지식, 자격",
    "문제해결경험": "문제를 발견하고 분석하여 해결한 구체적 경험",
    "협업태도":    "팀원과의 소통, 갈등 조율, 협력 방식에 관한 경험",
    "리더십":      "팀을 이끌거나 주도적으로 역할을 맡은 경험",
    "자기소개":    "성격, 장단점, 가치관 등 자신에 대한 소개",
    "성장과정":    "성장 배경, 가치관 형성에 영향을 준 사건이나 인물",
    "취미":        "여가 활동, 관심사",
    "프로젝트경험": "참여한 프로젝트의 역할, 과정, 결과",
    "경력":        "인턴, 직장 경험, 학력, 자격증, 수상 이력",
    "사회이슈":    "사회적 현상이나 이슈에 대한 본인의 견해",
    "기타":        "위 카테고리에 해당하지 않는 내용",
}

_CATEGORY_GUIDE = "\n".join(
    f"  - {k}: {v}" for k, v in CATEGORY_DEFINITIONS.items()
)

_VALID_CATEGORIES = set(CATEGORY_DEFINITIONS.keys())


# ═══════════════════════════════════════════════════════════════
# Pydantic 스키마
# ═══════════════════════════════════════════════════════════════

class _Meta(BaseModel):
    category:        str
    sub_categories:  list[str]   # 보조 카테고리 (검색 확장용)
    key_points:      list[str]   # STAR 구조 서술
    achievements:    list[str]
    keywords:        list[str]


class _Section(BaseModel):
    title:      str
    start_line: int
    end_line:   int
    meta:       _Meta


class _SectionList(BaseModel):
    sections: list[_Section]


# ═══════════════════════════════════════════════════════════════
# 텍스트 전처리
# ═══════════════════════════════════════════════════════════════

_RE_BLANK = re.compile(r"\n{3,}")
_RE_CTRL  = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _clean(text: str) -> str:
    text = _RE_CTRL.sub("", text)
    text = _RE_BLANK.sub("\n\n", text)
    return html.unescape(text).strip()


def _number_lines(text: str) -> tuple[str, list[str]]:
    """줄 번호를 붙인 텍스트와 원본 줄 목록을 반환."""
    lines = text.split("\n")
    numbered = "\n".join(f"{i + 1:04d} | {line}" for i, line in enumerate(lines))
    return numbered, lines


def _slice_lines(lines: list[str], start: int, end: int) -> str:
    s = max(0, start - 1)
    e = min(len(lines), end)
    return "\n".join(lines[s:e]).strip()


# ═══════════════════════════════════════════════════════════════
# 형식 자동 감지
# ═══════════════════════════════════════════════════════════════

_RE_SYMBOL = re.compile(r"[■●\[\]【】◆▶❑▣△]")
_RE_NUMBER = re.compile(r"^\s*\d+[\.\)]\s", re.MULTILINE)
_RE_TABLE  = re.compile(r"\|")


def _detect_format_hint(text: str) -> str:
    """자소서 형식을 감지해 LLM 힌트 문자열을 반환."""
    if _RE_SYMBOL.search(text):
        return "특수기호(■ ● [ ] ◆ 등) 패턴으로 문항이 구분되어 있습니다."
    if len(_RE_TABLE.findall(text)) > 5:
        return "표(|) 형식으로 작성되어 있습니다. 행 단위로 문항을 분리하세요."
    if _RE_NUMBER.search(text):
        return "번호(1. 2. 3.) 패턴으로 문항이 구분되어 있습니다."
    return (
        "구분자 없이 자유 서술 형식입니다. "
        "주제 전환(문단 흐름, 소제목 등)을 기준으로 섹션을 분리하세요."
    )


# ═══════════════════════════════════════════════════════════════
# 경계 검증
# ═══════════════════════════════════════════════════════════════

@dataclass
class _ValidationResult:
    ok:      bool
    errors:  list[str] = field(default_factory=list)


def _validate_coverage(sections: list[dict], total_lines: int) -> _ValidationResult:
    """섹션 경계가 전체 줄을 빠짐없이 1회 커버하는지 검증."""
    covered: dict[int, str] = {}   # line_no → section title
    errors: list[str] = []

    for sec in sections:
        title = sec.get("title", "?")
        for ln in range(int(sec["start_line"]), int(sec["end_line"]) + 1):
            if ln in covered:
                errors.append(f"줄 {ln} 중복: '{covered[ln]}' vs '{title}'")
            covered[ln] = title

    missing = sorted(set(range(1, total_lines + 1)) - covered.keys())
    if missing:
        sample = missing[:10]
        suffix = f" 외 {len(missing) - 10}줄" if len(missing) > 10 else ""
        errors.append(f"누락 줄: {sample}{suffix}")

    return _ValidationResult(ok=not errors, errors=errors)


def _validate_categories(sections: list[dict]) -> list[str]:
    """존재하지 않는 category 값 탐지."""
    errors = []
    for sec in sections:
        cat = sec.get("meta", {}).get("category", "")
        if cat not in _VALID_CATEGORIES:
            errors.append(f"'{sec.get('title')}' — 알 수 없는 category: '{cat}'")
        for sub in sec.get("meta", {}).get("sub_categories", []):
            if sub not in _VALID_CATEGORIES:
                errors.append(f"'{sec.get('title')}' — 알 수 없는 sub_category: '{sub}'")
    return errors


# ═══════════════════════════════════════════════════════════════
# 프롬프트 빌더
# ═══════════════════════════════════════════════════════════════

def _build_prompt(numbered: str, total: int, format_hint: str) -> str:
    return (
        "다음은 자기소개서 텍스트입니다. 각 줄 앞에 줄 번호가 붙어 있습니다.\n"
        "문항(질문) 단위로 분리하고, 각 문항의 메타데이터도 함께 추출해주세요.\n\n"
        f"[형식 힌트] {format_hint}\n\n"
        "[분리 기준]\n"
        "- 각 자소서 문항(질문)은 독립적인 섹션으로 분리\n"
        "- 질문 제목과 답변을 같은 섹션에 포함\n"
        f"- start_line과 end_line은 실제 줄 번호(1~{total})여야 하며 누락 없이 커버\n"
        "- 모든 줄은 정확히 하나의 섹션에만 속해야 합니다\n\n"
        "[category / sub_categories 선택 기준]\n"
        "- category: 섹션 내용을 가장 잘 나타내는 주 카테고리 1개\n"
        "- sub_categories: 보조로 해당하는 카테고리 목록 (없으면 빈 배열)\n"
        "- 반드시 아래 목록 중에서만 선택:\n"
        f"{_CATEGORY_GUIDE}\n\n"
        "[key_points 작성 기준 — STAR 구조]\n"
        "- Situation(상황) + Action(행동) + Result(결과) 흐름으로 2~3문장 작성\n"
        "- 예시: '레거시 배치 시스템의 처리 지연이 반복되는 상황에서(S), "
        "Kafka 기반 이벤트 스트리밍으로 아키텍처를 전환하여(A), "
        "처리 속도를 3배 향상시켰다(R)'\n"
        "- 단순 요약이 아닌 무엇을 했고 어떤 변화가 있었는지 구체적으로 서술\n\n"
        "[achievements 작성 기준]\n"
        "- 수치(%, 배수, 개수 등)가 포함된 정량적 성과\n"
        "- 수치 없어도 명확한 결과(출시, 수상, 채택, 합격 등)면 포함\n"
        "- 없으면 빈 배열 []\n\n"
        "[keywords 작성 기준]\n"
        "- 기술 스택, 직무 역량, 도메인 키워드 (중복 없이, 핵심만)\n\n"
        f"자기소개서:\n{numbered}"
    )


_SYSTEM_INSTRUCTION = (
    "당신은 채용 전문가이자 자기소개서 분석 AI입니다. "
    "줄 번호가 붙은 자소서를 읽고 문항 경계(start_line, end_line)와 "
    "메타데이터(category, sub_categories, key_points, achievements, keywords)를 정확히 추출합니다. "
    "텍스트를 그대로 복사하지 말고 줄 번호와 정제된 메타값만 반환하세요. "
    "start_line ~ end_line 범위가 전체 줄을 빠짐없이 커버해야 합니다."
)


# ═══════════════════════════════════════════════════════════════
# Gemini 호출 (재시도 포함)
# ═══════════════════════════════════════════════════════════════

def _call_gemini(client: _genai.Client, prompt: str) -> list[dict]:
    """Gemini API를 호출하고 파싱된 sections 목록을 반환."""
    for attempt in range(_LLM_RETRIES):
        try:
            response = client.models.generate_content(
                model=_LLM_MODEL,
                contents=prompt,
                config=_types.GenerateContentConfig(
                    system_instruction=_SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=_SectionList,
                ),
            )
            return json.loads(response.text)["sections"]
        except Exception as exc:
            err = str(exc)
            if attempt < _LLM_RETRIES - 1:
                wait = 30 if "429" in err else 5
                print(f"  [WARN] LLM 호출 실패 ({err[:60]}). {wait}초 후 재시도...")
                time.sleep(wait)
            else:
                raise


# ═══════════════════════════════════════════════════════════════
# 섹션 → 청크 변환
# ═══════════════════════════════════════════════════════════════

def _sections_to_chunks(
    sections: list[dict],
    raw_lines: list[str],
    source: str,
) -> Iterator[dict]:
    """섹션 목록을 DB 삽입용 청크 dict로 변환."""
    for item in sections:
        start = int(item["start_line"])
        end   = int(item["end_line"])
        raw   = _slice_lines(raw_lines, start, end)
        text_clean = _clean(raw)

        if len(text_clean) < MIN_CHUNK_CHARS:
            continue

        meta            = item.get("meta", {})
        keywords: list  = meta.get("keywords", [])
        sub_categories  = [
            c for c in meta.get("sub_categories", [])
            if c in _VALID_CATEGORIES and c != meta.get("category")
        ]

        yield {
            "source":          source,
            "doc_type":        "cover_letter",
            "section":         item["title"],
            "category":        meta.get("category", "기타"),
            "sub_categories":  sub_categories,
            "key_points":      meta.get("key_points", []),
            "achievements":    meta.get("achievements", []),
            "keywords":        keywords,
            "keywords_str":    ", ".join(keywords),
            "text":            text_clean,
            "char_count":      len(text_clean),
        }


# ═══════════════════════════════════════════════════════════════
# 핵심 파이프라인
# ═══════════════════════════════════════════════════════════════

def _gemini_split(text: str, source: str) -> list[dict]:
    client = _get_gemini_client()
    numbered, raw_lines = _number_lines(text)
    total        = len(raw_lines)
    format_hint  = _detect_format_hint(text)
    prompt       = _build_prompt(numbered, total, format_hint)

    sections: list[dict] = []

    # ── 경계 검증 루프 ──────────────────────────────────────────
    for v_attempt in range(_VALIDATE_RETRIES + 1):
        sections = _call_gemini(client, prompt)

        coverage = _validate_coverage(sections, total)
        cat_errs = _validate_categories(sections)
        all_errors = coverage.errors + cat_errs

        if not all_errors:
            break

        if v_attempt < _VALIDATE_RETRIES:
            err_summary = "\n".join(f"  - {e}" for e in all_errors[:5])
            print(
                f"  [WARN] 검증 실패 (시도 {v_attempt + 1}/{_VALIDATE_RETRIES}):\n"
                f"{err_summary}\n  → 재시도..."
            )
            # 오류 내용을 프롬프트에 추가해 재시도
            prompt = (
                _build_prompt(numbered, total, format_hint)
                + f"\n\n[이전 응답의 오류 — 반드시 수정]\n{err_summary}"
            )
        else:
            print(
                f"  [ERROR] {_VALIDATE_RETRIES}회 재시도 후에도 검증 실패. "
                "현재 결과로 진행합니다."
            )

    return list(_sections_to_chunks(sections, raw_lines, source))


# ═══════════════════════════════════════════════════════════════
# 공개 API
# ═══════════════════════════════════════════════════════════════

def chunk(text: str, source: str) -> list[dict]:
    """자소서 텍스트를 Gemini 1-pass로 문항 단위 청킹 + 메타 추출."""
    return _gemini_split(text, source)