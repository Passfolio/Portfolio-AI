"""
자소서 평가 엔진 - Gemini-3-Flash preview
규칙 기반(작성품질) + LLM(지원동기·직무역량·인재상·AI의심도) 하이브리드
"""

from __future__ import annotations

import os
import re
from typing import Any

from dotenv import load_dotenv
load_dotenv()

from google import genai as _genai
from google.genai import types as _genai_types
from pydantic import BaseModel

# ── 설정 ──────────────────────────────────────────────────────────────────────
MODEL   = "gemini-3-flash-preview"
WEIGHTS = {"A": 0.15, "B": 0.35, "C": 0.20, "D": 0.15, "E": 0.15}
LABELS  = {"A": "지원동기", "B": "직무역량", "C": "인재상", "D": "작성품질", "E": "AI 의심도"}

_RE_NUMBER = re.compile(
    r"\d+\s*[%％]"           # 퍼센트
    r"|\d+\s*(ms|초|분|시간)"  # 시간
    r"|\d+\s*(배|배수)"        # 배수
    r"|\d+\s*(명|팀|개|건|회|번|줄|행|열|종|개월|주)"  # 단위
    r"|\d+\s*(억|만|원|달러)"  # 금액
    r"|\d+\s*(년|월|일)"       # 기간
    r"|\d+\s*(위|등|점)"       # 순위/점수
)

_SYSTEM_PROMPT = """\
당신은 대한민국 채용 전문가입니다. 자소서를 아래 기준으로 평가하세요.
각 점수는 0~100 정수로 반환하세요.

평가 기준:
A. 지원동기(15%): 두괄식 구성 여부, 기업 이해도 반영, 입사 설득력
B. 직무역량(35%): STAR 구조(상황-과제-행동-결과) 충족, 수치/정성 성과 포함, 경험→성장→기여 연결
C. 인재상(20%): 핵심 가치관 부합, 소통/협업 경험, 갈등 해결 사례
E. AI의심도: 아래 패턴이 많을수록 높은 점수(높으면 나쁨)
  - 무견해/판단 회피
  - 구조적 전형성
  - 지나친 과장/편중
  - 구체적 근거 부족
  - 복잡한 서술 구조\
"""


# ── 구조화 출력 스키마 ─────────────────────────────────────────────────────────

class _ScoreA(BaseModel):
    score:  int
    reason: str
    fix:    str

class _ScoreB(BaseModel):
    score:      int
    star_score: int
    num_score:  int
    reason:     str
    fix:        str

class _ScoreC(BaseModel):
    score:  int
    reason: str
    fix:    str

class _ScoreE(BaseModel):
    score:    int
    detected: list[str]
    reason:   str

class _EvalResult(BaseModel):
    A:       _ScoreA
    B:       _ScoreB
    C:       _ScoreC
    E:       _ScoreE
    overall: str


# ── Gemini 클라이언트 ─────────────────────────────────────────────────────────

def _get_gemini_client() -> _genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        return _genai.Client(api_key=api_key)
    project = os.getenv("GCP_PROJECT_ID")
    if project:
        return _genai.Client(vertexai=True, project=project, location=os.getenv("GCP_LOCATION", "global"))
    raise ValueError("GEMINI_API_KEY 또는 GCP_PROJECT_ID 환경변수를 설정하세요.")


# ── 1단계: 규칙 기반 (작성품질 D) ────────────────────────────────────────────

_RE_FREE_FORMAT = re.compile(r"제한\s*없음")

_RE_CHAR_LIMIT_PATTERNS = [
    re.compile(r"최대\s*([\d,]+)\s*자(?:\s*입력\s*가능)?"),  # 최대 n자 / 최대 n자 입력 가능
    re.compile(r"[\(\（]([\d,]+)\s*자[\)\）]"),               # (n자) （n자）
    re.compile(r"([\d,]+)\s*자\s*이내"),                       # n자 이내
    re.compile(r"([\d,]+)\s*자"),                              # n자 (fallback)
]

_COMPETENCY_KEYWORDS = re.compile(
    r"직무|업무|역량|경험|성과|프로젝트|도전|팀워크|협업|갈등|문제\s*해결|리더십|성취|과제|활동"
)


def parse_char_limit(section: str) -> int | None:
    """섹션 제목에서 글자 수 제한을 파싱. '제한없음' 또는 제한이 없으면 None."""
    if _RE_FREE_FORMAT.search(section):
        return None
    for pat in _RE_CHAR_LIMIT_PATTERNS:
        m = pat.search(section)
        if m:
            return int(m.group(1).replace(",", ""))
    return None


def is_competency_question(question: str) -> bool:
    """직무역량·경험 관련 문항 여부 판별."""
    return bool(_COMPETENCY_KEYWORDS.search(question))


def rule_based_score(text: str, char_limit: int | None = None, is_competency: bool = False) -> tuple[int, dict]:
    """분량·반복·수치 포함 여부를 규칙으로 분석해 D 점수 반환."""
    details: dict[str, Any] = {}

    length = len(text.strip())
    details["length"] = length

    if char_limit:
        ratio = length / char_limit
        if   ratio >= 0.9: vol_score = 10
        elif ratio >= 0.7: vol_score = 8
        elif ratio >= 0.5: vol_score = 6
        elif ratio >= 0.3: vol_score = 4
        else:              vol_score = 2
        details["char_limit"]  = char_limit
        details["fill_ratio"]  = round(ratio * 100, 1)
        details["vol_status"]  = "적정" if ratio >= 0.7 else ("내용 부족" if ratio < 0.5 else "보통")
    elif is_competency:
        # 직무역량·경험 문항: 800자 기준
        if   length > 800:  vol_score, status = 7,  "가독성 저하"
        elif length >= 600: vol_score, status = 10, "적정"
        elif length >= 400: vol_score, status = 6,  "내용 부족"
        elif length >= 200: vol_score, status = 4,  "내용 부족"
        else:               vol_score, status = 2,  "내용 부족"
        details["free_standard"] = 800
        details["is_competency"] = True
        details["vol_status"]    = status
    else:
        # 그 외 문항: 500자 기준
        if   length >= 600: vol_score, status = 7,  "가독성 저하"
        elif length >= 400: vol_score, status = 10, "적정"
        elif length >= 200: vol_score, status = 6,  "내용 부족"
        elif length >= 100: vol_score, status = 4,  "내용 부족"
        else:               vol_score, status = 2,  "내용 부족"
        details["free_standard"] = 500
        details["is_competency"] = False
        details["vol_status"]    = status
    details["vol_score"] = vol_score

    sentences = [s.strip() for s in re.split(r"[.!?。]", text) if len(s.strip()) > 10]
    seen, dup_count = set(), 0
    for s in sentences:
        key = s[:20]
        if key in seen:
            dup_count += 1
        seen.add(key)
    details["dup_count"] = dup_count
    penalty = 2 if dup_count >= 3 else (1 if dup_count >= 1 else 0)

    has_number = bool(_RE_NUMBER.search(text))
    details["has_number"] = has_number
    bonus = 1 if has_number else 0

    score = max(1, min(10, vol_score - penalty + bonus))
    details["final_score"] = score
    return score, details


# ── 2단계: LLM 평가 ───────────────────────────────────────────────────────────

def llm_evaluate(text: str) -> _EvalResult:
    """Gemini-3-Flash preview로 A·B·C·E 평가 (구조화 출력)."""
    client = _get_gemini_client()
    resp = client.models.generate_content(
        model=MODEL,
        contents=f"다음 자소서를 평가해주세요:\n\n{text}",
        config=_genai_types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=_EvalResult,
            temperature=0,
            max_output_tokens=4096,
        ),
    )
    return _EvalResult.model_validate_json(resp.text)


# ── 3단계: 가중 합산 ──────────────────────────────────────────────────────────

def compute_weighted(llm: _EvalResult, d_score: int) -> float:
    """전 항목 0~100 기준으로 가중 합산. 결과 범위 0~100.
    - A·B·C·E: LLM이 0~100으로 반환
    - D: 0~10 → ×10으로 정규화
    - E: AI 의심도이므로 역산 (100 - score)
    """
    e_adj  = 100 - llm.E.score  # 높을수록 나쁨 → 역산
    d_norm = d_score * 10        # 0~10 → 0~100
    total = (
        llm.A.score * WEIGHTS["A"] +
        llm.B.score * WEIGHTS["B"] +
        llm.C.score * WEIGHTS["C"] +
        d_norm      * WEIGHTS["D"] +
        e_adj       * WEIGHTS["E"]
    )
    return round(total, 2)


# ── 출력 포매터 ───────────────────────────────────────────────────────────────

def grade_label(score: float) -> str:
    if   score >= 85: return "우수 ★★★"
    elif score >= 70: return "양호 ★★"
    elif score >= 55: return "보통 ★"
    else:             return "미흡"


def print_result(llm: _EvalResult, d_score: int, d_detail: dict, weighted: float) -> None:
    sep = "─" * 52

    print(f"\n{'═'*52}")
    print(f"  자소서 평가 결과  |  모델: {MODEL}")
    print(f"{'═'*52}")
    print(f"  최종 점수  {weighted:6.2f} / 100   {grade_label(weighted)}")
    print(f"{'═'*52}\n")

    print(f"[A] {LABELS['A']}  ({int(WEIGHTS['A']*100)}%)  →  {llm.A.score}/100")
    print(f"    근거: {llm.A.reason}")
    print(f"    개선: {llm.A.fix}")
    print(sep)

    print(f"[B] {LABELS['B']}  ({int(WEIGHTS['B']*100)}%)  →  {llm.B.score}/100")
    print(f"    STAR 구조: {llm.B.star_score}/100  |  수치 성과: {llm.B.num_score}/100")
    print(f"    근거: {llm.B.reason}")
    print(f"    개선: {llm.B.fix}")
    print(sep)

    print(f"[C] {LABELS['C']}  ({int(WEIGHTS['C']*100)}%)  →  {llm.C.score}/100")
    print(f"    근거: {llm.C.reason}")
    print(f"    개선: {llm.C.fix}")
    print(sep)

    print(f"[D] {LABELS['D']}  ({int(WEIGHTS['D']*100)}%)  →  {d_score * 10}/100")
    if "char_limit" in d_detail:
        limit_info = f"  |  제한 {d_detail['char_limit']}자 ({d_detail['fill_ratio']}% 충족, {d_detail['vol_status']})"
    elif "free_standard" in d_detail:
        category = "직무역량" if d_detail["is_competency"] else "일반"
        limit_info = f"  |  자유형식 ({category} {d_detail['free_standard']}자 기준, {d_detail['vol_status']})"
    else:
        limit_info = ""
    print(f"    글자 수: {d_detail['length']}자{limit_info}  |  반복 문장: {d_detail['dup_count']}건  |  수치 포함: {'✓' if d_detail['has_number'] else '✗'}")
    print(sep)

    e_adj = 100 - llm.E.score
    print(f"[E] {LABELS['E']}  ({int(WEIGHTS['E']*100)}%)  →  원점수 {llm.E.score}/100  (역산 적용: {e_adj}/100)")
    print(f"    감지 패턴: {', '.join(llm.E.detected) if llm.E.detected else '없음'}")
    print(f"    근거: {llm.E.reason}")
    print(sep)

    print(f"\n  총평: {llm.overall}\n")


# ── 전후 비교 ─────────────────────────────────────────────────────────────────

def evaluate_comparison(original: str, improved: str, char_limit: int | None = None, question: str = "") -> dict:
    """원문과 개선안을 각각 평가해 점수 delta를 출력."""
    print("\n  ── 원문 평가 ──")
    before = evaluate(original, char_limit, question)
    print("\n  ── 개선안 평가 ──")
    after  = evaluate(improved, char_limit, question)

    delta = round(after["weighted"] - before["weighted"], 2)
    sign  = "+" if delta >= 0 else ""

    sep = "═" * 52
    print(f"\n{sep}")
    print(f"  전후 비교 요약")
    print(f"{sep}")
    print(f"  {'항목':<12} {'원문':>6}  {'개선안':>6}  {'변화':>6}")
    print(f"  {'─'*40}")

    llm_keys = ["A", "B", "C", "E"]
    for k in llm_keys:
        b = before["llm"][k]["score"]
        a = after["llm"][k]["score"]
        d = a - b
        arrow = "+" if d > 0 else ("" if d == 0 else "")
        print(f"  {k}. {LABELS[k]:<10} {b:>6}  {a:>6}  {arrow}{d:>+5}")

    b_d = before["D"]["score"]
    a_d = after["D"]["score"]
    d_d = a_d - b_d
    print(f"  D. {LABELS['D']:<10} {b_d*10:>6}  {a_d*10:>6}  {d_d*10:>+6}")

    print(f"  {'─'*40}")
    print(f"  {'최종 점수':<12} {before['weighted']:>6.2f}  {after['weighted']:>6.2f}  {sign}{delta:>+5.2f}")
    print(f"  {'등급':<12} {grade_label(before['weighted']):>8}  {grade_label(after['weighted']):>8}")
    print(f"{sep}\n")

    return {
        "before":  before,
        "after":   after,
        "delta":   delta,
        "per_category": {
            k: {
                "before": before["llm"][k]["score"],
                "after":  after["llm"][k]["score"],
                "delta":  after["llm"][k]["score"] - before["llm"][k]["score"],
                "before_detail": {kk: vv for kk, vv in before["llm"][k].items() if kk != "score"},
                "after_detail":  {kk: vv for kk, vv in after["llm"][k].items()  if kk != "score"},
            }
            for k in llm_keys
        } | {
            "D": {
                "before": b_d,
                "after":  a_d,
                "delta":  d_d,
                "before_detail": before["D"]["detail"],
                "after_detail":  after["D"]["detail"],
            },
            "overall": {
                "before": before["llm"]["overall"],
                "after":  after["llm"]["overall"],
            },
        },
    }


# ── 메인 ──────────────────────────────────────────────────────────────────────

def evaluate(text: str, char_limit: int | None = None, question: str = "") -> dict:
    if len(text.strip()) < 100:
        raise ValueError("자소서를 100자 이상 입력해주세요.")

    print("  [1/3] 규칙 기반 분석 중...")
    d_score, d_detail = rule_based_score(text, char_limit, is_competency_question(question))

    print("  [2/3] Gemini-3-Flash preview 평가 중...")
    llm = llm_evaluate(text)

    print("  [3/3] 점수 합산 중...")
    weighted = compute_weighted(llm, d_score)

    print_result(llm, d_score, d_detail, weighted)
    return {"llm": llm.model_dump(), "D": {"score": d_score, "detail": d_detail}, "weighted": weighted}

