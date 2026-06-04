"""
자소서 평가 엔진 - Gemini-3-Flash preview
LLM 통합 평가: 지원동기·직무역량·인재상·작성품질 전 항목 LLM 채점
"""

from __future__ import annotations

import concurrent.futures
import os
import re

from dotenv import load_dotenv
load_dotenv()

from google import genai as _genai
from google.genai import types as _genai_types
from pydantic import BaseModel

from app.core.metrics import track_metrics


# ─── 설정 ─────────────────────────────────────────────────────────────────────
MODEL   = "gemini-3-flash-preview"
WEIGHTS = {"A": 0.20, "B": 0.40, "C": 0.25, "D": 0.15}
LABELS  = {"A": "지원동기", "B": "직무역량", "C": "인재상", "D": "작성품질"}

_SYSTEM_PROMPT = """\
당신은 대한민국 채용 전문가입니다. 자소서를 아래 기준으로 평가하세요.
각 점수는 0~100 정수로 반환하세요 (단, D 항목은 아래 산출 방식 따름).

[문항 유형 인식]
입력에 [문항: ...] 이 주어지면 해당 문항이 요구하는 내용에 맞게 각 항목을 평가하세요.
예) 성장과정·자기소개 문항에서는 A(지원동기) 기준 중 '기업 이해도'보다 '서술 구조·진정성'을 중점 평가하세요.
예) 직무역량 문항에서는 B 항목에 더 높은 기대치를 적용하세요.

평가 기준:
A. 지원동기(20%): 두괄식 구성 여부, 기업 이해도 반영, 입사 설득력
B. 직무역량(40%): STAR 구조(상황-과제-행동-결과) 충족, 수치/정성 성과 포함, 경험→성장→기여 연결
C. 인재상(25%): 핵심 가치관 부합, 소통/협업 경험, 갈등 해결 사례

D. 작성품질(15%): 아래 세 항목을 엄격히 채점해 합산하세요.
  [D1. 분량] 0~40점
  - 글자 수 제한이 주어진 경우: 제한의 90% 이상→40점 / 70~89%→25점 / 50~69%→15점 / 50% 미만→0점
  - 글자 수 제한이 없는 경우: 800자 이상→40점 / 600~799자→20점 / 600자 미만→0점
  [D2. 성과 밀도] 0~40점
  정량·정성 성과를 합산해 채점하세요.
  정량 성과(%, ms, 배수, 건수 등): 3개 이상→30점 / 2개→20점 / 1개→10점 / 0개→0점
  구체적 정성 성과("무엇이→어떻게→어떤 변화" 3요소가 갖춰진 것만 인정): 2개 이상→10점 / 1개→5점 / 0개→0점
  → 합산 최대 40점. 정량 우선이나 정량 없어도 구체적 정성으로 최대 10점 획득 가능.
  인정 예시: "중재 역할로 2주 내 합의 도출 → 일정 지연 없이 마감"
  불인정 예시: "최선을 다해 기여했습니다" / "팀워크가 향상됐습니다" / "많은 것을 배웠습니다"
  날짜(2023년, 1월), 기간(N개월), 전화번호는 정량 성과에서 제외
  [D3. 감점] -20~0점
  - 추상 표현("열심히", "최선을 다" 등) 1개당 -3점 (최대 -10점)
  - 동일 의미 중복 표현 1건당 -2점 (최대 -8점)
  - 내용 없는 bullet 단순 나열 4개 이상: -5점
  → 최종 D = D1 + D2 + D3 (최솟값 0, 최댓값 80)\
"""


# ─── 구조화 출력 스키마 ────────────────────────────────────────────────────────

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

class _ScoreD(BaseModel):
    score:      int   # 최종 0~80 (D1+D2+D3, 최솟값 0)
    d1_volume:  int   # 0~40
    d2_quant:   int   # 0~40
    d3_penalty: int   # -20~0
    reason:     str
    fix:        str

class _EvalResult(BaseModel):
    A:       _ScoreA
    B:       _ScoreB
    C:       _ScoreC
    D:       _ScoreD
    overall: str


# ─── 보조 유틸 (서비스 레이어에서 사용) ────────────────────────────────────────

_RE_FREE_FORMAT = re.compile(r"제한\s*없음")

_RE_CHAR_LIMIT_PATTERNS = [
    re.compile(r"최대\s*([\d,]+)\s*자(?:\s*입력\s*가능)?"),
    re.compile(r"[\(\（]([\d,]+)\s*자[\)\）]"),
    re.compile(r"([\d,]+)\s*자\s*이내"),
    re.compile(r"([\d,]+)\s*자"),
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


# ─── Gemini 클라이언트 ─────────────────────────────────────────────────────────

def _get_gemini_client() -> _genai.Client:
    project = os.getenv("GCP_PROJECT_ID")
    if project:
        return _genai.Client(vertexai=True, project=project, location=os.getenv("GCP_LOCATION", "global"))
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        return _genai.Client(api_key=api_key)
    raise ValueError("GCP_PROJECT_ID 또는 GEMINI_API_KEY 환경변수를 설정하세요.")


# ─── LLM 평가 ─────────────────────────────────────────────────────────────────

@track_metrics
def llm_evaluate(text: str, char_limit: int | None = None, question: str = "") -> _EvalResult:
    client = _get_gemini_client()
    context = ""
    if question:
        context += f"[문항: {question}]\n"
    if char_limit:
        context += f"[글자 수 제한: {char_limit}자, 실제 글자 수: {len(text.strip())}자]\n"
    if context:
        context += "\n"

    for attempt in range(3):
        resp = client.models.generate_content(
            model=MODEL,
            contents=f"{context}다음 자소서를 평가해주세요:\n\n{text}",
            config=_genai_types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=_EvalResult,
                temperature=0,
            ),
        )
        try:
            return _EvalResult.model_validate_json(resp.text)
        except Exception:
            if attempt == 2:
                raise
    raise RuntimeError("llm_evaluate 재시도 실패")


# ─── 가중 합산 ─────────────────────────────────────────────────────────────────

def compute_weighted(llm: _EvalResult) -> float:
    total = (
        llm.A.score * WEIGHTS["A"] +
        llm.B.score * WEIGHTS["B"] +
        llm.C.score * WEIGHTS["C"] +
        llm.D.score * WEIGHTS["D"]
    )
    return round(total, 2)


# ─── 출력 포매터 ───────────────────────────────────────────────────────────────

def grade_label(score: float) -> str:
    if   score >= 85: return "우수 ★★★"
    elif score >= 70: return "양호 ★★"
    elif score >= 55: return "보통 ★"
    else:             return "미흡"


def print_result(llm: _EvalResult, weighted: float) -> None:
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
    print(f"[D] {LABELS['D']}  ({int(WEIGHTS['D']*100)}%)  →  {llm.D.score}/80")
    print(f"    D1 분량: {llm.D.d1_volume}/40  |  D2 수치성과: {llm.D.d2_quant}/40  |  D3 감점: {llm.D.d3_penalty}")
    print(f"    근거: {llm.D.reason}")
    print(f"    개선: {llm.D.fix}")
    print(sep)
    print(f"\n  총평: {llm.overall}\n")


# ─── 전후 비교 ─────────────────────────────────────────────────────────────────

@track_metrics
def evaluate_comparison(original: str, improved: str, char_limit: int | None = None, question: str = "") -> dict:
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f_before = executor.submit(evaluate, original, char_limit, question)
        f_after  = executor.submit(evaluate, improved, char_limit, question)
        before   = f_before.result()
        after    = f_after.result()

    delta = round(after["weighted"] - before["weighted"], 2)
    sign  = "+" if delta >= 0 else ""

    sep = "═" * 52
    print(f"\n{sep}")
    print(f"  전후 비교 요약")
    print(f"{sep}")
    print(f"  {'항목':<12} {'원문':>6}  {'개선안':>6}  {'변화':>6}")
    print(f"  {'─'*40}")
    for k in ["A", "B", "C", "D"]:
        b = before["llm"][k]["score"]
        a = after["llm"][k]["score"]
        print(f"  {k}. {LABELS[k]:<10} {b:>6}  {a:>6}  {a-b:>+6}")
    print(f"  {'─'*40}")
    print(f"  {'최종 점수':<12} {before['weighted']:>6.2f}  {after['weighted']:>6.2f}  {sign}{delta:>+5.2f}")
    print(f"  {'등급':<12} {grade_label(before['weighted']):>8}  {grade_label(after['weighted']):>8}")
    print(f"{sep}\n")

    return {
        "before": before,
        "after":  after,
        "delta":  delta,
        "per_category": {
            k: {
                "before":        before["llm"][k]["score"],
                "after":         after["llm"][k]["score"],
                "delta":         after["llm"][k]["score"] - before["llm"][k]["score"],
                "before_detail": {kk: vv for kk, vv in before["llm"][k].items() if kk != "score"},
                "after_detail":  {kk: vv for kk, vv in after["llm"][k].items()  if kk != "score"},
            }
            for k in ["A", "B", "C", "D"]
        } | {
            "overall": {
                "before": before["llm"]["overall"],
                "after":  after["llm"]["overall"],
            },
        },
    }


# ─── 메인 평가 함수 ────────────────────────────────────────────────────────────

@track_metrics
def evaluate(text: str, char_limit: int | None = None, question: str = "") -> dict:
    if len(text.strip()) < 100:
        raise ValueError("자소서를 100자 이상 입력해주세요.")

    print("  [1/2] Gemini-3-Flash preview 평가 중...")
    llm = llm_evaluate(text, char_limit, question=question)
    print("  [2/2] 점수 합산 중...")
    weighted = compute_weighted(llm)
    print_result(llm, weighted)
    return {"llm": llm.model_dump(), "weighted": weighted}
