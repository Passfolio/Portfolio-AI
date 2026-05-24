"""
포트폴리오 평가 엔진 - Gemini-3-Flash preview
LLM 통합 평가: 과정/판단력·역할/기여도·성과/인사이트·작성품질·직무연관성 전 항목 LLM 채점
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
load_dotenv()

from google import genai as _genai
from google.genai import types as _genai_types
from pydantic import BaseModel


# ═══════════════════════════════════════════════════════════════
# 설정
# ═══════════════════════════════════════════════════════════════

MODEL   = "gemini-3-flash-preview"
WEIGHTS = {"A": 0.35, "B": 0.25, "C": 0.20, "D": 0.10, "E": 0.10}
LABELS  = {
    "A": "과정/판단력",
    "B": "역할/기여도",
    "C": "성과/인사이트",
    "D": "작성품질",
    "E": "직무연관성",
}

_SYSTEM_PROMPT = """\
당신은 대한민국 채용 전문가입니다. 포트폴리오 내용을 아래 기준으로 평가하세요.
각 점수는 0~100 정수로 반환하세요.

평가 기준:
A. 과정/판단력(35%): 배경→문제인식→선택/판단→실행 흐름의 명확성, '왜 그 선택을 했는지'가 드러나는지
  - 단순 나열이 아닌 사고방식과 판단 근거 표현
  - 결과보다 과정과 의사결정 과정이 핵심
  - 배경→문제인식→판단→실행→결과 흐름이 단계별로 명확하게 서술되어 있는가
  - 기술/도구 선택 이유가 구체적으로 설명되어 있는가 ('왜 이 기술인가', 대안 대비 근거)

B. 역할/기여도(25%): 본인이 맡은 역할의 구체성, 팀 내 주도적 참여
  - 본인이 주도적으로 수행한 부분이 명시되는가
  - '내가/담당/주도/직접 설계' 등 1인칭 기여 표현이 충분히 나타나는가
  - 트러블슈팅 경험이 구체적으로 서술되어 있는가 (문제 발생 → 원인 파악 → 해결 과정)

C. 성과/인사이트(20%): 정량/정성 성과 구체성, 경험을 통한 배움과 성장
  - 수치로 드러나는 정량적 성과 우선
  - 경험 → 성장한 점 → 직무 기여 방향 연결

D. 작성품질(10%): 아래 세 항목을 엄격히 채점해 합산하세요.
  [D1. 분량] 0~40점
  - 400자 이상: 40점 / 200~399자: 20점 / 200자 미만: 0점
  [D2. 수치 성과 밀도] 0~40점
  - 정량 성과(%, ms, 배수, 건수 등) 3개 이상: 40점 / 2개: 25점 / 1개: 10점 / 0개: 0점
  - 날짜(2023년, 1월), 기간(N개월), 전화번호는 정량 성과에서 제외
  [D3. 감점] -20~0점
  - 추상 표현("열심히", "최선을 다" 등) 1개당 -3점 (최대 -10점)
  - 동일 의미 중복 표현 1건당 -4점 (최대 -8점)
  - bullet 단순 나열(판단 근거·성과 없는 짧은 항목 4개 이상): -5점
  → 최종 D = D1 + D2 + D3 (최솟값 0, 최댓값 80)

E. 직무연관성(10%): 목표 직무와의 연관성, 직무 역량 증명 여부
  - 직무 관련 기술·경험·역량이 드러나는가\
"""


# ═══════════════════════════════════════════════════════════════
# 구조화 출력 스키마
# ═══════════════════════════════════════════════════════════════

class _ScoreA(BaseModel):
    score:  int
    reason: str
    fix:    str

class _ScoreB(BaseModel):
    score:  int
    reason: str
    fix:    str

class _ScoreC(BaseModel):
    score:            int
    has_quantitative: bool
    reason:           str
    fix:              str

class _ScoreD(BaseModel):
    score:      int   # 최종 0~80 (D1+D2+D3, 최솟값 0)
    d1_volume:  int   # 0~40
    d2_quant:   int   # 0~40
    d3_penalty: int   # -20~0
    reason:     str
    fix:        str

class _ScoreE(BaseModel):
    score:  int
    reason: str
    fix:    str

class _EvalResult(BaseModel):
    A:       _ScoreA
    B:       _ScoreB
    C:       _ScoreC
    D:       _ScoreD
    E:       _ScoreE
    overall: str


# ═══════════════════════════════════════════════════════════════
# Gemini 클라이언트
# ═══════════════════════════════════════════════════════════════

def _get_gemini_client() -> _genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        return _genai.Client(api_key=api_key)
    project = os.getenv("GCP_PROJECT_ID")
    if project:
        return _genai.Client(vertexai=True, project=project, location=os.getenv("GCP_LOCATION", "global"))
    raise ValueError("GEMINI_API_KEY 또는 GCP_PROJECT_ID 환경변수를 설정하세요.")


# ═══════════════════════════════════════════════════════════════
# LLM 평가
# ═══════════════════════════════════════════════════════════════

def llm_evaluate(text: str) -> _EvalResult:
    client = _get_gemini_client()
    resp = client.models.generate_content(
        model=MODEL,
        contents=f"다음 포트폴리오 내용을 평가해주세요:\n\n{text}",
        config=_genai_types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=_EvalResult,
            temperature=0,
            max_output_tokens=4096,
        ),
    )
    return _EvalResult.model_validate_json(resp.text)


# ═══════════════════════════════════════════════════════════════
# 가중 합산
# ═══════════════════════════════════════════════════════════════

def compute_weighted(llm: _EvalResult) -> float:
    total = (
        llm.A.score * WEIGHTS["A"]
        + llm.B.score * WEIGHTS["B"]
        + llm.C.score * WEIGHTS["C"]
        + llm.D.score * WEIGHTS["D"]
        + llm.E.score * WEIGHTS["E"]
    )
    return round(total, 2)


# ═══════════════════════════════════════════════════════════════
# 출력 포매터
# ═══════════════════════════════════════════════════════════════

def grade_label(score: float) -> str:
    if   score >= 85: return "우수 ★★★"
    elif score >= 70: return "양호 ★★"
    elif score >= 55: return "보통 ★"
    else:             return "미흡"


def print_result(llm: _EvalResult, weighted: float) -> None:
    sep = "─" * 56

    print(f"\n{'═'*56}")
    print(f"  포트폴리오 평가 결과  |  모델: {MODEL}")
    print(f"{'═'*56}")
    print(f"  최종 점수  {weighted:6.2f} / 100   {grade_label(weighted)}")
    print(f"{'═'*56}\n")

    print(f"[A] {LABELS['A']}  ({int(WEIGHTS['A']*100)}%)  →  {llm.A.score}/100")
    print(f"    근거: {llm.A.reason}")
    print(f"    개선: {llm.A.fix}")
    print(sep)

    print(f"[B] {LABELS['B']}  ({int(WEIGHTS['B']*100)}%)  →  {llm.B.score}/100")
    print(f"    근거: {llm.B.reason}")
    print(f"    개선: {llm.B.fix}")
    print(sep)

    print(f"[C] {LABELS['C']}  ({int(WEIGHTS['C']*100)}%)  →  {llm.C.score}/100"
          f"  (정량성과: {'✓' if llm.C.has_quantitative else '✗'})")
    print(f"    근거: {llm.C.reason}")
    print(f"    개선: {llm.C.fix}")
    print(sep)

    print(f"[D] {LABELS['D']}  ({int(WEIGHTS['D']*100)}%)  →  {llm.D.score}/80")
    print(f"    D1 분량: {llm.D.d1_volume}/40  |  D2 수치성과: {llm.D.d2_quant}/40  |  D3 감점: {llm.D.d3_penalty}")
    print(f"    근거: {llm.D.reason}")
    print(f"    개선: {llm.D.fix}")
    print(sep)

    print(f"[E] {LABELS['E']}  ({int(WEIGHTS['E']*100)}%)  →  {llm.E.score}/100")
    print(f"    근거: {llm.E.reason}")
    print(f"    개선: {llm.E.fix}")
    print(sep)

    print(f"\n  총평: {llm.overall}\n")


# ═══════════════════════════════════════════════════════════════
# 메인 평가 함수
# ═══════════════════════════════════════════════════════════════

def evaluate(text: str, meta: dict | None = None) -> dict:
    if len(text.strip()) < 50:
        raise ValueError("포트폴리오 내용을 50자 이상 입력해주세요.")

    print("  [1/2] Gemini-3-Flash preview 평가 중...")
    llm = llm_evaluate(text)

    print("  [2/2] 점수 합산 중...")
    weighted = compute_weighted(llm)

    print_result(llm, weighted)
    return {"llm": llm.model_dump(), "weighted": weighted}


# ═══════════════════════════════════════════════════════════════
# 전후 비교
# ═══════════════════════════════════════════════════════════════

def evaluate_comparison(
    original: str,
    improved: str,
    meta: dict | None = None,
) -> dict:
    print("\n  ── 원문 평가 ──")
    before = evaluate(original, meta=meta)
    print("\n  ── 개선안 평가 ──")
    after  = evaluate(improved, meta=meta)

    delta = round(after["weighted"] - before["weighted"], 2)
    sign  = "+" if delta >= 0 else ""

    sep = "═" * 56
    print(f"\n{sep}")
    print(f"  전후 비교 요약")
    print(f"{sep}")
    print(f"  {'항목':<16} {'원문':>6}  {'개선안':>6}  {'변화':>6}")
    print(f"  {'─'*42}")

    for k in ["A", "B", "C", "D", "E"]:
        b = before["llm"][k]["score"]
        a = after["llm"][k]["score"]
        print(f"  {k}. {LABELS[k]:<14} {b:>6}  {a:>6}  {a-b:>+6}")

    print(f"  {'─'*42}")
    print(f"  {'최종 점수':<16} {before['weighted']:>6.2f}  {after['weighted']:>6.2f}  {sign}{delta:>+5.2f}")
    print(f"  {'등급':<16} {grade_label(before['weighted']):>8}  {grade_label(after['weighted']):>8}")
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
            for k in ["A", "B", "C", "D", "E"]
        } | {
            "overall": {
                "before": before["llm"]["overall"],
                "after":  after["llm"]["overall"],
            },
        },
    }


# ═══════════════════════════════════════════════════════════════
# CLI 테스트
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    sample = """
    사용자 인증 시스템의 응답 속도가 평균 1.2초로 느려 UX 불만이 발생하고 있었습니다.
    원인을 분석한 결과, 매 요청마다 DB에서 사용자 정보를 조회하는 구조가 문제였습니다.
    Redis 캐싱 도입을 검토했으나, 팀 인프라 환경상 추가 서버 비용이 부담이었습니다.
    대신 JWT 토큰에 필수 정보를 포함해 DB 조회를 최소화하는 방향으로 설계를 변경했습니다.
    그 결과 응답 속도가 평균 0.3초로 75% 개선되었고, 서버 비용 증가 없이 성능을 확보했습니다.
    이 경험을 통해 최적의 기술 선택이 항상 최신 기술이 아니라 상황에 맞는 판단임을 배웠습니다.
    """
    evaluate(sample)
