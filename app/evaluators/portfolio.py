"""
포트폴리오 평가 엔진 - Gemini-3-Flash preview
규칙 기반(작성품질 D) + LLM(과정/판단력·역할/기여도·성과/인사이트·직무연관성) 하이브리드

[D 규칙 기반 - 3개 지표]
  D1. 전체 분량 적절성  (글자수 기반, 400~1000자 적정, 초과·부족 감점)
  D2. 수치 성과 밀도    (정량 수치 패턴 등장 횟수)
  D7. 감점 요인         (추상 표현·중복 문장·단순 나열 bullet 비율, 패널티)

[LLM 평가 - 문제해결구조·역할명확성·기술선택근거·트러블슈팅 포함]
  A. 과정/판단력(35%): 배경→문제→판단→실행 흐름, 기술 선택 근거
  B. 역할/기여도(25%): 1인칭 기여 명확성, 트러블슈팅 경험
  C. 성과/인사이트(20%): 정량/정성 성과, 배움과 성장
  E. 직무연관성(10%): 목표 직무 연관성
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

# D 서브 항목 가중치 (합 = 1.0)
_D_SUB_WEIGHTS = {
    "d1_volume": 0.50,   # 전체 분량 적절성
    "d2_quant":  0.50,   # 수치 성과 밀도
}

_SYSTEM_PROMPT = """\
당신은 대한민국 채용 전문가입니다. 포트폴리오 내용을 아래 기준으로 평가하세요.
각 점수는 0~100 정수로 반환하세요.

평가 기준:
A. 과정/판단력(35%): 배경→문제인식→선택/판단→실행 흐름의 명확성, '왜 그 선택을 했는지'가 드러나는지
  - 단순 나열이 아닌 사고방식과 판단 근거 표현
  - 결과보다 과정과 의사결정 과정이 핵심
  - 아이디어·판단 기준·가치관이 드러나는가
  - 배경→문제인식→판단→실행→결과 흐름이 단계별로 명확하게 서술되어 있는가
  - 기술/도구 선택 이유가 구체적으로 설명되어 있는가 ('왜 이 기술인가', 대안 대비 근거)

B. 역할/기여도(25%): 본인이 맡은 역할의 구체성, 팀 내 주도적 참여
  - 단순 참여가 아닌 본인만의 기여 명확성
  - 본인이 주도적으로 수행한 부분이 명시되는가
  - 프로젝트 내에서 무엇을 보여주었는지
  - '내가/담당/주도/직접 설계' 등 1인칭 기여 표현이 충분히 나타나는가
  - 트러블슈팅 경험이 구체적으로 서술되어 있는가 (문제 발생 → 원인 파악 → 해결 과정)

C. 성과/인사이트(20%): 정량/정성 성과 구체성, 경험을 통한 배움과 성장
  - 수치로 드러나는 정량적 성과 우선
  - 결과물이 아닌 인사이트와 배움 중심
  - 경험 → 성장한 점 → 직무 기여 방향 연결

E. 직무연관성(10%): 목표 직무와의 연관성, 직무 역량 증명 여부
  - 대외활동도 '직무 역량을 보여줄 수 있느냐'가 판단 기준
  - 직무 관련 기술·경험·역량이 드러나는가\
"""


# ═══════════════════════════════════════════════════════════════
# 규칙 패턴 정의
# ═══════════════════════════════════════════════════════════════

# D2: 정량 수치 패턴
_RE_QUANT = re.compile(
    r"\d+\s*[%％]"
    r"|\d+\s*(ms|초|분|시간|fps|rps|tps)"
    r"|\d+\s*(배|배수|배속)"
    r"|\d+\s*(명|팀|개|건|회|번|줄|행|열|종|개월|주|페이지|req)"
    r"|\d+\s*(억|만|원|달러|\$|₩)"
    r"|\d+\s*(위|등|점|점수)"
    r"|\d+[,\d]*\s*(건|개|명|회)"     # 1,200건 등 천단위 콤마 포함
    r"|\d+\s*(GB|MB|KB|TB)"
)

# D7: 감점 - 추상 표현
_RE_ABSTRACT = re.compile(
    r"열심히|최선을\s*다|다양한\s*경험|많은\s*것을\s*배|성장할\s*수\s*있|노력했|도움이\s*됐|뜻깊"
    r"|좋은\s*경험|의미\s*있는\s*경험|값진\s*경험|소중한\s*경험"
)

# D7: 감점 - 단순 나열 문장 (동사 없이 명사·기술 나열)
_RE_NAIVE_LIST = re.compile(
    r"^[-•*]\s*.{0,40}$",   # bullet 뒤 40자 이하 짧은 항목이 연속되는 경우 체크용
    re.MULTILINE,
)


# ═══════════════════════════════════════════════════════════════
# 구조화 출력 스키마 (LLM)
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

class _ScoreE(BaseModel):
    score:  int
    reason: str
    fix:    str

class _EvalResult(BaseModel):
    A:       _ScoreA
    B:       _ScoreB
    C:       _ScoreC
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
# D1. 분량 적절성 (0~10)
# ═══════════════════════════════════════════════════════════════

def _score_d1_volume(text: str) -> tuple[int, str]:
    """프로젝트 섹션 기준: 400~1000자 적정."""
    length = len(text.strip())
    if   length > 1200: return 6,  f"{length}자 (가독성 저하, 밀도 있게 압축 권장)"
    elif length >= 400:  return 10, f"{length}자 (적정)"
    elif length >= 200:  return 6,  f"{length}자 (내용 부족)"
    elif length >= 100:  return 4,  f"{length}자 (내용 부족)"
    else:                return 2,  f"{length}자 (내용 부족)"


# ═══════════════════════════════════════════════════════════════
# D2. 수치 성과 밀도 (0~10)
# ═══════════════════════════════════════════════════════════════

def _score_d2_quant(text: str, meta: dict | None = None) -> tuple[int, str]:
    """텍스트 내 정량 수치 패턴 감지 + meta.achievements 연동."""
    hits = _RE_QUANT.findall(text)
    count = len(hits)

    # meta.achievements가 있으면 추가 카운트
    if meta:
        ach = meta.get("achievements", [])
        for a in ach:
            bonus = len(_RE_QUANT.findall(a))
            count += bonus

    if   count >= 4: return 10, f"수치 표현 {count}건 (풍부)"
    elif count >= 2: return 8,  f"수치 표현 {count}건"
    elif count == 1: return 5,  f"수치 표현 {count}건 (부족)"
    else:            return 2,  "수치 표현 없음"


# ═══════════════════════════════════════════════════════════════
# D7. 감점 패널티 계산
# ═══════════════════════════════════════════════════════════════

def _penalty_d7(text: str) -> tuple[int, list[str]]:
    """추상 표현·중복 문장·단순 나열 bullet 비율로 패널티(-3~0)."""
    reasons: list[str] = []
    penalty = 0

    # 추상 표현
    abstract_hits = _RE_ABSTRACT.findall(text)
    if len(abstract_hits) >= 3:
        penalty += 2
        reasons.append(f"추상 표현 {len(abstract_hits)}건 (열심히·최선 등)")
    elif len(abstract_hits) >= 1:
        penalty += 1
        reasons.append(f"추상 표현 {len(abstract_hits)}건")

    # 중복 문장 (앞 20자 기준)
    sentences = [s.strip() for s in re.split(r"[.!?。\n]", text) if len(s.strip()) > 10]
    seen, dup_count = set(), 0
    for s in sentences:
        key = s[:20]
        if key in seen:
            dup_count += 1
        seen.add(key)
    if dup_count >= 2:
        penalty += 1
        reasons.append(f"중복 문장 {dup_count}건")

    # 단순 bullet 나열 비율 (전체 줄 중 bullet이 70% 이상이면 감점)
    lines = [l for l in text.splitlines() if l.strip()]
    bullet_lines = [l for l in lines if re.match(r"^[-•*]\s+", l.strip())]
    if lines and len(bullet_lines) / len(lines) >= 0.7 and len(bullet_lines) >= 5:
        penalty += 1
        reasons.append(f"bullet 나열 비율 높음 ({len(bullet_lines)}/{len(lines)}줄)")

    return min(penalty, 3), reasons


# ═══════════════════════════════════════════════════════════════
# 규칙 기반 통합 (D 전체)
# ═══════════════════════════════════════════════════════════════

def rule_based_score(
    text: str,
    meta: dict | None = None,
) -> tuple[int, dict]:
    """
    D1(분량) + D2(수치 패턴) + D7(패널티) 기반 D 점수 (0~10) 반환.

    Args:
        text: 평가할 포트폴리오 섹션 텍스트
        meta: portfolio.py chunk()의 meta 딕셔너리 (선택, achievements 연동)

    Returns:
        (score_0_to_10, detail_dict)
    """
    d1, d1_msg = _score_d1_volume(text)
    d2, d2_msg = _score_d2_quant(text, meta)
    penalty, penalty_reasons = _penalty_d7(text)

    raw = d1 * _D_SUB_WEIGHTS["d1_volume"] + d2 * _D_SUB_WEIGHTS["d2_quant"]
    final = max(1, min(10, round(raw) - penalty))

    detail: dict[str, Any] = {
        "final_score": final,
        "d1_volume":   {"score": d1, "msg": d1_msg},
        "d2_quant":    {"score": d2, "msg": d2_msg},
        "d7_penalty":  {"penalty": penalty, "reasons": penalty_reasons},
        "length":      len(text.strip()),
        "has_number":  d2 >= 5,
        "dup_count":   sum(1 for r in penalty_reasons if "중복" in r),
        "vol_status":  d1_msg,
    }
    return final, detail


# ═══════════════════════════════════════════════════════════════
# LLM 평가 (A·B·C·E)
# ═══════════════════════════════════════════════════════════════

def llm_evaluate(text: str) -> _EvalResult:
    """Gemini-3-Flash preview로 A·B·C·E 평가 (구조화 출력)."""
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

def compute_weighted(llm: _EvalResult, d_score: int) -> float:
    """전 항목 0~100 기준 가중 합산.
    - A·B·C·E: LLM이 0~100으로 반환
    - D: 0~10 → ×10 정규화
    """
    d_norm = d_score * 10
    total = (
        llm.A.score * WEIGHTS["A"]
        + llm.B.score * WEIGHTS["B"]
        + llm.C.score * WEIGHTS["C"]
        + d_norm      * WEIGHTS["D"]
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


def print_result(
    llm: _EvalResult,
    d_score: int,
    d_detail: dict,
    weighted: float,
) -> None:
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

    # D 상세 출력
    d = d_detail
    d_norm = d_score * 10
    print(f"[D] {LABELS['D']}  ({int(WEIGHTS['D']*100)}%)  →  {d_norm}/100")
    print(f"    D1 분량     {d['d1_volume']['score']*10:>3}/100  {d['d1_volume']['msg']}")
    print(f"    D2 수치성과 {d['d2_quant']['score']*10:>3}/100  {d['d2_quant']['msg']}")
    if d["d7_penalty"]["penalty"] > 0:
        print(f"    D7 감점     -{d['d7_penalty']['penalty']}점  {', '.join(d['d7_penalty']['reasons'])}")
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
    """
    Args:
        text: 포트폴리오 섹션 텍스트
        meta: portfolio.py chunk()의 meta (선택, 있으면 규칙 정밀도 향상)
    """
    if len(text.strip()) < 50:
        raise ValueError("포트폴리오 내용을 50자 이상 입력해주세요.")

    print("  [1/3] 규칙 기반 분석 중...")
    d_score, d_detail = rule_based_score(text, meta=meta)

    print("  [2/3] Gemini-3-Flash preview평가 중...")
    llm = llm_evaluate(text)

    print("  [3/3] 점수 합산 중...")
    weighted = compute_weighted(llm, d_score)

    print_result(llm, d_score, d_detail, weighted)

    return {
        "llm":      llm.model_dump(),
        "D":        {"score": d_score, "detail": d_detail},
        "weighted": weighted,
    }


# ═══════════════════════════════════════════════════════════════
# 전후 비교
# ═══════════════════════════════════════════════════════════════

def evaluate_comparison(
    original: str,
    improved: str,
    meta: dict | None = None,
) -> dict:
    """원문과 개선안을 각각 평가해 점수 delta를 출력."""
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

    llm_keys = ["A", "B", "C", "E"]
    for k in llm_keys:
        b = before["llm"][k]["score"]
        a = after["llm"][k]["score"]
        print(f"  {k}. {LABELS[k]:<14} {b:>6}  {a:>6}  {a-b:>+6}")

    b_d, a_d = before["D"]["score"], after["D"]["score"]
    print(f"  D. {LABELS['D']:<14} {b_d*10:>6}  {a_d*10:>6}  {(a_d-b_d)*10:>+6}")

    # D 서브 항목 비교
    sub_keys   = ["d1_volume", "d2_quant"]
    sub_labels = {"d1_volume": "  분량", "d2_quant": "  수치성과"}
    for sk in sub_keys:
        bs  = before["D"]["detail"][sk]["score"]
        as_ = after["D"]["detail"][sk]["score"]
        print(f"  {sub_labels[sk]:<16} {bs*10:>6}  {as_*10:>6}  {(as_-bs)*10:>+6}")

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
                "before": before["llm"][k]["score"],
                "after":  after["llm"][k]["score"],
                "delta":  after["llm"][k]["score"] - before["llm"][k]["score"],
            }
            for k in llm_keys
        } | {
            "D": {
                "before": b_d,
                "after":  a_d,
                "delta":  a_d - b_d,
                "before_detail": before["D"]["detail"],
                "after_detail":  after["D"]["detail"],
            },
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
    # 메타 없이 텍스트만으로 평가
    sample_plain = """
    사용자 인증 시스템의 응답 속도가 평균 1.2초로 느려 UX 불만이 발생하고 있었습니다.
    원인을 분석한 결과, 매 요청마다 DB에서 사용자 정보를 조회하는 구조가 문제였습니다.
    Redis 캐싱 도입을 검토했으나, 팀 인프라 환경상 추가 서버 비용이 부담이었습니다.
    대신 JWT 토큰에 필수 정보를 포함해 DB 조회를 최소화하는 방향으로 설계를 변경했습니다.
    그 결과 응답 속도가 평균 0.3초로 75% 개선되었고, 서버 비용 증가 없이 성능을 확보했습니다.
    이 경험을 통해 최적의 기술 선택이 항상 최신 기술이 아니라 상황에 맞는 판단임을 배웠습니다.
    """

    # portfolio.py chunk() 메타 연동 예시
    sample_meta = {
        "role": "백엔드 인증 모듈 단독 설계 및 구현",
        "tech_stack": ["FastAPI", "JWT", "PostgreSQL", "Redis"],
        "contributions": ["JWT 기반 인증 구조 설계", "DB 조회 최소화 로직 구현"],
        "achievements": ["응답속도 75% 개선 (1.2s → 0.3s)", "서버 비용 증가 없음"],
    }

    print("\n[테스트 1] 텍스트만 입력")
    evaluate(sample_plain)

    print("\n[테스트 2] 메타 연동 입력")
    evaluate(sample_plain, meta=sample_meta)