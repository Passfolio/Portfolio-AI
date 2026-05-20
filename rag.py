"""
rag.py
─────────────────────────────────────────────────────────────────
RAG 검색 + Gemini LLM 응답 파이프라인

목적별 2종:
  1. rag_cover_letter(query)            → 유사 자소서 검색 후 자소서 개선
  2. rag_portfolio_to_cover_letter(question) → 포트폴리오 검색 후 자소서 문항 작성

하이브리드 검색: BM25 + pgvector 코사인 유사도 → RRF 융합
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pg8000
from dotenv import load_dotenv
load_dotenv()

from google import genai as _genai
from google.genai import types as _genai_types
from kiwipiepy import Kiwi
from pydantic import BaseModel
from rank_bm25 import BM25Okapi


# ═══════════════════════════════════════════════════════════════
# 구조화 출력 스키마
# ═══════════════════════════════════════════════════════════════

class _ImprovedResult(BaseModel):
    improved:  str        # 개선된 자소서 본문
    reasoning: str        # 개선 근거 (2~3문장)
    changes:   list[str]  # 주요 변경 사항 목록


class _SectionGap(BaseModel):
    field:       str  # 부족한 항목명 (예: "성과 수치", "기술 스택 버전", "GitHub 링크")
    reason:      str  # 왜 비었는지 — 자소서 원문에 무엇이 없었는지 1문장
    user_action: str  # 지원자가 직접 보완해야 할 구체적 요청 1문장
    example:     str  # 참고 포트폴리오에서 발췌한 표현 예시 (없으면 "")


class _PortfolioSubSections(BaseModel):
    overview:    str  # 개요: 프로젝트 배경·목적·기술스택 (없으면 "")
    development: str  # 개발: 핵심 구현 내용 (없으면 "")
    issue:       str  # 이슈: 주요 문제 및 해결 과정 (없으면 "")
    result:      str  # 성과: 수치·정성 결과 (없으면 "")


class _PortfolioGenResult(BaseModel):
    project:          str
    period:           str
    role:             str
    team:             str
    tech_stack:       list[str]
    sections:         _PortfolioSubSections
    gaps:             list[_SectionGap]  # 자소서 원문 부족으로 채우지 못한 항목들
    image_suggestion: str               # 이 섹션에 넣으면 좋을 이미지 유형 제안 (없으면 "")


# ═══════════════════════════════════════════════════════════════
# 설정
# ═══════════════════════════════════════════════════════════════

TOP_K_VECTOR = 10
TOP_K_BM25   = 10
TOP_K_FINAL  = 5

_LLM_MODEL = "gemini-3.1-pro-preview"  # "gemini-3.0-flash" --- IGNORE ---

OUTPUT_DIR          = Path(__file__).parent / "output"
CL_BM25_PATH        = OUTPUT_DIR / "bm25_cover_letters.pkl"
PORTFOLIO_BM25_PATH = OUTPUT_DIR / "bm25_portfolios.pkl"

DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "database": "postgres",
    "user":     os.getenv("PG_USER", "parkjunwoo"),
    "password": os.getenv("PG_PASSWORD"),
}


# ═══════════════════════════════════════════════════════════════
# 임베딩 (lazy load)
# ═══════════════════════════════════════════════════════════════

_kiwi = Kiwi()


def _tokenize_ko(text: str) -> list[str]:
    keep = {"NNG", "NNP", "NNB", "NR", "NP", "VV", "VA", "SL"}
    return [t.form for t in _kiwi.tokenize(text) if t.tag in keep]


def _embed_query(text: str) -> list[float]:
    project = os.getenv("GCP_PROJECT_ID")
    if not project:
        raise ValueError("GCP_PROJECT_ID 환경변수를 설정하세요.")
    client = _genai.Client(vertexai=True, project=project, location="global")
    resp = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text,
        config=_genai_types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=1024,
        ),
    )
    return resp.embeddings[0].values


# ═══════════════════════════════════════════════════════════════
# BM25 검색
# ═══════════════════════════════════════════════════════════════

def _bm25_search(
    query: str,
    bm25_path: Path,
    top_k: int,
    sub_section_filter: str | None = None,
) -> list[tuple[str, float]]:
    """BM25 검색.

    Args:
        sub_section_filter: 지정하면 해당 sub_section 인덱스만 대상으로 검색.
                            pickle에 'sub_sections' 키가 없으면 무시.
    """
    with open(bm25_path, "rb") as f:
        data = pickle.load(f)
    tokens = _tokenize_ko(query)

    sub_sections = data.get("sub_sections")  # list[str] | None
    if sub_section_filter and sub_sections:
        # 필터에 해당하는 인덱스만 추출
        target_idx = [i for i, s in enumerate(sub_sections) if s == sub_section_filter]
        if target_idx:
            all_scores = data["bm25"].get_scores(tokens)
            filtered = [(i, float(all_scores[i])) for i in target_idx if all_scores[i] > 0]
            filtered.sort(key=lambda x: x[1], reverse=True)
            return [(data["ids"][i], score) for i, score in filtered[:top_k]]
        # 해당 sub_section 데이터 없으면 필터 없이 fallback
    scores = data["bm25"].get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(data["ids"][i], float(scores[i])) for i in top_idx if scores[i] > 0]


# ═══════════════════════════════════════════════════════════════
# 벡터 검색 (pgvector)
# ═══════════════════════════════════════════════════════════════

def _vector_search(
    query_emb: list[float],
    table: str,
    top_k: int,
    sub_section_filter: str | None = None,
) -> list[tuple[str, float]]:
    """pgvector 코사인 유사도 검색.

    Args:
        sub_section_filter: 포트폴리오 테이블 전용. 지정하면 해당 sub_section 행만 검색.
    """
    emb_str = "[" + ",".join(str(x) for x in query_emb) + "]"
    conn = pg8000.connect(**DB_CONFIG)
    cur  = conn.cursor()
    if sub_section_filter:
        cur.execute(
            f"SELECT id, 1 - (embedding <=> %s::vector) AS score "
            f"FROM {table} WHERE sub_section = %s "
            f"ORDER BY embedding <=> %s::vector LIMIT %s",
            (emb_str, sub_section_filter, emb_str, top_k),
        )
    else:
        cur.execute(
            f"SELECT id, 1 - (embedding <=> %s::vector) AS score "
            f"FROM {table} ORDER BY embedding <=> %s::vector LIMIT %s",
            (emb_str, emb_str, top_k),
        )
    rows = [(row[0], float(row[1])) for row in cur.fetchall()]
    cur.close()
    conn.close()
    return rows


# ═══════════════════════════════════════════════════════════════
# RRF 융합
# ═══════════════════════════════════════════════════════════════

def _rrf_fusion(
    bm25_res:   list[tuple[str, float]],
    vector_res: list[tuple[str, float]],
    k: int = 60,
) -> list[str]:
    scores: dict[str, float] = {}
    for rank, (id_, _) in enumerate(bm25_res):
        scores[id_] = scores.get(id_, 0.0) + 1 / (k + rank + 1)
    for rank, (id_, _) in enumerate(vector_res):
        scores[id_] = scores.get(id_, 0.0) + 1 / (k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


# ═══════════════════════════════════════════════════════════════
# DB 청크 조회
# ═══════════════════════════════════════════════════════════════

def _fetch_chunks(ids: list[str], table: str, cols: list[str]) -> list[dict]:
    if not ids:
        return []
    placeholders = ",".join(["%s"] * len(ids))
    col_str = ", ".join(cols)
    conn = pg8000.connect(**DB_CONFIG)
    cur  = conn.cursor()
    cur.execute(f"SELECT {col_str} FROM {table} WHERE id IN ({placeholders})", ids)
    rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    cur.close()
    conn.close()
    id_to_row = {r["id"]: r for r in rows}
    return [id_to_row[id_] for id_ in ids if id_ in id_to_row]


# ═══════════════════════════════════════════════════════════════
# 포트폴리오 생성용 레퍼런스 검색
# ═══════════════════════════════════════════════════════════════

_PF_REF_COLS = [
    "id", "section", "sub_section", "project",
    "period", "role", "team", "tech_stack",
    "contributions", "achievements", "text",
]


def _search_portfolio_refs(cl_chunk: dict, top_k: int) -> list[dict]:
    """자소서 청크와 유사한 포트폴리오 레퍼런스 검색 (단일 검색).

    contributions/achievements/tech_stack 등 구조화 컬럼을 함께 가져와
    _generate_portfolio_section 에서 서브섹션별 예시로 분리 활용.
    """
    query_emb  = _embed_query(cl_chunk["text"])
    bm25_res   = _bm25_search(cl_chunk["text"], PORTFOLIO_BM25_PATH, TOP_K_BM25)
    vector_res = _vector_search(query_emb, "portfolio_chunks", TOP_K_VECTOR)
    fused_ids  = _rrf_fusion(bm25_res, vector_res)[:top_k]
    return _fetch_chunks(fused_ids, "portfolio_chunks", _PF_REF_COLS)


# ═══════════════════════════════════════════════════════════════
# 자소서 평가 기준 (프롬프트 공통 참조)
# ═══════════════════════════════════════════════════════════════

_PORTFOLIO_CRITERIA = """
[포트폴리오 작성 핵심 기준]

▶ 구조
- 프로젝트 목적 / 문제 정의 → 역할/기여 → 판단과 실행 → 성과 → 인사이트 순으로 서술
- 결론보다 과정과 사고방식이 핵심

▶ 내용
- '왜 그 선택을 했는지' 판단 근거를 반드시 명시
- 수치로 드러나는 정량적 성과 우선, 어렵다면 정성적 성과라도 구체적으로
- 본인이 주도적으로 수행한 역할을 명확히 구분
- 경험 → 성장한 점 → 직무 기여 방향 순으로 연결
- 대외활동은 직무 역량 관점에서 의미 있는 경험만 포함

▶ 반드시 피해야 할 감점 요인
- 단순 기술 나열 (무엇을 했다만 나열, 왜·어떻게 없음)
- 팀 성과를 본인 성과처럼 모호하게 서술
- 결과만 있고 과정·판단이 없는 서술
- 추상적 자기 평가 ("열심히 했다", "많이 배웠다")
"""

_EVALUATION_CRITERIA = """
[자소서 작성 핵심 기준]

▶ 구조
- 결론을 가장 앞에 쓰는 두괄식으로 작성
- 직무역량 문항은 60~70%를 전문지식·경험에, 나머지를 직무 적합 사유에 할당

▶ 내용
- 프로젝트·팀·역할 등 상황을 구체적으로 언급
- 수치로 환산된 정량적 성과 우선 (%, ms, 건수, 배수 등), 어렵다면 정성적 성과라도 구체적으로
- 경험 → 성장한 점 → 직무 기여 방향 순으로 연결
- 기업 미션·비전·사업과 본인 가치관을 연결하는 자신만의 스토리 포함
- 각 경험마다 "이것이 직무에 어떻게 활용되는지" 명시

▶ 반드시 피해야 할 감점 요인
- 같은 내용 반복
- 원론적·추상적 표현 (예: "최선을 다하겠습니다", "열정적으로 임하겠습니다")
- 분량 70% 미만
- 질문과 무관한 내용
- 타사 지원서 복사 흔적

▶ AI 생성 문체 특징 — 반드시 탈피
- 주관 없는 중립적 서술
- 짜여진 흐름, 구조적 전형성
- 구체적 근거 없는 주장
- 과도한 반복·과장
- 복잡하고 긴 문장 구조
"""

# ═══════════════════════════════════════════════════════════════
# Gemini 클라이언트
# ═══════════════════════════════════════════════════════════════

def _get_gemini_client() -> _genai.Client:
    project = os.getenv("GCP_PROJECT_ID")
    if not project:
        raise ValueError("GCP_PROJECT_ID 환경변수를 설정하세요.")
    return _genai.Client(vertexai=True, project=project, location="global")


# ═══════════════════════════════════════════════════════════════
# RAG-1: 자소서 수정/생성
# ═══════════════════════════════════════════════════════════════

def rag_cover_letter(
    query: str,
    top_k: int = TOP_K_FINAL,
    char_limit: int | None = None,
    section: str = "",
) -> str:
    """유사 자소서 검색 → Gemini로 자소서 개선안 생성.

    Args:
        query:      사용자가 작성한 자소서 원문
        top_k:      참고할 유사 자소서 수
        char_limit: 문항 글자 수 제한 (있으면 90% 이상 채우도록 지시)
        section:    문항 제목 (직무역량 여부 판별용)
    """
    from evaluators.cover_letter import is_competency_question

    query_emb  = _embed_query(query)
    bm25_res   = _bm25_search(query, CL_BM25_PATH, TOP_K_BM25)
    vector_res = _vector_search(query_emb, "cover_letter_chunks", TOP_K_VECTOR)
    fused_ids  = _rrf_fusion(bm25_res, vector_res)[:top_k]
    chunks     = _fetch_chunks(fused_ids, "cover_letter_chunks", ["id", "sub_section", "category", "text"])

    context_block = "\n\n---\n\n".join(
        f"[예시 {i+1} | {c['category']} — {c['sub_section']}]\n{c['text']}"
        for i, c in enumerate(chunks)
    )

    # D1 분량 목표 동적 생성
    if char_limit:
        target_min = int(char_limit * 0.9)
        volume_rule = f"개선안은 {target_min}자 이상 {char_limit}자 이하로 작성하세요 (제한의 90% 이상 채워야 만점)."
    elif is_competency_question(section):
        volume_rule = "개선안은 600~800자로 작성하세요 (직무역량 문항 기준)."
    else:
        volume_rule = "개선안은 400~600자로 작성하세요."

    prompt = (
        f"다음은 실제 합격자 수준의 자소서 예시들입니다 (구조·표현 참고용):\n\n"
        f"{context_block}\n\n"
        f"{_EVALUATION_CRITERIA}\n\n"
        f"위 예시들과 평가 기준을 참고하여 아래 자소서를 개선해주세요.\n\n"
        f"[절대 준수 사항 — 위반 시 무효]\n"
        f"1. 원문에 없는 프로젝트·경험·수치·기술을 절대 추가하지 마세요.\n"
        f"2. 원문에 명시된 사실(프로젝트명, 역할, 수치, 기술스택 등)만 사용하세요.\n"
        f"3. 수치가 원문에 없으면 임의로 만들지 말고 정성적 표현을 구체화하는 데 집중하세요.\n"
        f"4. 예시 자소서의 내용(경험, 수치 등)을 원문에 옮겨 쓰지 마세요 — 구조와 표현 방식만 참고하세요.\n"
        f"5. 허용 범위: 문장 구조 재배치, 두괄식 전환, 표현 구체화, 불필요한 문장 제거\n"
        f"6. [분량 필수] {volume_rule}\n"
        f"7. [수치 보존] 원문에 있는 숫자·단위(%, ms, 배, 건 등)는 개선안에 반드시 유지하세요.\n\n"
        f"[개선할 자소서]\n{query}\n\n"
        f"다음 JSON 형식으로 응답하세요:\n"
        f"- improved: 개선된 자소서 전문 (원문 사실만 사용)\n"
        f"- reasoning: 개선 근거 (왜 이렇게 바꿨는지 2~3문장)\n"
        f"- changes: 주요 변경 사항 목록 (항목당 한 줄, 추가된 내용 없이 수정만)"
    )

    client = _get_gemini_client()

    _MAX_VOLUME_RETRY = 2
    result = None

    for attempt in range(_MAX_VOLUME_RETRY + 1):
        current_prompt = prompt

        # 재시도 시: 이전 결과의 글자 수 부족을 명시해 강하게 지시
        if attempt > 0 and result is not None:
            short_by = target_min - len(result.improved)
            current_prompt = (
                f"[분량 미달 재작성 요청 — {attempt}회차]\n"
                f"이전 결과는 {len(result.improved)}자로, 목표 {target_min}자에 {short_by}자 부족합니다.\n"
                f"원문 사실 범위 안에서 문장을 더 구체화하고 설명을 보강해 "
                f"{target_min}자 이상 {char_limit}자 이하로 반드시 완성하세요.\n\n"
                f"[보강할 자소서 (이전 결과)]\n{result.improved}\n\n"
                f"다음 JSON 형식으로 응답하세요:\n"
                f"- improved: 보강된 자소서 전문\n"
                f"- reasoning: 보강 근거 (2~3문장)\n"
                f"- changes: 주요 변경 사항 목록"
            )

        resp = client.models.generate_content(
            model=_LLM_MODEL,
            contents=current_prompt,
            config=_genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_ImprovedResult,
            ),
        )
        result = _ImprovedResult.model_validate_json(resp.text)

        # char_limit 없거나 90% 충족하면 완료
        if not char_limit or len(result.improved) >= target_min:
            break

        print(f"  [분량 미달] {len(result.improved)}자 / 목표 {target_min}자 — 재생성 ({attempt + 1}/{_MAX_VOLUME_RETRY})")

    return {
        "improved":  result.improved,
        "reasoning": result.reasoning,
        "changes":   result.changes,
    }


# ═══════════════════════════════════════════════════════════════
# RAG-2: 포트폴리오 PDF → 자소서 생성 파이프라인
# ═══════════════════════════════════════════════════════════════

# 4개 고정 섹션: 포트폴리오에서 직접 도출 가능한 경험 기반 섹션만
_CL_GEN_SECTIONS = [
    {
        "label":      "직무역량",
        "char_limit": 800,
        "question":   "본인의 직무 역량과 이를 발휘한 프로젝트 경험을 구체적으로 서술하세요.",
        "query":      "기술 역량 개발 구현 설계 성능 최적화 시스템 아키텍처 직무",
        "writing_guide": (
            "행동(A) 단락에서 기술을 선택한 이유를 반드시 명시하세요.\n"
            "예) '~보다 ~를 선택한 이유는 ~이었습니다.'\n"
            "결과(R) 단락에는 포트폴리오에 있는 수치(%, ms, 배수 등)를 그대로 인용하세요."
        ),
    },
    {
        "label":      "협업경험",
        "char_limit": 700,
        "question":   "팀 프로젝트에서의 협업 경험과 본인이 기여한 역할을 서술하세요.",
        "query":      "팀 협업 소통 팀원 협력 의사소통 갈등 조율 리더 역할",
        "writing_guide": (
            "행동(A) 단락에서 팀 내 의견 충돌이나 조율 경험을 1가지 이상 구체적으로 서술하세요.\n"
            "본인 기여와 팀원 기여를 명확히 구분하고, 팀원 작업은 '협업했다'로만 언급하세요."
        ),
    },
    {
        "label":      "문제해결경험",
        "char_limit": 800,
        "question":   "개발 과정에서 가장 어려운 문제를 발견하고 해결한 경험을 서술하세요.",
        "query":      "문제 해결 트러블슈팅 장애 원인 개선 디버깅 병목 오류 이슈",
        "writing_guide": (
            "상황(S)에는 문제 증상을, 과제(T)에는 근본 원인을 각각 구분해 서술하세요.\n"
            "행동(A)에서 다른 해결 방법 대신 이 방법을 선택한 이유를 반드시 포함하세요."
        ),
    },
    {
        "label":      "성장경험",
        "char_limit": 700,
        "question":   "도전적인 경험을 통해 성장한 사례와 이를 직무에 어떻게 기여할지 서술하세요.",
        "query":      "성장 배움 도전 실패 극복 인사이트 개선 발전 경험 깨달음",
        "writing_guide": (
            "결과(R) 단락에서 수치 성과보다 '무엇을 새롭게 깨달았는가'를 중심으로 서술하세요.\n"
            "마무리에서 그 깨달음이 지원 직무에서 어떻게 발현될지 1문장으로 구체적으로 연결하세요."
        ),
    },
]

_CL_GEN_SYSTEM = """\
당신은 IT 직군 자소서 작성 전문가입니다. 지원자의 포트폴리오 내용만을 근거로 자소서를 작성합니다.

[원칙]
- 포트폴리오에 없는 사실(경험, 수치, 기술, 결과)을 절대 추가하지 마세요.
- 원문이 '분석했다'면 '분석했다'로만 쓰세요. '해결했다'로 격상하지 마세요.
- 본인이 직접 한 일만 1인칭으로 쓰고, 팀원·타 직군의 작업은 '협업했다' 수준으로만 언급하세요.
"""

_CL_GEN_WRITING_METHOD = """\
[작성 방법 — 아래 순서대로 내용을 구성하세요]

① 첫 문장: 두괄식 결론
   가장 중요한 역량·경험을 1문장으로 요약해 시작하세요.

② 상황(S): 1~2문장
   언제, 어떤 프로젝트에서, 어떤 문제 상황이었는지 구체적으로 쓰세요.
   팀 규모·역할·기술 환경을 포함하세요.

③ 과제(T): 1문장
   "내가 해결해야 했던 핵심 과제는 ~였습니다" 형식으로 명확하게 쓰세요.

④ 행동(A): 2~3문장
   "직접/담당/설계/구현" 등 1인칭 표현으로 수행한 내용을 쓰세요.
   왜 그 방법을 선택했는지 판단 근거를 반드시 포함하세요.

⑤ 결과(R): 1~2문장
   포트폴리오에 수치가 있으면 반드시 인용하세요 (%, ms, 배수, 건수 등).
   수치가 없으면 정성적 성과를 구체적으로 서술하세요. 수치를 임의로 만들지 마세요.

⑥ 마무리: 1~2문장
   이 경험에서 얻은 관점·역량을 쓰고, 지원 직무에서 어떻게 활용할지 연결하세요.

[형식]
- 단락 구분 없이 산문체로 이어서 작성하세요. 개행 문자(\\n)를 절대 사용하지 마세요.
- 전체가 하나의 연속된 문단이어야 합니다.

[금지]
- "열심히", "최선을 다", "성장할 수 있었습니다", "뜻깊은 경험" 등 추상 표현
- 같은 내용을 다른 표현으로 반복
- "이를 통해 ~하였습니다", "이러한 경험을 바탕으로" 등 AI 특유의 전형적 문체
- 총 글자 수 600자 미만 또는 800자 초과
"""


class _CoverLetterGenResult(BaseModel):
    text: str             # 생성된 자소서 답변 (600~800자)
    image_suggestion: str # 이 섹션을 강화할 이미지 유형 제안 (없으면 "")


def _select_portfolio_chunks(
    portfolio_chunks: list[dict],
    query: str,
    top_k: int = 3,
) -> list[dict]:
    """사용자 포트폴리오 청크에서 인-메모리 BM25로 관련 섹션 선택."""
    text_chunks = [
        c for c in portfolio_chunks
        if c.get("sub_section") != "이미지"
    ]
    if not text_chunks:
        return []
    corpus = [_tokenize_ko(c.get("text", "")) for c in text_chunks]
    bm25   = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize_ko(query))
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [text_chunks[i] for i in top_idx if scores[i] > 0]



def generate_cl_section(
    section_def: dict,
    portfolio_chunks: list[dict],
    client: _genai.Client,
    top_k: int = TOP_K_FINAL,
    used_projects: list[str] | None = None,
    img_ctx_by_project: dict[str, str] | None = None,
) -> dict:
    """포트폴리오 청크 + 자소서 예시 DB → 자소서 섹션 1개 생성.

    Args:
        used_projects:       앞 섹션에서 이미 주소재로 사용한 프로젝트명 목록.
        img_ctx_by_project:  프로젝트명 → 이미지 캡션 문자열 매핑.
                             선택된 청크의 프로젝트에 해당하는 캡션만 프롬프트에 삽입.
    """
    pf_chunks    = _select_portfolio_chunks(portfolio_chunks, section_def["query"], top_k=5)
    top_project  = pf_chunks[0].get("project", "") if pf_chunks else ""
    pf_block     = "\n\n---\n\n".join(
        f"[{c.get('section','')} — {c.get('project','')}]\n{c['text']}"
        for c in pf_chunks
    ) if pf_chunks else "포트폴리오 관련 섹션 없음"

    # 선택된 청크의 프로젝트에 해당하는 이미지 캡션만 수집
    img_block = ""
    if img_ctx_by_project:
        relevant_projects = {c.get("project", "") for c in pf_chunks if c.get("project")}
        img_parts = [
            f"[{proj}]\n{img_ctx_by_project[proj]}"
            for proj in relevant_projects
            if proj in img_ctx_by_project
        ]
        if img_parts:
            img_block = (
                "\n\n[포트폴리오 이미지 보조 자료 — 수치·구조 참고용, 내용 복사 금지]\n"
                + "\n\n".join(img_parts)
            )

    # 자소서 예시 검색 (스타일 레퍼런스용)
    query_emb  = _embed_query(section_def["question"])
    bm25_res   = _bm25_search(section_def["question"], CL_BM25_PATH, TOP_K_BM25)
    vector_res = _vector_search(query_emb, "cover_letter_chunks", TOP_K_VECTOR)
    fused_ids  = _rrf_fusion(bm25_res, vector_res)[:top_k]
    ref_chunks = _fetch_chunks(fused_ids, "cover_letter_chunks", ["id", "category", "sub_section", "text"])
    ref_block  = "\n\n---\n\n".join(
        f"[예시 {i+1} | {c.get('category','')}]\n{c['text']}"
        for i, c in enumerate(ref_chunks)
    )

    char_limit = section_def["char_limit"]
    target_min = int(char_limit * 0.9)

    # 앞 섹션 소재 힌트
    diversity_hint = ""
    if used_projects:
        names = ", ".join(f"'{p}'" for p in used_projects)
        diversity_hint = (
            f"\n[소재 다양성] 앞 섹션에서 이미 {names} 경험을 주소재로 사용했습니다. "
            f"포트폴리오에 다른 프로젝트·경험이 있다면 그것을 이 섹션의 주소재로 활용하세요. "
            f"적합한 다른 소재가 없을 경우에는 같은 프로젝트의 다른 측면(협업 방식, 의사결정 과정 등)을 부각하세요.\n"
        )

    prompt = (
        f"[지원자 포트폴리오]\n{pf_block}\n"
        f"{img_block}\n\n"
        f"[자소서 스타일 예시 — 구조·표현만 참고, 내용은 절대 복사 금지]\n{ref_block}\n\n"
        f"{_CL_GEN_WRITING_METHOD}\n\n"
        f"[이 섹션 추가 지침]\n{section_def['writing_guide']}\n"
        f"{diversity_hint}\n"
        f"[글자 수 기준] {target_min}자 이상 {char_limit}자 이하로 작성하세요. "
        f"이 범위를 벗어나면 D 평가 점수가 감점됩니다.\n\n"
        f"[image_suggestion 작성 기준]\n"
        f"이 자소서 섹션의 내용을 포트폴리오에서 시각적으로 뒷받침할 이미지 유형을 1~2문장으로 제안하세요.\n"
        f"예) '시스템 아키텍처 다이어그램: 데이터 파이프라인 전체 흐름을 도식화하면 구현 규모를 직관적으로 전달할 수 있습니다.'\n"
        f"예) '성능 지표 차트: 개선 전후 응답 시간(2875ms→149ms)을 그래프로 표현하면 수치 성과의 임팩트가 배가됩니다.'\n"
        f"적합한 이미지가 없으면 빈 문자열(\"\")을 반환하세요.\n\n"
        f"[자소서 문항]\n{section_def['question']}"
    )

    resp = client.models.generate_content(
        model=_LLM_MODEL,
        contents=prompt,
        config=_genai_types.GenerateContentConfig(
            system_instruction=_CL_GEN_SYSTEM,
            response_mime_type="application/json",
            response_schema=_CoverLetterGenResult,
        ),
    )
    parsed   = _CoverLetterGenResult.model_validate_json(resp.text)
    # 개행 문자 후처리: 산문체 유지
    clean_text = parsed.text.replace("\n\n", " ").replace("\n", " ").strip()
    return {
        "label":            section_def["label"],
        "question":         section_def["question"],
        "text":             clean_text,
        "top_project":      top_project,
        "image_suggestion": parsed.image_suggestion,
    }


def run_portfolio_to_cover_letter(pdf_path: str, top_k: int = TOP_K_FINAL) -> list[dict]:
    """포트폴리오 PDF → 4개 경험 기반 자소서 섹션 자동 생성 + 평가.

    생성 섹션: 직무역량 / 협업경험 / 문제해결경험 / 성장경험
    지원동기·입사포부는 회사 정보 없이 생성 불가하여 제외.

    Args:
        pdf_path: 포트폴리오 PDF 파일 경로
        top_k:    참고할 유사 자소서 수
    Returns:
        [{"label", "question", "text", "eval"}, ...]
    """
    from chunkers.portfolio import chunk
    from evaluators.cover_letter import evaluate

    print(f"\nPDF 변환 중: {pdf_path}")
    portfolio_chunks = chunk(pdf_path)
    text_chunks = [c for c in portfolio_chunks if c.get("sub_section") != "이미지"]
    img_chunks  = [c for c in portfolio_chunks if c.get("sub_section") == "이미지"]
    print(f"청킹 완료: {len(text_chunks)}개 텍스트 + {len(img_chunks)}개 이미지 섹션\n")

    # 프로젝트별 이미지 캡션 맵 구성
    img_ctx_by_project: dict[str, str] = {}
    for ic in img_chunks:
        proj  = ic.get("project", "")
        entry = f"[{ic.get('content_type', 'other')}] {ic['text']}"
        if proj in img_ctx_by_project:
            img_ctx_by_project[proj] += f"\n{entry}"
        else:
            img_ctx_by_project[proj] = entry
    if img_ctx_by_project:
        print(f"이미지 컨텍스트 보유 프로젝트: {list(img_ctx_by_project.keys())}\n")

    client       = _get_gemini_client()
    results      = []
    used_projects: list[str] = []
    sep          = "=" * 60

    for i, sec_def in enumerate(_CL_GEN_SECTIONS):
        print(f"\n{sep}")
        print(f"[{i+1}/{len(_CL_GEN_SECTIONS)}] {sec_def['label']} 생성 중...")

        generated = generate_cl_section(
            sec_def, text_chunks, client, top_k=top_k,
            used_projects=used_projects if used_projects else None,
            img_ctx_by_project=img_ctx_by_project or None,
        )

        if generated.get("top_project"):
            used_projects.append(generated["top_project"])

        print(f"\n{'─'*26} 생성 결과 {'─'*24}")
        print(generated["text"])
        print(f"({len(generated['text'])}자)")

        print(f"\n  평가 중...")
        eval_result = evaluate(generated["text"], char_limit=sec_def["char_limit"],
                               question=sec_def["question"])

        results.append({
            "label":    generated["label"],
            "question": generated["question"],
            "text":     generated["text"],
            "char_count": len(generated["text"]),
            "eval":     {
                "weighted":    eval_result["weighted"],
                "llm":         eval_result["llm"],
                "D":           eval_result["D"],
            },
        })

    stem = Path(pdf_path).stem
    OUTPUT_DIR.mkdir(exist_ok=True)

    out_json = OUTPUT_DIR / f"{stem}_cl_from_portfolio.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    from exporters.cover_letter_pdf import save_pdf
    out_pdf = str(OUTPUT_DIR / f"{stem}_cl_from_portfolio.pdf")
    save_pdf(results, out_pdf)

    print(f"\n{sep}")
    print(f"완료: 총 {len(results)}개 섹션 생성")
    print(f"  JSON → {out_json}")
    print(f"  PDF  → {out_pdf}")
    for r in results:
        from evaluators.cover_letter import grade_label
        print(f"  [{r['label']}] {r['char_count']}자 | {r['eval']['weighted']:.1f}점 {grade_label(r['eval']['weighted'])}")
    return results


# ═══════════════════════════════════════════════════════════════
# RAG-3: 포트폴리오 수정
# ═══════════════════════════════════════════════════════════════

def rag_portfolio(query: str, top_k: int = TOP_K_FINAL, img_context: str = "") -> dict:
    """유사 포트폴리오 검색 → Gemini로 포트폴리오 내용 개선.

    Args:
        query: 개선할 포트폴리오 섹션 원문
        top_k: 참고할 유사 포트폴리오 수
        img_context: 해당 포트폴리오의 이미지 캡션 컨텍스트 (있을 때만 전달)
    """
    query_emb  = _embed_query(query)
    bm25_res   = _bm25_search(query, PORTFOLIO_BM25_PATH, TOP_K_BM25)
    vector_res = _vector_search(query_emb, "portfolio_chunks", TOP_K_VECTOR)
    fused_ids  = _rrf_fusion(bm25_res, vector_res)[:top_k]
    chunks     = _fetch_chunks(fused_ids, "portfolio_chunks", ["id", "section", "sub_section", "project", "text"])

    context_block = "\n\n---\n\n".join(
        f"[예시 {i+1} | {c['section']} — {c.get('project', '')}]\n{c['text']}"
        for i, c in enumerate(chunks)
        if c.get("sub_section") != "이미지"
    )

    img_block = (
        f"\n\n[포트폴리오 이미지 컨텍스트 — 아래 이미지 설명을 참고해 개선 방향을 보완하세요]\n{img_context}"
        if img_context else ""
    )

    prompt = (
        f"다음은 실제 포트폴리오 예시들입니다 (구조·표현 참고용):\n\n"
        f"{context_block}\n\n"
        f"{_PORTFOLIO_CRITERIA}\n\n"
        f"위 예시들과 작성 기준을 참고하여 아래 포트폴리오 내용을 개선해주세요."
        f"{img_block}\n\n"
        f"[절대 준수 사항 — 위반 시 무효]\n"
        f"1. 원문에 없는 프로젝트·경험·수치·기술을 절대 추가하지 마세요.\n"
        f"2. 원문에 명시된 사실(프로젝트명, 역할, 수치, 기술스택 등)만 사용하세요.\n"
        f"3. 수치가 원문에 없으면 임의로 만들지 말고 정성적 표현을 구체화하는 데 집중하세요.\n"
        f"4. 허용 범위: 문장 구조 재배치, 과정·판단 근거 부각, 표현 구체화, 불필요한 문장 제거\n"
        f"5. [수치 보존 필수] 원문에 포함된 숫자·단위(%, ms, 배, 건, GB 등)는 개선안에 반드시 유지하세요. 수치 삭제는 감점 요인입니다.\n"
        f"6. [분량 기준] 개선안은 400~1000자로 작성하세요. 초과 시 핵심만 남기고 압축하세요.\n"
        f"7. [완료·결과 표현 제한] '해결했다', '완료했다', '성공했다' 등의 표현은 원문에 그 결과가 명시된 경우에만 사용하세요. 원문이 '분석했다', '시도했다', '발견했다' 수준이면 동일 수준으로만 서술하세요.\n"
        f"8. [역할 경계 준수] 본인이 직접 수행한 역할만 1인칭으로 서술하세요. 팀원·다른 직군(프론트엔드·디자이너 등)이 수행한 작업을 본인 성과로 표현하지 마세요.\n\n"
        f"[개선할 포트폴리오]\n{query}\n\n"
        f"다음 JSON 형식으로 응답하세요:\n"
        f"- improved: 개선된 포트폴리오 전문 (원문 사실만 사용, 400~1000자)\n"
        f"- reasoning: 개선 근거 (왜 이렇게 바꿨는지 2~3문장)\n"
        f"- changes: 주요 변경 사항 목록 (항목당 한 줄)"
    )

    client = _get_gemini_client()
    resp = client.models.generate_content(
        model=_LLM_MODEL,
        contents=prompt,
        config=_genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_ImprovedResult,
        ),
    )
    result = _ImprovedResult.model_validate_json(resp.text)
    return {
        "improved":  result.improved,
        "reasoning": result.reasoning,
        "changes":   result.changes,
    }


# ═══════════════════════════════════════════════════════════════
# PDF → 청킹 → RAG 전체 파이프라인
# ═══════════════════════════════════════════════════════════════

def run_from_pdf(pdf_path: str, top_k: int = TOP_K_FINAL) -> list[dict]:
    """PDF 자소서 → 청킹 → 각 문항별 RAG 개선안 생성.

    Args:
        pdf_path: 자소서 PDF 파일 경로
        top_k:    참고할 유사 자소서 수
    Returns:
        [{"section", "category", "original", "improved"}, ...]
    """
    from docling.document_converter import DocumentConverter
    from chunkers import cover_letter as cl_chunker
    from evaluators.cover_letter import evaluate_comparison, parse_char_limit

    print(f"\nPDF 변환 중: {pdf_path}")
    converter = DocumentConverter()
    text      = converter.convert(pdf_path).document.export_to_text()
    source    = Path(pdf_path).stem

    print("청킹 중...")
    chunks = cl_chunker.chunk(text, source)
    print(f"청킹 완료: {len(chunks)}개 문항\n")

    results = []
    sep = "=" * 60

    for i, chunk in enumerate(chunks):
        print(f"\n{sep}")
        print(f"[{i+1}/{len(chunks)}] {chunk['section']}  ({chunk['char_count']}자)")
        print(f"카테고리: {chunk['category']}")

        char_limit = parse_char_limit(chunk["section"])
        result = rag_cover_letter(
            chunk["text"], top_k=top_k,
            char_limit=char_limit, section=chunk["section"],
        )

        print(f"\n{'─'*28} 원문 {'─'*28}")
        print(chunk["text"])
        print(f"\n{'─'*26} 개선안 {'─'*26}")
        print(result["improved"])
        print(f"\n{'─'*26} 개선 근거 {'─'*24}")
        print(result["reasoning"])
        print(f"\n{'─'*26} 주요 변경사항 {'─'*21}")
        for j, change in enumerate(result["changes"], 1):
            print(f"  {j}. {change}")

        eval_result = evaluate_comparison(chunk["text"], result["improved"], char_limit, chunk["section"])

        results.append({
            "section":    chunk["section"],
            "category":   chunk["category"],
            "original":   chunk["text"],
            "improved":   result["improved"],
            "reasoning":  result["reasoning"],
            "changes":    result["changes"],
            "eval_before": eval_result["before"]["weighted"],
            "eval_after":  eval_result["after"]["weighted"],
            "eval_delta":  eval_result["delta"],
            "eval_detail": eval_result["per_category"],
        })

    stem = Path(pdf_path).stem
    OUTPUT_DIR.mkdir(exist_ok=True)

    out_json = OUTPUT_DIR / f"{stem}_rag_result.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    from exporters.cover_letter_pdf import save_cl_improvement_pdf
    out_pdf = str(OUTPUT_DIR / f"{stem}_rag_result.pdf")
    save_cl_improvement_pdf(results, out_pdf)

    print(f"\n{sep}")
    print(f"완료: 총 {len(results)}개 문항 개선")
    print(f"  JSON → {out_json}")
    print(f"  PDF  → {out_pdf}")
    return results


def run_portfolio_from_pdf(pdf_path: str, top_k: int = TOP_K_FINAL) -> list[dict]:
    """PDF 포트폴리오 → 청킹 → 각 섹션별 RAG 개선안 생성 + 평가.

    Args:
        pdf_path: 포트폴리오 PDF 파일 경로
        top_k:    참고할 유사 포트폴리오 수
    Returns:
        [{"section", "project", "original", "improved", ...}, ...]
    """
    from chunkers.portfolio import chunk
    from evaluators.portfolio import evaluate_comparison as pf_evaluate_comparison

    source = Path(pdf_path).stem

    print(f"\nPDF 변환 중: {pdf_path}")
    all_chunks = chunk(pdf_path)
    print(f"청킹 완료: {len(all_chunks)}개 섹션\n")

    # 이미지 청크와 텍스트 청크 분리
    img_chunks  = [c for c in all_chunks if c.get("sub_section") == "이미지"]
    text_chunks = [c for c in all_chunks if c.get("sub_section") != "이미지"]

    # 프로젝트별 이미지 컨텍스트 사전 구성
    img_ctx_by_project: dict[str, str] = {}
    for ic in img_chunks:
        proj = ic.get("project", "")
        entry = f"[{ic.get('content_type', '')}] {ic['text']}"
        if proj in img_ctx_by_project:
            img_ctx_by_project[proj] += f"\n{entry}"
        else:
            img_ctx_by_project[proj] = entry

    results = []
    sep = "=" * 60

    for i, c in enumerate(text_chunks):
        section     = c.get("section", "")
        project     = c.get("project", "")
        sub_section = c.get("sub_section", "")
        meta        = c.get("meta") or None
        label_parts = [project or section]
        if sub_section:
            label_parts.append(sub_section)
        label = " > ".join(label_parts)

        print(f"\n{sep}")
        print(f"[{i+1}/{len(text_chunks)}] {label}  ({c.get('char_count', len(c['text']))}자)")

        img_context = img_ctx_by_project.get(project, "")
        result = rag_portfolio(c["text"], top_k=top_k, img_context=img_context)

        print(f"\n{'─'*28} 원문 {'─'*28}")
        print(c["text"])
        print(f"\n{'─'*26} 개선안 {'─'*26}")
        print(result["improved"])
        print(f"\n{'─'*26} 개선 근거 {'─'*24}")
        print(result["reasoning"])
        print(f"\n{'─'*26} 주요 변경사항 {'─'*21}")
        for j, change in enumerate(result["changes"], 1):
            print(f"  {j}. {change}")

        eval_result = pf_evaluate_comparison(c["text"], result["improved"], meta=meta)

        results.append({
            "section":     section,
            "project":     project,
            "sub_section": sub_section,
            "original":    c["text"],
            "improved":    result["improved"],
            "reasoning":   result["reasoning"],
            "changes":     result["changes"],
            "eval_before": eval_result["before"]["weighted"],
            "eval_after":  eval_result["after"]["weighted"],
            "eval_delta":  eval_result["delta"],
            "eval_detail": eval_result["per_category"],
        })

    # 이미지 청크는 개선 없이 원문 그대로 결과에 포함
    for ic in img_chunks:
        results.append({
            "section":      ic.get("section", ""),
            "project":      ic.get("project", ""),
            "sub_section":  "이미지",
            "content_type": ic.get("content_type", ""),
            "image_path":   ic.get("image_path", ""),
            "original":     ic["text"],
            "improved":     ic["text"],
            "reasoning":    "이미지 캡션은 개선 대상에서 제외됩니다.",
            "changes":      [],
            "eval_before":  None,
            "eval_after":   None,
            "eval_delta":   None,
            "eval_detail":  None,
        })

    OUTPUT_DIR.mkdir(exist_ok=True)

    out_json = OUTPUT_DIR / f"{source}_portfolio_rag_result.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    from exporters.portfolio_pdf import save_improvement_pdf
    out_pdf = str(OUTPUT_DIR / f"{source}_portfolio_rag_result.pdf")
    save_improvement_pdf(results, out_pdf)

    print(f"\n{sep}")
    print(f"완료: 총 {len(results)}개 섹션 개선")
    print(f"  JSON → {out_json}")
    print(f"  PDF  → {out_pdf}")
    return results


# ═══════════════════════════════════════════════════════════════
# RAG-5: 자소서 → 포트폴리오 생성
# ═══════════════════════════════════════════════════════════════

_PF_GEN_SYSTEM = """\
당신은 IT 직군 포트폴리오 작성 전문가입니다.
자소서에 기술된 경험만을 근거로 포트폴리오 섹션을 작성합니다.

[절대 원칙]
- 자소서에 없는 사실(수치, 기술, 프로젝트명, 역할)을 절대 추가하지 마세요.
- 각 서브섹션은 자소서 내용에서 추출 가능한 경우에만 작성하고, 없으면 반드시 "" (빈 문자열)로 두세요.
- 포트폴리오 특유의 명사형·개조식 문체로 작성하세요 (자소서 산문체 금지).
- 수치가 있으면 반드시 그대로 인용하세요.

[평가 기준 반영 — 아래 기준으로 채점되므로 반드시 충족]
A. 과정/판단력(35%): 배경→문제인식→선택/판단→실행 흐름이 명확해야 함
   - '왜 그 기술/방법을 선택했는지' 판단 근거를 반드시 명시
   - 대안과 비교한 선택 이유, 트레이드오프 포함 시 고점
B. 역할/기여도(25%): 1인칭 기여가 명확해야 함
   - '내가/담당/주도/직접 설계·구현' 등 본인 행동을 구체적으로 서술
   - 트러블슈팅은 문제발생→원인파악→해결과정 순으로 서술
C. 성과/인사이트(20%): 정량 수치 우선, 배움과 성장 연결
   - %, ms, 배수, 건수 등 수치가 있으면 반드시 포함
   - 수치 없으면 명확한 정성 결과(배포, 채택, 기한 준수 등) 서술
D. 작성품질 감점 요인 — 반드시 회피:
   - '열심히', '최선을 다', '다양한 경험', '많은 것을 배' 등 추상 표현 금지
   - 동사 없는 단순 기술 나열 bullet 금지 (bullet마다 판단 근거나 성과 포함)
   - 중복 표현 금지
E. 직무연관성(10%): 직무 역량과 연결되는 경험임을 명시

[project 필드 작성 원칙]
- 자소서에 명시된 프로젝트 고유 명칭(예: '쇼핑몰 프로젝트', '전기차 커뮤니티', '병원 예약 시스템')을 그대로 사용하세요.
- 여러 프로젝트가 언급된 경우 본문에서 가장 구체적으로 서술된 프로젝트 1개를 선택하세요.
  → 절대로 여러 프로젝트를 묶어 임의의 대표 이름을 만들지 마세요.
- 단, 자소서 원문 자체가 '다수 개인 프로젝트', '여러 프로젝트' 등으로 묶어 서술하고 단일 프로젝트를 특정할 수 없는 경우에만 해당 표현을 그대로 인용하세요.
- 경력 카테고리 문항에서는 학력(학점, 학교), 자격증, 수상 이력, 인턴/아르바이트 이력을 포트폴리오 내용으로 사용하지 마세요. 직접 구현한 프로젝트 경험만 활용하세요.

[gaps 작성 원칙 — 자소서 원문 한계로 채우지 못한 항목을 지원자에게 안내]
- 서브섹션(result, development 등)이 "" 가 된 이유, tech_stack이 버전 없이 나열된 경우,
  수치 없이 정성 표현만 쓴 경우, GitHub·배포 링크가 없는 경우 등을 빠짐없이 기록하세요.
- field: 부족한 항목명 (예: "result 수치", "GitHub 링크", "기술 스택 버전", "기간")
- reason: 자소서 원문에 무엇이 없었는지 1문장
- user_action: 지원자가 직접 보완해야 할 구체적 요청 1문장
- example: 프롬프트에 제공된 포트폴리오 레퍼런스에서 이 gap을 잘 채운 실제 표현을 1~2문장 직접 인용하세요.
  인용 시 출처 프로젝트명을 앞에 표기하세요. (예: "[프로젝트A] 쿼리 50→4개(92%↓), 응답시간 2875→149ms(95%↓)")
  레퍼런스에 적합한 예시가 없으면 "" 반환.
- gaps가 없으면 빈 배열 []을 반환하세요.

[image_suggestion 작성 기준 — 포트폴리오 이미지 추천]
이 섹션의 구체적인 기술·구현 내용을 바탕으로 실제로 만들 수 있는 이미지를 2~4가지 제안하세요.
각 제안은 "이미지 유형: 구체적인 내용 + 왜 설득력이 높아지는지" 형태로 작성하세요.

활용 가능한 이미지 유형 (내용에 맞는 것 선택):
- 시스템 아키텍처 다이어그램: 클라이언트-서버-DB-외부API 연결 흐름을 draw.io·Mermaid로 도식화
  → 구현 규모와 전체 설계 사고를 한눈에 전달
- 코드 스니펫: 본문에 언급된 핵심 로직(예: JWT 필터, RestControllerAdvice, 쿼리 최적화 코드)
  10~20줄을 carbon.sh 또는 IDE 캡처로 이미지화 → 실제 구현 능력 직접 증명
- 성능·개선 전후 차트: 수치가 있다면 개선 전후를 막대/선 그래프로 시각화 → 정량 성과 임팩트 극대화
- ERD / DB 스키마: 설계한 테이블 구조와 관계를 ERDCloud·dbdiagram.io로 도식화 → 데이터 모델링 능력 강조
- API 명세 캡처: Swagger UI 또는 Postman 테스트 결과 스크린샷 → REST API 설계 완성도 증명
- UI 스크린샷: 실제 동작하는 주요 화면(로그인, 핵심 기능 페이지) 캡처 → 프로젝트 완성도 증명
- 시퀀스 다이어그램: JWT 인증 흐름, 결제 프로세스 등 복잡한 요청-응답 흐름 → 로직 이해도 증명
- 플로우차트: 핵심 알고리즘·분기 로직 흐름 → 판단 과정 시각화
- 테스트 커버리지 리포트: JaCoCo 등 테스트 결과 캡처 → 코드 품질과 책임감 증명
- Git 기여 그래프 / 커밋 로그: 실제 작업 이력 캡처 → 개발 기간·기여도 신뢰성 강화

내용에 적합한 이미지가 전혀 없으면 "" 반환.
"""

_PF_GEN_SUBSECTION_GUIDE_OVERVIEW = """\
  - 프로젝트 배경·목적·기간·팀 구성·기술스택을 개조식으로 기술
  - 직무와 연관된 역할·기술임을 드러내는 한 줄 포함
  - 예) "• 가계부+일기 통합 웹앱 / 5인 팀 / 42일 / Spring Boot, MySQL, AWS"
  - 예) "• 팀장 겸 백엔드 API 개발 — 연간·월간 보고서, 엑셀 출력, CI/CD 담당"\
"""

_PF_GEN_SUBSECTION_GUIDE_DEVELOPMENT = """\
  - 본인이 직접 구현한 핵심 기능, 설계 결정, 기술적 접근법
  - '왜 그 방법을 선택했는지' 판단 근거 필수 포함 (대안 대비 이유)
  - 배경→문제인식→판단→실행 흐름으로 서술
  - 예) "• YearlyReportDto 신규 설계 — 기존 월별 메서드 재사용 시 쿼리 50회 발생,
         단일 쿼리+서비스 레이어 루프 방식으로 전환하여 DB 병목 해소"\
"""

_PF_GEN_SUBSECTION_GUIDE_ISSUE = """\
  - 문제 발생 → 원인 파악 → 본인이 수행한 해결 과정 순으로 서술
  - '내가/직접/담당' 등 1인칭 행동 동사 사용
  - 자소서에 트러블슈팅 내용이 없으면 ""
  - 예) "• [문제] API 수정 요청 시 직군 간 원인 파악 지연
         [원인] 프론트-백엔드 네트워크 요청 시각 차이
         [해결] HTTP 요청 직접 확인 후 원인 근거 팀 공유 → 이슈 해소"\
"""

_PF_GEN_SUBSECTION_GUIDE_RESULT = """\
  - 수치 기반 성과 최우선 (%, ms, 배수, 건수 — 자소서 원문 그대로 인용)
  - 정성 성과(배포 완료, 기한 준수 등)도 포함
  - 이 경험에서 얻은 인사이트·배움 1줄 포함
  - 예) "• 쿼리 50→4개(92%↓), 응답시간 2875→149ms(95%↓)
         • 의존성 최소화 설계 → 테스트 코드 수정 없이 성능 향상 가능함을 확인"\
"""


def _generate_portfolio_section(
    cl_chunk: dict,
    refs:     list[dict],  # _search_portfolio_refs() 결과 (구조화 컬럼 포함)
    client:   _genai.Client,
) -> _PortfolioGenResult:
    """자소서 청크 1개 + 포트폴리오 레퍼런스 → 포트폴리오 섹션 생성.

    refs 각 항목의 contributions/achievements/tech_stack 등 구조화 컬럼을
    서브섹션별 예시로 분리해 프롬프트에 삽입 → RAG 실질 활용.
    """

    def _sub_example(key: str) -> str:
        items = refs[:3]
        if not items:
            return "  (레퍼런스 없음)"
        lines = []
        for c in items:
            proj = c.get("project", "")
            if key == "overview":
                parts = []
                if c.get("period"):     parts.append(f"기간: {c['period']}")
                if c.get("role"):       parts.append(f"역할: {c['role']}")
                if c.get("team"):       parts.append(f"팀: {c['team']}")
                ts = c.get("tech_stack") or []
                if ts: parts.append(f"기술: {', '.join(ts[:5])}")
                content = " / ".join(parts) if parts else c["text"][:150]
            elif key == "development":
                contribs = c.get("contributions") or []
                content = ("\n  ".join(f"• {x}" for x in contribs[:4])
                           if contribs else c["text"][:300])
            elif key == "result":
                achs = c.get("achievements") or []
                content = ("\n  ".join(f"• {x}" for x in achs[:4])
                           if achs else c["text"][-200:])
            else:  # issue — 전체 서술에 문제해결 내용이 포함됨
                content = c["text"][:400]
            lines.append(f"  [{proj}]\n  {content}")
        return "\n\n".join(lines)

    kp_block  = "\n".join(f"  - {kp}" for kp in cl_chunk.get("key_points", []))
    ach_block = "\n".join(f"  - {a}"  for a in cl_chunk.get("achievements", []))
    kw_block  = ", ".join(cl_chunk.get("keywords", []))

    prompt = (
        f"[자소서 원문 — 이 내용만 근거로 사용]\n"
        f"카테고리: {cl_chunk.get('category', '')}\n"
        f"섹션 제목: {cl_chunk.get('section', '')}\n\n"
        f"{cl_chunk['text']}\n\n"
        f"[추출된 메타 정보]\n"
        f"핵심 포인트(STAR):\n{kp_block}\n"
        f"성과:\n{ach_block}\n"
        f"키워드: {kw_block}\n\n"
        f"[서브섹션별 작성 기준 + 실제 포트폴리오 표현 예시]\n"
        f"아래 각 서브섹션을 작성할 때 '실제 예시'의 표현 방식·문체·구조를 참고하세요.\n"
        f"단, 예시의 내용(프로젝트명·수치·기술)은 절대 복사하지 말고 자소서 원문 사실만 사용하세요.\n\n"
        f"── overview (개요) — E. 직무연관성 ──\n"
        f"{_PF_GEN_SUBSECTION_GUIDE_OVERVIEW}\n"
        f"실제 예시:\n{_sub_example('overview')}\n\n"
        f"── development (개발) — A. 과정/판단력 ──\n"
        f"{_PF_GEN_SUBSECTION_GUIDE_DEVELOPMENT}\n"
        f"실제 예시:\n{_sub_example('development')}\n\n"
        f"── issue (이슈 및 해결) — B. 역할/기여도 ──\n"
        f"{_PF_GEN_SUBSECTION_GUIDE_ISSUE}\n"
        f"실제 예시:\n{_sub_example('issue')}\n\n"
        f"── result (성과) — C. 성과/인사이트 ──\n"
        f"{_PF_GEN_SUBSECTION_GUIDE_RESULT}\n"
        f"실제 예시:\n{_sub_example('result')}\n\n"
        f"gaps의 example 필드: 위 실제 예시 중 해당 gap을 잘 보완한 표현을 직접 인용하세요.\n\n"
        f"위 자소서 내용을 바탕으로 포트폴리오 섹션을 작성하세요.\n"
        f"자소서에 언급되지 않은 내용은 절대 추가하지 말고 해당 서브섹션을 \"\"로 두세요."
    )

    resp = client.models.generate_content(
        model=_LLM_MODEL,
        contents=prompt,
        config=_genai_types.GenerateContentConfig(
            system_instruction=_PF_GEN_SYSTEM,
            response_mime_type="application/json",
            response_schema=_PortfolioGenResult,
        ),
    )
    return _PortfolioGenResult.model_validate_json(resp.text)


def run_cover_letter_to_portfolio(pdf_path: str, top_k: int = TOP_K_FINAL) -> list[dict]:
    """자소서 PDF → 청킹 → 각 문항별 포트폴리오 섹션 생성."""
    from docling.document_converter import DocumentConverter
    from chunkers import cover_letter as cl_chunker

    print(f"\nPDF 변환 중: {pdf_path}")
    converter = DocumentConverter()
    text      = converter.convert(pdf_path).document.export_to_text()
    source    = Path(pdf_path).stem

    print("청킹 중...")
    chunks = cl_chunker.chunk(text, source)
    print(f"청킹 완료: {len(chunks)}개 문항")

    # 프로젝트 관련 카테고리만 포트폴리오 생성 대상으로 필터
    # 경력 카테고리는 프로젝트 관련 키워드가 있을 때만 포함 (학벌/자격증/수상/인턴 제외)
    _PROJECT_CATEGORIES  = {"직무역량", "문제해결경험", "프로젝트경험"}
    _CAREER_PROJECT_KW   = {"프로젝트", "구현", "개발", "서비스", "시스템", "앱", "웹", "API"}

    def _is_target(c: dict) -> bool:
        cat = c.get("category", "")
        if cat in _PROJECT_CATEGORIES:
            return True
        if cat == "경력":
            text = c.get("text", "")
            return any(kw in text for kw in _CAREER_PROJECT_KW)
        return False

    chunks = [c for c in chunks if _is_target(c)]
    print(f"필터 후 대상: {len(chunks)}개 문항 (직무역량·문제해결경험·프로젝트경험·경력 중 프로젝트 포함)\n")

    client  = _get_gemini_client()
    results = []
    sep     = "=" * 60

    for i, chunk in enumerate(chunks):
        print(f"\n{sep}")
        print(f"[{i+1}/{len(chunks)}] {chunk['section']}  ({chunk['char_count']}자)")
        print(f"카테고리: {chunk['category']}")

        # 단일 하이브리드 검색 → contributions/achievements 등 구조화 컬럼 포함
        refs = _search_portfolio_refs(chunk, top_k=top_k)
        ref_projects = [r.get("project", "?") for r in refs if r.get("project")]
        print(f"  레퍼런스 {len(refs)}개: {', '.join(ref_projects[:5])}")

        gen = _generate_portfolio_section(chunk, refs, client)
        print(f"  → 프로젝트: {gen.project}")
        for key, label in [("overview","개요"),("development","개발"),("issue","이슈"),("result","성과")]:
            print(f"     {label}: {'있음' if getattr(gen.sections, key) else '없음'}")

        if gen.gaps:
            print(f"  ⚠ 개선 필요 항목 ({len(gen.gaps)}개):")
            for g in gen.gaps:
                print(f"     [{g.field}] {g.reason}")
                print(f"       → 보완 요청: {g.user_action}")

        # 생성된 포트폴리오 평가
        full_text = "\n\n".join(filter(None, [
            gen.sections.overview,
            gen.sections.development,
            gen.sections.issue,
            gen.sections.result,
        ]))
        eval_result = None
        if len(full_text.strip()) >= 50:
            from evaluators.portfolio import evaluate as pf_evaluate, grade_label as pf_grade_label
            print(f"  평가 중...")
            try:
                eval_result = pf_evaluate(full_text, meta={
                    "period":     gen.period,
                    "role":       gen.role,
                    "team":       gen.team,
                    "tech_stack": gen.tech_stack,
                })
                print(f"  최종 점수: {eval_result['weighted']:.1f}점  {pf_grade_label(eval_result['weighted'])}")
            except Exception as e:
                print(f"  평가 실패: {e}")

        results.append({
            "section":          chunk["section"],
            "category":         chunk["category"],
            "project":          gen.project,
            "period":           gen.period,
            "role":             gen.role,
            "team":             gen.team,
            "tech_stack":       gen.tech_stack,
            "overview":         gen.sections.overview,
            "development":      gen.sections.development,
            "issue":            gen.sections.issue,
            "result":           gen.sections.result,
            "image_suggestion": gen.image_suggestion,
            "gaps": [
                {
                    "field":       g.field,
                    "reason":      g.reason,
                    "user_action": g.user_action,
                    "example":     g.example,
                }
                for g in gen.gaps
            ],
            "eval": eval_result,
        })

    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = Path(pdf_path).stem

    out_json = OUTPUT_DIR / f"{stem}_cl_to_portfolio.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    from exporters.portfolio_gen_pdf import save_generated_portfolio_pdf
    out_pdf = str(OUTPUT_DIR / f"{stem}_cl_to_portfolio.pdf")
    save_generated_portfolio_pdf(results, out_pdf)

    # 전체 gaps 요약 출력
    all_gaps = [(r["section"], r["project"], g) for r in results for g in r.get("gaps", [])]
    print(f"\n{sep}")
    print(f"완료: 총 {len(results)}개 섹션 생성")
    print(f"  JSON → {out_json}")
    print(f"  PDF  → {out_pdf}")

    if all_gaps:
        print(f"\n{'─'*28} 자소서 보완 요청 사항 ({len(all_gaps)}건) {'─'*10}")
        for section, project, g in all_gaps:
            header = f"[{section}]" + (f" {project}" if project else "")
            print(f"\n  {header}")
            print(f"    항목: {g['field']}")
            print(f"    이유: {g['reason']}")
            print(f"    요청: {g['user_action']}")
    else:
        print("\n  ✓ 모든 섹션이 자소서 원문으로 완성되었습니다.")

    return results


# ═══════════════════════════════════════════════════════════════
# CLI (테스트용)
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG 테스트")
    parser.add_argument("--pdf",              type=str, default=None,
                        help="자소서 PDF 파일 경로")
    parser.add_argument("--portfolio",        type=str, default=None,
                        help="포트폴리오 PDF 파일 경로 (개선)")
    parser.add_argument("--to-cover-letter",  type=str, default=None,
                        metavar="PORTFOLIO_PDF",
                        help="포트폴리오 PDF → 자소서 4개 섹션 자동 생성")
    parser.add_argument("--cl-to-portfolio", type=str, default=None,
                        metavar="CL_PDF",
                        help="자소서 PDF → 포트폴리오 섹션 자동 생성")
    parser.add_argument("--query",            type=str, default=None,
                        help="텍스트 쿼리 직접 입력 (자소서 개선)")
    parser.add_argument("--pf-query",         type=str, default=None,
                        help="텍스트 쿼리 직접 입력 (포트폴리오 개선)")
    parser.add_argument("--top-k",            type=int, default=TOP_K_FINAL,
                        help=f"참고 청크 수 (기본: {TOP_K_FINAL})")
    args = parser.parse_args()

    if args.to_cover_letter:
        run_portfolio_to_cover_letter(args.to_cover_letter, top_k=args.top_k)
    elif args.cl_to_portfolio:
        run_cover_letter_to_portfolio(args.cl_to_portfolio, top_k=args.top_k)
    elif args.pdf:
        run_from_pdf(args.pdf, top_k=args.top_k)
    elif args.portfolio:
        run_portfolio_from_pdf(args.portfolio, top_k=args.top_k)
    elif args.query:
        sep = "=" * 60
        result = rag_cover_letter(args.query, top_k=args.top_k)
        print(f"\n{'─'*28} 원문 {'─'*28}")
        print(args.query)
        print(f"\n{'─'*26} 개선안 {'─'*26}")
        print(result["improved"])
        print(f"\n{'─'*26} 개선 근거 {'─'*24}")
        print(result["reasoning"])
        print(f"\n{'─'*26} 주요 변경사항 {'─'*21}")
        for j, change in enumerate(result["changes"], 1):
            print(f"  {j}. {change}")
    elif args.pf_query:
        result = rag_portfolio(args.pf_query, top_k=args.top_k)
        print(f"\n{'─'*28} 원문 {'─'*28}")
        print(args.pf_query)
        print(f"\n{'─'*26} 개선안 {'─'*26}")
        print(result["improved"])
        print(f"\n{'─'*26} 개선 근거 {'─'*24}")
        print(result["reasoning"])
        print(f"\n{'─'*26} 주요 변경사항 {'─'*21}")
        for j, change in enumerate(result["changes"], 1):
            print(f"  {j}. {change}")
    else:
        parser.error("--pdf / --portfolio / --query / --pf-query 중 하나를 입력하세요.")