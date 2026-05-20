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

def _bm25_search(query: str, bm25_path: Path, top_k: int) -> list[tuple[str, float]]:
    with open(bm25_path, "rb") as f:
        data = pickle.load(f)
    tokens = _tokenize_ko(query)
    scores = data["bm25"].get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(data["ids"][i], float(scores[i])) for i in top_idx if scores[i] > 0]


# ═══════════════════════════════════════════════════════════════
# 벡터 검색 (pgvector)
# ═══════════════════════════════════════════════════════════════

def _vector_search(query_emb: list[float], table: str, top_k: int) -> list[tuple[str, float]]:
    emb_str = "[" + ",".join(str(x) for x in query_emb) + "]"
    conn = pg8000.connect(**DB_CONFIG)
    cur  = conn.cursor()
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
    parser.add_argument("--query",            type=str, default=None,
                        help="텍스트 쿼리 직접 입력 (자소서 개선)")
    parser.add_argument("--pf-query",         type=str, default=None,
                        help="텍스트 쿼리 직접 입력 (포트폴리오 개선)")
    parser.add_argument("--top-k",            type=int, default=TOP_K_FINAL,
                        help=f"참고 청크 수 (기본: {TOP_K_FINAL})")
    args = parser.parse_args()

    if args.to_cover_letter:
        run_portfolio_to_cover_letter(args.to_cover_letter, top_k=args.top_k)
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
