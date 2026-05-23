"""rag.py 공유 내부 유틸리티 — cover_letter / portfolio 서비스에서 공통 사용."""
from __future__ import annotations

import asyncio
import pickle
import threading
import time
from pathlib import Path

import asyncpg
import numpy as np
from google import genai as _genai
from google.genai import types as _genai_types
from kiwipiepy import Kiwi
from openai import OpenAI as _OpenAI
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from app.core.config import get_settings

# ───────────────────────────────────────────────────────────────
# 상수
# ───────────────────────────────────────────────────────────────

TOP_K_VECTOR = 10
TOP_K_BM25   = 10
TOP_K_FINAL  = 5

_LLM_MODEL = "gemini-3-flash-preview"

OUTPUT_DIR          = Path(__file__).parent.parent.parent / "output"
CL_BM25_PATH        = OUTPUT_DIR / "bm25_cover_letters.pkl"
PORTFOLIO_BM25_PATH = OUTPUT_DIR / "bm25_portfolios.pkl"

# ───────────────────────────────────────────────────────────────
# 구조화 출력 Pydantic 스키마 (rag.py 원본)
# ───────────────────────────────────────────────────────────────

class _ImprovedResult(BaseModel):
    improved:  str
    reasoning: str
    changes:   list[str]


class _SectionGap(BaseModel):
    field:       str
    reason:      str
    user_action: str
    example:     str


class _PortfolioSubSections(BaseModel):
    overview:    str
    development: str
    issue:       str
    result:      str


class _PortfolioGenResult(BaseModel):
    project:          str
    period:           str
    role:             str
    team:             str
    tech_stack:       list[str]
    sections:         _PortfolioSubSections
    gaps:             list[_SectionGap]
    image_suggestion: str


class _CoverLetterGenResult(BaseModel):
    text:             str
    image_suggestion: str


# ───────────────────────────────────────────────────────────────
# 평가 기준 텍스트 상수
# ───────────────────────────────────────────────────────────────

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

# ───────────────────────────────────────────────────────────────
# 포트폴리오 레퍼런스 컬럼
# ───────────────────────────────────────────────────────────────

_PF_REF_COLS = [
    "id", "section", "sub_section", "project",
    "period", "role", "team", "tech_stack",
    "contributions", "achievements", "text",
]

# ───────────────────────────────────────────────────────────────
# Kiwi 형태소 분석기 (모듈 로드 시 1회 초기화)
# ───────────────────────────────────────────────────────────────

_kiwi = Kiwi()


def _tokenize_ko(text: str) -> list[str]:
    keep = {"NNG", "NNP", "NNB", "NR", "NP", "VV", "VA", "SL"}
    return [t.form for t in _kiwi.tokenize(text) if t.tag in keep]


# ───────────────────────────────────────────────────────────────
# Gemini 클라이언트 싱글톤
# ───────────────────────────────────────────────────────────────

_gemini_client: _genai.Client | None = None
_gemini_lock = threading.Lock()


def _get_gemini_client() -> _genai.Client:
    global _gemini_client
    if _gemini_client is None:
        with _gemini_lock:
            if _gemini_client is None:
                project = get_settings().gcp_project_id
                if not project:
                    raise ValueError("GCP_PROJECT_ID 환경변수를 설정하세요.")
                _gemini_client = _genai.Client(
                    vertexai=True,
                    project=project,
                    location="global",
                )
    return _gemini_client


# ───────────────────────────────────────────────────────────────
# BM25 메모리 캐시 (최초 로드 후 재사용)
# ───────────────────────────────────────────────────────────────

_bm25_cache: dict[str, dict] = {}
_bm25_lock = threading.Lock()


def _load_bm25(path: Path) -> dict:
    key = str(path)
    if key not in _bm25_cache:
        with _bm25_lock:
            if key not in _bm25_cache:
                with open(path, "rb") as f:
                    _bm25_cache[key] = pickle.load(f)
    return _bm25_cache[key]


# ───────────────────────────────────────────────────────────────
# asyncpg 커넥션 풀
# ───────────────────────────────────────────────────────────────

_db_pool: asyncpg.Pool | None = None


async def init_db_pool() -> None:
    global _db_pool
    _db_pool = await asyncpg.create_pool(
        get_settings().db_dsn,
        min_size=2,
        max_size=10,
        command_timeout=30,
    )


async def close_db_pool() -> None:
    global _db_pool
    if _db_pool:
        await _db_pool.close()
        _db_pool = None


def _get_db_pool() -> asyncpg.Pool:
    if _db_pool is None:
        raise RuntimeError("DB 풀이 초기화되지 않았습니다.")
    return _db_pool


# ───────────────────────────────────────────────────────────────
# OpenAI 클라이언트 싱글톤 (임베딩 전용)
# ───────────────────────────────────────────────────────────────

_openai_client: _OpenAI | None = None
_openai_lock = threading.Lock()


def _get_openai_client() -> _OpenAI:
    global _openai_client
    if _openai_client is None:
        with _openai_lock:
            if _openai_client is None:
                api_key = get_settings().openai_api_key
                if not api_key:
                    raise ValueError("OPENAI_API_KEY 환경변수를 설정하세요.")
                _openai_client = _OpenAI(api_key=api_key)
    return _openai_client


# ───────────────────────────────────────────────────────────────
# 임베딩 (async — 동기 HTTP 호출을 스레드풀에서 실행)
# model="openai"  → text-embedding-3-small (cover_letter_chunks)
# model="gemini"  → gemini-embedding-2     (portfolio_chunks)
# ───────────────────────────────────────────────────────────────

async def _embed_query(text: str, model: str = "openai") -> list[float]:
    if model == "gemini":
        client = _get_gemini_client()
        resp = await asyncio.to_thread(
            client.models.embed_content,
            model="gemini-embedding-2",
            contents=text,
            config=_genai_types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=1024,
            ),
        )
        return resp.embeddings[0].values
    client = _get_openai_client()
    resp = await asyncio.to_thread(
        client.embeddings.create,
        model="text-embedding-3-small",
        input=text,
        dimensions=1024,
    )
    return resp.data[0].embedding


# ───────────────────────────────────────────────────────────────
# BM25 검색 (동기 — 순수 연산, DB 없음)
# ───────────────────────────────────────────────────────────────

def _bm25_search(
    query: str,
    bm25_path: Path,
    top_k: int,
    sub_section_filter: str | None = None,
) -> list[tuple[str, float]]:
    data = _load_bm25(bm25_path)
    tokens = _tokenize_ko(query)

    sub_sections = data.get("sub_sections")
    if sub_section_filter and sub_sections:
        target_idx = [i for i, s in enumerate(sub_sections) if s == sub_section_filter]
        if target_idx:
            all_scores = data["bm25"].get_scores(tokens)
            filtered = [(i, float(all_scores[i])) for i in target_idx if all_scores[i] > 0]
            filtered.sort(key=lambda x: x[1], reverse=True)
            return [(data["ids"][i], score) for i, score in filtered[:top_k]]
    scores  = data["bm25"].get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(data["ids"][i], float(scores[i])) for i in top_idx if scores[i] > 0]


# ───────────────────────────────────────────────────────────────
# 벡터 검색 (async — asyncpg)
# ───────────────────────────────────────────────────────────────

async def _vector_search(
    query_emb: list[float],
    table: str,
    top_k: int,
    sub_section_filter: str | None = None,
) -> list[tuple[str, float]]:
    emb_str = "[" + ",".join(str(x) for x in query_emb) + "]"
    pool = _get_db_pool()
    async with pool.acquire() as conn:
        if sub_section_filter:
            rows = await conn.fetch(
                f"SELECT id, 1 - (embedding <=> $1::vector) AS score "
                f"FROM {table} WHERE sub_section = $2 "
                f"ORDER BY embedding <=> $1::vector LIMIT $3",
                emb_str, sub_section_filter, top_k,
            )
        else:
            rows = await conn.fetch(
                f"SELECT id, 1 - (embedding <=> $1::vector) AS score "
                f"FROM {table} ORDER BY embedding <=> $1::vector LIMIT $2",
                emb_str, top_k,
            )
    return [(row["id"], float(row["score"])) for row in rows]


# ───────────────────────────────────────────────────────────────
# RRF 융합
# ───────────────────────────────────────────────────────────────

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


# ───────────────────────────────────────────────────────────────
# DB 청크 조회 (async — asyncpg)
# ───────────────────────────────────────────────────────────────

async def _fetch_chunks(ids: list[str], table: str, cols: list[str]) -> list[dict]:
    if not ids:
        return []
    placeholders = ",".join(f"${i + 1}" for i in range(len(ids)))
    col_str = ", ".join(cols)
    pool = _get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"SELECT {col_str} FROM {table} WHERE id IN ({placeholders})",
            *ids,
        )
    id_to_row = {row["id"]: dict(row) for row in rows}
    return [id_to_row[id_] for id_ in ids if id_ in id_to_row]


# ───────────────────────────────────────────────────────────────
# 포트폴리오 레퍼런스 검색
# ───────────────────────────────────────────────────────────────

async def _search_portfolio_refs(cl_chunk: dict, top_k: int) -> list[dict]:
    query_emb  = await _embed_query(cl_chunk["text"], model="gemini")
    bm25_res   = _bm25_search(cl_chunk["text"], PORTFOLIO_BM25_PATH, TOP_K_BM25)
    vector_res = await _vector_search(query_emb, "portfolio_chunks", TOP_K_VECTOR)
    fused_ids  = _rrf_fusion(bm25_res, vector_res)[:top_k]
    return await _fetch_chunks(fused_ids, "portfolio_chunks", _PF_REF_COLS)


# ───────────────────────────────────────────────────────────────
# LLM 호출 래퍼 (async — 동기 SDK를 스레드풀에서 실행, rate limit 시 비동기 대기)
# ───────────────────────────────────────────────────────────────

async def _generate_with_retry(
    client: _genai.Client,
    model: str,
    contents,
    config,
    max_attempts: int = 3,
):
    for attempt in range(max_attempts):
        try:
            return await asyncio.to_thread(
                client.models.generate_content,
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            err = str(e)
            if attempt < max_attempts - 1 and ("429" in err or "quota" in err.lower()):
                wait = 30 * (attempt + 1)
                print(f"  [Gemini rate limit] {wait}초 대기 후 재시도 ({attempt + 1}/{max_attempts - 1})...")
                await asyncio.sleep(wait)
            elif attempt < max_attempts - 1 and ("503" in err or "500" in err):
                wait = 10 * (attempt + 1)
                print(f"  [Gemini 서버 오류] {wait}초 대기 후 재시도 ({attempt + 1}/{max_attempts - 1})...")
                await asyncio.sleep(wait)
            else:
                raise


# ───────────────────────────────────────────────────────────────
# 자소서 → 포트폴리오 생성 시스템 프롬프트 & 서브섹션 가이드
# ───────────────────────────────────────────────────────────────

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
- 자소서에 명시된 프로젝트 고유 명칭을 그대로 사용하세요.
- 여러 프로젝트가 언급된 경우 본문에서 가장 구체적으로 서술된 프로젝트 1개를 선택하세요.
- 단일 프로젝트를 특정할 수 없는 경우에만 해당 표현을 그대로 인용하세요.

[gaps 작성 원칙]
- field: 부족한 항목명
- reason: 자소서 원문에 무엇이 없었는지 1문장
- user_action: 지원자가 직접 보완해야 할 구체적 요청 1문장
- example: 레퍼런스에서 이 gap을 잘 채운 표현 1~2문장 직접 인용 (없으면 "")
- gaps가 없으면 빈 배열 []을 반환하세요.
"""

_PF_GEN_SUBSECTION_GUIDE_OVERVIEW = """\
  - 프로젝트 배경·목적·기간·팀 구성·기술스택을 개조식으로 기술
  - 직무와 연관된 역할·기술임을 드러내는 한 줄 포함\"""

_PF_GEN_SUBSECTION_GUIDE_DEVELOPMENT = """\
  - 본인이 직접 구현한 핵심 기능, 설계 결정, 기술적 접근법
  - '왜 그 방법을 선택했는지' 판단 근거 필수 포함 (대안 대비 이유)
  - 배경→문제인식→판단→실행 흐름으로 서술\"""

_PF_GEN_SUBSECTION_GUIDE_ISSUE = """\
  - 문제 발생 → 원인 파악 → 본인이 수행한 해결 과정 순으로 서술
  - '내가/직접/담당' 등 1인칭 행동 동사 사용
  - 자소서에 트러블슈팅 내용이 없으면 ""\"""

_PF_GEN_SUBSECTION_GUIDE_RESULT = """\
  - 수치 기반 성과 최우선 (%, ms, 배수, 건수 — 자소서 원문 그대로 인용)
  - 정성 성과(배포 완료, 기한 준수 등)도 포함
  - 이 경험에서 얻은 인사이트·배움 1줄 포함\"""


async def _generate_portfolio_section(
    cl_chunk: dict,
    refs: list[dict],
    client: _genai.Client,
) -> _PortfolioGenResult:
    """자소서 청크 1개 + 포트폴리오 레퍼런스 → 포트폴리오 섹션 생성."""

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
            else:
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
        f"단, 예시의 내용(프로젝트명·수치·기술)은 절대 복사하지 말고 자소서 원문 사실만 사용하세요.\n\n"
        f"── overview (개요) ──\n"
        f"{_PF_GEN_SUBSECTION_GUIDE_OVERVIEW}\n"
        f"실제 예시:\n{_sub_example('overview')}\n\n"
        f"── development (개발) ──\n"
        f"{_PF_GEN_SUBSECTION_GUIDE_DEVELOPMENT}\n"
        f"실제 예시:\n{_sub_example('development')}\n\n"
        f"── issue (이슈 및 해결) ──\n"
        f"{_PF_GEN_SUBSECTION_GUIDE_ISSUE}\n"
        f"실제 예시:\n{_sub_example('issue')}\n\n"
        f"── result (성과) ──\n"
        f"{_PF_GEN_SUBSECTION_GUIDE_RESULT}\n"
        f"실제 예시:\n{_sub_example('result')}\n\n"
        f"위 자소서 내용을 바탕으로 포트폴리오 섹션을 작성하세요.\n"
        f"자소서에 언급되지 않은 내용은 절대 추가하지 말고 해당 서브섹션을 \"\"로 두세요."
    )

    resp = await _generate_with_retry(
        client,
        model=_LLM_MODEL,
        contents=prompt,
        config=_genai_types.GenerateContentConfig(
            system_instruction=_PF_GEN_SYSTEM,
            response_mime_type="application/json",
            response_schema=_PortfolioGenResult,
        ),
    )
    return _PortfolioGenResult.model_validate_json(resp.text)
