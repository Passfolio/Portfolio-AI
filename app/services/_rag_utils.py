"""rag.py 공유 내부 유틸리티 — cover_letter / portfolio 서비스에서 공통 사용."""
from __future__ import annotations

import os
import pickle
import re
import threading
import time
from pathlib import Path

import numpy as np
import pg8000
from google import genai as _genai
from google.genai import types as _genai_types
from kiwipiepy import Kiwi
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from app.core.config import get_settings
from app.core.metrics import track_metrics

# ───────────────────────────────────────────────────────────────
# 상수
# ───────────────────────────────────────────────────────────────

TOP_K_VECTOR = 10
TOP_K_BM25   = 10
TOP_K_FINAL  = 5

_LLM_MODEL = "gemini-3.1-flash-lite"

OUTPUT_DIR          = Path(__file__).parent.parent.parent / "output"
CL_BM25_PATH        = OUTPUT_DIR / "bm25_cover_letters.pkl"
PORTFOLIO_BM25_PATH = OUTPUT_DIR / "bm25_portfolios.pkl"

# BM25 인덱스 모듈 레벨 캐시 (첫 로드 후 재사용)
_bm25_cache: dict = {}

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
# 자소서 job/career 필터 유틸
# ───────────────────────────────────────────────────────────────

def map_career_input(years_str: str) -> str:
    """'n년' 형식 입력 → DB career ID 값으로 변환.
    0년 → 신입 / 1~3년 → 주니어 / 4년 이상 → 미드레벨 (5년 이상 시니어 취급, 데이터 없어 미드레벨 폴백).
    """
    m = re.match(r"(\d+)", years_str.strip())
    if not m:
        return ""
    n = int(m.group(1))
    if n == 0:
        return "신입"
    elif n <= 3:
        return "주니어(1~3년)"
    else:
        return "미드레벨(3~5년)"


def _cl_id_matches(id_: str, job: str | None, career: str | None) -> bool:
    """자소서 청크 ID에서 job/career 일치 여부 확인. mock_{company}_{job}_{career}_{hash}_{seq}"""
    parts = id_.split("_")
    if len(parts) < 6:
        return True
    if job and parts[2] != job:
        return False
    if career and parts[3] != career:
        return False
    return True


# ───────────────────────────────────────────────────────────────
# 임베딩
# ───────────────────────────────────────────────────────────────

@track_metrics
def _embed_query(text: str) -> list[float]:
    """포트폴리오 쿼리 임베딩 — Gemini embedding-2 (1024차원)."""
    client = _get_gemini_client()
    resp = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text,
        config=_genai_types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=1024,
        ),
    )
    return resp.embeddings[0].values



# ───────────────────────────────────────────────────────────────
# BM25 검색
# ───────────────────────────────────────────────────────────────

_bm25_cache: dict[Path, dict] = {}

_db_local = threading.local()


def _get_conn() -> pg8000.Connection:
    conn = getattr(_db_local, "conn", None)
    if conn is None:
        _db_local.conn = pg8000.connect(**get_settings().db_config)
        return _db_local.conn
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        _db_local.conn = pg8000.connect(**get_settings().db_config)
    return _db_local.conn


@track_metrics
def _bm25_search(
    query: str,
    bm25_path: Path,
    top_k: int,
    sub_section_filter: str | None = None,
    job: str | None = None,
    career: str | None = None,
) -> list[tuple[str, float]]:
    if bm25_path not in _bm25_cache:
        if not Path(bm25_path).exists():
            return []
        with open(bm25_path, "rb") as f:
            _bm25_cache[bm25_path] = pickle.load(f)
    data = _bm25_cache[bm25_path]
    tokens = _tokenize_ko(query)

    sub_sections = data.get("sub_sections")
    if sub_section_filter and sub_sections:
        target_idx = [i for i, s in enumerate(sub_sections) if s == sub_section_filter]
        if target_idx:
            all_scores = data["bm25"].get_scores(tokens)
            filtered = [(i, float(all_scores[i])) for i in target_idx if all_scores[i] > 0]
            filtered.sort(key=lambda x: x[1], reverse=True)
            results = [(data["ids"][i], score) for i, score in filtered[:top_k * 3]]
            if job or career:
                results = [(id_, s) for id_, s in results if _cl_id_matches(id_, job, career)]
            return results[:top_k]
    scores  = data["bm25"].get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:top_k * 3]
    results = [(data["ids"][i], float(scores[i])) for i in top_idx if scores[i] > 0]
    if job or career:
        results = [(id_, s) for id_, s in results if _cl_id_matches(id_, job, career)]
    return results[:top_k]


# ───────────────────────────────────────────────────────────────
# 벡터 검색 (pgvector)
# ───────────────────────────────────────────────────────────────

@track_metrics
def _vector_search(
    query_emb: list[float],
    table: str,
    top_k: int,
    sub_section_filter: str | None = None,
    job: str | None = None,
    career: str | None = None,
) -> list[tuple[str, float]]:
    emb_str = "[" + ",".join(str(x) for x in query_emb) + "]"
    conn = _get_conn()
    cur  = conn.cursor()

    fetch_k = top_k * 3 if (job or career) and table == "cover_letter_chunks" else top_k
    if sub_section_filter:
        cur.execute(
            f"SELECT id, 1 - (embedding <=> %s::vector) AS score "
            f"FROM {table} WHERE sub_section = %s "
            f"ORDER BY embedding <=> %s::vector LIMIT %s",
            (emb_str, sub_section_filter, emb_str, fetch_k),
        )
    else:
        cur.execute(
            f"SELECT id, 1 - (embedding <=> %s::vector) AS score "
            f"FROM {table} ORDER BY embedding <=> %s::vector LIMIT %s",
            (emb_str, emb_str, fetch_k),
        )
    rows = [(row[0], float(row[1])) for row in cur.fetchall()]
    cur.close()
    if (job or career) and table == "cover_letter_chunks":
        rows = [(id_, s) for id_, s in rows if _cl_id_matches(id_, job, career)]
    return rows


# ───────────────────────────────────────────────────────────────
# RRF 융합
# ───────────────────────────────────────────────────────────────

@track_metrics
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
# DB 청크 조회
# ───────────────────────────────────────────────────────────────

@track_metrics
def _fetch_chunks(ids: list[str], table: str, cols: list[str]) -> list[dict]:
    if not ids:
        return []
    placeholders = ",".join(["%s"] * len(ids))
    col_str      = ", ".join(cols)
    conn = _get_conn()
    cur  = conn.cursor()
    cur.execute(f"SELECT {col_str} FROM {table} WHERE id IN ({placeholders})", ids)
    rows       = [dict(zip(cols, row)) for row in cur.fetchall()]
    cur.close()
    id_to_row = {r["id"]: r for r in rows}
    return [id_to_row[id_] for id_ in ids if id_ in id_to_row]


# ───────────────────────────────────────────────────────────────
# 포트폴리오 레퍼런스 검색
# ───────────────────────────────────────────────────────────────

def _search_portfolio_refs(cl_chunk: dict, top_k: int) -> list[dict]:
    query_emb  = _embed_query(cl_chunk["text"])
    bm25_res   = _bm25_search(cl_chunk["text"], PORTFOLIO_BM25_PATH, TOP_K_BM25)
    vector_res = _vector_search(query_emb, "portfolio_chunks", TOP_K_VECTOR)
    fused_ids  = _rrf_fusion(bm25_res, vector_res)[:top_k]
    return _fetch_chunks(fused_ids, "portfolio_chunks", _PF_REF_COLS)


# ───────────────────────────────────────────────────────────────
# Gemini 클라이언트
# ───────────────────────────────────────────────────────────────

def _get_gemini_client() -> _genai.Client:
    project = os.getenv("GCP_PROJECT_ID")
    if project:
        return _genai.Client(vertexai=True, project=project, location=os.getenv("GCP_LOCATION", "global"))
    api_key = get_settings().gemini_api_key
    if api_key:
        return _genai.Client(api_key=api_key)
    raise ValueError("GCP_PROJECT_ID 또는 GEMINI_API_KEY 환경변수를 설정하세요.")


@track_metrics
def _generate_with_retry(
    client: _genai.Client,
    model: str,
    contents,
    config,
    max_attempts: int = 3,
):
    """generate_content 호출 래퍼 — 429/503 에러 시 재시도."""
    for attempt in range(max_attempts):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            err = str(e)
            if attempt < max_attempts - 1 and ("429" in err or "quota" in err.lower()):
                wait = 30 * (attempt + 1)
                print(f"  [Gemini 429 Rate Limit] model={model} attempt={attempt+1}/{max_attempts} → {wait}초 대기")
                time.sleep(wait)
            elif attempt < max_attempts - 1 and ("503" in err or "500" in err):
                wait = 10 * (attempt + 1)
                print(f"  [Gemini 503 서버오류] model={model} attempt={attempt+1}/{max_attempts} → {wait}초 대기")
                time.sleep(wait)
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
  - 직무와 연관된 역할·기술임을 드러내는 한 줄 포함\
"""

_PF_GEN_SUBSECTION_GUIDE_DEVELOPMENT = """\
  - 본인이 직접 구현한 핵심 기능, 설계 결정, 기술적 접근법
  - '왜 그 방법을 선택했는지' 판단 근거 필수 포함 (대안 대비 이유)
  - 배경→문제인식→판단→실행 흐름으로 서술\
"""

_PF_GEN_SUBSECTION_GUIDE_ISSUE = """\
  - 문제 발생 → 원인 파악 → 본인이 수행한 해결 과정 순으로 서술
  - '내가/직접/담당' 등 1인칭 행동 동사 사용
  - 자소서에 트러블슈팅 내용이 없으면 ""\
"""

_PF_GEN_SUBSECTION_GUIDE_RESULT = """\
  - 수치 기반 성과 최우선 (%, ms, 배수, 건수 — 자소서 원문 그대로 인용)
  - 정성 성과(배포 완료, 기한 준수 등)도 포함
  - 이 경험에서 얻은 인사이트·배움 1줄 포함\
"""


# ───────────────────────────────────────────────────────────────
# 코드 분석 유틸
# ───────────────────────────────────────────────────────────────

def _fetch_code_analysis(url: str) -> dict:
    """코드 분석 결과 JSON을 URL에서 fetch."""
    import httpx
    resp = httpx.get(url, timeout=15.0)
    resp.raise_for_status()
    return resp.json()


def _fetch_code_analyses(urls: list[str]) -> list[dict]:
    """복수의 코드 분석 URL을 fetch해 list[dict]로 반환."""
    return [_fetch_code_analysis(url) for url in urls if url]


def _build_code_analyses_block(code_analyses: list[dict]) -> str:
    """list[dict] 코드 분석 결과 → LLM 프롬프트 삽입용 텍스트 블록."""
    if not code_analyses:
        return ""
    parts = [
        f"[코드 분석 {i+1}/{len(code_analyses)} — {ca.get('service_name', '')}]\n"
        + _build_code_analysis_block(ca)
        for i, ca in enumerate(code_analyses)
    ]
    return "\n\n".join(parts)


def _build_code_analysis_block(code_analysis: dict) -> str:
    """코드 분석 결과 JSON (CDN 스키마) → LLM 프롬프트 삽입용 텍스트 블록.

    스키마:
      service_name, service_description, dev_type, frameworks, skills,
      user_role, contribute_share_percent, analysis_period,
      analysis.core_feat, analysis.core_perf, analysis.feedback, analysis.contribute
    """
    from collections import defaultdict

    lines = [
        f"[프로젝트] {code_analysis.get('service_name', '')}",
        f"[설명] {code_analysis.get('service_description', '')}",
    ]

    dev_type = code_analysis.get("dev_type", [])
    if dev_type:
        lines.append(f"[개발 유형] {', '.join(dev_type)}")

    # 기술스택 — frameworks + skills (category별)
    frameworks = [f["name"] for f in code_analysis.get("frameworks", [])]
    skills_by_cat: dict[str, list[str]] = defaultdict(list)
    for s in code_analysis.get("skills", []):
        skills_by_cat[s.get("category", "기타")].append(s["name"])

    skill_lines = []
    if frameworks:
        skill_lines.append(f"  · 프레임워크: {', '.join(frameworks)}")
    for cat, names in skills_by_cat.items():
        skill_lines.append(f"  · {cat}: {', '.join(names)}")
    if skill_lines:
        lines.append("[기술스택]\n" + "\n".join(skill_lines))

    # 기여 역할 요약
    user_role = code_analysis.get("user_role", "")
    if user_role:
        lines.append(f"\n[기여 역할]\n{user_role}")

    share  = code_analysis.get("contribute_share_percent")
    period = code_analysis.get("analysis_period", {})
    meta_parts = []
    if share is not None:
        meta_parts.append(f"기여 비율: {share}%")
    if period:
        meta_parts.append(
            f"분석 기간: {period.get('start', '')} ~ {period.get('end', '')} ({period.get('days', '')}일)"
        )
    if meta_parts:
        lines.append(f"[기여 메타] {' / '.join(meta_parts)}")

    analysis = code_analysis.get("analysis", {})

    # 핵심 기능
    core_feat = analysis.get("core_feat", [])
    if core_feat:
        lines.append("\n[핵심 기능 — 프로젝트가 제공하는 기능]")
        for f in core_feat:
            lines.append(f"• {f['feat_title']}: {f['description']}")

    # 기여한 기능 목록
    contrib_titles = analysis.get("contribute", {}).get("contribute_titles", [])
    if contrib_titles:
        lines.append("\n[본인이 기여한 기능]\n" + "\n".join(f"  · {t}" for t in contrib_titles))

    # 핵심 기술 구현 패턴
    core_perf = analysis.get("core_perf", [])
    if core_perf:
        lines.append("\n[핵심 기술 구현 패턴 — 코드에서 직접 확인된 사실]")
        for p in core_perf:
            lines.append(f"• {p['perf_title']} (관련 기능: {p.get('about_feat_title', '')})")
            lines.append(f"  {p['description']}")

    # 검증 피드백
    feedbacks = analysis.get("feedback", [])
    if feedbacks:
        lines.append("\n[수치화·검증 가능 항목 — 포트폴리오 성과 보강 힌트]")
        for fb in feedbacks:
            lines.append(f"• {fb['feedback_title']} (관련: {fb.get('about_perf_title', '')})")
            lines.append(f"  {fb['feedback']}")

    return "\n".join(lines)


_PF_GEN_SYSTEM_WITH_CODE = """\
당신은 IT 직군 포트폴리오 작성 전문가입니다.
자소서와 GitHub 코드 분석 결과를 함께 활용하여 포트폴리오 섹션을 작성합니다.

[절대 원칙]
- 자소서 또는 코드 분석 결과에서 확인된 사실만 사용하세요. 둘 다에 없는 내용은 절대 추가 금지.
- 코드 분석의 pattern_summary·tech_stack에 있는 기술적 구현 세부사항은 포트폴리오 보강에 활용 가능합니다.
- 코드 분석의 feedback.how_to_verify는 "수치화 방법" 힌트로 활용하되, 측정하지 않은 수치를 임의 생성 금지.
- 포트폴리오 특유의 명사형·개조식 문체로 작성하세요 (자소서 산문체 금지).
- 수치가 있으면 반드시 그대로 인용하세요.

[평가 기준 반영 — 아래 기준으로 채점되므로 반드시 충족]
A. 과정/판단력(35%): 배경→문제인식→선택/판단→실행 흐름이 명확해야 함
   - '왜 그 기술/방법을 선택했는지' 판단 근거를 반드시 명시
   - 코드 분석의 why_likely_matters를 판단 근거 서술에 활용 가능
B. 역할/기여도(25%): 1인칭 기여가 명확해야 함
   - author_contribution에서 확인된 책임 영역을 근거로 역할 서술 가능
C. 성과/인사이트(20%): 정량 수치 우선, 배움과 성장 연결
D. 작성품질 감점 요인 — 반드시 회피:
   - 추상 표현, 단순 기술 나열, 중복 표현 금지
E. 직무연관성(10%): 직무 역량과 연결되는 경험임을 명시

[project 필드] 코드 분석의 title을 우선 사용하되, 자소서 원문에 다른 명칭이 있으면 그것을 사용하세요.

[gaps 작성 원칙]
- 자소서와 코드 분석 모두에서 확인되지 않은 항목만 gap으로 기재하세요.
- user_action: 지원자가 직접 보완해야 할 구체적 요청 1문장
- example: 레퍼런스에서 이 gap을 잘 채운 표현 직접 인용 (없으면 "")
"""


@track_metrics
def _generate_portfolio_section(
    cl_chunk: dict,
    refs: list[dict],
    client: _genai.Client,
    job: str | None = None,
    career: str | None = None,
    code_analyses: list[dict] = [],
) -> _PortfolioGenResult:
    """자소서 청크 + 포트폴리오 레퍼런스 → 포트폴리오 섹션 생성.
    code_analyses가 있으면 코드 분석 결과를 프롬프트에 추가해 강화.
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
            else:
                content = c["text"][:400]
            lines.append(f"  [{proj}]\n  {content}")
        return "\n\n".join(lines)

    kp_block  = "\n".join(f"  - {kp}" for kp in cl_chunk.get("key_points", []))
    ach_block = "\n".join(f"  - {a}"  for a in cl_chunk.get("achievements", []))
    kw_block  = ", ".join(cl_chunk.get("keywords", []))

    job_career_ctx = ""
    if job or career:
        job_career_ctx = (
            f"[지원자 정보]\n"
            + (f"직군: {job}\n" if job else "")
            + (f"경력: {career}\n" if career else "")
            + "\n"
        )

    code_block_section = ""
    if code_analyses:
        code_block_section = (
            f"[GitHub 코드 분석 결과 — 자소서 내용 보강에 활용 가능한 기술적 사실]\n"
            f"{_build_code_analyses_block(code_analyses)}\n\n"
        )

    origin_note = (
        "자소서+코드분석 사실만 사용하세요." if code_analyses
        else "자소서 원문 사실만 사용하세요."
    )
    closing_note = (
        "위 자소서 내용과 코드 분석을 바탕으로 포트폴리오 섹션을 작성하세요.\n"
        "자소서에도 코드 분석에도 없는 내용은 절대 추가하지 말고 해당 서브섹션을 \"\"로 두세요."
        if code_analyses else
        "위 자소서 내용을 바탕으로 포트폴리오 섹션을 작성하세요.\n"
        "자소서에 언급되지 않은 내용은 절대 추가하지 말고 해당 서브섹션을 \"\"로 두세요."
    )

    prompt = (
        f"{job_career_ctx}"
        f"[자소서 원문]\n"
        f"카테고리: {cl_chunk.get('category', '')}\n"
        f"섹션 제목: {cl_chunk.get('section', '')}\n\n"
        f"{cl_chunk['text']}\n\n"
        f"[추출된 메타 정보]\n"
        f"핵심 포인트(STAR):\n{kp_block}\n"
        f"성과:\n{ach_block}\n"
        f"키워드: {kw_block}\n\n"
        f"{code_block_section}"
        f"[서브섹션별 작성 기준 + 실제 포트폴리오 표현 예시]\n"
        f"단, 예시의 내용(프로젝트명·수치·기술)은 절대 복사하지 말고 {origin_note}\n\n"
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
        f"{closing_note}"
    )

    system = _PF_GEN_SYSTEM_WITH_CODE if code_analyses else _PF_GEN_SYSTEM
    resp = _generate_with_retry(
        client,
        model=_LLM_MODEL,
        contents=prompt,
        config=_genai_types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=_PortfolioGenResult,
        ),
    )
    return _PortfolioGenResult.model_validate_json(resp.text)
