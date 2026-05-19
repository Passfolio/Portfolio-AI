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

def rag_cover_letter(query: str, top_k: int = TOP_K_FINAL) -> str:
    """유사 자소서 검색 → Gemini로 자소서 개선안 생성.

    Args:
        query: 사용자가 작성한 자소서 원문 또는 문항
        top_k: 참고할 유사 자소서 수
    """
    query_emb  = _embed_query(query)
    bm25_res   = _bm25_search(query, CL_BM25_PATH, TOP_K_BM25)
    vector_res = _vector_search(query_emb, "cover_letter_chunks", TOP_K_VECTOR)
    fused_ids  = _rrf_fusion(bm25_res, vector_res)[:top_k]
    chunks     = _fetch_chunks(fused_ids, "cover_letter_chunks", ["id", "sub_section", "category", "text"])

    context_block = "\n\n---\n\n".join(
        f"[예시 {i+1} | {c['category']} — {c['sub_section']}]\n{c['text']}"
        for i, c in enumerate(chunks)
    )

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
        f"5. 허용 범위: 문장 구조 재배치, 두괄식 전환, 표현 구체화, 불필요한 문장 제거\n\n"
        f"[개선할 자소서]\n{query}\n\n"
        f"다음 JSON 형식으로 응답하세요:\n"
        f"- improved: 개선된 자소서 전문 (원문 사실만 사용)\n"
        f"- reasoning: 개선 근거 (왜 이렇게 바꿨는지 2~3문장)\n"
        f"- changes: 주요 변경 사항 목록 (항목당 한 줄, 추가된 내용 없이 수정만)"
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
# RAG-2: 포트폴리오 기반 자소서 생성
# ═══════════════════════════════════════════════════════════════

def rag_portfolio_to_cover_letter(question: str, top_k: int = TOP_K_FINAL) -> str:
    """자소서 문항 → 포트폴리오 검색 → Gemini로 자소서 답변 작성.

    Args:
        question: 자소서 문항 (예: "가장 도전적이었던 프로젝트 경험을 작성해주세요")
        top_k: 참고할 포트폴리오 섹션 수
    """
    query_emb  = _embed_query(question)
    bm25_res   = _bm25_search(question, PORTFOLIO_BM25_PATH, TOP_K_BM25)
    vector_res = _vector_search(query_emb, "portfolio_chunks", TOP_K_VECTOR)
    fused_ids  = _rrf_fusion(bm25_res, vector_res)[:top_k]
    chunks     = _fetch_chunks(fused_ids, "portfolio_chunks", ["id", "section", "project", "text"])

    context_block = "\n\n---\n\n".join(
        f"[{c['section']} — {c.get('project', '')}]\n{c['text']}"
        for i, c in enumerate(chunks)
    )

    prompt = (
        f"다음은 지원자의 포트폴리오 내용입니다:\n\n"
        f"{context_block}\n\n"
        f"위 포트폴리오의 실제 경험과 수치를 근거로 삼아 "
        f"아래 자소서 문항에 대한 답변을 300~500자로 작성해주세요. "
        f"STAR 구조(상황→행동→결과)를 따르고, 구체적인 수치를 반드시 포함하세요.\n\n"
        f"[자소서 문항]\n{question}"
    )

    client = _get_gemini_client()
    resp = client.models.generate_content(model=_LLM_MODEL, contents=prompt)
    return resp.text


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
        f"4. 허용 범위: 문장 구조 재배치, 과정·판단 근거 부각, 표현 구체화, 불필요한 문장 제거\n\n"
        f"[개선할 포트폴리오]\n{query}\n\n"
        f"다음 JSON 형식으로 응답하세요:\n"
        f"- improved: 개선된 포트폴리오 전문 (원문 사실만 사용)\n"
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

        result = rag_cover_letter(chunk["text"], top_k=top_k)

        print(f"\n{'─'*28} 원문 {'─'*28}")
        print(chunk["text"])
        print(f"\n{'─'*26} 개선안 {'─'*26}")
        print(result["improved"])
        print(f"\n{'─'*26} 개선 근거 {'─'*24}")
        print(result["reasoning"])
        print(f"\n{'─'*26} 주요 변경사항 {'─'*21}")
        for j, change in enumerate(result["changes"], 1):
            print(f"  {j}. {change}")

        char_limit  = parse_char_limit(chunk["section"])
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

    out_path = OUTPUT_DIR / f"{Path(pdf_path).stem}_rag_result.json"
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{sep}")
    print(f"완료: 총 {len(results)}개 문항 개선 → {out_path}")
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
            "section":     ic.get("section", ""),
            "project":     ic.get("project", ""),
            "sub_section": "이미지",
            "original":    ic["text"],
            "improved":    ic["text"],
            "reasoning":   "이미지 캡션은 개선 대상에서 제외됩니다.",
            "changes":     [],
            "eval_before": None,
            "eval_after":  None,
            "eval_delta":  None,
            "eval_detail": None,
        })

    out_path = OUTPUT_DIR / f"{source}_portfolio_rag_result.json"
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{sep}")
    print(f"완료: 총 {len(results)}개 섹션 개선 → {out_path}")
    return results


# ═══════════════════════════════════════════════════════════════
# CLI (테스트용)
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG 테스트")
    parser.add_argument("--pdf",       type=str, default=None,
                        help="자소서 PDF 파일 경로")
    parser.add_argument("--portfolio", type=str, default=None,
                        help="포트폴리오 PDF 파일 경로")
    parser.add_argument("--query",     type=str, default=None,
                        help="텍스트 쿼리 직접 입력 (자소서 개선)")
    parser.add_argument("--pf-query",  type=str, default=None,
                        help="텍스트 쿼리 직접 입력 (포트폴리오 개선)")
    parser.add_argument("--top-k",     type=int, default=TOP_K_FINAL,
                        help=f"참고 청크 수 (기본: {TOP_K_FINAL})")
    args = parser.parse_args()

    if args.pdf:
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
