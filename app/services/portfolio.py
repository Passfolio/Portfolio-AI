"""포트폴리오 관련 RAG 서비스 함수 + BackgroundTask 래퍼."""
from __future__ import annotations

import logging

from google.genai import types as _genai_types

from app.services.pdf_pipeline import (
    download_pdf_to_temp, make_output_path,
    upload_pdf_file, cleanup_files, run_job_pipeline,
)
from app.services._rag_utils import (
    TOP_K_BM25,
    TOP_K_FINAL,
    TOP_K_VECTOR,
    PORTFOLIO_BM25_PATH,
    _ImprovedResult,
    _LLM_MODEL,
    _bm25_search,
    _embed_query,
    _fetch_chunks,
    _generate_with_retry,
    _get_gemini_client,
    _rrf_fusion,
    _vector_search,
    _PORTFOLIO_CRITERIA,
)

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────
# 내부 helper: 포트폴리오 텍스트 개선
# ───────────────────────────────────────────────────────────────

def _improve_portfolio_text(query: str, top_k: int = TOP_K_FINAL, img_context: str = "") -> dict:
    query_emb  = _embed_query(query)
    bm25_res   = _bm25_search(query, PORTFOLIO_BM25_PATH, TOP_K_BM25)
    vector_res = _vector_search(query_emb, "portfolio_chunks", TOP_K_VECTOR)
    fused_ids  = _rrf_fusion(bm25_res, vector_res)[:top_k]
    chunks     = _fetch_chunks(
        fused_ids, "portfolio_chunks",
        ["id", "section", "sub_section", "project", "text"],
    )

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
        f"5. [수치 보존 필수] 원문에 포함된 숫자·단위(%, ms, 배, 건, GB 등)는 개선안에 반드시 유지하세요.\n"
        f"6. [분량 기준] 개선안은 400~1000자로 작성하세요.\n"
        f"7. [완료·결과 표현 제한] 원문에 그 결과가 명시된 경우에만 '해결했다', '완료했다' 등을 사용하세요.\n"
        f"8. [역할 경계 준수] 본인이 직접 수행한 역할만 1인칭으로 서술하세요.\n\n"
        f"[개선할 포트폴리오]\n{query}\n\n"
        f"다음 JSON 형식으로 응답하세요:\n"
        f"- improved: 개선된 포트폴리오 전문 (원문 사실만 사용, 400~1000자)\n"
        f"- reasoning: 개선 근거 (왜 이렇게 바꿨는지 2~3문장)\n"
        f"- changes: 주요 변경 사항 목록 (항목당 한 줄)"
    )

    client = _get_gemini_client()
    resp = _generate_with_retry(
        client,
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


# ───────────────────────────────────────────────────────────────
# RAG-4: 포트폴리오 PDF → 섹션별 개선
# ───────────────────────────────────────────────────────────────

def run_portfolio_from_pdf(pdf_s3_url: str, user_id: int | None = None, top_k: int = TOP_K_FINAL) -> dict:
    from app.chunkers.portfolio import chunk
    from app.evaluators.portfolio import evaluate_comparison as pf_evaluate_comparison
    from app.exporters.portfolio_pdf import save_improvement_pdf

    tmp_path = download_pdf_to_temp(pdf_s3_url)
    out_pdf  = make_output_path("portfolio_rag_result")
    try:
        logger.info("[RAG-4] PDF 청킹 중...")
        all_chunks = chunk(tmp_path)
        logger.info("[RAG-4] 청킹 완료: %d개 청크", len(all_chunks))

        img_chunks  = [c for c in all_chunks if c.get("sub_section") == "이미지"]
        text_chunks = [c for c in all_chunks if c.get("sub_section") != "이미지"]

        img_ctx_by_project: dict[str, str] = {}
        for ic in img_chunks:
            proj  = ic.get("project", "")
            entry = f"[{ic.get('content_type', '')}] {ic['text']}"
            img_ctx_by_project[proj] = (
                img_ctx_by_project[proj] + f"\n{entry}" if proj in img_ctx_by_project else entry
            )

        results = []

        for idx, c in enumerate(text_chunks):
            section     = c.get("section", "")
            project     = c.get("project", "")
            sub_section = c.get("sub_section", "")
            meta        = c.get("meta") or None

            logger.info("[RAG-4] [%d/%d] 텍스트 개선: %s / %s", idx + 1, len(text_chunks), project, sub_section)
            img_context = img_ctx_by_project.get(project, "")
            result      = _improve_portfolio_text(c["text"], top_k=top_k, img_context=img_context)
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

        logger.info("[RAG-4] PDF 생성 중...")
        save_improvement_pdf(results, out_pdf)
        output_s3_url = upload_pdf_file(out_pdf, user_id)
        return {"sections": results, "outputPdfS3Url": output_s3_url}
    finally:
        cleanup_files(tmp_path, out_pdf)


# ───────────────────────────────────────────────────────────────
# BackgroundTask 래퍼
# ───────────────────────────────────────────────────────────────

async def run_portfolio_from_pdf_task(
    job_id: str,
    pdf_s3_url: str,
    user_id: int | None = None,
    top_k: int = TOP_K_FINAL,
) -> None:
    run_job_pipeline(
        job_id,
        lambda: run_portfolio_from_pdf(pdf_s3_url, user_id=user_id, top_k=top_k),
        tag="RAG-4",
    )
