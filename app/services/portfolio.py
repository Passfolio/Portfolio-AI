"""포트폴리오 관련 RAG 서비스 함수 + BackgroundTask 래퍼."""
from __future__ import annotations

import concurrent.futures
import logging

from google.genai import types as _genai_types
from app.core.metrics import track_metrics

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
    _build_code_analyses_block,
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

@track_metrics
def _improve_portfolio_text(
    query: str,
    top_k: int = TOP_K_FINAL,
    img_context: str = "",
    code_analyses: list[dict] = [],
) -> dict:
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

    code_block_section = ""
    if code_analyses:
        code_block_section = (
            f"[GitHub 코드 분석 결과 — 아래 기술적 사실은 포트폴리오 보강에 활용 가능]\n"
            f"{_build_code_analyses_block(code_analyses)}\n\n"
        )

    if code_analyses:
        constraints = (
            f"[준수 사항]\n"
            f"1. 원문 또는 코드 분석에서 확인된 사실만 사용하세요 (둘 다에 없는 내용 추가 금지).\n"
            f"2. 코드 분석의 pattern_summary에 있는 기술 구현 세부사항은 구체화·추가 가능합니다.\n"
            f"3. 코드 분석의 feedback(how_to_verify)은 힌트로 활용하되, 측정 안 된 수치 생성 금지.\n"
            f"4. [수치 보존 필수] 원문의 숫자·단위(%, ms, 배, 건, GB 등)는 반드시 유지하세요.\n"
            f"5. [분량 기준] 개선안은 최대 1000자로 작성하세요.\n"
            f"6. 본인이 직접 수행한 역할만 1인칭으로 서술하세요.\n\n"
        )
        result_note = (
            f"- reasoning: 개선 근거 (코드 분석에서 보강한 내용 명시, 2~3문장)\n"
            f"- changes: 주요 변경 사항 목록 (코드분석 추가분은 '[코드분석]' 태그 표시)"
        )
    else:
        constraints = (
            f"[절대 준수 사항 — 위반 시 무효]\n"
            f"1. 원문에 없는 프로젝트·경험·수치·기술을 절대 추가하지 마세요.\n"
            f"2. 원문에 명시된 사실(프로젝트명, 역할, 수치, 기술스택 등)만 사용하세요.\n"
            f"3. 수치가 원문에 없으면 임의로 만들지 말고 정성적 표현을 구체화하는 데 집중하세요.\n"
            f"4. 허용 범위: 문장 구조 재배치, 과정·판단 근거 부각, 표현 구체화, 불필요한 문장 제거\n"
            f"5. [수치 보존 필수] 원문에 포함된 숫자·단위(%, ms, 배, 건, GB 등)는 개선안에 반드시 유지하세요.\n"
            f"6. [분량 기준] 개선안은 1000자 이내로 작성하세요.\n"
            f"7. [완료·결과 표현 제한] 원문에 그 결과가 명시된 경우에만 '해결했다', '완료했다' 등을 사용하세요.\n"
            f"8. [역할 경계 준수] 본인이 직접 수행한 역할만 1인칭으로 서술하세요.\n\n"
        )
        result_note = (
            f"- reasoning: 개선 근거 (왜 이렇게 바꿨는지 2~3문장)\n"
            f"- changes: 주요 변경 사항 목록 (항목당 한 줄)"
        )

    prompt = (
        f"다음은 실제 포트폴리오 예시들입니다 (구조·표현 참고용):\n\n"
        f"{context_block}\n\n"
        f"{_PORTFOLIO_CRITERIA}\n\n"
        f"{code_block_section}"
        f"위 예시들과 작성 기준을 참고하여 아래 포트폴리오 내용을 개선해주세요."
        f"{img_block}\n\n"
        f"{constraints}"
        f"[개선할 포트폴리오]\n{query}\n\n"
        f"다음 JSON 형식으로 응답하세요:\n"
        f"- improved: 개선된 포트폴리오 전문 (400~1000자)\n"
        f"{result_note}"
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
# 코드 분석 프로젝트 매칭 + 신규 섹션 생성
# ───────────────────────────────────────────────────────────────

def _match_project(portfolio_project: str, service_name: str) -> bool:
    """포트폴리오 프로젝트명과 코드분석 service_name이 같은 프로젝트인지 판별.
    대소문자·괄호 표기 차이를 허용하는 포함 관계로 비교.
    """
    a = portfolio_project.lower().strip()
    b = service_name.lower().strip()
    # 괄호 안 부가 설명 제거 후 비교 (예: "Deokive (덕이브)" → "deokive")
    import re
    a_clean = re.sub(r"\(.*?\)", "", a).strip()
    b_clean = re.sub(r"\(.*?\)", "", b).strip()
    return a_clean == b_clean or a_clean in b_clean or b_clean in a_clean


@track_metrics
def _generate_portfolio_from_code_analysis(
    code_analysis: dict,
    top_k: int = TOP_K_FINAL,
) -> dict:
    """단일 코드 분석 결과로 포트폴리오 신규 섹션 텍스트 생성."""
    service_name = code_analysis.get("service_name", "")
    query        = f"{service_name} {code_analysis.get('service_description', '')}"

    query_emb  = _embed_query(query)
    bm25_res   = _bm25_search(query, PORTFOLIO_BM25_PATH, TOP_K_BM25)
    vector_res = _vector_search(query_emb, "portfolio_chunks", TOP_K_VECTOR)
    fused_ids  = _rrf_fusion(bm25_res, vector_res)[:top_k]
    chunks     = _fetch_chunks(fused_ids, "portfolio_chunks", ["id", "section", "project", "text"])

    ref_block = "\n\n---\n\n".join(
        f"[예시 {i+1} | {c['section']} — {c.get('project', '')}]\n{c['text']}"
        for i, c in enumerate(chunks)
    )

    from app.services._rag_utils import _build_code_analysis_block
    code_block = _build_code_analysis_block(code_analysis)

    prompt = (
        f"다음은 실제 포트폴리오 예시들입니다 (구조·표현 참고용, 내용 복사 금지):\n\n"
        f"{ref_block}\n\n"
        f"{_PORTFOLIO_CRITERIA}\n\n"
        f"[GitHub 코드 분석 결과 — 아래 내용을 근거로 포트폴리오를 새로 작성하세요]\n"
        f"{code_block}\n\n"
        f"[작성 원칙]\n"
        f"1. 코드 분석에서 확인된 사실(service_name, core_perf, user_role, contribute 등)만 사용하세요.\n"
        f"2. '왜 그 기술/방법을 선택했는지' 판단 근거를 반드시 포함하세요 (core_perf.description 참고).\n"
        f"3. 본인이 기여한 기능(contribute_titles)을 중심으로 역할을 명확히 서술하세요.\n"
        f"4. 수치가 코드 분석에 없으면 임의로 만들지 말고 정성적 성과로 구체화하세요.\n"
        f"5. [분량] 600~1000자로 작성하세요.\n"
        f"6. 포트폴리오 특유의 개조식·명사형 문체로 작성하세요.\n\n"
        f"다음 JSON 형식으로 응답하세요:\n"
        f"- improved: 작성된 포트폴리오 전문 (600~1000자)\n"
        f"- reasoning: 작성 근거 (코드 분석에서 근거한 내용 2~3문장)\n"
        f"- changes: 포함된 주요 내용 목록"
    )

    client = _get_gemini_client()
    resp   = _generate_with_retry(
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
        "section":     "프로젝트 경험",
        "project":     service_name,
        "sub_section": "개발",
        "original":    "",
        "improved":    result.improved,
        "reasoning":   result.reasoning,
        "changes":     result.changes,
        "eval_before": None,
        "eval_after":  None,
        "eval_delta":  None,
        "eval_detail": None,
        "is_generated_from_code_analysis": True,
    }


# ───────────────────────────────────────────────────────────────
# RAG-4: 포트폴리오 PDF → 섹션별 개선
# ───────────────────────────────────────────────────────────────

@track_metrics
def run_portfolio_from_pdf(
    pdf_s3_url: str,
    user_id: int | None = None,
    top_k: int = TOP_K_FINAL,
    code_analyses: list[dict] = [],
) -> dict:
    from app.chunkers.portfolio import chunk
    from app.evaluators.portfolio import evaluate_comparison as pf_evaluate_comparison
    from app.exporters.portfolio_pdf import save_improvement_pdf

    tmp_path = download_pdf_to_temp(pdf_s3_url)
    out_pdf  = make_output_path("portfolio_rag_result")
    try:
        logger.info("[RAG-4] PDF 청킹 중...")
        all_chunks = chunk(tmp_path)
        logger.info("[RAG-4] 청킹 완료: %d개 청크 (코드분석: %s)", len(all_chunks), "있음" if code_analyses else "없음")

        img_chunks  = [c for c in all_chunks if c.get("sub_section") == "이미지"]
        text_chunks = [c for c in all_chunks if c.get("sub_section") != "이미지"]

        img_ctx_by_project: dict[str, str] = {}
        for ic in img_chunks:
            proj  = ic.get("project", "")
            entry = f"[{ic.get('content_type', '')}] {ic['text']}"
            img_ctx_by_project[proj] = (
                img_ctx_by_project[proj] + f"\n{entry}" if proj in img_ctx_by_project else entry
            )

        total_chunks = len(text_chunks)
        matched_service_names: set[str] = set()

        def _process_chunk(args: tuple) -> dict:
            idx, c = args
            section     = c.get("section", "")
            project     = c.get("project", "")
            sub_section = c.get("sub_section", "")
            meta        = c.get("meta") or None

            matched_ca = next(
                (ca for ca in code_analyses if _match_project(project, ca.get("service_name", ""))),
                None,
            )
            if matched_ca:
                matched_service_names.add(matched_ca.get("service_name", ""))

            logger.info(
                "[RAG-4] [%d/%d] 텍스트 개선: %s / %s (코드분석: %s)",
                idx, total_chunks, project, sub_section,
                matched_ca.get("service_name", "") if matched_ca else "N",
            )
            img_context = img_ctx_by_project.get(project, "")
            result      = _improve_portfolio_text(
                c["text"], top_k=top_k, img_context=img_context,
                code_analyses=[matched_ca] if matched_ca else [],
            )
            eval_result = pf_evaluate_comparison(c["text"], result["improved"], meta=meta)
            logger.info("[RAG-4] [%d/%d] 완료: %s / %s", idx, total_chunks, project, sub_section)
            return {
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
            }

        max_workers = min(5, total_chunks)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_chunk, (i + 1, c)) for i, c in enumerate(text_chunks)]
            results = [f.result() for f in futures]

        # 포트폴리오에 없는 코드분석 프로젝트 → 신규 섹션 생성
        for ca in code_analyses:
            if ca.get("service_name", "") not in matched_service_names:
                sn = ca.get("service_name", "")
                logger.info("[RAG-4] 코드분석 프로젝트 '%s'가 포트폴리오에 없음 → 신규 섹션 생성", sn)
                results.append(_generate_portfolio_from_code_analysis(ca, top_k=top_k))

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
# BackgroundTask 래퍼 — code_analysis_url 있으면 fetch 후 전달
# ───────────────────────────────────────────────────────────────

async def run_portfolio_from_pdf_task(
    job_id: str,
    pdf_s3_url: str,
    user_id: int | None = None,
    top_k: int = TOP_K_FINAL,
    code_analysis_urls: list[str] = [],
) -> None:
    from app.services._rag_utils import _fetch_code_analyses
    code_analyses = _fetch_code_analyses(code_analysis_urls)
    await run_job_pipeline(
        job_id,
        lambda: run_portfolio_from_pdf(pdf_s3_url, user_id=user_id, top_k=top_k, code_analyses=code_analyses),
        tag="RAG-4",
    )
