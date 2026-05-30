"""자소서 관련 RAG 서비스 함수 + BackgroundTask 래퍼."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from google.genai import types as _genai_types

if TYPE_CHECKING:
    from google import genai as _genai

logger = logging.getLogger(__name__)

from app.services.pdf_pipeline import (
    download_pdf_to_temp, make_output_path,
    upload_pdf_file, cleanup_files, run_job_pipeline,
)
from app.services._rag_utils import (
    TOP_K_BM25,
    TOP_K_FINAL,
    TOP_K_VECTOR,
    CL_BM25_PATH,
    _CoverLetterGenResult,
    _ImprovedResult,
    _LLM_MODEL,
    _bm25_search,
    _embed_query_cl,
    _fetch_chunks,
    _generate_portfolio_section,
    _generate_with_retry,
    _get_gemini_client,
    _rrf_fusion,
    _search_portfolio_refs,
    _tokenize_ko,
    _vector_search,
    _EVALUATION_CRITERIA,
)
from rank_bm25 import BM25Okapi

# ───────────────────────────────────────────────────────────────
# 자소서 생성 섹션 정의
# ───────────────────────────────────────────────────────────────

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
   수치가 없으면 "무엇이→어떻게→어떤 변화"가 드러나는 구체적 정성 성과를 서술하세요.
   예) "팀 내 리뷰 문화 정착 → PR 병합 속도 체감 향상" / "중재 역할로 2주 내 합의 도출 → 일정 지연 없이 마감"
   수치를 임의로 만들지 마세요. "기여했습니다", "향상됐습니다" 같은 막연한 표현만으로 마무리하지 마세요.

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


# ───────────────────────────────────────────────────────────────
# 내부 helper: 자소서 텍스트 개선
# ───────────────────────────────────────────────────────────────

def _improve_cover_letter_text(
    query: str,
    top_k: int = TOP_K_FINAL,
    char_limit: int | None = None,
    section: str = "",
) -> dict:
    from app.evaluators.cover_letter import is_competency_question

    query_emb  = _embed_query_cl(query)
    bm25_res   = _bm25_search(query, CL_BM25_PATH, TOP_K_BM25)
    vector_res = _vector_search(query_emb, "cover_letter_chunks", TOP_K_VECTOR)
    fused_ids  = _rrf_fusion(bm25_res, vector_res)[:top_k]
    chunks     = _fetch_chunks(fused_ids, "cover_letter_chunks", ["id", "sub_section", "category", "text"])

    context_block = "\n\n---\n\n".join(
        f"[예시 {i+1} | {c['category']} — {c['sub_section']}]\n{c['text']}"
        for i, c in enumerate(chunks)
    )

    if char_limit:
        target_min   = int(char_limit * 0.9)
        volume_rule  = f"개선안은 {target_min}자 이상 {char_limit}자 이하로 작성하세요 (제한의 90% 이상 채워야 만점)."
    elif is_competency_question(section):
        target_min   = None
        volume_rule  = "개선안은 600~800자로 작성하세요 (직무역량 문항 기준)."
    else:
        target_min   = None
        volume_rule  = "개선안은 400~600자로 작성하세요."

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

        if attempt > 0 and result is not None and target_min is not None:
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

        resp = _generate_with_retry(
            client,
            model=_LLM_MODEL,
            contents=current_prompt,
            config=_genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_ImprovedResult,
            ),
        )
        result = _ImprovedResult.model_validate_json(resp.text)

        if not char_limit or target_min is None or len(result.improved) >= target_min:
            break

    return {
        "improved":  result.improved,
        "reasoning": result.reasoning,
        "changes":   result.changes,
    }


# ───────────────────────────────────────────────────────────────
# RAG-1: 자소서 PDF → 섹션별 개선
# ───────────────────────────────────────────────────────────────

def run_cover_letter_from_pdf(pdf_s3_url: str, user_id: int | None = None, top_k: int = TOP_K_FINAL) -> dict:
    from docling.document_converter import DocumentConverter
    from app.chunkers import cover_letter as cl_chunker
    from app.evaluators.cover_letter import evaluate_comparison, parse_char_limit
    from app.exporters.cover_letter_pdf import save_cl_improvement_pdf

    tmp_path = download_pdf_to_temp(pdf_s3_url)
    out_pdf  = make_output_path("cl_improvement")
    try:
        converter = DocumentConverter()
        text      = converter.convert(tmp_path).document.export_to_text()

        logger.info("[RAG-1] 자소서 청킹 중...")
        chunks = cl_chunker.chunk(text, source="cover_letter")
        logger.info("[RAG-1] 청킹 완료: %d개 청크", len(chunks))

        results = []
        for idx, c in enumerate(chunks):
            section    = c.get("section", "")
            category   = c.get("category", "")
            char_limit = parse_char_limit(section)

            logger.info("[RAG-1] [%d/%d] 텍스트 개선: %s / %s", idx + 1, len(chunks), section, category)
            result      = _improve_cover_letter_text(c["text"], top_k=top_k, char_limit=char_limit, section=section)
            eval_result = evaluate_comparison(c["text"], result["improved"], char_limit=char_limit, question=section)

            results.append({
                "section":     section,
                "category":    category,
                "original":    c["text"],
                "improved":    result["improved"],
                "reasoning":   result["reasoning"],
                "changes":     result["changes"],
                "eval_before": eval_result["before"]["weighted"],
                "eval_after":  eval_result["after"]["weighted"],
                "eval_delta":  eval_result["delta"],
                "eval_detail": eval_result["per_category"],
            })

        logger.info("[RAG-1] PDF 생성 중...")
        save_cl_improvement_pdf(results, out_pdf)
        output_s3_url = upload_pdf_file(out_pdf, user_id)
        return {"sections": results, "outputPdfS3Url": output_s3_url}
    finally:
        cleanup_files(tmp_path, out_pdf)


# ───────────────────────────────────────────────────────────────
# 포트폴리오 → 자소서 생성 헬퍼
# ───────────────────────────────────────────────────────────────

def _select_portfolio_chunks(
    portfolio_chunks: list[dict],
    query: str,
    top_k: int = 3,
) -> list[dict]:
    text_chunks = [c for c in portfolio_chunks if c.get("sub_section") != "이미지"]
    if not text_chunks:
        return []
    corpus  = [_tokenize_ko(c.get("text", "")) for c in text_chunks]
    bm25    = BM25Okapi(corpus)
    scores  = bm25.get_scores(_tokenize_ko(query))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [text_chunks[i] for i in top_idx if scores[i] > 0]


def generate_cl_section(
    section_def: dict,
    portfolio_chunks: list[dict],
    client: _genai.Client,
    top_k: int = TOP_K_FINAL,
    used_projects: list[str] | None = None,
    img_ctx_by_project: dict[str, str] | None = None,
) -> dict:
    pf_chunks   = _select_portfolio_chunks(portfolio_chunks, section_def["query"], top_k=5)
    top_project = pf_chunks[0].get("project", "") if pf_chunks else ""
    pf_block    = "\n\n---\n\n".join(
        f"[{c.get('section','')} — {c.get('project','')}]\n{c['text']}"
        for c in pf_chunks
    ) if pf_chunks else "포트폴리오 관련 섹션 없음"

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

    query_emb  = _embed_query_cl(section_def["question"])
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
        f"적합한 이미지가 없으면 빈 문자열(\"\")을 반환하세요.\n\n"
        f"[자소서 문항]\n{section_def['question']}"
    )

    resp = _generate_with_retry(
        client,
        model=_LLM_MODEL,
        contents=prompt,
        config=_genai_types.GenerateContentConfig(
            system_instruction=_CL_GEN_SYSTEM,
            response_mime_type="application/json",
            response_schema=_CoverLetterGenResult,
        ),
    )
    parsed     = _CoverLetterGenResult.model_validate_json(resp.text)
    clean_text = parsed.text.replace("\n\n", " ").replace("\n", " ").strip()
    return {
        "label":            section_def["label"],
        "question":         section_def["question"],
        "text":             clean_text,
        "top_project":      top_project,
        "image_suggestion": parsed.image_suggestion,
    }


# ───────────────────────────────────────────────────────────────
# RAG-3: 포트폴리오 PDF → 자소서 생성
# ───────────────────────────────────────────────────────────────

def run_portfolio_to_cover_letter(pdf_s3_url: str, user_id: int | None = None, top_k: int = TOP_K_FINAL) -> dict:
    from app.chunkers.portfolio import chunk
    from app.evaluators.cover_letter import evaluate
    from app.exporters.cover_letter_pdf import save_pdf

    tmp_path = download_pdf_to_temp(pdf_s3_url)
    out_pdf  = make_output_path("cl_from_portfolio")
    try:
        logger.info("[RAG-3] PDF 청킹 중...")
        portfolio_chunks = chunk(tmp_path)
        text_chunks = [c for c in portfolio_chunks if c.get("sub_section") != "이미지"]
        img_chunks  = [c for c in portfolio_chunks if c.get("sub_section") == "이미지"]
        logger.info("[RAG-3] 청킹 완료: %d개 텍스트, %d개 이미지", len(text_chunks), len(img_chunks))

        img_ctx_by_project: dict[str, str] = {}
        for ic in img_chunks:
            proj  = ic.get("project", "")
            entry = f"[{ic.get('content_type', 'other')}] {ic['text']}"
            img_ctx_by_project[proj] = (
                img_ctx_by_project[proj] + f"\n{entry}" if proj in img_ctx_by_project else entry
            )

        client        = _get_gemini_client()
        results       = []
        used_projects: list[str] = []

        for idx, sec_def in enumerate(_CL_GEN_SECTIONS):
            logger.info("[RAG-3] [%d/%d] 섹션 생성: %s", idx + 1, len(_CL_GEN_SECTIONS), sec_def["label"])
            generated = generate_cl_section(
                sec_def, text_chunks, client, top_k=top_k,
                used_projects=used_projects or None,
                img_ctx_by_project=img_ctx_by_project or None,
            )
            if generated.get("top_project"):
                used_projects.append(generated["top_project"])

            eval_result = evaluate(
                generated["text"],
                char_limit=sec_def["char_limit"],
                question=sec_def["question"],
            )
            results.append({
                "label":      generated["label"],
                "question":   generated["question"],
                "text":       generated["text"],
                "char_count": len(generated["text"]),
                "eval": {
                    "weighted": eval_result["weighted"],
                    "llm":      eval_result["llm"],
                },
            })

        logger.info("[RAG-3] PDF 생성 중...")
        save_pdf(results, out_pdf)
        output_s3_url = upload_pdf_file(out_pdf, user_id)
        return {"sections": results, "outputPdfS3Url": output_s3_url}
    finally:
        cleanup_files(tmp_path, out_pdf)


# ───────────────────────────────────────────────────────────────
# RAG-2: 자소서 PDF → 포트폴리오 생성
# ───────────────────────────────────────────────────────────────

def run_cover_letter_to_portfolio(pdf_s3_url: str, user_id: int | None = None, top_k: int = TOP_K_FINAL) -> dict:
    from docling.document_converter import DocumentConverter
    from app.chunkers import cover_letter as cl_chunker
    from app.exporters.portfolio_gen_pdf import save_generated_portfolio_pdf

    tmp_path = download_pdf_to_temp(pdf_s3_url)
    out_pdf  = make_output_path("cl_to_portfolio")
    try:
        converter = DocumentConverter()
        text      = converter.convert(tmp_path).document.export_to_text()
        source    = "cover_letter"

        logger.info("[RAG-2] 자소서 청킹 중...")
        chunks = cl_chunker.chunk(text, source)
        logger.info("[RAG-2] 청킹 완료: %d개 청크", len(chunks))

        _PROJECT_CATEGORIES = {"직무역량", "문제해결경험", "프로젝트경험"}
        _CAREER_PROJECT_KW  = {"프로젝트", "구현", "개발", "서비스", "시스템", "앱", "웹", "API"}

        def _is_target(c: dict) -> bool:
            cat = c.get("category", "")
            if cat in _PROJECT_CATEGORIES:
                return True
            if cat == "경력":
                return any(kw in c.get("text", "") for kw in _CAREER_PROJECT_KW)
            return False

        chunks  = [c for c in chunks if _is_target(c)]
        client  = _get_gemini_client()
        results = []

        for idx, chunk in enumerate(chunks):
            logger.info("[RAG-2] [%d/%d] 포트폴리오 섹션 생성: %s / %s", idx + 1, len(chunks), chunk.get("section", ""), chunk.get("category", ""))
            refs = _search_portfolio_refs(chunk, top_k=top_k)
            gen  = _generate_portfolio_section(chunk, refs, client)

            full_text  = "\n\n".join(filter(None, [
                gen.sections.overview,
                gen.sections.development,
                gen.sections.issue,
                gen.sections.result,
            ]))
            eval_result = None
            if len(full_text.strip()) >= 50:
                from app.evaluators.portfolio import evaluate as pf_evaluate
                try:
                    eval_result = pf_evaluate(full_text, meta={
                        "period":     gen.period,
                        "role":       gen.role,
                        "team":       gen.team,
                        "tech_stack": gen.tech_stack,
                    })
                except Exception:
                    pass

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

        logger.info("[RAG-2] PDF 생성 중...")
        save_generated_portfolio_pdf(results, out_pdf)
        output_s3_url = upload_pdf_file(out_pdf, user_id)
        return {"sections": results, "outputPdfS3Url": output_s3_url}
    finally:
        cleanup_files(tmp_path, out_pdf)


# ───────────────────────────────────────────────────────────────
# BackgroundTask 래퍼
# ───────────────────────────────────────────────────────────────

async def run_portfolio_to_cover_letter_task(
    job_id: str,
    pdf_s3_url: str,
    user_id: int | None = None,
    top_k: int = TOP_K_FINAL,
) -> None:
    await run_job_pipeline(
        job_id,
        lambda: run_portfolio_to_cover_letter(pdf_s3_url, user_id=user_id, top_k=top_k),
        tag="RAG-3",
    )


async def run_cover_letter_to_portfolio_task(
    job_id: str,
    pdf_s3_url: str,
    user_id: int | None = None,
    top_k: int = TOP_K_FINAL,
) -> None:
    await run_job_pipeline(
        job_id,
        lambda: run_cover_letter_to_portfolio(pdf_s3_url, user_id=user_id, top_k=top_k),
        tag="RAG-2",
    )


async def run_cover_letter_from_pdf_task(
    job_id: str,
    pdf_s3_url: str,
    user_id: int | None = None,
    top_k: int = TOP_K_FINAL,
) -> None:
    await run_job_pipeline(
        job_id,
        lambda: run_cover_letter_from_pdf(pdf_s3_url, user_id=user_id, top_k=top_k),
        tag="RAG-1",
    )
