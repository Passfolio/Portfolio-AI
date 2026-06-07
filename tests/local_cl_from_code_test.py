"""
로컬 코드분석 → 자소서 RAG 테스트
코드분석 CDN URL → 직무역량 + 문제해결경험 자소서 섹션 생성

사용법:
  python -m tests.local_cl_from_code_test
"""
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

CODE_ANALYSIS_URL = "https://cdn.passfolio.dev/analyses/hooby/deokive-be-youcu-1/final.json"
JOB    = "백엔드"          # 백엔드 / 프론트엔드 / 풀스택
CAREER = "신입"  # 신입 / 주니어(1~3년) / 미드레벨(3~5년)
OUT_PREFIX = "output_cl_from_code"


def main():
    from app.services._rag_utils import _fetch_code_analyses
    from app.services.cover_letter import run_code_analysis_to_cover_letter

    print("[코드분석 fetch 중...]")
    code_analyses = _fetch_code_analyses([CODE_ANALYSIS_URL])
    print(f"  → {len(code_analyses)}개 로드 완료\n")

    if not code_analyses:
        print("코드 분석 fetch 실패. URL 및 내부 API 키를 확인하세요.")
        return

    ca = code_analyses[0]
    print(f"  프로젝트: {ca.get('service_name', '')}")
    print(f"  설명:     {ca.get('service_description', '')}\n")

    out_pdf  = f"{OUT_PREFIX}.pdf"
    out_json = f"{OUT_PREFIX}.json"

    print(f"[자소서 생성 중... job={JOB}, career={CAREER}]")
    result = run_code_analysis_to_cover_letter(
        code_analyses=code_analyses,
        job=JOB,
        career=CAREER,
        out_pdf=out_pdf,
    )

    Path(out_json).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n  JSON → {out_json}")

    for sec in result.get("sections", []):
        weighted = sec.get("eval", {}).get("weighted", 0)
        print(f"  [{sec['label']}] {sec['char_count']}자  평가: {weighted:.1f}/100")

    if Path(out_pdf).exists():
        print(f"  PDF  → {out_pdf}")


if __name__ == "__main__":
    main()
