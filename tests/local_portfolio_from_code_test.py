"""
로컬 코드분석 → 포트폴리오 RAG 테스트 (RAG-5)
코드분석 CDN URL → 기여 기능별 포트폴리오 섹션 생성

사용법:
  python -m tests.local_portfolio_from_code_test
"""
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

CODE_ANALYSIS_URL = "https://cdn.passfolio.dev/analyses/hooby/deokive-be-youcu-1/final.json"
JOB    = "백엔드"
CAREER = "신입"
OUT_PREFIX = "output_portfolio_from_code"


def main():
    from app.services._rag_utils import _fetch_code_analyses, _code_analyses_to_cl_chunks
    from app.services.cover_letter import run_code_analysis_to_portfolio

    print("[코드분석 fetch 중...]")
    code_analyses = _fetch_code_analyses([CODE_ANALYSIS_URL])
    print(f"  → {len(code_analyses)}개 로드 완료")

    if not code_analyses:
        print("코드 분석 fetch 실패.")
        return

    cl_chunks = _code_analyses_to_cl_chunks(code_analyses)
    print(f"  → 생성 예정 섹션 {len(cl_chunks)}개:")
    for c in cl_chunks:
        print(f"     · {c['section']}")
    print()

    out_pdf  = f"{OUT_PREFIX}.pdf"
    out_json = f"{OUT_PREFIX}.json"

    print(f"[포트폴리오 생성 중... job={JOB}, career={CAREER}]")
    result = run_code_analysis_to_portfolio(
        code_analyses=code_analyses,
        job=JOB,
        career=CAREER,
        out_pdf=out_pdf,
    )

    Path(out_json).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n  JSON → {out_json}")

    for sec in result.get("sections", []):
        eval_w = (sec.get("eval") or {}).get("weighted", "-")
        weighted_str = f"{eval_w:.1f}/100" if isinstance(eval_w, float) else "-"
        print(f"  [{sec['section']}]  project={sec['project']}  평가={weighted_str}")

    if Path(out_pdf).exists():
        print(f"  PDF  → {out_pdf}")


if __name__ == "__main__":
    main()
