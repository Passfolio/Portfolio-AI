"""
로컬 자소서 → 포트폴리오 RAG 테스트
코드분석 기반 자소서(output_cl_from_code.json)를 입력으로 포트폴리오 생성
1. 코드분석 없이 RAG만으로 포트폴리오 생성
2. 코드분석 포함 RAG 포트폴리오 생성
"""
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv
load_dotenv()

CL_JSON_PATH      = Path(__file__).resolve().parents[1] / "output_cl_from_code.json"
CODE_ANALYSIS_URL = "https://cdn.passfolio.dev/analyses/hooby/deokive-be-youcu-1/final.json"


def _load_cl_text(json_path: Path) -> str:
    """output_cl_from_code.json에서 자소서 본문 텍스트만 추출하여 합침.
    '1. 섹션명' 형식으로 번호를 붙여 2-phase 청커의 섹션 분리를 활성화한다.
    """
    data = json.loads(json_path.read_text())
    parts = []
    for idx, sec in enumerate(data.get("sections", []), 1):
        label = sec.get("label", "")
        text  = sec.get("text", "")
        if text:
            parts.append(f"{idx}. {label}\n{text}")
    return "\n\n".join(parts)


def _run(cl_text: str, code_analyses: list, out_prefix: str) -> None:
    out_pdf  = Path(f"{out_prefix}.pdf")
    out_json = Path(f"{out_prefix}.json")

    with ExitStack() as stack:
        stack.enter_context(patch("app.services.cover_letter.download_pdf_to_temp", return_value="/tmp/fake_cl.pdf"))
        stack.enter_context(patch("app.services.cover_letter.make_output_path",     return_value=str(out_pdf.resolve())))
        stack.enter_context(patch("app.services.cover_letter.upload_pdf_file",      return_value=str(out_pdf.resolve())))
        stack.enter_context(patch("app.services.cover_letter.cleanup_files",        return_value=None))

        MockConverter = stack.enter_context(patch("docling.document_converter.DocumentConverter"))
        MockConverter.return_value.convert.return_value.document.export_to_text.return_value = cl_text

        from app.services.cover_letter import run_cover_letter_to_portfolio
        result = run_cover_letter_to_portfolio(
            pdf_s3_url="dummy",
            user_id=None,
            code_analyses=code_analyses,
        )

    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"  JSON → {out_json}")
    if out_pdf.exists():
        print(f"  PDF  → {out_pdf}")


def main():
    if not CL_JSON_PATH.exists():
        print(f"자소서 JSON 없음: {CL_JSON_PATH}")
        print("먼저 python -m tests.local_cl_from_code_test 를 실행하세요.")
        return

    cl_text = _load_cl_text(CL_JSON_PATH)
    print(f"[자소서 텍스트 로드] {len(cl_text)}자\n")

    from app.services._rag_utils import _fetch_code_analyses
    print("[코드분석 fetch 중...]")
    code_analyses = _fetch_code_analyses([CODE_ANALYSIS_URL])
    print(f"  → {len(code_analyses)}개 로드 완료\n")

    print("=" * 50)
    print("[1] 코드분석 없이 RAG 실행")
    print("=" * 50)
    _run(cl_text, code_analyses=[], out_prefix="output_portfolio_no_code")

    print()
    print("=" * 50)
    print("[2] 코드분석 포함 RAG 실행")
    print("=" * 50)
    _run(cl_text, code_analyses=code_analyses, out_prefix="output_portfolio_with_code")


if __name__ == "__main__":
    main()
