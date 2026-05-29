"""
로컬 PDF 파일로 RAG 기능 테스트 (S3 없이 실행).

사용법:
  python scripts/test_local.py cl_improve       --pdf pdfsample/펄어비스_웹개발(백엔드).pdf
  python scripts/test_local.py cl_improve       --pdf pdfsample/펄어비스_웹개발(백엔드).pdf --no-rag
  python scripts/test_local.py cl_generate      --pdf portfoliosample/박중헌_포트폴리오.pdf --job 백엔드 --career 2년
  python scripts/test_local.py pf_improve       --pdf portfoliosample/박중헌_포트폴리오.pdf
  python scripts/test_local.py cl_to_pf         --pdf pdfsample/펄어비스_웹개발(백엔드).pdf
  python scripts/test_local.py cl_improve_code  --pdf pdfsample/펄어비스_웹개발(백엔드).pdf --code scripts/final-3.json
  python scripts/test_local.py pf_improve_code  --pdf portfoliosample/박중헌_포트폴리오.pdf --code scripts/final-3.json
  python scripts/test_local.py cl_to_pf_code    --pdf pdfsample/펄어비스_웹개발(백엔드).pdf --code scripts/final-3.json

옵션:
  --pdf      로컬 PDF 경로 (생략 시 기본값 사용)
  --job      직군: 백엔드 | 프론트엔드(웹) | 프론트엔드(앱) | 풀스택
  --career   경력: n년 형식 (0년=신입, 1~3년=주니어, 4년이상=미드레벨)
  --no-rag   RAG 없이 프롬프트만으로 실행 (cl_improve 전용)
  --code     코드 분석 JSON 경로 (pf_improve_code / cl_to_pf_code 전용)
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

DEFAULT_CL_PDF = ROOT / "pdfsample" / "펄어비스_웹개발(백엔드).pdf"
DEFAULT_PF_PDF = ROOT / "portfoliosample" / "박중헌_포트폴리오.pdf"


def _to_temp(pdf_path: Path) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    shutil.copy2(str(pdf_path), tmp.name)
    tmp.close()
    return tmp.name


def _save(result: dict, label: str):
    import json
    out = OUTPUT_DIR / f"{label}_result.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장 → {out}")


# ── 1. 자소서 개선 ───────────────────────────────────────────────
def test_cl_improve(pdf: Path, no_rag: bool = False, **_):
    mode = "프롬프트 전용" if no_rag else "RAG 포함"
    print(f"\n[자소서 개선 — {mode}] {pdf.name}")
    from unittest.mock import patch
    from app.services.cover_letter import run_cover_letter_from_pdf

    tmp = _to_temp(pdf)
    try:
        with patch("app.services.cover_letter.download_pdf_to_temp", return_value=tmp), \
             patch("app.services.cover_letter.upload_pdf_file", return_value="local_output.pdf"), \
             patch("app.services.cover_letter.cleanup_files"):
            result = run_cover_letter_from_pdf(pdf_s3_url="local", use_rag=not no_rag)
        label = "cl_improve_no_rag" if no_rag else "cl_improve"
        _save(result, label)
        print(f"섹션 수: {len(result.get('sections', []))}")
    finally:
        os.unlink(tmp)


# ── 2. 포트폴리오 → 자소서 생성 ─────────────────────────────────
def test_cl_generate(pdf: Path, job: str | None, career: str | None, **_):
    from app.services._rag_utils import map_career_input
    career_mapped = map_career_input(career) if career else None
    print(f"\n[포트폴리오 → 자소서 생성] {pdf.name}")
    print(f"  job={job or '미지정'}  career={career or '미지정'}" +
          (f" → {career_mapped}" if career_mapped else ""))

    from unittest.mock import patch
    from app.services.cover_letter import run_portfolio_to_cover_letter

    tmp = _to_temp(pdf)
    try:
        with patch("app.services.cover_letter.download_pdf_to_temp", return_value=tmp), \
             patch("app.services.cover_letter.upload_pdf_file", return_value="local_output.pdf"), \
             patch("app.services.cover_letter.cleanup_files"):
            result = run_portfolio_to_cover_letter(
                pdf_s3_url="local", job=job, career=career_mapped,
            )
        _save(result, "cl_generate")
        print(f"섹션 수: {len(result.get('sections', []))}")
    finally:
        os.unlink(tmp)


# ── 3. 포트폴리오 개선 ───────────────────────────────────────────
def test_pf_improve(pdf: Path, **_):
    print(f"\n[포트폴리오 개선] {pdf.name}")
    from unittest.mock import patch
    from app.services.portfolio import run_portfolio_from_pdf

    tmp = _to_temp(pdf)
    try:
        with patch("app.services.portfolio.download_pdf_to_temp", return_value=tmp), \
             patch("app.services.portfolio.upload_pdf_file", return_value="local_output.pdf"), \
             patch("app.services.portfolio.cleanup_files"):
            result = run_portfolio_from_pdf(pdf_s3_url="local")
        _save(result, "pf_improve")
        print(f"섹션 수: {len(result.get('sections', []))}")
    finally:
        os.unlink(tmp)


# ── 4. 자소서 → 포트폴리오 생성 ─────────────────────────────────
def test_cl_to_pf(pdf: Path, job: str | None, career: str | None, **_):
    from app.services._rag_utils import map_career_input
    career_mapped = map_career_input(career) if career else None
    print(f"\n[자소서 → 포트폴리오 생성] {pdf.name}")
    print(f"  job={job or '미지정'}  career={career or '미지정'}" +
          (f" → {career_mapped}" if career_mapped else ""))

    from unittest.mock import patch
    from app.services.cover_letter import run_cover_letter_to_portfolio

    tmp = _to_temp(pdf)
    try:
        with patch("app.services.cover_letter.download_pdf_to_temp", return_value=tmp), \
             patch("app.services.cover_letter.upload_pdf_file", return_value="local_output.pdf"), \
             patch("app.services.cover_letter.cleanup_files"):
            result = run_cover_letter_to_portfolio(pdf_s3_url="local", job=job, career=career_mapped)
        _save(result, "cl_to_pf")
        print(f"섹션 수: {len(result.get('sections', []))}")
    finally:
        os.unlink(tmp)


# ── 5. 자소서 개선 + 코드 분석 (RAG-7) ─────────────────────────
def test_cl_improve_code(pdf: Path, code: Path | None, job: str | None, career: str | None, **_):
    if code is None:
        print("--code 옵션으로 코드 분석 JSON 경로를 지정하세요.")
        sys.exit(1)
    from app.services._rag_utils import map_career_input
    career_mapped = map_career_input(career) if career else None
    print(f"\n[자소서 개선 + 코드분석] {pdf.name}  code={code.name}")
    print(f"  job={job or '미지정'}  career={career or '미지정'}" +
          (f" → {career_mapped}" if career_mapped else ""))
    code_analysis = json.loads(code.read_text(encoding="utf-8"))

    from unittest.mock import patch
    from app.services.cover_letter import run_cover_letter_from_pdf_with_code

    tmp = _to_temp(pdf)
    try:
        with patch("app.services.cover_letter.download_pdf_to_temp", return_value=tmp), \
             patch("app.services.cover_letter.upload_pdf_file", return_value="local_output.pdf"), \
             patch("app.services.cover_letter.cleanup_files"):
            result = run_cover_letter_from_pdf_with_code(
                pdf_s3_url="local", code_analysis=code_analysis, job=job, career=career_mapped,
            )
        _save(result, "cl_improve_code")
        print(f"섹션 수: {len(result.get('sections', []))}")
    finally:
        os.unlink(tmp)


# ── 6. 포트폴리오 개선 + 코드 분석 (RAG-5) ──────────────────────
def test_pf_improve_code(pdf: Path, code: Path | None, **_):
    if code is None:
        print("--code 옵션으로 코드 분석 JSON 경로를 지정하세요.")
        sys.exit(1)
    print(f"\n[포트폴리오 개선 + 코드분석] {pdf.name}  code={code.name}")
    code_analysis = json.loads(code.read_text(encoding="utf-8"))

    from unittest.mock import patch
    from app.services.portfolio import run_portfolio_from_pdf_with_code

    tmp = _to_temp(pdf)
    try:
        with patch("app.services.portfolio.download_pdf_to_temp", return_value=tmp), \
             patch("app.services.portfolio.upload_pdf_file", return_value="local_output.pdf"), \
             patch("app.services.portfolio.cleanup_files"):
            result = run_portfolio_from_pdf_with_code(pdf_s3_url="local", code_analysis=code_analysis)
        _save(result, "pf_improve_code")
        print(f"섹션 수: {len(result.get('sections', []))}")
    finally:
        os.unlink(tmp)


# ── 6. 자소서 → 포트폴리오 생성 + 코드 분석 (RAG-6) ────────────
def test_cl_to_pf_code(pdf: Path, code: Path | None, job: str | None, career: str | None, **_):
    if code is None:
        print("--code 옵션으로 코드 분석 JSON 경로를 지정하세요.")
        sys.exit(1)
    from app.services._rag_utils import map_career_input
    career_mapped = map_career_input(career) if career else None
    print(f"\n[자소서 → 포트폴리오 생성 + 코드분석] {pdf.name}  code={code.name}")
    print(f"  job={job or '미지정'}  career={career or '미지정'}" +
          (f" → {career_mapped}" if career_mapped else ""))
    code_analysis = json.loads(code.read_text(encoding="utf-8"))

    from unittest.mock import patch
    from app.services.cover_letter import run_cover_letter_to_portfolio_with_code

    tmp = _to_temp(pdf)
    try:
        with patch("app.services.cover_letter.download_pdf_to_temp", return_value=tmp), \
             patch("app.services.cover_letter.upload_pdf_file", return_value="local_output.pdf"), \
             patch("app.services.cover_letter.cleanup_files"):
            result = run_cover_letter_to_portfolio_with_code(
                pdf_s3_url="local", code_analysis=code_analysis, job=job, career=career_mapped,
            )
        _save(result, "cl_to_pf_code")
        print(f"섹션 수: {len(result.get('sections', []))}")
    finally:
        os.unlink(tmp)


# ── 진입점 ───────────────────────────────────────────────────────
COMMANDS = {
    "cl_improve":      (test_cl_improve,      DEFAULT_CL_PDF),
    "cl_generate":     (test_cl_generate,     DEFAULT_PF_PDF),
    "pf_improve":      (test_pf_improve,      DEFAULT_PF_PDF),
    "cl_to_pf":        (test_cl_to_pf,        DEFAULT_CL_PDF),
    "cl_improve_code": (test_cl_improve_code, DEFAULT_CL_PDF),
    "pf_improve_code": (test_pf_improve_code, DEFAULT_PF_PDF),
    "cl_to_pf_code":   (test_cl_to_pf_code,   DEFAULT_CL_PDF),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=list(COMMANDS))
    parser.add_argument("--pdf",    help="로컬 PDF 경로")
    parser.add_argument("--job",    help="직군 (백엔드|프론트엔드(웹)|프론트엔드(앱)|풀스택)")
    parser.add_argument("--career", help="경력 (n년 형식, 예: 2년)")
    parser.add_argument("--no-rag", action="store_true", help="RAG 없이 프롬프트만으로 실행 (cl_improve 전용)")
    parser.add_argument("--code",   help="코드 분석 JSON 경로 (pf_improve_code / cl_to_pf_code 전용)")
    args = parser.parse_args()

    fn, default_pdf = COMMANDS[args.command]
    pdf  = Path(args.pdf)  if args.pdf  else default_pdf
    code = Path(args.code) if args.code else None
    if not pdf.exists():
        print(f"PDF 파일을 찾을 수 없습니다: {pdf}")
        sys.exit(1)
    if code and not code.exists():
        print(f"코드 분석 JSON을 찾을 수 없습니다: {code}")
        sys.exit(1)

    fn(pdf=pdf, job=args.job, career=args.career, no_rag=args.no_rag, code=code)
