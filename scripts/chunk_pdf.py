import os
import json
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from docling.document_converter import DocumentConverter
from app.chunkers import cover_letter, portfolio

_converter = DocumentConverter()

_ROOT = Path(__file__).parent.parent
PDF_DIR = _ROOT / "pdfsample"
PORTFOLIO_DIR = _ROOT / "portfoliosample"
OUTPUT_DIR = _ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

REFERENCE_PDFS = [
    (PDF_DIR / "CJ프레시웨이_IT전략.pdf",             "cover_letter"),
    (PDF_DIR / "네이버_TechSw개발.pdf",               "cover_letter"),
    (PDF_DIR / "네이버_TechSw개발_2.pdf",             "cover_letter"),
    (PDF_DIR / "삼성전자_AI센터_SW개발.pdf",           "cover_letter"),
    (PDF_DIR / "삼성전자_DS부문_SW개발.pdf",           "cover_letter"),
    (PDF_DIR / "삼성전자_SW개발.pdf",                 "cover_letter"),
    (PDF_DIR / "삼성전자_SW개발_2.pdf",               "cover_letter"),
    (PDF_DIR / "삼성증권_IT.pdf",                     "cover_letter"),
    (PDF_DIR / "아이나비시스템즈_AI개발.pdf",           "cover_letter"),
    (PDF_DIR / "에코인사이트글로벌_웹개발.pdf",          "cover_letter"),
    (PDF_DIR / "엔테크서비스_풀스택 웹개발자.pdf",       "cover_letter"),
    (PDF_DIR / "오픈노트_웹기획,개발.pdf",             "cover_letter"),
    (PDF_DIR / "코오롱 인더스트리_시스템개발.pdf",       "cover_letter"),
    (PDF_DIR / "퍼시스 그룹_ITERP 시스템 개발.pdf",     "cover_letter"),
    (PDF_DIR / "펄어비스_웹개발(백엔드).pdf",           "cover_letter"),
    (PDF_DIR / "펄어비스_웹개발[프로그래밍].pdf",        "cover_letter"),
    (PDF_DIR / "한화시스템_서비스 개발_운영.pdf",        "cover_letter"),
]

PORTFOLIO_PDFS = [
    PORTFOLIO_DIR / "output예시 포폴.pdf",
    PORTFOLIO_DIR / "박중헌_포트폴리오.pdf",
]

# ── 메인 ──────────────────────────────────────────────────────────────
def run():
    cover_letter_chunks = []
    portfolio_chunks = []

    for pdf_path, doc_type in REFERENCE_PDFS:
        print(f"\n{'='*60}")
        print(f"파일: {pdf_path.name}  [{doc_type}]")
        print('='*60)

        text = _converter.convert(str(pdf_path)).document.export_to_text()
        chunks = cover_letter.chunk(text, pdf_path.stem)
        print(f"청킹 완료: {len(chunks)}개\n")
        for i, c in enumerate(chunks):
            print(f"  [{i+1}/{len(chunks)}] {c['category']} / {c['section'][:40]} ({c['char_count']}자)")
            cover_letter_chunks.append({"id": f"{c['source']}_{i:03d}", **c})

    for pdf_path in PORTFOLIO_PDFS:
        print(f"\n{'='*60}")
        print(f"파일: {pdf_path.name}  [portfolio]")
        print('='*60)

        chunks = portfolio.chunk(str(pdf_path), pdf_path.stem)
        if not chunks:
            print("  ⚠ 추출된 텍스트 없음, 건너뜀")
            continue
        print(f"청킹 완료: {len(chunks)}개\n")
        for i, c in enumerate(chunks):
            project = c.get("project", c["section"])
            print(f"  [{i+1}/{len(chunks)}] {c['section']} / {project} ({c['char_count']}자)")
            portfolio_chunks.append({"id": f"{c['source']}_{i:03d}", **c})

    # ── JSON 저장 ──────────────────────────────────────────────────────
    cl_path        = OUTPUT_DIR / "coverletter_chunks.json"
    portfolio_path = OUTPUT_DIR / "portfolio_chunks.json"

    with open(cl_path, "w", encoding="utf-8") as f:
        json.dump(cover_letter_chunks, f, ensure_ascii=False, indent=2)
    with open(portfolio_path, "w", encoding="utf-8") as f:
        json.dump(portfolio_chunks, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"저장 완료:")
    print(f"  자소서     → {cl_path}  ({len(cover_letter_chunks)}개)")
    print(f"  포트폴리오 → {portfolio_path}  ({len(portfolio_chunks)}개)")
    print('='*60)

    return cover_letter_chunks, portfolio_chunks


if __name__ == "__main__":
    run()
