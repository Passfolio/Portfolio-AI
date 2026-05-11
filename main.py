import os
import json
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from docling.document_converter import DocumentConverter
from chunkers import resume, cover_letter, portfolio

_converter = DocumentConverter()

PDF_DIR = Path(__file__).parent / "pdfsample"
PORTFOLIO_DIR = Path(__file__).parent / "portfoliosample"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

REFERENCE_PDFS = [
    (PDF_DIR / "삼성전자_DS부문_SW개발.pdf", "cover_letter"),
]

PORTFOLIO_PDFS = [
    PORTFOLIO_DIR / name for name in [
        # "output예시 포폴.pdf",
    ]
]

# ── 메인 ──────────────────────────────────────────────────────────────
def run():
    resume_chunks = []
    cover_letter_chunks = []
    portfolio_chunks = []

    for pdf_path, doc_type in REFERENCE_PDFS:
        print(f"\n{'='*60}")
        print(f"파일: {pdf_path.name}  [{doc_type}]")
        print('='*60)

        if doc_type == "resume":
            chunks = resume.chunk(str(pdf_path), pdf_path.stem)
            if not chunks:
                print("  ⚠ 추출된 텍스트 없음, 건너뜀")
                continue
            print(f"청킹 완료: {len(chunks)}개\n")
            for i, c in enumerate(chunks):
                print(f"  [{i+1}/{len(chunks)}] {c['section']} ({c['char_count']}자)")
                resume_chunks.append({"id": f"{c['source']}_{i:03d}", **c})

        else:
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
    resume_path    = OUTPUT_DIR / "resume_chunks.json"
    cl_path        = OUTPUT_DIR / "coverletter_chunks.json"
    portfolio_path = OUTPUT_DIR / "portfolio_chunks.json"

    with open(resume_path, "w", encoding="utf-8") as f:
        json.dump(resume_chunks, f, ensure_ascii=False, indent=2)
    with open(cl_path, "w", encoding="utf-8") as f:
        json.dump(cover_letter_chunks, f, ensure_ascii=False, indent=2)
    with open(portfolio_path, "w", encoding="utf-8") as f:
        json.dump(portfolio_chunks, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"저장 완료:")
    print(f"  이력서     → {resume_path}  ({len(resume_chunks)}개)")
    print(f"  자소서     → {cl_path}  ({len(cover_letter_chunks)}개)")
    print(f"  포트폴리오 → {portfolio_path}  ({len(portfolio_chunks)}개)")
    print('='*60)

    return resume_chunks, cover_letter_chunks, portfolio_chunks


if __name__ == "__main__":
    run()
