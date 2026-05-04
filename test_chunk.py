"""
청킹 결과 확인용 스크립트. LLM 호출 없이 청크 구조만 출력/저장.

사용법:
  python test_chunk.py resume   <pdf경로>
  python test_chunk.py cl       <pdf경로>
"""
import sys
import json
import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from pathlib import Path
from docling.document_converter import DocumentConverter
from chunkers import resume, cover_letter

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

_converter = DocumentConverter()


def test_resume(pdf_path: str):
    path = Path(pdf_path)
    print(f"\n[이력서 청킹] {path.name}")
    print("=" * 60)

    chunks = resume.chunk(str(path), path.stem)

    for i, c in enumerate(chunks):
        print(f"\n── 청크 {i+1:02d} | section={c['section']} | {c['char_count']}자")
        print(c["text"])
        print()

    out = OUTPUT_DIR / f"{path.stem}_resume_chunks.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\n총 {len(chunks)}개 청크 → {out}")


def test_cover_letter(pdf_path: str):
    path = Path(pdf_path)
    print(f"\n[자소서 청킹] {path.name}")
    print("=" * 60)

    text = _converter.convert(str(path)).document.export_to_text()
    chunks = cover_letter.chunk(text, path.stem)

    for i, c in enumerate(chunks):
        print(f"\n── 청크 {i+1:02d} | section={c['section']} | sub={c['sub_section']} | {c['char_count']}자")
        print(c["text"])
        print()

    out = OUTPUT_DIR / f"{path.stem}_cl_chunks.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\n총 {len(chunks)}개 청크 → {out}")


def resolve_pdf(arg: str) -> str:
    """파일명만 입력해도 pdfsample/ 에서 탐색."""
    candidates = [
        Path(arg),
        Path(__file__).parent / arg,
        Path(__file__).parent / "pdfsample" / arg,
        Path(__file__).parent / "pdfsample" / Path(arg).name,
    ]
    for p in candidates:
        if p.exists():
            return str(p.resolve())

    pdf_dir = Path(__file__).parent / "pdfsample"
    available = [f.name for f in sorted(pdf_dir.glob("*.pdf"))]
    print(f"파일을 찾을 수 없습니다: {arg}")
    print("pdfsample/ 에 있는 파일:")
    for name in available:
        print(f"  {name}")
    sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("사용법: python test_chunk.py [resume|cl] <pdf경로 또는 파일명>")
        print("예시:   python test_chunk.py resume 삼성전자_SW개발.pdf")
        sys.exit(1)

    mode, pdf = sys.argv[1], sys.argv[2]
    pdf = resolve_pdf(pdf)

    if mode == "resume":
        test_resume(pdf)
    elif mode == "cl":
        test_cover_letter(pdf)
    else:
        print(f"알 수 없는 타입: {mode}  (resume 또는 cl)")
        sys.exit(1)
