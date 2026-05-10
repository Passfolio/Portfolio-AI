"""
portfolio_chunker.py
────────────────────────────────────────────────────────────
Docling 기반 포트폴리오 PDF → 프로젝트 단위 청킹

파이프라인:
  1. PyMuPDF로 스캔본 여부 자동 감지
  2. Docling 배치 변환 (BATCH_SIZE 페이지씩, bad_alloc 방지)
  3. 병합 마크다운 → Gemini LLM으로 프로젝트 섹션 분리 + 메타 동시 추출
  4. 섹션별 청크 dict 반환

수정 사항 (v2):
  - meta를 정규식 대신 LLM이 직접 추출 → 형식에 무관한 정확한 값 반환
  - <!-- image --> 태그 및 연속 빈 줄 후처리 제거

OCR 전략:
  - 텍스트 PDF  → OCR 생략
  - 스캔본 PDF  → Tesseract (LSTM 엔진) + 전처리 파이프라인

전처리 파이프라인 (저해상도/CPU 특화):
  1. 130 DPI 렌더링
  2. 1.5배 업스케일 (INTER_CUBIC)
  3. 노이즈 제거 (fastNlMeansDenoising, h=7)
  4. CLAHE 적응형 히스토그램 균등화
  5. 다크 배경 자동 감지 후 반전
  6. 적응형 이진화 (ADAPTIVE_THRESH_GAUSSIAN_C, blockSize=15)
  7. 형태학적 획 복원 (MORPH_CLOSE 2×2)
"""

from __future__ import annotations

import gc
import json
import os
import re
import time
from pathlib import Path
from typing import Optional
import html

import cv2
import fitz  # PyMuPDF
import numpy as np
from pydantic import BaseModel

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.datamodel.base_models import InputFormat
from google import genai as _genai
from google.genai import types as _types


# ═══════════════════════════════════════════════════════════════
# 설정값
# ═══════════════════════════════════════════════════════════════

IMAGES_SCALE = 0.7
BATCH_SIZE   = 5
RENDER_DPI   = 130

TESSERACT_CMD        = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
_TEXT_CHAR_THRESHOLD = 50
MIN_CHUNK_CHARS      = 80

_LLM_MODEL   = "gemini-3.1-flash-lite"
_LLM_RETRIES = 3


# ═══════════════════════════════════════════════════════════════
# 1. 스캔본 자동 감지
# ═══════════════════════════════════════════════════════════════

def is_scanned_pdf(pdf_path: str, sample_pages: int = 3) -> bool:
    doc = fitz.open(pdf_path)
    pages_to_check = min(sample_pages, len(doc))
    scanned_count = sum(
        1 for i in range(pages_to_check)
        if len(doc[i].get_text().strip()) < _TEXT_CHAR_THRESHOLD
    )
    doc.close()
    return scanned_count >= pages_to_check / 2


def get_page_count(pdf_path: str) -> int:
    doc = fitz.open(pdf_path)
    n = len(doc)
    doc.close()
    return n


# ═══════════════════════════════════════════════════════════════
# 2. OCR 전처리 파이프라인
# ═══════════════════════════════════════════════════════════════

def _detect_dark_background(gray: np.ndarray) -> bool:
    return float(np.mean(gray)) < 127


def _ocr_preprocess(gray: np.ndarray) -> np.ndarray:
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.fastNlMeansDenoising(gray, h=7)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    if _detect_dark_background(gray):
        gray = cv2.bitwise_not(gray)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=10,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def preprocess_pdf_to_image_pdf(src_path: str, dst_path: str) -> None:
    src = fitz.open(src_path)
    dst = fitz.open()
    mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
    for page in src:
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        processed = _ocr_preprocess(img)
        h, w = processed.shape
        new_page = dst.new_page(width=w, height=h)
        _, buf = cv2.imencode(".png", processed)
        new_page.insert_image(new_page.rect, stream=buf.tobytes())
    dst.save(dst_path)
    src.close()
    dst.close()


# ═══════════════════════════════════════════════════════════════
# 3. 컨버터 팩토리
# ═══════════════════════════════════════════════════════════════

def _make_converter(use_ocr: bool) -> DocumentConverter:
    pipeline = PdfPipelineOptions()
    pipeline.images_scale = IMAGES_SCALE
    if use_ocr:
        pipeline.do_ocr = True
        pipeline.do_table_structure = False
        pipeline.ocr_options = TesseractCliOcrOptions(
            lang=["kor", "eng"],
            tesseract_cmd=TESSERACT_CMD,
            psm=6,
        )
    else:
        pipeline.do_ocr = False
        pipeline.do_table_structure = True
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline)}
    )


_converter_no_ocr: Optional[DocumentConverter] = None
_converter_ocr:    Optional[DocumentConverter] = None


def _get_converter(pdf_path: str) -> tuple[DocumentConverter, bool]:
    global _converter_no_ocr, _converter_ocr
    scanned = is_scanned_pdf(pdf_path)
    if scanned:
        print(f"[OCR ] 스캔본 감지 → Tesseract + 전처리: {Path(pdf_path).name}")
        if _converter_ocr is None:
            _converter_ocr = _make_converter(use_ocr=True)
        return _converter_ocr, True
    else:
        print(f"[SKIP] 텍스트 PDF 감지 → OCR 생략: {Path(pdf_path).name}")
        if _converter_no_ocr is None:
            _converter_no_ocr = _make_converter(use_ocr=False)
        return _converter_no_ocr, False


# ═══════════════════════════════════════════════════════════════
# 4. 배치 변환 → 마크다운 문자열 반환
# ═══════════════════════════════════════════════════════════════

def _convert_in_batches(
    pdf_path: str,
    converter: DocumentConverter,
    use_ocr: bool = True,
) -> str:
    """BATCH_SIZE 페이지씩 변환 후 마크다운 문자열로 병합."""
    total = get_page_count(pdf_path)

    if total <= BATCH_SIZE:
        if use_ocr:
            tmp_pre = Path(pdf_path).with_suffix(".tmp_pre.pdf")
            try:
                preprocess_pdf_to_image_pdf(pdf_path, str(tmp_pre))
                return converter.convert(str(tmp_pre)).document.export_to_markdown()
            finally:
                if tmp_pre.exists():
                    tmp_pre.unlink()
        else:
            return converter.convert(pdf_path).document.export_to_markdown()

    print(f"[BATCH] 총 {total}페이지 → {BATCH_SIZE}페이지씩 분할 처리")
    markdown_parts: list[str] = []
    src_doc = fitz.open(pdf_path)

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        print(f"  처리 중: {start + 1}~{end}페이지 / {total}")

        tmp_raw = Path(pdf_path).with_suffix(f".tmp_{start}_raw.pdf")
        tmp_pre = Path(pdf_path).with_suffix(f".tmp_{start}_pre.pdf")

        try:
            sub = fitz.open()
            sub.insert_pdf(src_doc, from_page=start, to_page=end - 1)
            sub.save(str(tmp_raw))
            sub.close()

            if use_ocr:
                preprocess_pdf_to_image_pdf(str(tmp_raw), str(tmp_pre))
                target = str(tmp_pre)
            else:
                target = str(tmp_raw)

            md = converter.convert(target).document.export_to_markdown()
            if md.strip():
                markdown_parts.append(md.strip())

        except Exception as e:
            print(f"  [WARN] {start + 1}~{end}페이지 처리 실패 (건너뜀): {e}")
        finally:
            for p in (tmp_raw, tmp_pre):
                if p.exists():
                    p.unlink()
            gc.collect()

    src_doc.close()

    if not markdown_parts:
        raise RuntimeError("모든 배치 변환 실패 — 텍스트를 추출할 수 없습니다.")

    return "\n\n".join(markdown_parts)


# ═══════════════════════════════════════════════════════════════
# 5. 텍스트 후처리
# ═══════════════════════════════════════════════════════════════

_RE_IMAGE_TAG    = re.compile(r"<!--\s*image\s*-->", re.I)
_RE_BLANK_LINES  = re.compile(r"\n{3,}")
# 한글 자모/완성형 문자 사이의 단일 공백 제거 (자간 넓은 디자인 폰트 OCR 오류 복원)
_RE_KO_SPACE     = re.compile(r"(?<=[가-힣ᄀ-ᇿ㄰-㆏]) (?=[가-힣ᄀ-ᇿ㄰-㆏])")


def _fix_korean_spacing(text: str) -> str:
    """OCR이 자간을 단어 경계로 잘못 분리한 한글 공백을 반복 제거."""
    prev = None
    while prev != text:
        prev = text
        text = _RE_KO_SPACE.sub("", text)
    return text


def _clean_text(text: str) -> str:
    text = _RE_IMAGE_TAG.sub("", text)
    text = _fix_korean_spacing(text)
    text = _RE_BLANK_LINES.sub("\n\n", text)
    text = html.unescape(text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════
# 6. Gemini LLM 기반 섹션 분리 + 메타 추출 (1-pass)
# ═══════════════════════════════════════════════════════════════

class _Meta(BaseModel):
    """프로젝트 섹션에서 LLM이 직접 추출하는 메타데이터.

    프로젝트가 아닌 섹션(자기소개, 기술스택 등)은 str 필드는 "", list 필드는 []로 반환.
    """
    period:        str        # 기간 (예: "2024.03 ~ 2025.11"). 없으면 ""
    role:          str        # 역할/담당 (예: "풀스택 개발"). 없으면 ""
    team:          str        # 팀 구성. 텍스트에 명시된 경우만 기재. 없으면 반드시 ""
    tech_stack:    list[str]  # 기술 스택 목록. 없으면 []
    contributions: list[str]  # 본인이 직접 수행한 역할/구현 내용 (수치 없는 역할). 없으면 []
    achievements:  list[str]  # 수치 포함 정량적 성과 또는 명확한 결과. 없으면 []
    keywords:      list[str]  # 기술·직무·도메인 키워드. 없으면 []


class _Section(BaseModel):
    section:    str   # 상위 분류 (프로젝트경험 / 기술스택 / 자기소개 / 경력 / 기타)
    project:    str   # 세부 프로젝트명 또는 섹션명
    start_line: int   # 섹션 시작 줄 번호 (1-based)
    end_line:   int   # 섹션 끝 줄 번호 (1-based, inclusive)
    meta:       _Meta # ← LLM이 섹션 텍스트를 읽고 직접 추출


class _SectionList(BaseModel):
    sections: list[_Section]


def _number_lines(markdown: str) -> tuple[str, list[str]]:
    """마크다운에 줄 번호를 붙여 LLM 전달용 문자열과 원본 줄 목록을 반환."""
    lines = markdown.split("\n")
    numbered = "\n".join(f"{i + 1:04d} | {line}" for i, line in enumerate(lines))
    return numbered, lines


def _slice_lines(lines: list[str], start: int, end: int) -> str:
    """1-based inclusive 범위로 원본 줄을 슬라이싱."""
    s = max(0, start - 1)
    e = min(len(lines), end)
    return "\n".join(lines[s:e]).strip()


def _gemini_split_sections(markdown: str, source: str) -> list[dict]:
    """Gemini로 포트폴리오 마크다운을 프로젝트/섹션 단위로 분리하고 메타도 함께 추출."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")

    client = _genai.Client(api_key=api_key)
    numbered_md, raw_lines = _number_lines(markdown)
    total_lines = len(raw_lines)

    prompt = f"""다음은 포트폴리오 PDF에서 추출한 마크다운입니다. 각 줄 앞에 줄 번호가 붙어 있습니다.
이 텍스트를 읽고 프로젝트/섹션 단위로 분리하고, 각 섹션의 메타데이터도 함께 추출해주세요.

[section 선택 기준]
- 프로젝트경험: 개발/기획/디자인 등 프로젝트 경험
- 기술스택: 보유 기술, 언어, 프레임워크 목록
- 자기소개: 프로필, 소개, 목표
- 경력: 인턴, 직장, 대외활동, 수상
- 기타: 위에 해당하지 않는 내용

[분리 기준]
- 각 프로젝트는 독립적인 섹션으로 분리
- 요약 슬라이드와 상세 슬라이드가 같은 프로젝트면 하나의 섹션으로 합침
- start_line과 end_line은 실제 줄 번호(1~{total_lines})여야 하며 누락 없이 커버
- 모든 줄은 정확히 하나의 섹션에 속해야 합니다

[meta 추출 기준] — section이 프로젝트경험일 때만 의미 있게 채우고, 나머지는 str → "" / list → []
- period:        "YYYY.MM ~ YYYY.MM" 형식으로 정제. 없으면 ""
- role:          역할/담당 설명을 간결하게. 없으면 ""
- team:          텍스트에 팀 구성이 명시된 경우에만 기재 (예: "4인 팀"). "개인프로젝트"라고 적혀 있으면 그대로 반환. 명시가 없으면 반드시 "". 추측 금지
- tech_stack:    언어·프레임워크·라이브러리·인프라·툴을 항목별로 추출. 없으면 []
- contributions: 수치 없이 본인이 직접 수행한 역할·구현·설계 내용을 "~구현", "~개발", "~설계" 형태로 항목별 추출. 없으면 []
- achievements:  수치(%, 배수, 건수 등) 포함 정량적 성과 또는 명확한 결과(출시, 수상 등)를 항목별 추출. contributions와 중복 금지. 없으면 []
- keywords:      기술 스택·직무 역량·도메인 키워드를 중복 없이 추출. 없으면 []

마크다운:
{numbered_md}"""

    for attempt in range(_LLM_RETRIES):
        try:
            response = client.models.generate_content(
                model=_LLM_MODEL,
                contents=prompt,
                config=_types.GenerateContentConfig(
                    system_instruction=(
                        "당신은 포트폴리오 문서 구조 분석 전문가입니다. "
                        "줄 번호가 붙은 마크다운을 읽고 섹션 경계(start_line, end_line)와 "
                        "메타데이터(period, role, tech_stack, outcome, team)를 정확히 추출합니다. "
                        "텍스트를 복사하지 말고 줄 번호와 정제된 메타값만 반환하세요."
                    ),
                    response_mime_type="application/json",
                    response_schema=_SectionList,
                ),
            )
            data = json.loads(response.text)
            break
        except Exception as e:
            err = str(e)
            if attempt < _LLM_RETRIES - 1:
                wait = 30 if "429" in err else 5
                print(f"  [WARN] LLM 호출 실패 ({err[:50]}). {wait}초 후 재시도...")
                time.sleep(wait)
            else:
                raise

    chunks: list[dict] = []
    for item in data["sections"]:
        start = int(item["start_line"])
        end   = int(item["end_line"])
        raw   = _slice_lines(raw_lines, start, end)
        text  = _clean_text(raw)           # ← <!-- image --> 제거 + 빈 줄 정리

        if len(text) < MIN_CHUNK_CHARS:
            continue

        # LLM이 반환한 meta dict에서 빈 문자열 필드 제거
        meta_raw: dict = item.get("meta", {})
        meta = {k: v for k, v in meta_raw.items() if isinstance(v, str) and v.strip()}

        chunks.append({
            "source":     source,
            "doc_type":   "portfolio",
            "section":    item["section"],
            "project":    item["project"],
            "text":       text,
            "meta":       meta,
            "char_count": len(text),
        })

    return chunks


# ═══════════════════════════════════════════════════════════════
# 7. 공개 API
# ═══════════════════════════════════════════════════════════════

def chunk(pdf_path: str, source: str) -> list[dict]:
    """PDF → 마크다운 → LLM 섹션 분리 + 메타 추출 → 청크 리스트 반환."""
    converter, use_ocr = _get_converter(pdf_path)
    markdown = _convert_in_batches(pdf_path, converter, use_ocr=use_ocr)

    if not markdown.strip():
        return []

    print(f"  마크다운 추출 완료 ({len(markdown)}자) → LLM 섹션 분리 중...")
    return _gemini_split_sections(markdown, source)


def get_markdown(pdf_path: str) -> str:
    """PDF를 마크다운 문자열로 변환 (디버깅용)."""
    converter, use_ocr = _get_converter(pdf_path)
    return _convert_in_batches(pdf_path, converter, use_ocr=use_ocr)


def get_structure_summary(pdf_path: str) -> str:
    """헤딩 구조만 추출 (split_level 결정 참고용)."""
    from docling_core.types.doc import SectionHeaderItem
    converter, _ = _get_converter(pdf_path)
    doc = converter.convert(pdf_path).document
    lines: list[str] = []
    for item, _ in doc.iterate_items():
        if isinstance(item, SectionHeaderItem):
            level = getattr(item, "level", 1)
            lines.append(f"{'  ' * (level - 1)}H{level}: {item.text.strip()}")
    return "\n".join(lines) if lines else "(헤딩 없음)"


# ═══════════════════════════════════════════════════════════════
# 8. CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()

    pdf = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).parent.parent / "portfoliosample" / "output예시 포폴.pdf"
    )
    mode = sys.argv[2] if len(sys.argv) > 2 else "chunk"

    if mode == "raw":
        print(get_markdown(str(pdf)))

    elif mode == "structure":
        print("=== 헤딩 구조 ===")
        print(get_structure_summary(str(pdf)))

    else:
        results = chunk(str(pdf), source=pdf.stem)
        print(f"\n총 {len(results)}개 청크 생성\n{'=' * 60}")

        output_dir = Path(__file__).parent.parent / "output"
        output_dir.mkdir(exist_ok=True)
        md_path = output_dir / f"{pdf.stem}_chunks.md"

        md_lines = [f"# {pdf.stem} 포트폴리오 청킹 결과\n\n총 {len(results)}개 청크\n"]
        for i, c in enumerate(results, 1):
            md_lines.append(
                f"---\n\n### [{i}/{len(results)}] "
                f"section: {c['section']}  |  project: {c['project']}  |  {c['char_count']}자\n"
            )
            if c.get("meta"):
                for k, v in c["meta"].items():
                    md_lines.append(f"> **{k}**: {v}  ")
                md_lines.append("")
            md_lines.append(c["text"])
            md_lines.append("")

        md_path.write_text("\n".join(md_lines), encoding="utf-8")
        print(f"MD 저장 완료: {md_path}")