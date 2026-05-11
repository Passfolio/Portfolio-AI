"""
portfolio_chunker.py
────────────────────────────────────────────────────────────
Docling 기반 포트폴리오 PDF → 프로젝트 단위 청킹

OCR + 청킹 전략 (환경변수 조합에 따라 자동 선택):
  ① UPSTAGE_API_KEY + GEMINI_API_KEY  → Upstage OCR + Gemini LLM 청킹  (최고 품질)
  ② UPSTAGE_API_KEY only              → Upstage OCR + 규칙 기반 청킹   (OCR 품질 우수)
  ③ GEMINI_API_KEY only               → Docling OCR + Gemini LLM 청킹  (청킹 품질 우수)
  ④ 둘 다 없음                         → Docling OCR + 규칙 기반 청킹   (폴백, 무료)

환경: Windows / CPU 전용 (Docling OCR 사용 시)
Docling OCR 전략:
  - PyMuPDF로 스캔본 여부 자동 감지
  - 텍스트 PDF  → OCR 생략 (빠름)
  - 스캔본 PDF  → Tesseract (LSTM 엔진)
  - 혼합 PDF    → 스캔 페이지 비율로 판단 후 OCR 적용

전처리 파이프라인 (저해상도/CPU 특화):
  1. 130 DPI 렌더링 (72 기본 대비 인식률 개선, 메모리 안전)
  2. 1.5배 업스케일 (INTER_CUBIC)
  3. 노이즈 제거 (fastNlMeansDenoising, h=7)
  4. CLAHE 적응형 히스토그램 균등화
  5. 다크 배경 자동 감지 후 반전
  6. 적응형 이진화 (ADAPTIVE_THRESH_GAUSSIAN_C, blockSize=15)
  7. 형태학적 획 복원 (MORPH_CLOSE 2×2)

메모리 최적화 (std::bad_alloc 방지):
  - IMAGES_SCALE 0.7 : 이미지 해상도 낮춤 (기본 2.0 → 0.7)
  - BATCH_SIZE       : 페이지 단위 배치 처리
  - do_table_structure: 스캔본은 표 구조 분석 생략

설치:
  pip install docling pymupdf rapidocr-onnxruntime opencv-python
  pip install docling-core[chunking] transformers
"""

from __future__ import annotations

import gc
import json
import os
import re
import time
import requests
from pathlib import Path
from typing import Optional

import cv2
import fitz  # PyMuPDF
import numpy as np
from pydantic import BaseModel

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker
from docling_core.types.doc import (
    DoclingDocument,
    SectionHeaderItem,
    TextItem,
    TableItem,
    ListItem,
)
from google import genai as _genai
from google.genai import types as _types


# ═══════════════════════════════════════════════════════════════
# 설정값
# ═══════════════════════════════════════════════════════════════

IMAGES_SCALE = 0.7   # Docling 내부 이미지 스케일 (메모리 절감용)
BATCH_SIZE   = 5     # 한 번에 처리할 최대 페이지 수 (OOM 발생 시 2~3으로 낮출 것)
RENDER_DPI   = 130   # 전처리용 렌더링 DPI (72=기본 / 130=안전 / 150=권장)
                     # 메모리 부족 시 110~120으로 낮추고, 여유 있으면 150까지 올려볼 것

TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

_TEXT_CHAR_THRESHOLD = 50  # 페이지당 텍스트 글자 수 기준 (이하면 스캔본으로 판단)
MIN_CHUNK_CHARS      = 80  # 청크 최소 길이 (이하면 드롭)

_LLM_MODEL   = "gemini-3.1-flash-lite"
_LLM_RETRIES = 3


# ═══════════════════════════════════════════════════════════════
# 1. 스캔본 자동 감지
# ═══════════════════════════════════════════════════════════════

def is_scanned_pdf(pdf_path: str, sample_pages: int = 3) -> bool:
    """앞 sample_pages 페이지의 텍스트량으로 스캔본 여부 판단."""
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
# 2. OCR 전처리 파이프라인 (저해상도 / CPU 특화)
# ═══════════════════════════════════════════════════════════════

def _detect_dark_background(gray: np.ndarray) -> bool:
    """평균 밝기로 다크 배경 여부 판단 (127 미만이면 다크)."""
    return float(np.mean(gray)) < 127


def _ocr_preprocess(gray: np.ndarray) -> np.ndarray:
    """
    저해상도 이미지를 Tesseract 인식에 최적화된 형태로 가공.

    체인 순서:
      업스케일 → 노이즈 제거 → CLAHE → 다크 반전 → 이진화 → 획 복원
    """
    # ── Step 1. 제한적 업스케일 ──────────────────────────────
    # 2.0배는 메모리 부담이 크므로 1.5배로 제한.
    # INTER_CUBIC이 엣지 보존에 유리 (INTER_LINEAR 대비 한글 획 선명도 향상).
    gray = cv2.resize(
        gray, None, fx=1.5, fy=1.5,
        interpolation=cv2.INTER_CUBIC,
    )

    # ── Step 2. 노이즈 제거 ──────────────────────────────────
    # h=7: 낮은 값으로 디테일(얇은 획) 보존 우선.
    # h=10~15로 올리면 노이즈는 더 제거되지만 글자 번짐 위험.
    gray = cv2.fastNlMeansDenoising(gray, h=7)

    # ── Step 3. CLAHE (적응형 히스토그램 균등화) ─────────────
    # 저대비·불균일 조명 슬라이드에서 가장 효과적인 단일 기법.
    # clipLimit=2.0: 과도한 노이즈 증폭 방지 (3.0 이상은 권장 안 함).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # ── Step 4. 다크 배경 감지 후 반전 ──────────────────────
    # 포트폴리오 슬라이드에 다크 테마가 많으므로 반드시 처리.
    if _detect_dark_background(gray):
        gray = cv2.bitwise_not(gray)

    # ── Step 5. 적응형 이진화 ────────────────────────────────
    # blockSize=15: 저해상도에서 11보다 안정적 (너무 크면 디테일 손실).
    # C=10: 배경 노이즈 억제 강도 (값이 클수록 배경 더 날림).
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=10,
    )

    # ── Step 6. 형태학적 획 복원 ─────────────────────────────
    # 저해상도에서 끊긴 한글 획을 연결. 2×2 커널로 과도한 뭉침 방지.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary


def preprocess_pdf_to_image_pdf(src_path: str, dst_path: str) -> None:
    """
    각 페이지를 그레이스케일 이미지로 렌더링 → 전처리 → 새 PDF로 재조립.
    Docling(Tesseract)은 이 전처리된 PDF를 OCR 대상으로 받는다.

    RENDER_DPI=130 선택 이유:
      72dpi(기본)보다 ~3.3배 선명, 150dpi 대비 메모리는 약 25% 절감.
      CPU 노트북 환경에서 속도/품질 균형점.
    """
    src = fitz.open(src_path)
    dst = fitz.open()
    mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)

    for page in src:
        # 그레이스케일로 렌더링 (RGB 대비 메모리 1/3)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width
        )

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
    """OCR 여부에 따라 Docling DocumentConverter 생성."""
    pipeline = PdfPipelineOptions()
    pipeline.images_scale = IMAGES_SCALE

    if use_ocr:
        pipeline.do_ocr = True
        pipeline.do_table_structure = False  # 스캔본은 TableFormer 생략 (메모리↓)
        pipeline.ocr_options = TesseractCliOcrOptions(
            lang=["kor", "eng"],
            tesseract_cmd=TESSERACT_CMD,
        )
    else:
        pipeline.do_ocr = False
        pipeline.do_table_structure = True

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline)
        }
    )


_converter_no_ocr: Optional[DocumentConverter] = None
_converter_ocr:    Optional[DocumentConverter] = None


def _get_converter(pdf_path: str) -> tuple[DocumentConverter, bool]:
    """PDF 특성을 자동 감지해 (컨버터, use_ocr) 반환. 싱글턴으로 재사용."""
    global _converter_no_ocr, _converter_ocr

    scanned = is_scanned_pdf(pdf_path)

    if scanned:
        print(f"[OCR ] 스캔본 감지 → Tesseract + 전처리 배치: {Path(pdf_path).name}")
        if _converter_ocr is None:
            _converter_ocr = _make_converter(use_ocr=True)
        return _converter_ocr, True
    else:
        print(f"[SKIP] 텍스트 PDF 감지 → OCR 생략: {Path(pdf_path).name}")
        if _converter_no_ocr is None:
            _converter_no_ocr = _make_converter(use_ocr=False)
        return _converter_no_ocr, False


# ═══════════════════════════════════════════════════════════════
# 4. 배치 변환 (전처리 포함)
# ═══════════════════════════════════════════════════════════════

def _convert_in_batches(
    pdf_path: str,
    converter: DocumentConverter,
    use_ocr: bool = True,
) -> DoclingDocument:
    """
    BATCH_SIZE 페이지씩 분할 → (OCR시 전처리) → 변환 → 마크다운 병합.

    전처리 흐름 (OCR 모드):
      원본 PDF
        └─ fitz로 N페이지 추출 → tmp_raw.pdf
              └─ preprocess_pdf_to_image_pdf → tmp_pre.pdf
                    └─ Docling(Tesseract) 변환 → 마크다운

    텍스트 모드는 전처리 없이 페이지 분할만 적용 (Docling ML 모델 OOM 방지).
    임시 파일은 배치마다 즉시 삭제해 디스크 사용 최소화.
    """
    total = get_page_count(pdf_path)

    if total <= BATCH_SIZE:
        if use_ocr:
            tmp_pre = Path(pdf_path).with_suffix(".tmp_pre.pdf")
            try:
                preprocess_pdf_to_image_pdf(pdf_path, str(tmp_pre))
                return converter.convert(str(tmp_pre)).document
            finally:
                if tmp_pre.exists():
                    tmp_pre.unlink()
        else:
            return converter.convert(pdf_path).document

    print(f"[BATCH] 총 {total}페이지 → {BATCH_SIZE}페이지씩 분할 처리")

    markdown_parts: list[str] = []
    src_doc = fitz.open(pdf_path)

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        print(f"  처리 중: {start + 1}~{end}페이지 / {total}")

        tmp_raw = Path(pdf_path).with_suffix(f".tmp_{start}_raw.pdf")
        tmp_pre = Path(pdf_path).with_suffix(f".tmp_{start}_pre.pdf")

        try:
            # ① 페이지 분할
            sub = fitz.open()
            sub.insert_pdf(src_doc, from_page=start, to_page=end - 1)
            sub.save(str(tmp_raw))
            sub.close()

            if use_ocr:
                # ② 전처리 적용 (OCR 모드)
                preprocess_pdf_to_image_pdf(str(tmp_raw), str(tmp_pre))
                target = str(tmp_pre)
            else:
                target = str(tmp_raw)

            # ③ Docling 변환
            result = converter.convert(target)
            md = result.document.export_to_markdown()
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

    # 병합 마크다운 → DoclingDocument 재조립
    combined_md = "\n\n".join(markdown_parts)
    tmp_md = Path(pdf_path).with_suffix(".tmp_combined.md")
    try:
        tmp_md.write_text(combined_md, encoding="utf-8")
        return DocumentConverter().convert(str(tmp_md)).document
    finally:
        if tmp_md.exists():
            tmp_md.unlink()



# ═══════════════════════════════════════════════════════════════
# 5. 텍스트 후처리
# ═══════════════════════════════════════════════════════════════

_RE_IMAGE_TAG   = re.compile(r"<!--\s*image\s*-->", re.I)
_RE_BLANK_LINES = re.compile(r"\n{3,}")
_RE_MD_IMAGE   = re.compile(r'!\[.*?\]\([^)]*\)')   # ![image](/image/placeholder)
_RE_CTRL_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')  # 제어문자

def _clean_text(text: str) -> str:
    """청크 텍스트에서 노이즈 제거: image 태그, 연속 빈 줄, HTML 엔티티."""
    import html
    text = _RE_IMAGE_TAG.sub("", text)       # <!-- image -->
    text = _RE_MD_IMAGE.sub("", text)        # ← 추가: ![image](...)
    text = _RE_CTRL_CHARS.sub("", text)      # ← 추가: 제어문자
    text = _RE_BLANK_LINES.sub("\n\n", text)
    text = html.unescape(text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════
# 5-2. LLM 기반 섹션 분리 + 메타 추출 (1-pass, Gemini)
# ═══════════════════════════════════════════════════════════════

class _Meta(BaseModel):
    """LLM이 섹션에서 직접 추출하는 메타데이터."""
    period:     str
    role:       str
    tech_stack: str
    outcome:    str
    team:       str


class _Section(BaseModel):
    section:    str
    project:    str
    start_line: int
    end_line:   int
    meta:       _Meta


class _SectionList(BaseModel):
    sections: list[_Section]


def _number_lines(markdown: str) -> tuple[str, list[str]]:
    """마크다운에 줄 번호를 붙여 LLM 전달용 문자열과 원본 줄 목록 반환."""
    lines = markdown.split("\n")
    numbered = "\n".join(f"{i + 1:04d} | {line}" for i, line in enumerate(lines))
    return numbered, lines


def _slice_lines(lines: list[str], start: int, end: int) -> str:
    """1-based inclusive 범위로 원본 줄 슬라이싱."""
    s = max(0, start - 1)
    e = min(len(lines), end)
    return "\n".join(lines[s:e]).strip()


def _gemini_split_sections(markdown: str, source: str) -> list[dict]:
    """Gemini로 포트폴리오 마크다운을 프로젝트/섹션 단위로 분리하고 메타도 추출 (1-pass)."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")

    client = _genai.Client(api_key=api_key)
    numbered_md, raw_lines = _number_lines(markdown)
    total_lines = len(raw_lines)

    prompt = (
        "다음은 포트폴리오 PDF에서 추출한 마크다운입니다. 각 줄 앞에 줄 번호가 붙어 있습니다.\n"
        "이 텍스트를 읽고 프로젝트/섹션 단위로 분리하고, 각 섹션의 메타데이터도 함께 추출해주세요.\n\n"
        "[section 선택 기준]\n"
        "- 프로젝트경험: 개발/기획/디자인 등 프로젝트 경험\n"
        "- 기술스택: 보유 기술, 언어, 프레임워크 목록\n"
        "- 자기소개: 프로필, 소개, 목표\n"
        "- 경력: 인턴, 직장, 대외활동, 수상\n"
        "- 기타: 위에 해당하지 않는 내용\n\n"
        "[분리 기준]\n"
        "- 각 프로젝트는 독립적인 섹션으로 분리\n"
        "- 요약 슬라이드와 상세 슬라이드가 같은 프로젝트면 하나의 섹션으로 합침\n"
        f"- start_line과 end_line은 실제 줄 번호(1~{total_lines})여야 하며 누락 없이 커버\n"
        "- 모든 줄은 정확히 하나의 섹션에 속해야 합니다\n\n"
        "[meta 추출 기준] — section이 프로젝트경험일 때만 의미 있게 채우고, 나머지는 빈 문자열\n"
        '- period:     "YYYY.MM ~ YYYY.MM" 형식으로 정제. 없으면 ""\n'
        '- role:       역할/담당 설명을 간결하게. 없으면 ""\n'
        '- tech_stack: 핵심 기술을 쉼표로 구분해 요약 (최대 5개). 없으면 ""\n'
        '- outcome:    가장 인상적인 성과 1문장. 없으면 ""\n'
        '- team:       텍스트에 팀 구성이 명시된 경우만 기재. 명시 없으면 반드시 ""\n\n'
        f"마크다운:\n{numbered_md}"
    )

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
        text  = _clean_text(raw)

        if len(text) < MIN_CHUNK_CHARS:
            continue

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
# 6. 유틸리티
# ═══════════════════════════════════════════════════════════════

def _serialize_item(item, doc: Optional[DoclingDocument] = None) -> str:
    if isinstance(item, TableItem):
        try:
            return item.export_to_markdown(doc) if doc is not None else item.export_to_markdown()
        except Exception:
            return "[표]"
    if isinstance(item, (TextItem, ListItem)):
        return item.text.strip()
    if hasattr(item, "text"):
        return item.text.strip()
    return ""


def _heading_level(item) -> Optional[int]:
    if isinstance(item, SectionHeaderItem):
        return getattr(item, "level", 1)
    return None


# ═══════════════════════════════════════════════════════════════
# 7. DoclingDocument → 프로젝트 단위 청크
# ═══════════════════════════════════════════════════════════════

def _build_project_chunks(
    doc: DoclingDocument,
    source: str,
    split_level: int = 2,
) -> list[dict]:
    """
    헤딩 계층에 따라 문서를 프로젝트 단위로 분할.
    split_level=2 기준:
      H1 이하 → section 구분자
      H2      → project 구분자 (청크 경계)
      H3+     → 청크 내부 소제목
    """
    chunks: list[dict] = []
    current_h1      = "포트폴리오"
    current_project = "소개"
    buffer_lines: list[str] = []

    def flush() -> None:
        content = "\n".join(buffer_lines).strip()
        if len(content) < MIN_CHUNK_CHARS:
            return
        chunks.append({
            "source":     source,
            "doc_type":   "portfolio",
            "section":    current_h1,
            "project":    current_project,
            "text":       content,
            "meta":       {},
            "char_count": len(content),
        })

    for item, _ in doc.iterate_items():
        level = _heading_level(item)

        if level is not None and level <= split_level - 1:
            flush()
            buffer_lines.clear()
            current_h1 = item.text.strip()
            buffer_lines.append(f"# {current_h1}")

        elif level is not None and level == split_level:
            flush()
            buffer_lines.clear()
            current_project = item.text.strip()
            buffer_lines.append(f"## {current_project}")

        elif level is not None and level > split_level:
            buffer_lines.append(f"{'#' * level} {item.text.strip()}")

        else:
            serialized = _serialize_item(item, doc)
            if serialized:
                buffer_lines.append(serialized)

    flush()
    return chunks


# ═══════════════════════════════════════════════════════════════
# 8. 큰 청크 추가 분할 (HybridChunker)
# ═══════════════════════════════════════════════════════════════

def _split_oversized(
    chunks: list[dict],
    doc: DoclingDocument,
    max_tokens: int = 8192,
    embed_model_id: str = "BAAI/bge-m3",
) -> list[dict]:
    """토큰 한도를 초과하는 청크를 HybridChunker로 추가 분할."""
    try:
        from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
        from transformers import AutoTokenizer

        tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(embed_model_id),
            max_tokens=max_tokens,
        )
        chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
    except Exception as e:
        print(f"[WARN] HybridChunker 초기화 실패, 원본 청크 반환: {e}")
        return chunks

    result: list[dict] = []
    for chunk in chunks:
        if tokenizer.count_tokens(chunk["text"]) <= max_tokens:
            result.append(chunk)
            continue
        for i, sc in enumerate(chunker.chunk(dl_doc=doc)):
            sub_text = chunker.contextualize(sc).strip()
            if len(sub_text) < MIN_CHUNK_CHARS:
                continue
            result.append({
                **chunk,
                "text":       sub_text,
                "project":    f"{chunk['project']} [{i + 1}]",
                "char_count": len(sub_text),
            })
    return result


# ═══════════════════════════════════════════════════════════════
# 9. Upstage Document Parse
# ═══════════════════════════════════════════════════════════════

def _parse_with_upstage(pdf_path: str, api_key: str) -> str:
    """Upstage Document Parse API로 PDF → 마크다운 변환."""
    with open(pdf_path, "rb") as f:
        resp = requests.post(
            "https://api.upstage.ai/v1/document-digitization",
            headers={"Authorization": f"Bearer {api_key}"},
            files={
                "document": (Path(pdf_path).name, f, "application/pdf"),
            },
            data={
                "model": "document-parse",
                "output_formats": '["markdown"]',
            },
        )
    if not resp.ok:
        print(f"[Upstage 오류] {resp.status_code}: {resp.text}")
        resp.raise_for_status()

    data = resp.json()
    # 응답 구조 확인용 (첫 실행 시)
    print(f"[Upstage] 응답 키: {list(data.keys())}")

    content = data.get("content", {})
    return content.get("markdown") or content.get("html") or content.get("text") or str(content)


def _markdown_to_docling(pdf_path: str, markdown: str) -> "DoclingDocument":
    """마크다운 문자열 → DoclingDocument (기존 청킹 로직 재사용)."""
    tmp_md = Path(pdf_path).with_suffix(".tmp_upstage.md")
    try:
        tmp_md.write_text(markdown, encoding="utf-8")
        return DocumentConverter().convert(str(tmp_md)).document
    finally:
        if tmp_md.exists():
            tmp_md.unlink()


# ═══════════════════════════════════════════════════════════════
# 10. 공개 API
# ═══════════════════════════════════════════════════════════════

def chunk(
    pdf_path: str,
    source: str,
    split_level: int = 2,
    embed_model_id: str = "BAAI/bge-m3",
    max_tokens: int = 8192,
) -> list[dict]:
    """
    PDF를 프로젝트 단위 청크 리스트로 변환.

    Args:
        pdf_path:       처리할 PDF 경로
        source:         청크 메타데이터에 기록할 출처 식별자
        split_level:    청크 경계로 쓸 헤딩 레벨 (기본 H2)
        embed_model_id: 토큰 카운팅용 임베딩 모델 ID
        max_tokens:     청크 최대 토큰 수

    Returns:
        [{"source", "doc_type", "section", "project", "text", "meta", "char_count"}, ...]
    """
    upstage_key = os.getenv("UPSTAGE_API_KEY")
    gemini_key  = os.getenv("GEMINI_API_KEY")

    if upstage_key and gemini_key:
        # ── 최우선: Upstage OCR + Gemini LLM 청킹 ──────────────
        print(f"[Upstage] Document Parse 사용: {Path(pdf_path).name}")
        markdown = _parse_with_upstage(pdf_path, upstage_key)
        print(f"  마크다운 추출 완료 ({len(markdown)}자) → Gemini LLM 섹션 분리 중...")
        return _gemini_split_sections(markdown, source)

    elif upstage_key:
        # ── Upstage OCR + 규칙 기반 청킹 (Gemini 키 없을 때 폴백) ─
        print(f"[Upstage] Document Parse 사용 (규칙 청킹): {Path(pdf_path).name}")
        markdown = _parse_with_upstage(pdf_path, upstage_key)
        doc = _markdown_to_docling(pdf_path, markdown)
        chunks = _build_project_chunks(doc, source, split_level=split_level)
        chunks = _split_oversized(chunks, doc, max_tokens=max_tokens, embed_model_id=embed_model_id)
        return chunks

    elif gemini_key:
        # ── Docling OCR + Gemini LLM 청킹 ──────────────────────
        converter, use_ocr = _get_converter(pdf_path)
        doc = _convert_in_batches(pdf_path, converter, use_ocr=use_ocr)
        markdown = doc.export_to_markdown()
        if not markdown.strip():
            return []
        print(f"  마크다운 추출 완료 ({len(markdown)}자) → Gemini LLM 섹션 분리 중...")
        return _gemini_split_sections(markdown, source)

    else:
        # ── 폴백: Docling OCR + 규칙 기반 청킹 ─────────────────
        converter, use_ocr = _get_converter(pdf_path)
        doc = _convert_in_batches(pdf_path, converter, use_ocr=use_ocr)
        chunks = _build_project_chunks(doc, source, split_level=split_level)
        chunks = _split_oversized(chunks, doc, max_tokens=max_tokens, embed_model_id=embed_model_id)
        return chunks


def get_markdown(pdf_path: str) -> str:
    """PDF를 마크다운 문자열로 변환 (디버깅 / 구조 확인용).
    Upstage API 키가 있으면 Upstage 우선, 없으면 Docling 사용."""
    upstage_key = os.getenv("UPSTAGE_API_KEY")
    if upstage_key:
        return _parse_with_upstage(pdf_path, upstage_key)
    converter, use_ocr = _get_converter(pdf_path)
    return _convert_in_batches(pdf_path, converter, use_ocr=use_ocr)


def get_structure_summary(pdf_path: str) -> str:
    """헤딩 구조만 추출 (split_level 결정 참고용)."""
    converter, _ = _get_converter(pdf_path)
    doc = converter.convert(pdf_path).document
    lines: list[str] = []
    for item, _ in doc.iterate_items():
        level = _heading_level(item)
        if level is not None:
            lines.append(f"{'  ' * (level - 1)}H{level}: {item.text.strip()}")
    return "\n".join(lines) if lines else "(헤딩 없음)"


# ═══════════════════════════════════════════════════════════════
# 10. CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    pdf = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).parent.parent / "portfoliosample" / "output예시 포폴.pdf"
    )
    mode = sys.argv[2] if len(sys.argv) > 2 else "chunk"

    if mode == "raw":
        print(get_markdown(str(pdf)))

    elif mode == "structure":
        print("=== 헤딩 구조 (split_level 결정 참고용) ===")
        print(get_structure_summary(str(pdf)))

    else:
        results = chunk(str(pdf), source=pdf.name)
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