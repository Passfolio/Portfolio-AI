"""
portfolio_chunker.py
────────────────────────────────────────────────────────────
Docling 기반 포트폴리오 PDF → 프로젝트 단위 청킹

개선 사항:
  - DoclingDocument 계층 구조 직접 순회 (정규식 파싱 제거)
  - 헤딩 레벨 기반 프로젝트 경계 자동 탐지
  - 테이블 / 리스트 / 텍스트 타입별 직렬화
  - 프로젝트 메타데이터 자동 추출 (기간·기술스택·역할·성과)
  - HybridChunker 로 토큰 한도 초과 청크 자동 분할
  - BGE-M3 토크나이저 연동 (max_tokens=8192)
"""

import re
from pathlib import Path
from typing import Optional

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker
from docling_core.types.doc import (
    DoclingDocument,
    SectionHeaderItem,
    TextItem,
    TableItem,
    ListItem,
)

# ── 1. 컨버터 (싱글톤) ──────────────────────────────────────
_pipeline = PdfPipelineOptions()
_pipeline.do_table_structure = True
_pipeline.do_ocr = True
_pipeline.ocr_options = EasyOcrOptions(lang=["ko", "en"])
_pipeline.images_scale = 1.5  # 기본 2.0 → 메모리 절약

_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=_pipeline)
    }
)

# ── 2. 메타데이터 패턴 ──────────────────────────────────────
# 포트폴리오에 자주 등장하는 키워드 → 메타 필드로 분리
_META_PATTERNS = {
    "period":     re.compile(r"(기간|period|duration)[^\n:：]*[:：]?\s*(.+)", re.I),
    "role":       re.compile(r"(역할|담당|role|position)[^\n:：]*[:：]?\s*(.+)", re.I),
    "tech_stack": re.compile(r"(기술\s*스택|tech\s*stack|사용\s*기술|skills?|tools?)[^\n:：]*[:：]?\s*(.+)", re.I),
    "outcome":    re.compile(r"(성과|결과|outcome|achievement|결과물)[^\n:：]*[:：]?\s*(.+)", re.I),
    "team":       re.compile(r"(팀\s*구성|인원|team\s*size|팀원)[^\n:：]*[:：]?\s*(.+)", re.I),
}

MIN_CHUNK_CHARS = 80


# ── 3. 유틸 ─────────────────────────────────────────────────

def _extract_meta(text: str) -> dict:
    """텍스트 블록에서 메타데이터 키-값 추출."""
    meta = {}
    for key, pat in _META_PATTERNS.items():
        m = pat.search(text)
        if m:
            meta[key] = m.group(2).strip()
    return meta


def _serialize_item(item, doc=None) -> str:
    """DoclingDocument 아이템 → 문자열 직렬화."""
    if isinstance(item, TableItem):
        try:
            return item.export_to_markdown(doc) if doc is not None else item.export_to_markdown()
        except Exception:
            return "[표]"

    if isinstance(item, (TextItem, ListItem)):
        return item.text.strip()

    # 기타 (수식, 캡션 등)
    if hasattr(item, "text"):
        return item.text.strip()

    return ""


def _heading_level(item) -> Optional[int]:
    """SectionHeaderItem 이면 레벨(1~6) 반환, 아니면 None."""
    if isinstance(item, SectionHeaderItem):
        return getattr(item, "level", 1)
    return None


# ── 4. 핵심: DoclingDocument → 프로젝트 단위 청크 ────────────

def _build_project_chunks(
    doc: DoclingDocument,
    source: str,
    split_level: int = 2,          # 이 레벨 헤딩에서 프로젝트 경계 인식
) -> list[dict]:
    """
    DoclingDocument를 순회하며 split_level 헤딩마다 새 프로젝트 청크 생성.

    split_level=2  → ## 헤딩이 프로젝트 경계
    split_level=1  → #  헤딩이 프로젝트 경계  (단일 프로젝트가 페이지 전체일 때)

    각 청크 구조:
    {
        "source":      str,          # 파일 경로
        "doc_type":    "portfolio",
        "section":     str,          # 상위(#) 섹션명
        "project":     str,          # 프로젝트명 (## 헤딩)
        "text":        str,          # 본문 전체
        "meta": {
            "period":     str | None,
            "role":       str | None,
            "tech_stack": str | None,
            "outcome":    str | None,
            "team":       str | None,
        },
        "char_count":  int,
    }
    """
    chunks: list[dict] = []

    current_h1 = "포트폴리오"
    current_project = "소개"
    buffer_lines: list[str] = []

    def flush():
        content = "\n".join(buffer_lines).strip()
        if len(content) < MIN_CHUNK_CHARS:
            return
        meta = _extract_meta(content)
        chunks.append({
            "source":     source,
            "doc_type":   "portfolio",
            "section":    current_h1,
            "project":    current_project,
            "text":       content,
            "meta":       meta,
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


# ── 5. HybridChunker로 큰 청크 추가 분할 ────────────────────

def _split_oversized(
    chunks: list[dict],
    doc: DoclingDocument,
    max_tokens: int = 8192,
    embed_model_id: str = "BAAI/bge-m3",
) -> list[dict]:
    """
    HybridChunker를 써서 토큰 초과 청크를 세분화.
    (대부분의 포트폴리오 청크는 8192 이하라 변화 없음)
    """
    try:
        from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
        from transformers import AutoTokenizer

        tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(embed_model_id),
            max_tokens=max_tokens,
        )
        chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
    except Exception as e:
        print(f"[WARN] HybridChunker 초기화 실패, 원본 청크 그대로 반환: {e}")
        return chunks

    result = []
    for chunk in chunks:
        token_count = tokenizer.count_tokens(chunk["text"])
        if token_count <= max_tokens:
            result.append(chunk)
            continue

        # 토큰 초과: HybridChunker로 세분화
        sub_chunks = list(chunker.chunk(dl_doc=doc))
        for i, sc in enumerate(sub_chunks):
            sub_text = chunker.contextualize(sc)
            if len(sub_text.strip()) < MIN_CHUNK_CHARS:
                continue
            result.append({
                **chunk,
                "text":      sub_text.strip(),
                "project":   f"{chunk['project']} [{i+1}]",
                "char_count": len(sub_text.strip()),
            })

    return result


# ── 6. 공개 API ──────────────────────────────────────────────

def chunk(
    pdf_path: str,
    source: str,
    split_level: int = 2,
    embed_model_id: str = "BAAI/bge-m3",
    max_tokens: int = 8192,
) -> list[dict]:
    """
    포트폴리오 PDF → 프로젝트 단위 청크 리스트 반환.

    Parameters
    ----------
    pdf_path      : PDF 파일 경로
    source        : 청크에 태깅할 출처 식별자 (파일명 등)
    split_level   : 이 헤딩 레벨에서 프로젝트 경계 인식 (기본 2 = ##)
    embed_model_id: HuggingFace 토크나이저 모델 ID
    max_tokens    : 청크 최대 토큰 수
    """
    result = _converter.convert(pdf_path)
    doc = result.document

    # Step 1: 문서 구조 기반 프로젝트 청킹
    chunks = _build_project_chunks(doc, source, split_level=split_level)

    # Step 2: 토큰 초과 청크 추가 분할 (선택적)
    chunks = _split_oversized(chunks, doc, max_tokens=max_tokens, embed_model_id=embed_model_id)

    return chunks


def get_markdown(pdf_path: str) -> str:
    """변환된 markdown 반환 (검증용)."""
    result = _converter.convert(pdf_path)
    return result.document.export_to_markdown()


def get_structure_summary(pdf_path: str) -> str:
    """
    문서 헤딩 트리 출력 (청킹 전 split_level 결정에 활용).
    어느 레벨이 프로젝트 경계인지 빠르게 파악 가능.
    """
    result = _converter.convert(pdf_path)
    doc = result.document

    lines = []
    for item, _ in doc.iterate_items():
        level = _heading_level(item)
        if level is not None:
            indent = "  " * (level - 1)
            lines.append(f"{indent}H{level}: {item.text.strip()}")

    return "\n".join(lines) if lines else "(헤딩 없음)"


# ── 7. CLI ───────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json

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

    else:  # "chunk" (기본)
        results = chunk(str(pdf), source=pdf.name)
        print(f"\n총 {len(results)}개 청크 생성\n{'=' * 60}")

        # MD 파일로 저장
        output_dir = Path(__file__).parent.parent / "output"
        output_dir.mkdir(exist_ok=True)
        md_path = output_dir / f"{pdf.stem}_chunks.md"

        md_lines = [f"# {pdf.stem} 포트폴리오 청킹 결과\n\n총 {len(results)}개 청크\n"]
        for i, c in enumerate(results, 1):
            md_lines.append(f"---\n\n### [{i}/{len(results)}] section: {c['section']}  |  project: {c['project']}  |  {c['char_count']}자\n")
            if c.get("meta"):
                for k, v in c["meta"].items():
                    md_lines.append(f"> **{k}**: {v}  ")
                md_lines.append("")
            md_lines.append(c["text"])
            md_lines.append("")

        md_path.write_text("\n".join(md_lines), encoding="utf-8")
        print(f"MD 저장 완료: {md_path}")