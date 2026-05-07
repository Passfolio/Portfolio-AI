import re
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

_pipeline = PdfPipelineOptions()
_pipeline.do_table_structure = True
_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=_pipeline)
    }
)

_STOP_PATTERNS = [
    re.compile(r'자기소개서|자소서|coverletter', re.IGNORECASE),
    re.compile(r'^\d+[\.\)]\s+.{10,}', re.MULTILINE),
]


def _cut_at_cover_letter(markdown: str) -> str:
    """자소서 시작 지점 이후 텍스트 제거."""
    for pat in _STOP_PATTERNS:
        m = pat.search(markdown)
        if m:
            markdown = markdown[:m.start()]
    return markdown.strip()


def chunk(pdf_path: str, source: str) -> list[dict]:
    """
    이력서 PDF를 markdown으로 변환 후 단일 청크 반환.
    Gemini가 섹션 분리와 사실 추출을 한 번에 처리.
    """
    result = _converter.convert(pdf_path)
    markdown = result.document.export_to_markdown()
    markdown = _cut_at_cover_letter(markdown)

    if not markdown:
        return []

    return [{
        "source": source,
        "doc_type": "resume",
        "section": "전체",
        "sub_section": "전체",
        "text": markdown,
        "char_count": len(markdown),
    }]
