import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import SectionHeaderItem, TextItem, TableItem, ListItem

pipeline = PdfPipelineOptions()
pipeline.do_table_structure = False
pipeline.do_ocr = False

conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline)})
result = conv.convert("portfoliosample/박중헌_포트폴리오.pdf")
doc = result.document

print("=== 아이템 타입 및 텍스트 ===")
for i, (item, _) in enumerate(doc.iterate_items()):
    t = type(item).__name__
    text = getattr(item, "text", "")
    level = getattr(item, "level", None) if isinstance(item, SectionHeaderItem) else None
    print(f"[{i:03d}] {t:25s} level={level}  text={repr(text[:80])}")
    if i > 60:
        print("... (이하 생략)")
        break

print("\n=== 전체 마크다운 (앞 3000자) ===")
md = doc.export_to_markdown()
print(md[:3000])
