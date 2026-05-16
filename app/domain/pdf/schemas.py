from dataclasses import dataclass


@dataclass
class PdfSection:
    section_title: str
    section_type: str      # "project" | "skill" | "experience" | "education" | "general"
    plain_text: str        # 임베딩용 — 마크업 제거
    markdown: str          # Gemini 입력용 — 구조 보존
    page: int
    order: int
    parent_title: str | None
    token_count: int
