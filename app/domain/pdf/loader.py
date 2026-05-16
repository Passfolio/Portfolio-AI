import asyncio
import json
from pathlib import Path

import tiktoken

from app.domain.pdf.schemas import PdfSection

_enc = tiktoken.get_encoding("cl100k_base")

CATEGORY_TYPE_MAP: dict[str, str] = {
    "지원동기": "general",
    "성장과정": "general",
    "장단점": "general",
    "직무역량": "skill",
    "프로젝트": "project",
    "경력": "experience",
    "학력": "education",
}


async def load_pdf(source: str) -> bytes:
    """로컬 파일 읽기. 배포 시 boto3 S3 다운로드로 교체."""
    return await asyncio.to_thread(Path(source).read_bytes)


async def load_mock_json(path: str) -> list[PdfSection]:
    """팀원 JSON 목데이터 → list[PdfSection].

    BE 완료 후 교체 지점:
        pdf_bytes = await load_pdf(source)
        return parse_pdf(pdf_bytes)
    """
    raw = await asyncio.to_thread(Path(path).read_text, encoding="utf-8")
    items: list[dict] = json.loads(raw)

    sections: list[PdfSection] = []
    for i, item in enumerate(items):
        text = item.get("text", "")

        key_points = item.get("key_points", [])
        kp_md = "\n".join(f"- {kp}" for kp in key_points)

        section_md = f"## {item['section']}\n\n{text}"
        if kp_md:
            section_md += f"\n\n**핵심 포인트:**\n{kp_md}"

        ach_list = item.get("achievements", [])
        ach_md = "\n".join(f"- {a}" for a in ach_list)
        if ach_md:
            section_md += f"\n\n**주요 성과:**\n{ach_md}"

        sections.append(
            PdfSection(
                section_title=item["section"],
                section_type=CATEGORY_TYPE_MAP.get(item.get("category", ""), "general"),
                plain_text=text,
                markdown=section_md,
                page=1,
                order=i,
                parent_title=None,
                token_count=len(_enc.encode(text)),
            )
        )

    return sections
