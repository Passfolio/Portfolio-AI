import google.generativeai as genai

from app.core.gemini_client import get_gemini_model
from app.domain.analysis.prompts import SYSTEM_PROMPT
from app.domain.analysis.schemas import GeminiOutput, ImprovedSection
from app.domain.pdf.schemas import PdfSection
from app.domain.retrieval.schemas import SearchResult


async def grounded_generate(
    section: PdfSection,
    context: list[SearchResult],
    tech_data: dict | None = None,
) -> ImprovedSection:
    if not context:
        return ImprovedSection(
            section_title=section.section_title,
            section_type=section.section_type,
            order=section.order,
            before=section.markdown,
            after=section.markdown,
            status="no_context",
        )

    user_message = _build_user_message(section, context, tech_data)
    model = get_gemini_model(SYSTEM_PROMPT)
    response = await model.generate_content_async(
        user_message,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=GeminiOutput,
        ),
    )
    output = GeminiOutput.model_validate_json(response.text)
    return ImprovedSection(
        section_title=section.section_title,
        section_type=section.section_type,
        order=section.order,
        before=section.markdown,
        after=output.after,
        status="improved",
    )


def _build_user_message(
    section: PdfSection,
    context: list[SearchResult],
    tech_data: dict | None,
) -> str:
    parts = [
        f"[원본 섹션]\n{section.markdown}",
        "[참고 문맥]\n" + "\n".join(f"{i + 1}. {r.content}" for i, r in enumerate(context)),
    ]
    if tech_data:
        parts.append(f"[기술스택 정보]\n{tech_data}")
    return "\n\n".join(parts)
