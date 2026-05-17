import google.generativeai as genai

from app.core.config import get_settings


def get_gemini_model(system_instruction: str) -> genai.GenerativeModel:
    genai.configure(api_key=get_settings().gemini_api_key)
    return genai.GenerativeModel(
        model_name="gemini-3-flash-preview",
        system_instruction=system_instruction,
    )
