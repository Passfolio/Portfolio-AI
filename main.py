import os
import json
import time
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from pydantic import BaseModel
from google import genai
from google.genai import types
from docling.document_converter import DocumentConverter
from chunkers import cover_letter

# llm 활용 자소서 json 변형

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

PDF_DIR = Path(__file__).parent / "pdfsample"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

REFERENCE_PDFS = [
    PDF_DIR / "에코인사이트글로벌_웹개발.pdf",
    PDF_DIR / "엔테크서비스_풀스택 웹개발자.pdf",
]

CATEGORIES = [
    "지원동기", "직무역량", "문제해결경험", "협업태도",
    "자기소개", "취미", "프로젝트경험", "경력", "기타"
]

converter = DocumentConverter()

RPM_DELAY = 13  # 분당 4회 (5 RPM 미만 유지)


# ── Gemini 응답 스키마 ─────────────────────────────────────────────────
class ChunkExtraction(BaseModel):
    category: str
    key_points: list[str]
    achievements: list[str]
    keywords: list[str]


def extract_with_gemini(text: str, retries: int = 3) -> dict:
    prompt = f"""다음 자기소개서 텍스트를 분석해서 JSON으로 변환해주세요.

category는 반드시 다음 중 하나만 선택: {", ".join(CATEGORIES)}
key_points: 핵심 내용 2~3문장
achievements: 수치나 성과가 포함된 문장 (없으면 빈 배열)
keywords: 기술스택, 역량 키워드

텍스트:
{text}"""

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ChunkExtraction,
                ),
            )
            return json.loads(response.text)
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                wait = 60
                print(f"    ⚠ Rate limit. {wait}초 대기 후 재시도 ({attempt+1}/{retries})...")
                time.sleep(wait)
            else:
                raise


# ── 메인 ──────────────────────────────────────────────────────────────
def run():
    all_chunks = []

    for pdf_path in REFERENCE_PDFS:
        print(f"\n{'='*60}")
        print(f"파일: {pdf_path.name}")
        print('='*60)

        raw_text = converter.convert(str(pdf_path)).document.export_to_text()
        chunks = cover_letter.chunk(raw_text, pdf_path.stem)
        print(f"청킹 완료: {len(chunks)}개 → Gemini 추출 시작\n")

        for i, c in enumerate(chunks):
            print(f"  [{i+1}/{len(chunks)}] {c['sub_section'][:40]} 처리 중...")
            extracted = extract_with_gemini(c["text"])

            full_chunk = {
                "id": f"{c['source']}_{i:03d}",
                "source": c["source"],
                "doc_type": c["doc_type"],
                "category": extracted["category"],
                "sub_section": c["sub_section"],
                "keywords_str": ", ".join(extracted["keywords"]),
                "text": c["text"],
                "key_points": extracted["key_points"],
                "achievements": extracted["achievements"],
                "keywords": extracted["keywords"],
                "char_count": c["char_count"],
            }
            all_chunks.append(full_chunk)
            time.sleep(RPM_DELAY)  # 분당 요청 제한 준수

    # JSON 저장
    output_path = OUTPUT_DIR / "chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    # 결과 출력
    print(f"\n{'='*60}")
    print(f"저장 완료: {output_path}  ({len(all_chunks)}개 청크)")
    print('='*60)

    for c in all_chunks:
        print(f"\n[{c['id']}]")
        print(f"  category    : {c['category']}")
        print(f"  sub_section : {c['sub_section']}")
        print(f"  key_points  : {c['key_points']}")
        print(f"  achievements: {c['achievements']}")
        print(f"  keywords    : {c['keywords']}")

    return all_chunks


if __name__ == "__main__":
    run()
