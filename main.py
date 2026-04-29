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
from chunkers import document

# llm 활용 자소서 json 변형

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

PDF_DIR = Path(__file__).parent / "pdfsample"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

REFERENCE_PDFS = [
    PDF_DIR / "네이버_TechSw개발.pdf",
]

CATEGORIES = [
    "지원동기", "입사포부", "직무역량", "문제해결경험",
    "협업태도", "리더십", "자기소개", "성장과정",
    "취미", "프로젝트경험", "경력", "사회이슈", "기타"
]

converter = DocumentConverter()

RPM_DELAY = 13  # 분당 4회 (5 RPM 미만 유지)


# ── Gemini 응답 스키마 ─────────────────────────────────────────────────
class ChunkExtraction(BaseModel):
    category: str
    key_points: list[str]
    achievements: list[str]
    keywords: list[str]



CATEGORY_DEFINITIONS = {
    "지원동기":    "해당 기업 또는 직무에 지원하게 된 이유, 관심 계기",
    "입사포부":    "입사 후 이루고 싶은 목표, 성장 계획, 기여 방향",
    "직무역량":    "직무 수행에 필요한 전문 기술, 지식, 자격",
    "문제해결경험": "문제를 발견하고 분석하여 해결한 구체적 경험",
    "협업태도":    "팀원과의 소통, 갈등 조율, 협력 방식에 관한 경험",
    "리더십":      "팀을 이끌거나 주도적으로 역할을 맡은 경험",
    "자기소개":    "성격, 장단점, 가치관 등 자신에 대한 소개",
    "성장과정":    "성장 배경, 가치관 형성에 영향을 준 사건이나 인물",
    "취미":        "여가 활동, 관심사",
    "프로젝트경험": "참여한 프로젝트의 역할, 과정, 결과",
    "경력":        "인턴, 직장 경험, 학력, 자격증, 수상 이력",
    "사회이슈":    "사회적 현상이나 이슈에 대한 본인의 견해",
    "기타":        "위 카테고리에 해당하지 않는 내용",
}

CATEGORY_GUIDE = "\n".join(
    f"  - {k}: {v}" for k, v in CATEGORY_DEFINITIONS.items()
)


def extract_with_gemini(text: str, retries: int = 5) -> dict:
    prompt = f"""다음 자기소개서 텍스트를 분석해서 JSON으로 변환해주세요.

[category 선택 기준]
반드시 아래 중 하나만 선택하세요. 각 항목의 정의를 참고하여 문맥에 가장 적합한 카테고리를 고르세요:
{CATEGORY_GUIDE}

[key_points 작성 기준]
- 지원자의 구체적인 행동과 그 결과 중심으로 2~3문장 작성
- 단순 요약이 아닌, 지원자가 무엇을 했고 어떤 변화/성과가 있었는지 명확히 서술

[achievements 작성 기준]
- 수치(%, 배수, 개수, 금액, 시간 등)가 포함된 정량적 성과
- 수치가 없더라도 명확한 결과(도입 완료, 구축 완료, 수주, 채택, 수상, 게재 등)가 있는 문장
- "~를 달성했다", "~로 이어졌다", "~를 수주했다" 등 긍정적 결론이 명시된 문장
- 단순 과정 설명이나 계획은 제외
- 해당 내용이 없으면 빈 배열 []

[keywords 작성 기준]
- 기술 스택(언어, 프레임워크, 툴)
- 직무 역량 키워드(예: 데이터 분석, 알고리즘 최적화)
- 도메인 키워드(예: MLOps, 스마트팩토리, 자율주행)
- 중복 없이 핵심 키워드만 추출

텍스트:
{text}"""

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "당신은 채용 전문가이자 자기소개서 분석 AI입니다. "
                        "지원자의 자기소개서를 읽고 핵심 내용, 성과, 기술 키워드를 정확하게 추출합니다. "
                        "카테고리 분류는 문맥을 깊이 이해하여 가장 적합한 항목을 선택하세요."
                    ),
                    response_mime_type="application/json",
                    response_schema=ChunkExtraction,
                ),
            )
            return json.loads(response.text)
        except Exception as e:
            err = str(e)
            if ("429" in err or "503" in err) and attempt < retries - 1:
                wait = 60
                print(f"    ⚠ 서버 응답 없음({err[:10]}). {wait}초 대기 후 재시도 ({attempt+1}/{retries})...")
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
        chunks = document.chunk(raw_text, pdf_path.stem)
        print(f"청킹 완료: {len(chunks)}개 → Gemini 추출 시작\n")

        for i, c in enumerate(chunks):
            label = f"{c['section']} / {c['sub_section']}"
            print(f"  [{i+1}/{len(chunks)}] {label[:40]} 처리 중...")
            extracted = extract_with_gemini(c["text"])

            full_chunk = {
                "id": f"{c['source']}_{i:03d}",
                "source": c["source"],
                "doc_type": c["doc_type"],
                "section": c["section"],
                "sub_section": c["sub_section"],
                "category": extracted["category"],
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
