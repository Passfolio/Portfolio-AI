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
from chunkers import resume, cover_letter, portfolio

_converter = DocumentConverter()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

PDF_DIR = Path(__file__).parent / "pdfsample"
PORTFOLIO_DIR = Path(__file__).parent / "portfoliosample"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

REFERENCE_PDFS = [
    # (PDF_DIR / "펄어비스_웹개발[프로그래밍].pdf", "cover_letter"),
]

PORTFOLIO_PDFS = [
    PORTFOLIO_DIR / name for name in [
        "박중헌_포트폴리오.pdf",
        # "output예시 포폴.pdf",
    ]
]

RPM_DELAY = 1  # 유료 티어 - 요청 간 최소 딜레이
CHUNK_ONLY = True  # True: 청킹 결과만 저장 (LLM 호출 없음), False: 전체 파이프라인 실행


# ── 자소서 스키마 ──────────────────────────────────────────────────────
class CoverLetterExtraction(BaseModel):
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


# ── 이력서 스키마 ──────────────────────────────────────────────────────
class ResumeSection(BaseModel):
    section: str
    facts: list[str]
    skills: list[str]
    period: str

class ResumeFullExtraction(BaseModel):
    sections: list[ResumeSection]


# ── 포트폴리오 스키마 ──────────────────────────────────────────────────
class PortfolioProject(BaseModel):
    section_type: str        # 프로젝트 / 기술스택 / 자기소개 / 경력 / 기타
    project_name: str
    period: str
    role: str
    tech_stack: list[str]
    summary: str
    contributions: list[str]
    achievements: list[str]
    keywords: list[str]


def _call_gemini(prompt: str, schema: type, system: str, retries: int = 5) -> dict:
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
            return json.loads(response.text)
        except Exception as e:
            err = str(e)
            if attempt < retries - 1:
                if "429" in err:
                    wait = 30
                elif "503" in err:
                    wait = 5
                else:
                    raise
                print(f"    ⚠ 서버 응답 없음({err[:10]}). {wait}초 대기 후 재시도 ({attempt+1}/{retries})...")
                time.sleep(wait)
            else:
                raise


def extract_cover_letter(text: str) -> dict:
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
- 해당 내용이 없으면 빈 배열 []

[keywords 작성 기준]
- 기술 스택(언어, 프레임워크, 툴)
- 직무 역량 키워드(예: 데이터 분석, 알고리즘 최적화)
- 도메인 키워드(예: MLOps, 스마트팩토리, 자율주행)
- 중복 없이 핵심 키워드만 추출

텍스트:
{text}"""

    return _call_gemini(
        prompt=prompt,
        schema=CoverLetterExtraction,
        system=(
            "당신은 채용 전문가이자 자기소개서 분석 AI입니다. "
            "지원자의 자기소개서를 읽고 핵심 내용, 성과, 기술 키워드를 정확하게 추출합니다. "
            "카테고리 분류는 문맥을 깊이 이해하여 가장 적합한 항목을 선택하세요."
        ),
    )


def extract_portfolio(text: str) -> dict:
    prompt = f"""다음은 포트폴리오의 한 섹션입니다. 내용을 분석해서 JSON으로 변환해주세요.

[section_type 선택 기준]
- 프로젝트: 개발/기획/디자인 프로젝트 경험
- 기술스택: 보유 기술, 언어, 프레임워크 목록
- 자기소개: 프로필, 소개, 목표
- 경력: 인턴, 직장, 대외활동, 수상
- 기타: 위에 해당하지 않는 내용

[contributions 작성 기준]
- 본인이 직접 수행한 역할과 구현 내용
- "~구현", "~개발", "~설계", "~담당" 형태로 구체적으로

[achievements 작성 기준]
- 수치(%, 배수, 시간, 건수 등)가 포함된 정량적 성과
- 수치가 없어도 명확한 결과(출시, 수상, 채택 등)가 있으면 포함
- 없으면 빈 배열 []

[tech_stack 작성 기준]
- 언어, 프레임워크, 라이브러리, 인프라, 툴 등
- 텍스트에 언급된 것만 추출, 없으면 빈 배열 []

텍스트:
{text}"""

    return _call_gemini(
        prompt=prompt,
        schema=PortfolioProject,
        system=(
            "당신은 포트폴리오 분석 전문 AI입니다. "
            "지원자의 포트폴리오 섹션을 읽고 프로젝트 정보, 기여 내용, 기술 스택, 성과를 정확하게 추출합니다. "
            "원문에 없는 내용은 절대 추가하지 마세요."
        ),
    )


def extract_resume_full(markdown: str) -> dict:
    prompt = f"""다음은 이력서 전체를 마크다운으로 변환한 텍스트입니다.
섹션별로 사실(fact) 정보를 추출해서 JSON으로 변환해주세요.

[section 작성 기준]
- 인적사항 / 학력사항 / 경력사항 / 수상및활동 / 자격및어학 / 병역사항 / 기타 중 적합한 이름 사용
- 마크다운 헤더나 테이블 헤더를 참고해 판단

[facts 작성 기준]
- 각 항목을 "기간 / 기관 / 내용" 형태로 정리 (해당 정보가 있을 경우)
- 인적사항은 이름, 연락처, 이메일, 주소 등 항목별로 한 줄씩
- 사실 그대로 추출하며 해석이나 요약 금지

[skills 작성 기준]
- 기술 스택, 자격증, 어학 점수 등 역량 관련 항목만 추출
- 해당 없으면 빈 배열 []

[period 작성 기준]
- 해당 섹션의 전체 기간 범위 (예: "2019.03 ~ 2023.02")
- 기간 정보가 없으면 빈 문자열 ""

이력서:
{markdown}"""

    return _call_gemini(
        prompt=prompt,
        schema=ResumeFullExtraction,
        system=(
            "당신은 이력서 파싱 전문 AI입니다. "
            "이력서 전체를 읽고 섹션을 스스로 파악한 뒤, 각 섹션의 사실 정보를 정확하게 추출합니다. "
            "해석이나 평가 없이 원문에 충실하게 추출하세요."
        ),
    )

# ── 메인 ──────────────────────────────────────────────────────────────
def run():
    resume_chunks = []
    cover_letter_chunks = []

    for pdf_path, doc_type in REFERENCE_PDFS:
        print(f"\n{'='*60}")
        print(f"파일: {pdf_path.name}  [{doc_type}]")
        print('='*60)

        if doc_type == "resume":
            md_chunks = resume.chunk(str(pdf_path), pdf_path.stem)
            if not md_chunks:
                print("  ⚠ 추출된 텍스트 없음, 건너뜀")
                continue
            print(f"markdown 변환 완료 → Gemini 1회 호출\n")
            extracted = extract_resume_full(md_chunks[0]["text"])
            for i, sec in enumerate(extracted["sections"]):
                resume_chunks.append({
                    "id": f"{pdf_path.stem}_{i:03d}",
                    "source": pdf_path.stem,
                    "doc_type": "resume",
                    "section": sec["section"],
                    "sub_section": sec["section"],
                    "facts": sec["facts"],
                    "skills": sec["skills"],
                    "period": sec["period"],
                    "char_count": sum(len(f) for f in sec["facts"]),
                })
                print(f"  섹션 추출: {sec['section']}")
            time.sleep(RPM_DELAY)

        else:
            text = _converter.convert(str(pdf_path)).document.export_to_text()
            chunks = cover_letter.chunk(text, pdf_path.stem)
            print(f"청킹 완료: {len(chunks)}개 → Gemini 추출 시작\n")

            for i, c in enumerate(chunks):
                label = f"{c['section']} / {c['sub_section']}"
                print(f"  [{i+1}/{len(chunks)}] {label[:50]} 처리 중...")
                extracted = extract_cover_letter(c["text"])
                cover_letter_chunks.append({
                    "id": f"{c['source']}_{i:03d}",
                    "source": c["source"],
                    "doc_type": "cover_letter",
                    "section": c["section"],
                    "sub_section": c["sub_section"],
                    "category": extracted["category"],
                    "keywords_str": ", ".join(extracted["keywords"]),
                    "text": c["text"],
                    "key_points": extracted["key_points"],
                    "achievements": extracted["achievements"],
                    "keywords": extracted["keywords"],
                    "char_count": c["char_count"],
                })
                time.sleep(RPM_DELAY)

    # ── 포트폴리오 파이프라인 ──────────────────────────────────────────
    portfolio_chunks = []

    for pdf_path in PORTFOLIO_PDFS:
        print(f"\n{'='*60}")
        print(f"파일: {pdf_path.name}  [portfolio]")
        print('='*60)

        chunks = portfolio.chunk(str(pdf_path), pdf_path.stem)
        if not chunks:
            print("  ⚠ 추출된 텍스트 없음, 건너뜀")
            continue
        print(f"청킹 완료: {len(chunks)}개\n")

        if CHUNK_ONLY:
            for i, c in enumerate(chunks):
                project = c.get("project", c["section"])
                print(f"  [{i+1}/{len(chunks)}] section={c['section']} / project={project} ({c['char_count']}자)")
                print(f"    {c['text'][:120].replace(chr(10), ' ')}...")
                print()
            raw_path = OUTPUT_DIR / f"{pdf_path.stem}_raw_chunks.json"
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            print(f"청킹 결과 저장 완료 → {raw_path}")
            continue

        print("Gemini 추출 시작\n")
        for i, c in enumerate(chunks):
            project = c.get("project", c["section"])
            label = f"{c['section']} / {project}"
            print(f"  [{i+1}/{len(chunks)}] {label[:50]} 처리 중...")
            extracted = extract_portfolio(c["text"])
            portfolio_chunks.append({
                "id": f"{c['source']}_{i:03d}",
                "source": c["source"],
                "doc_type": "portfolio",
                "section": c["section"],
                "project": project,
                "section_type": extracted["section_type"],
                "project_name": extracted["project_name"],
                "period": extracted["period"],
                "role": extracted["role"],
                "tech_stack": extracted["tech_stack"],
                "summary": extracted["summary"],
                "contributions": extracted["contributions"],
                "achievements": extracted["achievements"],
                "keywords": extracted["keywords"],
                "char_count": c["char_count"],
            })
            time.sleep(RPM_DELAY)

    # JSON 저장
    resume_path = OUTPUT_DIR / "resume_chunks.json"
    cl_path = OUTPUT_DIR / "coverletter_chunks.json"
    portfolio_path = OUTPUT_DIR / "portfolio_chunks.json"

    with open(resume_path, "w", encoding="utf-8") as f:
        json.dump(resume_chunks, f, ensure_ascii=False, indent=2)
    with open(cl_path, "w", encoding="utf-8") as f:
        json.dump(cover_letter_chunks, f, ensure_ascii=False, indent=2)
    with open(portfolio_path, "w", encoding="utf-8") as f:
        json.dump(portfolio_chunks, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"저장 완료:")
    print(f"  이력서    → {resume_path}  ({len(resume_chunks)}개)")
    print(f"  자소서    → {cl_path}  ({len(cover_letter_chunks)}개)")
    print(f"  포트폴리오 → {portfolio_path}  ({len(portfolio_chunks)}개)")
    print('='*60)

    for c in resume_chunks:
        print(f"\n[{c['id']}] {c['section']}")
        print(f"  facts  : {c['facts']}")
        print(f"  skills : {c['skills']}")
        print(f"  period : {c['period']}")

    for c in cover_letter_chunks:
        print(f"\n[{c['id']}] {c['section']}")
        print(f"  category    : {c['category']}")
        print(f"  key_points  : {c['key_points']}")
        print(f"  achievements: {c['achievements']}")
        print(f"  keywords    : {c['keywords']}")

    for c in portfolio_chunks:
        print(f"\n[{c['id']}] {c['section_type']} / {c['project_name']}")
        print(f"  project       : {c['project']}")
        print(f"  period        : {c['period']}")
        print(f"  role          : {c['role']}")
        print(f"  tech_stack    : {c['tech_stack']}")
        print(f"  contributions : {c['contributions']}")
        print(f"  achievements  : {c['achievements']}")

    return resume_chunks, cover_letter_chunks, portfolio_chunks


if __name__ == "__main__":
    run()
