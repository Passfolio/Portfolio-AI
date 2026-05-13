"""
generate_cover_letters.py
─────────────────────────────────────────────────────────────────
IT 직군(프론트엔드/백엔드/풀스택) 자소서 합성 데이터 생성기

Gemini 1-pass로 청크 JSON을 직접 생성 → embed.py에 바로 투입 가능한 포맷 출력
(raw 텍스트 생성 후 별도 청킹 없음, API 호출 1회로 4~5개 청크 생성)

사용법:
    python generate_cover_letters.py
    python generate_cover_letters.py --count 100 --out output/mock_cover_letters.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from google import genai as _genai
from google.genai import types as _types
from pydantic import BaseModel
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════
# 상수
# ═══════════════════════════════════════════════════════════════

_LLM_MODEL = "gemini-3.1-flash-lite"

VALID_CATEGORIES = {
    "지원동기", "입사포부", "직무역량", "문제해결경험",
    "협업태도", "리더십", "자기소개", "성장과정",
    "취미", "프로젝트경험", "경력", "사회이슈", "기타",
}


# ═══════════════════════════════════════════════════════════════
# Pydantic 스키마 (Gemini 구조화 출력용)
# ═══════════════════════════════════════════════════════════════

class _MockChunk(BaseModel):
    section:        str         # 문항 제목 (예: "삼성전자를 선택한 이유")
    category:       str         # 위 VALID_CATEGORIES 중 1개
    sub_categories: list[str]   # 보조 카테고리 (없으면 [])
    key_points:     list[str]   # STAR 구조 2~3문장
    achievements:   list[str]   # 수치 포함 성과 (없으면 [])
    keywords:       list[str]   # 기술/역량 키워드
    text:           str         # 자소서 답변 본문 (300~500자)


class _MockChunkList(BaseModel):
    chunks: list[_MockChunk]


# ═══════════════════════════════════════════════════════════════
# 변수 풀 (다양성 확보)
# ═══════════════════════════════════════════════════════════════

JOBS = {
    "프론트엔드(웹)": {
        "stacks": [
            ["React", "TypeScript", "Next.js", "Tailwind CSS"],
            ["Vue.js", "Nuxt.js", "Pinia", "SCSS"],
            ["React", "Redux", "Webpack", "Jest"],
            ["Svelte", "SvelteKit", "TypeScript"],
            ["Angular", "RxJS", "NgRx", "TypeScript"],
        ],
        "keywords": ["UI/UX", "반응형 웹", "웹 성능 최적화", "크로스브라우징", "접근성", "Core Web Vitals"],
    },
    "프론트엔드(앱)": {
        "stacks": [
            ["React Native", "TypeScript", "Expo"],
            ["Flutter", "Dart", "Riverpod"],
            ["Swift", "UIKit", "SwiftUI"],
            ["Kotlin", "Jetpack Compose", "Coroutines"],
            ["React Native", "Redux", "Firebase"],
        ],
        "keywords": ["모바일 UX", "앱 성능 최적화", "푸시 알림", "오프라인 지원", "앱스토어 배포", "딥링크"],
    },
    "백엔드": {
        "stacks": [
            ["Spring Boot", "Java", "JPA", "MySQL"],
            ["FastAPI", "Python", "SQLAlchemy", "PostgreSQL"],
            ["Node.js", "Express", "TypeScript", "MongoDB"],
            ["Django", "Python", "DRF", "Redis"],
            ["Go", "Gin", "GORM", "PostgreSQL"],
            ["NestJS", "TypeScript", "Prisma", "MySQL"],
        ],
        "keywords": ["RESTful API", "MSA", "DB 설계", "캐싱", "인증/인가", "성능 튜닝", "CI/CD"],
    },
    "풀스택": {
        "stacks": [
            ["React", "Node.js", "TypeScript", "PostgreSQL"],
            ["Next.js", "FastAPI", "Python", "MySQL"],
            ["Vue.js", "Spring Boot", "Java", "MongoDB"],
            ["React", "Django", "Python", "Redis"],
            ["Next.js", "NestJS", "TypeScript", "PostgreSQL"],
        ],
        "keywords": ["풀스택 개발", "API 설계", "배포 자동화", "클라우드", "DevOps", "모노레포"],
    },
}

CAREER_LEVELS = {
    "신입": "비전공/전공 무관 취준생, 부트캠프 또는 학부 졸업 예정자",
    "주니어(1~3년)": "소규모 스타트업 또는 SI 경력 1~3년",
    "미드레벨(3~5년)": "중견 이상 기업 경력 3~5년, 팀 리딩 경험",
}

COMPANY_TYPES = {
    "대기업": {
        "examples": ["삼성전자", "카카오", "네이버", "LG CNS", "SK C&C", "현대오토에버"],
        "tone": "안정성, 대규모 시스템, 조직 문화 기여",
    },
    "스타트업": {
        "examples": ["토스", "당근마켓", "야놀자", "무신사", "직방", "크래프톤"],
        "tone": "빠른 성장, 주도적 역할, 임팩트, 스케일업",
    },
    "중견기업": {
        "examples": ["더존비즈온", "이스트소프트", "NHN", "카페24", "메가존클라우드"],
        "tone": "전문성, 기술 깊이, 안정적 성장",
    },
}

QUESTION_POOLS = {
    "지원동기":    [
        "저희 회사에 지원하게 된 동기를 작성해주세요.",
        "{company}를 선택한 이유와 입사 후 기여하고 싶은 점을 서술하세요.",
        "해당 직무에 지원하게 된 계기와 이유를 설명해주세요.",
    ],
    "직무역량":    [
        "본인의 기술 역량과 직무에 어떻게 기여할 수 있는지 작성해주세요.",
        "지원 직무와 관련된 본인의 핵심 역량을 구체적으로 서술하세요.",
    ],
    "프로젝트경험": [
        "가장 도전적이었던 프로젝트 경험을 작성해주세요.",
        "개발 과정에서 기술적 문제를 해결한 경험을 구체적으로 서술하세요.",
        "팀 프로젝트에서 본인의 역할과 기여한 바를 작성해주세요.",
    ],
    "협업태도":    [
        "팀원과 협업하면서 갈등을 해결한 경험을 작성해주세요.",
        "다양한 직군과 협력했던 경험과 느낀 점을 작성해주세요.",
    ],
    "입사포부":    [
        "입사 후 3년 안에 이루고 싶은 목표를 작성해주세요.",
        "개발자로서의 커리어 목표와 우리 회사에서의 성장 계획을 서술하세요.",
    ],
    "문제해결경험": [
        "개발 중 예상치 못한 문제를 발견하고 해결한 경험을 작성해주세요.",
        "성능 이슈나 장애를 해결한 구체적 경험을 작성해주세요.",
    ],
}

PERSONAS = {
    "names": [
        "김민준", "이서연", "박지훈", "최수아", "정도현",
        "강예진", "조성민", "윤하은", "장태양", "임나리",
        "한승우", "오지은", "송민재", "권아름", "류성현",
    ],
    "universities": [
        "한국대학교 컴퓨터공학과", "서울과학기술대학교 소프트웨어학과",
        "경희대학교 컴퓨터공학부", "숭실대학교 IT학부",
        "세종대학교 컴퓨터공학과", "국민대학교 소프트웨어학부",
        "부트캠프(패스트캠퍼스) 수료", "부트캠프(코드스테이츠) 수료",
        "인하대학교 정보통신공학과", "아주대학교 소프트웨어학과",
    ],
}


# ═══════════════════════════════════════════════════════════════
# 시나리오 조합
# ═══════════════════════════════════════════════════════════════

@dataclass
class Scenario:
    job:          str
    career:       str
    company_type: str
    company:      str
    stack:        list[str]
    questions:    list[tuple[str, str]]   # (category, question_text)
    persona_name: str
    persona_univ: str

    @property
    def source(self) -> str:
        return f"mock_{self.company}_{self.job}_{self.career}"


def _make_scenario() -> Scenario:
    job          = random.choice(list(JOBS.keys()))
    career       = random.choice(list(CAREER_LEVELS.keys()))
    company_type = random.choice(list(COMPANY_TYPES.keys()))
    company      = random.choice(COMPANY_TYPES[company_type]["examples"])
    stack        = random.choice(JOBS[job]["stacks"])

    mandatory = ["지원동기", "입사포부"]
    optional  = random.sample(
        ["직무역량", "프로젝트경험", "협업태도", "문제해결경험"],
        k=random.randint(2, 3),
    )
    categories = mandatory + optional
    random.shuffle(categories)

    questions = [
        (cat, random.choice(QUESTION_POOLS[cat]).replace("{company}", company))
        for cat in categories
    ]

    return Scenario(
        job=job,
        career=career,
        company_type=company_type,
        company=company,
        stack=stack,
        questions=questions,
        persona_name=random.choice(PERSONAS["names"]),
        persona_univ=random.choice(PERSONAS["universities"]),
    )


# ═══════════════════════════════════════════════════════════════
# 프롬프트 빌더
# ═══════════════════════════════════════════════════════════════

_CATEGORY_GUIDE = "\n".join(f"  - {c}" for c in sorted(VALID_CATEGORIES))

_SYSTEM_INSTRUCTION = (
    "당신은 IT 직군 자기소개서 데이터 생성 AI입니다. "
    "주어진 조건에 맞게 자소서 문항별 청크 JSON을 생성합니다. "
    "각 청크의 text는 실제 합격자 수준의 자소서 답변이어야 하며, "
    "category는 반드시 제시된 목록 중에서만 선택하세요."
)


def _build_prompt(s: Scenario) -> str:
    stack_str = ", ".join(s.stack)
    q_block   = "\n".join(
        f"{i+1}. [{cat}] {q}" for i, (cat, q) in enumerate(s.questions)
    )
    company_tone = COMPANY_TYPES[s.company_type]["tone"]

    return (
        f"아래 조건의 취업 준비생 자소서를 문항별 청크 JSON으로 생성해주세요.\n\n"
        f"[지원자]\n"
        f"- 이름: {s.persona_name} / {s.persona_univ}\n"
        f"- 지원 직군: {s.job} ({s.career}, {CAREER_LEVELS[s.career]})\n"
        f"- 기술 스택: {stack_str}\n\n"
        f"[지원 회사]\n"
        f"- {s.company} ({s.company_type}) — 강조 톤: {company_tone}\n\n"
        f"[생성할 문항 — 각 문항이 독립 청크 1개]\n"
        f"{q_block}\n\n"
        f"[각 청크 작성 규칙]\n"
        f"- section: 문항 제목을 간결하게 (예: '삼성전자 지원 동기')\n"
        f"- category: 아래 목록 중 정확히 1개\n"
        f"{_CATEGORY_GUIDE}\n"
        f"- sub_categories: 보조 카테고리 (없으면 빈 배열)\n"
        f"- key_points: STAR 구조(상황→행동→결과) 2~3문장\n"
        f"- achievements: 수치 포함 성과 (없으면 빈 배열)\n"
        f"- keywords: 기술/역량 키워드 3~7개\n"
        f"- text: 실제 자소서 답변 300~500자, 구체적 수치·기술명 포함"
    )


# ═══════════════════════════════════════════════════════════════
# 생성 파이프라인
# ═══════════════════════════════════════════════════════════════

def generate_chunks(client: _genai.Client, scenario: Scenario) -> list[dict]:
    """시나리오 1개 → coverletter_chunks.json 포맷 청크 목록 반환."""
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=_LLM_MODEL,
                contents=_build_prompt(scenario),
                config=_types.GenerateContentConfig(
                    system_instruction=_SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=_MockChunkList,
                ),
            )
            break
        except Exception as e:
            err = str(e)
            if attempt < 4:
                wait = 60 if "429" in err else 10 * (attempt + 1)  # 503: 10→20→30→40초
                tqdm.write(f"    [재시도 {attempt+1}/4] {err[:60]} → {wait}초 대기...")
                time.sleep(wait)
            else:
                raise

    raw_chunks: list[dict] = json.loads(response.text)["chunks"]

    result = []
    for i, c in enumerate(raw_chunks):
        keywords = [k for k in c.get("keywords", []) if k]
        result.append({
            "id":             f"{scenario.source}_{i:03d}",
            "source":         scenario.source,
            "doc_type":       "cover_letter",
            "section":        c["section"],
            "category":       c.get("category", "기타"),
            "sub_categories": [s for s in c.get("sub_categories", []) if s in VALID_CATEGORIES],
            "key_points":     c.get("key_points", []),
            "achievements":   c.get("achievements", []),
            "keywords":       keywords,
            "keywords_str":   ", ".join(keywords),
            "text":           c["text"],
            "char_count":     len(c["text"]),
        })
    return result


def run(total: int, out_path: Path, delay: float = 0.5) -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY 환경변수를 설정하세요.")

    client = _genai.Client(api_key=api_key)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_chunks: list[dict] = []
    # 이어쓰기: 기존 파일이 있으면 로드
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            all_chunks = json.load(f)
        print(f"기존 데이터 {len(all_chunks)}개 로드, 이어서 생성합니다.")

    success = 0
    fail    = 0

    with tqdm(total=total, desc="시나리오 생성 중") as bar:
        for i in range(total):
            scenario = _make_scenario()
            try:
                chunks = generate_chunks(client, scenario)
                all_chunks.extend(chunks)
                # 매 시나리오마다 덮어쓰기 (중간 결과 보존)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(all_chunks, f, ensure_ascii=False, indent=2)
                success += 1
                tqdm.write(f"  ✓ {scenario.source} → {len(chunks)}개 청크")
            except Exception as e:
                fail += 1
                err = str(e)
                wait = 60 if "429" in err else 5
                tqdm.write(f"  [WARN] {i+1}번 실패: {err[:80]}. {wait}초 후 재시도...")
                time.sleep(wait)
            finally:
                bar.update(1)
                bar.set_postfix(success=success, fail=fail, total_chunks=len(all_chunks))
                time.sleep(delay)

    print(f"\n완료: 시나리오 {success}개 성공 / {fail}개 실패")
    print(f"총 청크 수: {len(all_chunks)}개 → {out_path}")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IT 직군 자소서 합성 청크 데이터 생성기")
    parser.add_argument("--count", type=int,   default=50,
                        help="생성할 시나리오 수 (시나리오당 4~5청크, 기본: 50)")
    parser.add_argument("--out",   type=str,   default="output/mock_cover_letters.json",
                        help="출력 JSON 파일 경로")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="요청 간격(초), 기본 0.5")
    args = parser.parse_args()

    run(total=args.count, out_path=Path(args.out), delay=args.delay)
