import re

# ── 카테고리 키워드 매핑 ───────────────────────────────────────────────
CATEGORY_KEYWORDS = {
    "지원동기":    ["지원동기", "지원 동기", "성장 방향", "목표", "지원하게"],
    "직무역량":    ["직무역량", "직무 역량", "기술역량", "역량"],
    "문제해결경험": ["문제해결", "문제 해결", "트러블", "개선", "해결"],
    "협업태도":    ["협업", "팀워크", "소통", "성장 태도", "협력"],
    "자기소개":    ["자기소개", "자기 소개", "성격", "장단점", "자신을 표현"],
    "취미":        ["취미"],
    "프로젝트경험": ["프로젝트", "개발 경험", "참여경험", "인턴", "경진대회"],
    "경력":        ["경력", "학력", "자격증", "수상"],
}

# ── 기술 키워드 사전 ───────────────────────────────────────────────────
TECH_KEYWORDS = [
    "Java", "Spring", "Python", "React", "Vue", "Node", "JavaScript",
    "TypeScript", "Docker", "AWS", "Redis", "MySQL", "PostgreSQL",
    "MongoDB", "Git", "CI/CD", "REST", "API", "JPA", "Kotlin",
    "FastAPI", "Django", "Flask", "Kubernetes", "Linux", "JWT",
    "OAuth", "GraphQL", "Nginx", "Jenkins", "GitHub Actions",
]


def classify_category(main_section: str, sub_section: str) -> str:
    """main_section과 sub_section 텍스트로 카테고리 분류."""
    combined = f"{main_section} {sub_section}".lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in combined:
                return category
    return "기타"


def extract_achievements(text: str) -> list[str]:
    """수치나 성과(%, 숫자+단위)가 포함된 문장 추출."""
    sentences = re.split(r'(?<=[.!?])\s+|(?<=다\.)\s+', text)
    result = []
    for s in sentences:
        s = s.strip()
        if s and re.search(r'\d+\s*[%배개명회억원ms초분]', s):
            result.append(s)
    return result


def extract_keywords(text: str) -> list[str]:
    """사전 기반 기술 키워드 추출."""
    found = []
    for kw in TECH_KEYWORDS:
        if kw.lower() in text.lower() and kw not in found:
            found.append(kw)
    return found
