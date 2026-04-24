from pydantic import BaseModel, Field
from typing import List, Optional

# --- 1. 세부 항목 정의 (Nested Models) ---

class PersonalInfo(BaseModel):
    name: str = Field(..., description="성명")
    contact: str = Field(..., description="연락처 (010-0000-0000)")
    email: str = Field(..., description="이메일 주소")
    RRN: str = Field(..., description="주민등록번호 (YYMMDD-XXXXXXX)")
    address: str = Field(..., description="거주지 주소")

class Education(BaseModel):
    admissionDate: str = Field(..., description="입학일 (YYYY.MM)")
    graduationDate: str = Field(..., description="졸업일 (YYYY.MM)")
    schoolName: str = Field(..., description="학교명")
    major: str = Field(..., description="전공")
    degree: str = Field(..., description="학위 (학사/석사 등)")

class Military(BaseModel):
    status: str = Field(..., description="병역구분 (군필/미필/면제/해당없음)")
    branch: Optional[str] = Field(None, description="군별")
    rank: Optional[str] = Field(None, description="계급")
    period: Optional[str] = Field(None, description="복무 기간")

class Certificate(BaseModel):
    name: str = Field(..., description="자격증/어학시험명")
    date: str = Field(..., description="취득일자")
    issuer: str = Field(..., description="발행처/기관")

class Experience(BaseModel):
    category: str = Field(..., description="경력 종류 (인턴/정규직/프로젝트 등)")
    content: str = Field(..., description="수행 업무 및 성과")
    period: str = Field(..., description="재직 기간")
    location: Optional[str] = Field(None, description="근무지/회사명")

# class SelfIntroduction(BaseModel):
#     # title: str = Field(..., description="자기소개서 문항 제 목 또는 소제목")
#     # content: str = Field(..., description="자기소개서 본문 내용")
#     category: str
#     key_points: list[str]
#     achievements: list[str]
#     keywords: list[str]

# ── [융합 포인트: a.py 스타일 상세 분석 스키마 이식] ──────────────────────────
# 자소서를 단순 텍스트가 아닌 category, achievements, keywords 등 정밀 필드로 정의합니다.
class SelfIntroItem(BaseModel):
    category: str = Field(..., description="지원동기, 직무역량, 문제해결경험, 협업태도, 자기소개, 취미, 프로젝트경험, 경력, 기타 중 택1")
    sub_section: str = Field(..., description="자기소개서 문항의 소제목 또는 질문")
    keywords_str: str = Field(..., description="키워드들을 쉼표로 구분한 문자열")
    text: str = Field(..., description="해당 문항의 원본 텍스트 본문 (절대 요약 금지)")
    key_points: List[str] = Field(..., description="핵심 내용 2~3문장 요약")
    achievements: List[str] = Field(..., description="수치나 구체적 성과가 포함된 문장 (없으면 빈 리스트)")
    keywords: List[str] = Field(..., description="기술 스택 및 주요 역량 키워드 리스트")

# --- 2. 최종 통합 스키마 ---

class ResumeExtraction(BaseModel):
    """이력서 및 자기소개서 정보 추출을 위한 최종 구조"""
    personalInfo: PersonalInfo
    education: List[Education]
    military: Military
    certificates: List[Certificate]
    experience: List[Experience]
    selfIntroduction: List[SelfIntroItem]
    # selfIntroduction: List[SelfIntroduction]

