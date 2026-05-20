"""
generate_portfolios.py
─────────────────────────────────────────────────────────────────
IT 직군 포트폴리오 합성 데이터 생성기

Gemini 1-pass로 포트폴리오 섹션 JSON 생성 → embed.py에 바로 투입 가능한 포맷
섹션 구성: 자기소개 1 + 기술스택 1 + 경력 0~1 + 프로젝트경험 2~4

직군: 풀스택 / 프론트엔드 / 백엔드 (3가지)

[리팩토링 내역 - v2]
  FIX-1  프로젝트 본문 500자 초과 → 재생성 루프 (최대 2회 재시도)
  FIX-2  직군×경력 조합 불균형 → ScenarioQueue round-robin 균등 샘플링
  FIX-3  achievements 수치 단위 % 편중 → 프롬프트 단위 다변화 지시 + 후처리 경고
  FIX-4  기술스택 줄바꿈 누락 → 프롬프트 명시 + 후처리 자동 보정
  FIX-5  회사 선택 빅테크 저빈도 → tier별 가중치 샘플링 (빅테크 1.5×)

사용법:
    python generate_portfolios.py
    python generate_portfolios.py --count 50 --out output/mock_portfolios.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import random
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from itertools import product as iterproduct
from pathlib import Path


def _prevent_sleep() -> None:
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)


def _allow_sleep() -> None:
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


from dotenv import load_dotenv
load_dotenv()

from google import genai as _genai
from google.genai import types as _types
from pydantic import BaseModel
from tqdm import tqdm


def _get_gemini_client() -> _genai.Client:
    project = os.getenv("GCP_PROJECT_ID")
    if project:
        print("vertex ai 실행")
        return _genai.Client(
            vertexai=True,
            project=project,
            location=os.getenv("GCP_LOCATION", "us-central1"),
        )
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GCP_PROJECT_ID 또는 GEMINI_API_KEY 환경변수를 설정하세요.")
    return _genai.Client(api_key=api_key)


# ═══════════════════════════════════════════════════════════════
# 상수
# ═══════════════════════════════════════════════════════════════

_LLM_MODEL = "gemini-2.5-flash"

VALID_SECTIONS = {"프로젝트경험", "기술스택", "자기소개", "경력", "기타"}

# 프로젝트 본문 글자 수 상·하한 및 재생성 최대 횟수
# D1 평가 기준: 400~1000자 만점, 1200자 초과 감점 → 400~800자 목표
_TEXT_MIN   = 400
_TEXT_MAX   = 800
_REGEN_MAX  = 2

# FIX-3: achievements % 비중 경고 임계값
_PCT_RATIO_WARN = 0.50


# ═══════════════════════════════════════════════════════════════
# Pydantic 스키마 (Gemini 구조화 출력용)
# ═══════════════════════════════════════════════════════════════

class _Meta(BaseModel):
    period:        str
    role:          str
    team:          str
    tech_stack:    list[str]
    contributions: list[str]
    achievements:  list[str]
    keywords:      list[str]


class _Section(BaseModel):
    section: str    # VALID_SECTIONS 중 1개
    project: str    # 프로젝트경험일 때 프로젝트명, 나머지는 ""
    meta:    _Meta  # 프로젝트경험일 때만 의미 있게 채움
    text:    str    # 실제 내용 본문


class _PortfolioList(BaseModel):
    sections: list[_Section]


class _ProjectImage(BaseModel):
    content_type: str  # architecture/erd/ui/chart/code_image/other
    caption: str       # 3~4문장 캡션


class _ProjectImageList(BaseModel):
    images: list[_ProjectImage]


# ═══════════════════════════════════════════════════════════════
# 변수 풀 — 풀스택 / 프론트엔드 / 백엔드 3직군
# FIX-5: companies를 tier별 dict로 분리
# ═══════════════════════════════════════════════════════════════

JOBS: dict[str, dict] = {
    # ── 풀스택 ──────────────────────────────────────────────
    "풀스택": {
        "stacks": [
            ["React", "Node.js", "TypeScript", "PostgreSQL", "Docker", "AWS"],
            ["Next.js", "FastAPI", "Python", "MySQL", "Redis", "Nginx"],
            ["Vue.js", "Spring Boot", "Java", "MongoDB", "Kubernetes", "GCP"],
            ["React", "Django", "Python", "Redis", "Celery", "AWS Lambda"],
            ["Next.js", "NestJS", "TypeScript", "PostgreSQL", "Prisma", "Vercel"],
            ["Remix", "Go", "gRPC", "CockroachDB", "Terraform", "AWS ECS"],
            ["SvelteKit", "Rust", "Axum", "SQLite", "Cloudflare Workers"],
            ["React", "Ruby on Rails", "PostgreSQL", "Sidekiq", "Heroku"],
            ["Next.js", "Supabase", "TypeScript", "Tailwind CSS", "Edge Functions"],
            ["Nuxt.js", "Laravel", "PHP", "MySQL", "Redis", "DigitalOcean"],
        ],
        "keywords": [
            "풀스택 개발", "API 설계", "배포 자동화", "클라우드 인프라", "DevOps",
            "모노레포", "마이크로서비스", "서버리스", "DB 설계", "CI/CD 파이프라인",
            "OAuth 인증", "WebSocket", "GraphQL", "REST API", "컨테이너화",
            "IaC", "비용 최적화", "확장성 설계", "E2E 테스트", "A/B 테스트",
            "성능 모니터링", "로깅/추적", "멀티테넌시", "이벤트 드리븐",
        ],
        "awards": [
            "AWS Hackathon 최우수상",
            "구글 솔루션 챌린지 2등",
            "정보통신진흥원 SW 개발자 대회 우수상",
            "카카오 개발자 공모전 장려상",
            "네이버 핵데이 우승",
            "사내 이노베이션 해커톤 1위",
            "공개SW 개발자대회 과학기술정보통신부 장관상",
            "스타트업 위크엔드 서울 Tech Track 1위",
            "GitHub Galaxy 오픈소스 기여자 선정",
            "GDSC(Google Developer Student Clubs) Solution Challenge Top 50",
        ],
        "companies": {
            "빅테크": [
                {"name": "카카오",    "type": "대기업", "domain": "플랫폼/메신저"},
                {"name": "네이버",    "type": "대기업", "domain": "검색/커머스"},
                {"name": "쿠팡",      "type": "대기업", "domain": "이커머스/물류"},
                {"name": "라인플러스","type": "대기업", "domain": "글로벌 메신저"},
                {"name": "삼성SDS",  "type": "대기업", "domain": "IT서비스/클라우드"},
                {"name": "SK텔레콤", "type": "대기업", "domain": "통신/AI"},
                {"name": "LG CNS",   "type": "대기업", "domain": "IT서비스/DX"},
            ],
            "중견/유니콘": [
                {"name": "토스(비바리퍼블리카)", "type": "유니콘", "domain": "핀테크"},
                {"name": "당근마켓",             "type": "유니콘", "domain": "로컬 커머스"},
                {"name": "야놀자",               "type": "유니콘", "domain": "여행/숙박"},
                {"name": "무신사",               "type": "유니콘", "domain": "패션 커머스"},
                {"name": "직방",                 "type": "중견",   "domain": "부동산 플랫폼"},
                {"name": "크래프톤",             "type": "대기업", "domain": "게임"},
                {"name": "넥슨코리아",           "type": "대기업", "domain": "게임"},
                {"name": "엔씨소프트",           "type": "대기업", "domain": "게임"},
                {"name": "카카오페이",           "type": "중견",   "domain": "핀테크"},
                {"name": "카카오뱅크",           "type": "중견",   "domain": "인터넷은행"},
                {"name": "네이버파이낸셜",       "type": "중견",   "domain": "핀테크"},
                {"name": "두나무",               "type": "중견",   "domain": "블록체인/핀테크"},
            ],
            "스타트업": [
                {"name": "리멤버(드라마앤컴퍼니)", "type": "스타트업", "domain": "HR/커리어"},
                {"name": "채널코퍼레이션",         "type": "스타트업", "domain": "SaaS/CX"},
                {"name": "스캐터랩",               "type": "스타트업", "domain": "AI/대화"},
                {"name": "뤼튼테크놀로지스",       "type": "스타트업", "domain": "생성형 AI"},
                {"name": "솔트룩스",               "type": "중견",     "domain": "AI/NLP"},
                {"name": "에이블리",               "type": "스타트업", "domain": "패션 커머스"},
                {"name": "버킷플레이스(오늘의집)", "type": "스타트업", "domain": "인테리어 커머스"},
                {"name": "클래스101",             "type": "스타트업", "domain": "교육 플랫폼"},
                {"name": "원티드랩",              "type": "스타트업", "domain": "HR/채용"},
                {"name": "트릿지",                "type": "스타트업", "domain": "애그테크/B2B"},
            ],
        },
    },

    # ── 프론트엔드 ─────────────────────────────────────────
    "프론트엔드": {
        "stacks": [
            ["React", "TypeScript", "Next.js", "Tailwind CSS", "Zustand", "React Query"],
            ["Vue.js", "Nuxt.js", "Pinia", "SCSS", "Vitest", "Storybook"],
            ["React", "Redux Toolkit", "Webpack", "Jest", "Cypress", "Emotion"],
            ["Svelte", "SvelteKit", "TypeScript", "TailwindCSS", "Playwright"],
            ["Angular", "RxJS", "NgRx", "TypeScript", "Jasmine", "Karma"],
            ["React", "Vite", "TanStack Query", "Jotai", "Framer Motion", "shadcn/ui"],
            ["Next.js", "TypeScript", "Three.js", "WebGL", "GSAP", "styled-components"],
            ["React", "TypeScript", "MSW", "Turborepo", "Storybook", "Chromatic"],
            ["Qwik", "TypeScript", "TailwindCSS", "Partytown"],
            ["Astro", "React", "TypeScript", "TailwindCSS", "Cloudflare Pages"],
        ],
        "keywords": [
            "UI/UX 구현", "반응형 웹", "웹 성능 최적화", "크로스브라우징",
            "웹 접근성(WCAG)", "Core Web Vitals", "디자인 시스템",
            "컴포넌트 아키텍처", "상태 관리", "코드 스플리팅", "SSR/SSG",
            "번들 사이즈 최적화", "SEO 최적화", "애니메이션/인터랙션",
            "Storybook 문서화", "A11y", "PWA", "WebAssembly",
            "마이크로 프론트엔드", "모노레포", "테스트 자동화", "E2E 테스트",
            "레이아웃 시프트 방지", "Lighthouse CI", "국제화(i18n)",
        ],
        "awards": [
            "네이버 FE 플랫폼 챌린지 최우수상",
            "카카오 FE 기술 공모전 우수상",
            "CSS Design Awards 수상",
            "Google Chrome DevSummit Hackathon 입상",
            "공개SW 개발자대회 은상",
            "DEVIEW 발표자 선정",
            "GDG WebTech 발표자 선정",
            "사내 UX 혁신 프로젝트 표창",
            "GDSC 프론트엔드 트랙 1위",
            "인프콘 라이트닝 톡 발표자 선정",
            "삼성 청년 SW 아카데미 최우수 프로젝트상",
            "Kakao Tech Campus 우수상",
        ],
        "companies": {
            "빅테크": [
                {"name": "네이버",             "type": "대기업", "domain": "검색/포털"},
                {"name": "카카오",             "type": "대기업", "domain": "플랫폼/메신저"},
                {"name": "라인플러스",         "type": "대기업", "domain": "글로벌 메신저"},
                {"name": "쿠팡",               "type": "대기업", "domain": "이커머스"},
                {"name": "삼성전자 MX사업부",  "type": "대기업", "domain": "스마트폰 SW"},
                {"name": "LG전자 웹OS팀",      "type": "대기업", "domain": "스마트TV OS"},
                {"name": "SKT AI서비스사업부", "type": "대기업", "domain": "AI 서비스"},
            ],
            "중견/유니콘": [
                {"name": "토스",                  "type": "유니콘", "domain": "핀테크"},
                {"name": "당근마켓",              "type": "유니콘", "domain": "로컬 커머스"},
                {"name": "무신사",                "type": "유니콘", "domain": "패션 커머스"},
                {"name": "야놀자",                "type": "유니콘", "domain": "여행/숙박"},
                {"name": "마켓컬리",              "type": "중견",   "domain": "신선 이커머스"},
                {"name": "29CM",                  "type": "중견",   "domain": "라이프스타일 커머스"},
                {"name": "지그재그(카카오스타일)", "type": "중견",   "domain": "패션 커머스"},
                {"name": "크래프톤",              "type": "대기업", "domain": "게임"},
                {"name": "넥슨코리아",            "type": "대기업", "domain": "게임"},
                {"name": "원스토어",              "type": "중견",   "domain": "앱마켓"},
                {"name": "펄어비스",              "type": "중견",   "domain": "게임"},
                {"name": "커넥트웨이브(다나와)",  "type": "중견",   "domain": "가격비교/커머스"},
            ],
            "스타트업": [
                {"name": "채널코퍼레이션",         "type": "스타트업", "domain": "B2B SaaS"},
                {"name": "버킷플레이스(오늘의집)", "type": "스타트업", "domain": "인테리어"},
                {"name": "에이블리",               "type": "스타트업", "domain": "패션 커머스"},
                {"name": "클래스101",             "type": "스타트업", "domain": "교육"},
                {"name": "리멤버(드라마앤컴퍼니)", "type": "스타트업", "domain": "HR"},
                {"name": "뤼튼테크놀로지스",       "type": "스타트업", "domain": "생성형 AI"},
                {"name": "미리캔버스",             "type": "스타트업", "domain": "디자인 SaaS"},
                {"name": "노션코리아",             "type": "외국계",   "domain": "생산성 SaaS"},
                {"name": "에어테이블코리아",       "type": "외국계",   "domain": "No-Code"},
                {"name": "웰로(wello)",            "type": "스타트업", "domain": "복지/HR"},
            ],
        },
    },

    # ── 백엔드 ─────────────────────────────────────────────
    "백엔드": {
        "stacks": [
            ["Spring Boot", "Java 17", "JPA/Hibernate", "MySQL", "Redis", "Kafka"],
            ["FastAPI", "Python 3.11", "SQLAlchemy", "PostgreSQL", "Celery", "RabbitMQ"],
            ["Node.js", "Express", "TypeScript", "MongoDB", "Redis", "AWS SQS"],
            ["Django", "Python", "DRF", "PostgreSQL", "Redis", "Elasticsearch"],
            ["Go", "Gin", "GORM", "PostgreSQL", "gRPC", "Prometheus"],
            ["NestJS", "TypeScript", "Prisma", "MySQL", "Bull Queue", "AWS Lambda"],
            ["Spring WebFlux", "Kotlin", "R2DBC", "PostgreSQL", "Kafka", "OpenTelemetry"],
            ["Ktor", "Kotlin", "Exposed", "PostgreSQL", "gRPC", "Kubernetes"],
            ["Rust", "Actix-web", "Diesel", "PostgreSQL", "Tokio", "Redis"],
            ["Phoenix", "Elixir", "Ecto", "PostgreSQL", "LiveView", "Oban"],
            ["Ruby on Rails", "PostgreSQL", "Sidekiq", "Redis", "AWS S3"],
            ["ASP.NET Core", "C#", "Entity Framework", "SQL Server", "Azure Service Bus"],
        ],
        "keywords": [
            "RESTful API 설계", "MSA", "DB 설계/최적화", "캐싱 전략",
            "인증/인가(JWT/OAuth)", "성능 튜닝", "CI/CD", "쿼리 최적화",
            "비동기 처리", "메시지 큐", "이벤트 드리븐 아키텍처",
            "분산 시스템", "트랜잭션 관리", "API 게이트웨이",
            "서버리스", "컨테이너 오케스트레이션", "모니터링/알러팅",
            "백프레셔 처리", "레이트 리미팅", "데이터 파이프라인",
            "샤딩/파티셔닝", "읽기/쓰기 분리", "Saga 패턴",
            "DDD(도메인 주도 설계)", "TDD", "보안 취약점 대응",
            "OpenAPI/Swagger", "gRPC", "WebSocket 서버",
        ],
        "awards": [
            "카카오 백엔드 챌린지 최우수상",
            "네이버 핵데이 서버 트랙 우승",
            "AWS re:Invent Hackathon 입상",
            "공개SW 개발자대회 과학기술정보통신부 장관상",
            "정보보안 취약점 신고 포상(KISA)",
            "Spring 오픈소스 컨트리뷰터 선정",
            "사내 DB 최적화 프로젝트 우수상",
            "ICPC 아시아 지역 예선 입상",
            "삼성 SW 역량 테스트 B형 합격",
            "구름 GROOM 해커톤 대상",
            "지역 정보화 공모전 최우수상(한국지능정보사회진흥원)",
            "Kafka Summit Asia 발표자 선정",
        ],
        "companies": {
            "빅테크": [
                {"name": "네이버",           "type": "대기업", "domain": "검색/AI"},
                {"name": "카카오",           "type": "대기업", "domain": "플랫폼"},
                {"name": "쿠팡",             "type": "대기업", "domain": "이커머스/물류"},
                {"name": "라인플러스",       "type": "대기업", "domain": "글로벌 메신저"},
                {"name": "삼성SDS",          "type": "대기업", "domain": "IT서비스"},
                {"name": "SKT AI R&D 센터",  "type": "대기업", "domain": "AI/플랫폼"},
                {"name": "LG CNS",           "type": "대기업", "domain": "DX/클라우드"},
                {"name": "KT NexR",          "type": "대기업", "domain": "빅데이터/AI"},
            ],
            "중견/유니콘": [
                {"name": "토스(비바리퍼블리카)", "type": "유니콘", "domain": "핀테크"},
                {"name": "당근마켓",             "type": "유니콘", "domain": "커뮤니티"},
                {"name": "야놀자",               "type": "유니콘", "domain": "여행/숙박"},
                {"name": "카카오뱅크",           "type": "중견",   "domain": "인터넷은행"},
                {"name": "카카오페이",           "type": "중견",   "domain": "결제"},
                {"name": "두나무(업비트)",       "type": "중견",   "domain": "블록체인/거래소"},
                {"name": "크래프톤",             "type": "대기업", "domain": "게임 서버"},
                {"name": "넥슨코리아",           "type": "대기업", "domain": "게임 서버"},
                {"name": "엔씨소프트",           "type": "대기업", "domain": "게임 서버"},
                {"name": "마켓컬리",             "type": "중견",   "domain": "물류/이커머스"},
                {"name": "무신사",               "type": "유니콘", "domain": "패션 커머스"},
            ],
            "스타트업": [
                {"name": "채널코퍼레이션",         "type": "스타트업", "domain": "B2B SaaS"},
                {"name": "버킷플레이스(오늘의집)", "type": "스타트업", "domain": "커머스"},
                {"name": "뤼튼테크놀로지스",       "type": "스타트업", "domain": "생성형 AI"},
                {"name": "솔트룩스",               "type": "중견",     "domain": "AI/NLP"},
                {"name": "루닛",                   "type": "스타트업", "domain": "의료AI"},
                {"name": "딥노이드",               "type": "스타트업", "domain": "AI 솔루션"},
                {"name": "에이모",                 "type": "스타트업", "domain": "자율주행 AI"},
                {"name": "원티드랩",              "type": "스타트업", "domain": "HR/채용"},
                {"name": "리멤버(드라마앤컴퍼니)", "type": "스타트업", "domain": "HR"},
                {"name": "스캐터랩(이루다)",       "type": "스타트업", "domain": "대화AI"},
                {"name": "클래스101",             "type": "스타트업", "domain": "교육"},
                {"name": "직방",                  "type": "중견",     "domain": "부동산"},
                {"name": "라온시큐어",            "type": "중견",     "domain": "보안"},
                {"name": "이스트시큐리티",        "type": "중견",     "domain": "보안"},
                {"name": "AWS코리아",             "type": "외국계",   "domain": "클라우드"},
                {"name": "구글코리아",            "type": "외국계",   "domain": "클라우드/AI"},
                {"name": "마이크로소프트코리아",  "type": "외국계",   "domain": "클라우드"},
            ],
        },
    },
}

# FIX-5: tier별 선택 가중치 — 빅테크 1.5× 보정으로 저빈도 해소
_COMPANY_TIER_WEIGHTS: dict[str, float] = {
    "빅테크":     1.5,
    "중견/유니콘": 1.0,
    "스타트업":   1.0,
}

CAREER_LEVELS: dict[str, str] = {
    "신입":            "부트캠프 수료 또는 학부 졸업 예정자, 개인/팀 프로젝트 중심",
    "주니어(1~3년)":   "스타트업 또는 SI 경력 1~3년, 실무 프로젝트 다수",
    "미드레벨(3~5년)": "중견 이상 경력 3~5년, 팀 리딩 및 아키텍처 설계 경험",
}

PROJECT_TYPES: list[str] = [
    "커머스/쇼핑 플랫폼",
    "SNS/커뮤니티 서비스",
    "관리자 대시보드",
    "실시간 채팅/알림 서비스",
    "인증/회원 관리 시스템",
    "검색/추천 서비스",
    "예약/스케줄링 서비스",
    "결제/정산 시스템",
    "파일/미디어 관리 서비스",
    "데이터 파이프라인/분석",
    "게임/엔터테인먼트 플랫폼",
    "헬스케어/피트니스 서비스",
    "교육/학습 플랫폼",
    "여행/숙박 예약 서비스",
    "음식 주문/배달 서비스",
    "핀테크/투자 서비스",
    "HR/채용 플랫폼",
    "부동산/매물 서비스",
    "B2B SaaS 솔루션",
    "공공데이터 활용 서비스",
    "이벤트/티켓팅 플랫폼",
    "구독/멤버십 서비스",
    "AI 챗봇/어시스턴트",
    "물류/배송 추적 시스템",
    "IoT 대시보드/모니터링",
]

CERTIFICATIONS: list[str] = [
    "AWS Certified Solutions Architect – Associate",
    "AWS Certified Developer – Associate",
    "Google Cloud Professional Cloud Developer",
    "CKA(Certified Kubernetes Administrator)",
    "정보처리기사",
    "SQLD(SQL 개발자)",
    "리눅스마스터 1급",
    "네트워크 관리사 2급",
    "OCP(Oracle Certified Professional)",
    "Azure Developer Associate",
]

PERSONAS: dict[str, list[str]] = {
    "names": [
        "김민준", "이서연", "박지훈", "최수아", "정도현",
        "강예진", "조성민", "윤하은", "장태양", "임나리",
        "한승우", "오지은", "송민재", "권아름", "류성현",
        "신동혁", "백지수", "문준호", "안소희", "황민철",
        "차은우", "노지원", "유승호", "홍아름", "전혜린",
        "배성준", "서지민", "고은혜", "남도현", "심유진",
    ],
    "universities": [
        "서울대학교 컴퓨터공학과",
        "KAIST 전산학부",
        "연세대학교 컴퓨터과학과",
        "고려대학교 컴퓨터학과",
        "성균관대학교 소프트웨어학과",
        "한양대학교 컴퓨터소프트웨어학부",
        "한국대학교 컴퓨터공학과",
        "서울과학기술대학교 소프트웨어학과",
        "경희대학교 컴퓨터공학부",
        "숭실대학교 IT학부",
        "세종대학교 컴퓨터공학과",
        "국민대학교 소프트웨어학부",
        "인하대학교 정보통신공학과",
        "아주대학교 소프트웨어학과",
        "동국대학교 컴퓨터공학과",
        "중앙대학교 소프트웨어학부",
        "부산대학교 정보컴퓨터공학부",
        "부트캠프(패스트캠퍼스) 수료",
        "부트캠프(코드스테이츠) 수료",
        "부트캠프(우아한테크코스) 수료",
        "부트캠프(항해99) 수료",
        "부트캠프(SSAFY) 수료",
        "부트캠프(크래프톤 정글) 수료",
    ],
}


# ═══════════════════════════════════════════════════════════════
# FIX-5: tier 가중치 회사 샘플러
# ═══════════════════════════════════════════════════════════════

def _pick_company(job: str) -> dict:
    """tier별 가중치를 반영해 회사를 랜덤 선택한다."""
    tiers        = list(JOBS[job]["companies"].keys())
    weights      = [_COMPANY_TIER_WEIGHTS.get(t, 1.0) for t in tiers]
    chosen_tier  = random.choices(tiers, weights=weights, k=1)[0]
    return random.choice(JOBS[job]["companies"][chosen_tier])


# ═══════════════════════════════════════════════════════════════
# FIX-2: 직군×경력 round-robin ScenarioQueue
# ═══════════════════════════════════════════════════════════════

class ScenarioQueue:
    """
    9가지 (직군 × 경력) 조합을 균등하게 순환 배출한다.
    큐가 소진되면 셔플 후 재충전 → 전체 실행 동안 편향 없이 분배.
    """

    def __init__(self) -> None:
        self._combos: list[tuple[str, str]] = list(
            iterproduct(JOBS.keys(), CAREER_LEVELS.keys())
        )
        self._queue: deque[tuple[str, str]] = deque()
        self._refill()

    def _refill(self) -> None:
        pool = self._combos.copy()
        random.shuffle(pool)
        self._queue.extend(pool)

    def next(self) -> tuple[str, str]:
        if not self._queue:
            self._refill()
        return self._queue.popleft()


# ═══════════════════════════════════════════════════════════════
# 시나리오 조합
# ═══════════════════════════════════════════════════════════════

@dataclass
class Scenario:
    job:            str
    career:         str
    stack:          list[str]
    project_types:  list[str]
    persona_name:   str
    persona_univ:   str
    company:        dict        # {"name": ..., "type": ..., "domain": ...}
    awards:         list[str]   # 0~2개 수상경력
    certifications: list[str]   # 0~2개 자격증
    job_keywords:   list[str]   # 직군 키워드 샘플
    run_id:         str = field(default_factory=lambda: uuid.uuid4().hex[:6])

    @property
    def source(self) -> str:
        return f"mock_portfolio_{self.job}_{self.career}_{self.run_id}"


def _make_scenario(queue: ScenarioQueue) -> Scenario:
    # FIX-2: round-robin으로 직군·경력 결정
    job, career = queue.next()

    stack      = random.choice(JOBS[job]["stacks"])
    n_proj     = random.randint(2, 4)
    proj_types = random.sample(PROJECT_TYPES, k=n_proj)

    # FIX-5: tier 가중치 회사 샘플링
    company = _pick_company(job)

    # 신입은 수상 0~1개, 주니어+ 0~2개
    max_awards = 1 if career == "신입" else 2
    n_awards   = random.randint(0, max_awards)
    awards     = random.sample(JOBS[job]["awards"], k=n_awards) if n_awards else []

    # 자격증 0~2개
    n_certs = random.randint(0, 2)
    certs   = random.sample(CERTIFICATIONS, k=n_certs)

    # 직군 키워드 4~7개
    n_kw     = random.randint(4, 7)
    kw_pool  = JOBS[job]["keywords"]
    keywords = random.sample(kw_pool, k=min(n_kw, len(kw_pool)))

    return Scenario(
        job=job,
        career=career,
        stack=stack,
        project_types=proj_types,
        persona_name=random.choice(PERSONAS["names"]),
        persona_univ=random.choice(PERSONAS["universities"]),
        company=company,
        awards=awards,
        certifications=certs,
        job_keywords=keywords,
    )


# ═══════════════════════════════════════════════════════════════
# 프롬프트 빌더
# ═══════════════════════════════════════════════════════════════

_SYSTEM_INSTRUCTION = """\
당신은 IT 직군 포트폴리오 훈련 데이터 생성 AI입니다.
생성되는 모든 포트폴리오는 아래 5개 평가 기준에서 높은 점수를 받도록 작성해야 합니다.

[A. 과정/판단력 — 35%: 가장 중요]
  - 반드시 '배경/상황 → 문제인식 → 기술선택 근거(왜?) → 실행 → 결과' 흐름으로 서술
  - '왜 이 기술을 선택했는가'를 대안 대비 근거와 함께 명시 (예: Redis 대신 JWT 선택 이유)
  - 단순 나열이 아닌 사고방식과 판단 기준이 문장 속에 드러나야 함

[B. 역할/기여도 — 25%]
  - '직접/담당/주도/설계' 등 1인칭 기여 표현을 충분히 사용
  - 트러블슈팅 1건 이상 포함: 문제 발생 → 원인 파악 → 해결 과정을 구체적으로 서술

[C. 성과/인사이트 — 20%]
  - 정량 수치 4개 이상 포함 (ms, %, rps, 배수×N, GB, 건/일 등 단위 다양하게)
  - 마지막에 이 경험을 통한 인사이트·배움·직무 기여 방향을 1~2문장 추가

[D. 작성품질 — 10% (규칙 기반 자동 감점)]
  - 본문 400~800자 유지 (400 미만·1200 초과 시 감점)
  - 금지: '열심히', '최선을 다', '다양한 경험', '성장할 수 있', '뜻깊', '좋은 경험' 등 추상 표현
  - 금지: bullet(-/•/*) 나열이 전체 줄의 70% 이상 차지하는 구성
  - 금지: 중복 문장 (앞 20자 기준 동일 문장 반복)

[E. 직무연관성 — 10%]
  - 목표 직군(풀스택/프론트엔드/백엔드)의 핵심 기술·역량이 명확히 드러나야 함
"""


def _build_prompt(s: Scenario) -> str:
    stack_str    = ", ".join(s.stack)
    projects_str = "\n".join(f"  {i+1}. {pt}" for i, pt in enumerate(s.project_types))
    has_career   = s.career != "신입"
    awards_str   = "\n".join(f"  - {a}" for a in s.awards) if s.awards else "  - 없음"
    certs_str    = "\n".join(f"  - {c}" for c in s.certifications) if s.certifications else "  - 없음"
    kw_str       = ", ".join(s.job_keywords)
    company      = s.company

    career_section = (
        f"3. 경력 (1개): 실무/인턴/대외활동 경력 요약 (200~300자)\n"
        f"   - 주요 근무 회사: {company['name']} ({company['type']}, {company['domain']})"
        if has_career else
        "3. 경력: 생략 (신입)"
    )

    return (
        f"아래 조건의 개발자 포트폴리오를 섹션별 JSON으로 생성해주세요.\n\n"
        f"[지원자]\n"
        f"- 이름: {s.persona_name} / {s.persona_univ}\n"
        f"- 직군: {s.job} ({s.career}, {CAREER_LEVELS[s.career]})\n"
        f"- 주요 기술 스택: {stack_str}\n"
        f"- 핵심 키워드: {kw_str}\n\n"
        f"[경력/이력]\n"
        f"- 주요 회사: {company['name']} ({company['type']}, {company['domain']})\n"
        f"- 수상경력:\n{awards_str}\n"
        f"- 자격증:\n{certs_str}\n\n"
        f"[생성할 섹션 구성]\n"
        f"1. 자기소개 (1개): 개발자로서의 철학·목표·강점 (200~300자)\n"
        f"2. 기술스택 (1개): 보유 기술을 카테고리별로 정리한 텍스트 (100~200자)\n"
        f"{career_section}\n"
        f"4. 프로젝트경험 ({len(s.project_types)}개): 아래 프로젝트 타입별로 각각 생성\n"
        f"{projects_str}\n\n"
        f"[자기소개 작성 규칙]\n"
        f"- 위 수상경력, 자격증, 핵심 키워드를 자연스럽게 녹여서 작성\n"
        f"- 개발 철학·협업 방식·성장 방향 포함\n\n"
        # FIX-4: 줄바꿈 명시
        f"[기술스택 작성 규칙]\n"
        f"- Language / Framework / Database / Infra / Tool 카테고리로 분류\n"
        f"- 각 카테고리는 반드시 줄바꿈(\\n)으로 구분 (예: 'Language: ...\\nFramework: ...')\n"
        f"- 위 자격증이 있으면 Certification 카테고리로 마지막에 추가\n\n"
        f"[경력 작성 규칙]\n"
        f"- 회사명·재직기간·담당 업무·주요 성과를 구체적으로 서술\n"
        f"- {company['name']}의 {company['domain']} 도메인 관련 업무 포함\n"
        f"- 수상경력이 있으면 경력 섹션 text에 자연스럽게 언급\n\n"
        f"[프로젝트경험 작성 규칙]\n"
        f"- section: 반드시 '프로젝트경험'\n"
        f"- project: 창의적인 프로젝트명 (영문 또는 한글)\n"
        f"- meta.period: 반드시 'YYYY.MM ~ YYYY.MM' 형식 ('현재' 사용 금지, 종료 미정이면 예상 종료월 기재)\n"
        + (f"- meta.period 추가: 신입이므로 모든 날짜를 과거 완료형으로 기재\n" if not has_career else "")
        + f"- meta.role: 본인 역할 (예: 백엔드 리드, 프론트엔드 개발)\n"
        f"- meta.team: 팀 구성 (예: '4인 팀', '개인프로젝트')\n"
        f"- meta.tech_stack: 사용 기술 목록\n"
        f"- meta.contributions: 수치 없이 직접 수행한 역할·구현 내용을 '~구현', '~개발' 형태로\n"
        f"- meta.achievements: 수치 포함 성과. contributions와 중복 금지\n"
        f"  · 단위를 반드시 고르게 사용: %, ms, rps/tps, 건/일, GB/MB, 배수(×N배), 점수\n"
        f"  · 동일 단위 3회 이상 연속 사용 금지. % 단위가 전체 achievements의 50% 이하가 되도록 조절\n"
        f"- meta.keywords: 기술·직무·도메인 키워드 (5~8개, 직군 핵심 키워드 포함)\n"
        f"\n"
        f"- text: 프로젝트 설명 본문. 반드시 400자 이상 800자 이하\n"
        f"  ┌ [필수 서술 구조 — 평가 A 35% 기준]\n"
        f"  │ ① 배경/상황: 이 프로젝트를 시작한 맥락·문제 상황 (1~2문장)\n"
        f"  │ ② 문제인식: 해결해야 할 핵심 기술 과제 구체적 명시\n"
        f"  │ ③ 기술선택 근거: '왜 이 기술·방법인가' 대안 대비 이유 포함 (예: A 대신 B 선택 이유)\n"
        f"  │ ④ 실행/구현: 직접 담당한 부분을 1인칭으로 ('직접/담당/주도/설계' 표현 사용)\n"
        f"  │ ⑤ 트러블슈팅: 문제 발생 → 원인 파악 → 해결 과정 1건 이상 (평가 B 25% 기준)\n"
        f"  │ ⑥ 성과: 정량 수치 4개 이상 (ms·%·rps·배수·건 등 단위 다양하게) (평가 C 20% 기준)\n"
        f"  └ ⑦ 인사이트: 이 경험을 통한 배움·성장·직무 기여 방향 1~2문장\n"
        f"  · [금지] 추상 표현: '열심히', '최선을 다', '다양한 경험', '성장할 수 있', '뜻깊', '좋은 경험'\n"
        f"  · [금지] bullet(-/•/*) 나열이 전체 줄의 70% 이상 차지하는 구성\n"
        f"  · [금지] 중복 문장 (앞 20자 동일 문장 반복)\n\n"
        f"[자기소개·기술스택·경력 작성 규칙]\n"
        f"- project: '' (빈 문자열)\n"
        f"- meta 전체: str 필드는 '', list 필드는 []\n"
    )


# ═══════════════════════════════════════════════════════════════
# FIX-4: 기술스택 줄바꿈 후처리 보정
# ═══════════════════════════════════════════════════════════════

_STACK_CATEGORY_RE = re.compile(
    r"(Language|Framework|Database|Infra|Tool|Certification)\s*:",
    re.IGNORECASE,
)


def _fix_techstack_newlines(text: str) -> str:
    """
    'Language: ...Framework: ...' 처럼 줄바꿈 없이 붙어 나온 기술스택 텍스트에
    카테고리 앞 줄바꿈을 삽입한다. 이미 줄바꿈이 있으면 원본 반환.
    """
    if "\n" in text:
        return text
    # 카테고리 키워드 기준으로 분할 후 재조합
    tokens = _STACK_CATEGORY_RE.split(text)
    if len(tokens) <= 1:
        return text
    lines: list[str] = []
    # tokens = [pre, key1, val1, key2, val2, ...]
    pre = tokens[0].strip()
    if pre:
        lines.append(pre)
    idx = 1
    while idx + 1 < len(tokens):
        key = tokens[idx].strip()
        val = tokens[idx + 1].strip()
        lines.append(f"{key}: {val}")
        idx += 2
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# FIX-3: achievements % 비중 후처리 경고
# ═══════════════════════════════════════════════════════════════

def _warn_pct_ratio(achievements: list[str], source: str) -> None:
    if not achievements:
        return
    pct_count = sum(1 for a in achievements if "%" in a)
    ratio = pct_count / len(achievements)
    if ratio > _PCT_RATIO_WARN:
        tqdm.write(
            f"  [WARN-수치] {source}: achievements % 비중 {ratio:.0%} "
            f"({pct_count}/{len(achievements)}) — 목표 50% 이하"
        )


# ═══════════════════════════════════════════════════════════════
# FIX-1: 500자 초과 프로젝트 본문 단독 재생성
# ═══════════════════════════════════════════════════════════════

def _regen_overlong_project(
    client: _genai.Client,
    project_name: str,
    original_text: str,
) -> str:
    """800자 초과 프로젝트 본문을 400~800자로 재작성 요청한다."""
    prompt = (
        f"아래 포트폴리오 프로젝트 설명이 {len(original_text)}자로 800자 상한을 초과했습니다.\n"
        f"핵심 내용(기술명, 수치 성과 4개 이상, 역할, 트러블슈팅)을 유지하면서 400~800자로 줄여 재작성해주세요.\n"
        f"배경→문제→기술선택 근거→실행→트러블슈팅→성과→인사이트 흐름을 유지하세요.\n"
        f"재작성한 본문 텍스트만 출력하세요. JSON·마크다운 형식 없이 순수 텍스트로.\n\n"
        f"[프로젝트명] {project_name}\n"
        f"[원본 텍스트]\n{original_text}"
    )
    try:
        response = client.models.generate_content(
            model=_LLM_MODEL,
            contents=prompt,
            config=_types.GenerateContentConfig(
                system_instruction=_SYSTEM_INSTRUCTION,
            ),
        )
        rewritten = response.text.strip()
        # 재생성 후에도 초과이면 앞 500자 truncate (최후 보루)
        return rewritten if len(rewritten) <= _TEXT_MAX else rewritten[:_TEXT_MAX]
    except Exception:
        return original_text[:_TEXT_MAX]


# ═══════════════════════════════════════════════════════════════
# 이미지 청크 생성
# ═══════════════════════════════════════════════════════════════

_IMAGE_CONTENT_TYPES = ["architecture", "erd", "ui", "chart", "code_image"]

_IMAGE_GEN_PROMPT = """\
다음 프로젝트 포트폴리오 설명을 읽고, 이 프로젝트에 포함됐을 법한 이미지를 1~2개 선택하여 각 이미지에 대한 캡션을 작성하세요.

[프로젝트명]
{project_name}

[프로젝트 설명]
{project_text}

[이미지 유형 선택 기준]
- architecture: 시스템 아키텍처 다이어그램 (서비스 간 관계, 인프라 구성)
- erd: 데이터베이스 ERD (테이블 구조, 관계)
- ui: 화면 UI/UX 스크린샷 (주요 페이지, 사용자 흐름)
- chart: 성능/지표 차트 (응답시간, 트래픽, 개선 효과)
- code_image: 핵심 코드 스니펫 (알고리즘, 구현 패턴)

[캡션 작성 기준]
- 3~4문장으로 이미지 내용 설명
- 기술 스택, 구체적 수치, 역할을 포함
- 프로젝트 설명에 실제로 등장하는 내용만 사용
- 한국어로 작성

적합한 이미지 유형 1~2개를 선택하고 캡션을 작성하세요."""


def _generate_project_image_chunks(
    client: _genai.Client,
    section: dict,
    base_index: int,
) -> list[dict]:
    """프로젝트 섹션에 대한 이미지 캡션 청크 1~2개 생성."""
    project_name = section.get("project", "")
    text = section.get("text", "")
    if not text or not project_name:
        return []

    prompt = _IMAGE_GEN_PROMPT.format(
        project_name=project_name,
        project_text=text,
    )

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=_LLM_MODEL,
                contents=prompt,
                config=_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=_ProjectImageList,
                ),
            )
            images = _ProjectImageList.model_validate_json(response.text).images
            break
        except Exception as e:
            err = str(e)
            if attempt < 2:
                wait = 60 if "429" in err else 5
                time.sleep(wait)
            else:
                return []

    chunks = []
    for j, img in enumerate(images[:2]):
        content_type = img.content_type if img.content_type in _IMAGE_CONTENT_TYPES else "other"
        source = section["source"]
        chunks.append({
            "id":           f"{source}_{base_index + j:03d}",
            "source":       source,
            "doc_type":     "portfolio",
            "job":          section.get("job", ""),
            "career":       section.get("career", ""),
            "company":      section.get("company", {}),
            "section":      "프로젝트경험",
            "project":      project_name,
            "sub_section":  "이미지",
            "meta":         section.get("meta", {}),
            "content_type": content_type,
            "fig_id":       "",
            "image_path":   "",
            "text":         img.caption,
            "char_count":   len(img.caption),
        })
    return chunks


# ═══════════════════════════════════════════════════════════════
# 생성 파이프라인
# ═══════════════════════════════════════════════════════════════

def _call_gemini(client: _genai.Client, scenario: Scenario) -> list[dict]:
    """Gemini API 호출 (5회 재시도). 성공 시 raw sections list 반환."""
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=_LLM_MODEL,
                contents=_build_prompt(scenario),
                config=_types.GenerateContentConfig(
                    system_instruction=_SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=_PortfolioList,
                ),
            )
            return json.loads(response.text)["sections"]
        except Exception as e:
            err = str(e)
            if attempt < 4:
                wait = 60 if "429" in err else 10 * (attempt + 1)
                tqdm.write(f"    [재시도 {attempt+1}/4] {err[:60]} → {wait}초 대기...")
                time.sleep(wait)
            else:
                raise
    return []  # unreachable


def generate_sections(client: _genai.Client, scenario: Scenario) -> list[dict]:
    raw_sections = _call_gemini(client, scenario)
    result: list[dict] = []

    for i, sec in enumerate(raw_sections):
        section = sec.get("section", "기타")
        if section not in VALID_SECTIONS:
            section = "기타"

        text = sec.get("text", "").strip()
        if not text:
            continue

        meta_raw = sec.get("meta", {})

        if section == "프로젝트경험":
            meta = {
                k: v for k, v in meta_raw.items()
                if (isinstance(v, str) and v.strip()) or (isinstance(v, list) and v)
            }

            # 신입 '현재' 후처리 (프롬프트 지시에도 간혹 발생)
            if scenario.career == "신입" and "period" in meta:
                meta["period"] = re.sub(r"~\s*현재", "~ 완료", meta["period"])

            # FIX-3: achievements % 비중 경고
            _warn_pct_ratio(meta.get("achievements", []), scenario.source)

            # 800자 초과 → 재생성 루프 (최대 _REGEN_MAX 회)
            project_name = sec.get("project", "")
            for _ in range(_REGEN_MAX):
                if len(text) <= _TEXT_MAX:
                    break
                tqdm.write(
                    f"  [REGEN] {scenario.source} / '{project_name}' "
                    f"{len(text)}자 초과 → 재작성 요청"
                )
                text = _regen_overlong_project(client, project_name, text)

        elif section == "기술스택":
            # FIX-4: 줄바꿈 누락 후처리 보정
            text = _fix_techstack_newlines(text)
            meta = {}

        else:
            meta = {}

        result.append({
            "id":         f"{scenario.source}_{i:03d}",
            "source":     scenario.source,
            "doc_type":   "portfolio",
            "job":        scenario.job,
            "career":     scenario.career,
            "company":    scenario.company,
            "section":    section,
            "project":    sec.get("project", ""),
            "meta":       meta,
            "text":       text,
            "char_count": len(text),
        })

    # 프로젝트경험 섹션마다 이미지 캡션 청크 추가
    img_chunks: list[dict] = []
    for sec_item in result:
        if sec_item["section"] == "프로젝트경험":
            imgs = _generate_project_image_chunks(
                client, sec_item, len(result) + len(img_chunks)
            )
            img_chunks.extend(imgs)
    result.extend(img_chunks)

    return result


# ═══════════════════════════════════════════════════════════════
# 메인 실행 루프
# ═══════════════════════════════════════════════════════════════

def run(total: int, out_path: Path, delay: float = 0.5) -> None:
    _prevent_sleep()
    client = _get_gemini_client()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_sections: list[dict] = []
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            all_sections = json.load(f)
        print(f"기존 데이터 {len(all_sections)}개 로드, 이어서 생성합니다.")

    # FIX-2: round-robin 시나리오 큐 초기화
    queue   = ScenarioQueue()
    success = 0
    fail    = 0

    with tqdm(total=total, desc="포트폴리오 생성 중") as bar:
        for i in range(total):
            scenario = _make_scenario(queue)
            try:
                sections = generate_sections(client, scenario)
                all_sections.extend(sections)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(all_sections, f, ensure_ascii=False, indent=2)
                success += 1
                tqdm.write(
                    f"  ✓ {scenario.source} [{scenario.job}/{scenario.career}] "
                    f"→ {len(sections)}개 섹션"
                )
            except Exception as e:
                fail += 1
                err  = str(e)
                wait = 60 if "429" in err else 5
                tqdm.write(f"  [WARN] {i+1}번 실패: {err[:80]}. {wait}초 후 재시도...")
                time.sleep(wait)
            finally:
                bar.update(1)
                bar.set_postfix(success=success, fail=fail, total_sections=len(all_sections))
                time.sleep(delay)

    _allow_sleep()
    print(f"\n완료: 시나리오 {success}개 성공 / {fail}개 실패")
    print(f"총 섹션 수: {len(all_sections)}개 → {out_path}")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IT 직군 포트폴리오 합성 데이터 생성기")
    parser.add_argument(
        "--count", type=int, default=50,
        help="생성할 포트폴리오 수 (포트폴리오당 4~7섹션, 기본: 50)",
    )
    parser.add_argument(
        "--out", type=str, default="output/mock_portfolios.json",
        help="출력 JSON 파일 경로",
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="요청 간격(초), 기본 0.5",
    )
    args = parser.parse_args()

    run(total=args.count, out_path=Path(args.out), delay=args.delay)