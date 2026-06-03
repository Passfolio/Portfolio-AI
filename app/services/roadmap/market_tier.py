"""
market_tier.py — DATASET 기반 국내 시장 기술 티어 분류

5개 티어 (우선순위 높은 순):
  필수    — 국내 채용공고 필수 요구 기술 (높은 빈도 + 안정적 수요)   [다이어그램 강조]
  급부상  — 빠르게 성장 중인 기술 (높은 수요 증가율)               [다이어그램 강조]
  주류    — 검증된 표준 기술 (안정적 수요, 보편화)
  성장    — 신흥 확인 기술 (낮은 빈도이나 증가세)
  중간    — 그 외 (틈새·하락·미분류)

우선순위: 필수 > 급부상 > 주류 > 성장 > 중간
"""
from __future__ import annotations
import json
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = ROOT / "DATASET"

ESSENTIAL   = "필수"
RISING      = "급부상"
MAINSTREAM  = "주류"
GROWING     = "성장"
MODERATE    = "중간"

HIGHLIGHT_TIERS = {ESSENTIAL, RISING}

# ── topic-level 노드 수동 매핑 (데이터셋에 직접 등장하지 않는 개념형 토픽) ──
_TOPIC_TIER: dict[str, str] = {
    # ── 공통 ──
    "Internet":                             ESSENTIAL,
    "Version Control Systems":              ESSENTIAL,
    "Repo Hosting Services":                ESSENTIAL,
    "VCS Hosting":                          ESSENTIAL,

    # ── kr_roadmap.csv: 백엔드 ──
    "Git / 협업":                           ESSENTIAL,
    "프로그래밍 언어":                          ESSENTIAL,
    "HTTP / 웹 기초":                        ESSENTIAL,
    "웹 프레임워크":                            ESSENTIAL,
    "관계형 데이터베이스":                         ESSENTIAL,
    "인증 / 보안":                            ESSENTIAL,
    "API 설계":                             ESSENTIAL,
    "테스트":                               MAINSTREAM,
    "컨테이너 / 배포":                          ESSENTIAL,
    "클라우드 기초":                            ESSENTIAL,
    "NoSQL / 캐싱":                         MAINSTREAM,
    "메시지 브로커":                            MAINSTREAM,
    "아키텍처 패턴":                            MAINSTREAM,
    "성능 / 스케일링":                          GROWING,
    "AI 연동 (선택)":                         GROWING,

    # ── kr_roadmap.csv: 프론트엔드 ──
    "HTML5 / CSS3 기초":                    ESSENTIAL,
    "JavaScript (ES6+)":                   ESSENTIAL,
    "TypeScript":                          RISING,
    "React":                               MAINSTREAM,
    "Next.js":                             RISING,
    "상태 관리":                              MAINSTREAM,
    "스타일링":                               MAINSTREAM,
    "빌드 도구":                              MAINSTREAM,
    "성능 최적화":                             GROWING,
    "백엔드 연동":                             MAINSTREAM,

    # ── kr_roadmap.csv: DevOps ──
    "Linux / 서버 운영":                      ESSENTIAL,
    "네트워크 기초":                            ESSENTIAL,
    "버전 관리":                              ESSENTIAL,
    "컨테이너화":                              ESSENTIAL,
    "클라우드 - AWS":                         ESSENTIAL,
    "클라우드 - 멀티클라우드":                     MAINSTREAM,
    "컨테이너 오케스트레이션":                      RISING,
    "IaC (인프라 코드화)":                     RISING,
    "CI/CD 파이프라인":                        MAINSTREAM,
    "모니터링 / 관측성":                         MAINSTREAM,
    "보안":                                 MAINSTREAM,
    "스크립팅 / 자동화":                         MAINSTREAM,
    "SRE 실무":                             GROWING,

    # ── kr_roadmap.csv: AI Engineer ──
    "Python 심화":                          ESSENTIAL,
    "LLM API 활용":                         RISING,
    "RAG 파이프라인":                          RISING,
    "AI 에이전트":                            RISING,
    "프롬프트 엔지니어링":                         RISING,
    "Fine-tuning":                         GROWING,
    "배포 / 서빙":                            MAINSTREAM,
    "데이터 처리":                             MAINSTREAM,

    # ── kr_roadmap.csv: Machine Learning ──
    "Python + 수학 기초":                     ESSENTIAL,
    "전통 ML":                              MAINSTREAM,
    "딥러닝 - PyTorch":                      MAINSTREAM,
    "딥러닝 응용":                             GROWING,
    "실험 관리":                              GROWING,
    "모델 서빙 / 최적화":                        MAINSTREAM,
    "ML 인프라":                             GROWING,

    # ── kr_roadmap.csv: Data Engineer ──
    "SQL 심화":                             ESSENTIAL,
    "Python 데이터 처리":                      ESSENTIAL,
    "파이프라인 오케스트레이션":                     MAINSTREAM,
    "이벤트 스트리밍":                           GROWING,
    "대규모 처리":                             GROWING,
    "클라우드 데이터 플랫폼":                      MAINSTREAM,
    "데이터 변환 / 품질":                        MAINSTREAM,

    # ── kr_roadmap.csv: Android ──
    "Kotlin 기초":                          ESSENTIAL,
    "Android 기본":                         ESSENTIAL,
    "Jetpack 컴포넌트":                       MAINSTREAM,
    "비동기 / 반응형":                          MAINSTREAM,
    "네트워크 연동":                            MAINSTREAM,
    "배포 / CI/CD":                         MAINSTREAM,

    # ── kr_roadmap.csv: iOS ──
    "Swift 기초":                           ESSENTIAL,
    "UIKit / SwiftUI":                     MAINSTREAM,
    "반응형 / 비동기":                          MAINSTREAM,
    "로컬 데이터":                             MAINSTREAM,
    "테스트 / 배포":                           MAINSTREAM,

    # ── 백엔드 ──
    "Pick a Language":                      ESSENTIAL,
    "Learn about APIs":                     ESSENTIAL,
    "Relational Databases":                 ESSENTIAL,
    "Containerization vs Virtualization":   ESSENTIAL,
    "Web Security":                         ESSENTIAL,
    "Caching":                              MAINSTREAM,
    "NoSQL Databases":                      MAINSTREAM,
    "Testing":                              MAINSTREAM,
    "CI / CD":                              MAINSTREAM,
    "More about Databases":                 MAINSTREAM,
    "Architectural Patterns":               MAINSTREAM,
    "Message Brokers":                      MAINSTREAM,
    "Web Servers":                          MAINSTREAM,
    "Design and Development Principles":    MAINSTREAM,
    "Scaling Databases":                    GROWING,
    "Search Engines":                       GROWING,
    "Real-Time Data":                       GROWING,
    "Building For Scale":                   GROWING,

    # ── 프론트엔드 ──
    "HTML":                                 ESSENTIAL,
    "CSS":                                  ESSENTIAL,
    "JavaScript":                           ESSENTIAL,
    "Pick a Framework":                     ESSENTIAL,
    "Package Managers":                     MAINSTREAM,
    "Build Tools":                          MAINSTREAM,
    "Linters and Formatters":               MAINSTREAM,
    "Module Bundlers":                      MAINSTREAM,
    "Authentication Strategies":            MAINSTREAM,
    "Web Security Basics":                  MAINSTREAM,
    "Type Checkers":                        RISING,
    "SSR":                                  RISING,
    "Writing CSS":                          MODERATE,
    "CSS Architecture":                     MODERATE,
    "CSS Preprocessors":                    MODERATE,
    "GraphQL":                              GROWING,
    "Static Site Generators":               GROWING,
    "PWAs":                                 MODERATE,
    "Mobile Apps":                          MODERATE,
    "Desktop Apps":                         MODERATE,
    "Web Components":                       MODERATE,

    # ── DevOps ──
    "Learn a Programming Language":         ESSENTIAL,
    "Operating System":                     ESSENTIAL,
    "Terminal Knowledge":                   ESSENTIAL,
    "Containers":                           ESSENTIAL,
    "Cloud Providers":                      ESSENTIAL,
    "Networking & Protocols":               ESSENTIAL,
    "CI / CD Tools":                        MAINSTREAM,
    "What is and how to setup X ?":         MAINSTREAM,
    "Provisioning":                         RISING,
    "Configuration Management":             MAINSTREAM,
    "Secret Management":                    MAINSTREAM,
    "Infrastructure Monitoring":            MAINSTREAM,
    "Logs Management":                      MAINSTREAM,
    "Container Orchestration":              RISING,
    "Artifact Management":                  MODERATE,
    "GitOps":                               GROWING,
    "Service Mesh":                         GROWING,
    "Cloud Design Patterns":                GROWING,
    "Application Monitoring":               GROWING,
    "Serverless":                           GROWING,
}

# ── 기술명 → 티어 (subtopic 매칭용, 소문자 정규화 키) ──
# 데이터셋에서 추출한 분류 + 보완
_TECH_TIER: dict[str, str] = {
    # 필수 ─────────────────────────────────────────────────────────
    "git":              ESSENTIAL,
    "github":           ESSENTIAL,
    "linux":            ESSENTIAL,
    "bash":             ESSENTIAL,
    "docker":           ESSENTIAL,
    "aws":              ESSENTIAL,
    "python":           ESSENTIAL,
    "java":             ESSENTIAL,
    "http":             ESSENTIAL,
    "https":            ESSENTIAL,
    "dns":              ESSENTIAL,
    "rest":             ESSENTIAL,
    "rest api":         ESSENTIAL,
    "sql":              ESSENTIAL,
    "mysql":            ESSENTIAL,
    "postgresql":       ESSENTIAL,
    "mariadb":          ESSENTIAL,
    "jwt":              ESSENTIAL,
    "oauth":            ESSENTIAL,
    "oauth2":           ESSENTIAL,
    "ssl":              ESSENTIAL,
    "tls":              ESSENTIAL,
    "cors":             ESSENTIAL,

    # 급부상 ───────────────────────────────────────────────────────
    "typescript":       RISING,
    "next.js":          RISING,
    "nextjs":           RISING,
    "fastapi":          RISING,
    "kubernetes":       RISING,
    "k8s":              RISING,
    "terraform":        RISING,
    "langchain":        RISING,
    "langgraph":        RISING,
    "rag":              RISING,
    "vite":             RISING,
    "zustand":          RISING,
    "dbt":              RISING,
    "tailwind":         RISING,
    "tailwindcss":      RISING,
    "tailwind css":     RISING,

    # 주류 ─────────────────────────────────────────────────────────
    "react":            MAINSTREAM,
    "spring boot":      MAINSTREAM,
    "spring":           MAINSTREAM,
    "node.js":          MAINSTREAM,
    "nodejs":           MAINSTREAM,
    "nestjs":           MAINSTREAM,
    "express":          MAINSTREAM,
    "javascript":       MAINSTREAM,
    "redis":            MAINSTREAM,
    "mongodb":          MAINSTREAM,
    "elasticsearch":    MAINSTREAM,
    "nginx":            MAINSTREAM,
    "jenkins":          MAINSTREAM,
    "gitlab":           MAINSTREAM,
    "gcp":              MAINSTREAM,
    "azure":            MAINSTREAM,
    "graphql":          MAINSTREAM,
    "grpc":             MAINSTREAM,
    "websocket":        MAINSTREAM,
    "websockets":       MAINSTREAM,
    "kotlin":           MAINSTREAM,
    "swift":            MAINSTREAM,
    "go":               MAINSTREAM,
    "vue":              MAINSTREAM,
    "webpack":          MAINSTREAM,
    "jest":             MAINSTREAM,
    "junit":            MAINSTREAM,
    "microservices":    MAINSTREAM,
    "gradle":           MAINSTREAM,
    "maven":            MAINSTREAM,
    "jpa":              MAINSTREAM,
    "hibernate":        MAINSTREAM,
    "orm":              MAINSTREAM,
    "orms":             MAINSTREAM,

    # 성장 ─────────────────────────────────────────────────────────
    "kafka":            GROWING,
    "rabbitmq":         GROWING,
    "airflow":          GROWING,
    "mlflow":           GROWING,
    "xgboost":          GROWING,
    "lightgbm":         GROWING,
    "pytorch":          GROWING,
    "tensorflow":       GROWING,
    "bigquery":         GROWING,
    "hadoop":           GROWING,
    "spark":            GROWING,
    "ansible":          GROWING,
    "github actions":   GROWING,
    "event sourcing":   GROWING,
    "cqrs":             GROWING,
    "domain driven":    GROWING,
    "ddd":              GROWING,

    # 중간 ─────────────────────────────────────────────────────────
    "php":              MODERATE,
    "ruby":             MODERATE,
    "rust":             MODERATE,
    "c#":               MODERATE,
    "angular":          MODERATE,
    "svn":              MODERATE,
    "jquery":           MODERATE,
    "solidity":         MODERATE,
    "erlang":           MODERATE,
    "elixir":           MODERATE,
    "memcached":        MODERATE,
    "sqlite":           MODERATE,
}


def _tokens(s: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9#+.]+", s.lower()))


def _match_tech(label: str) -> str | None:
    """라벨을 _TECH_TIER에서 토큰 기반으로 찾는다. 없으면 None."""
    label_l = label.lower().strip()

    # 완전 일치 우선
    if label_l in _TECH_TIER:
        return _TECH_TIER[label_l]

    lt = _tokens(label_l)
    best: str | None = None
    best_len = 0

    for tech, tier in _TECH_TIER.items():
        tt = _tokens(tech)
        if not tt:
            continue
        if tt <= lt or lt <= tt:
            if len(tt) > best_len:
                best = tier
                best_len = len(tt)

    return best


@lru_cache(maxsize=None)
def get_tier(label: str, node_type: str = "subtopic") -> str:
    """
    roadmap 노드 라벨 → 시장 티어 반환.

    topic 타입은 _TOPIC_TIER 우선, subtopic은 _TECH_TIER 우선.
    """
    if node_type == "topic":
        if label in _TOPIC_TIER:
            return _TOPIC_TIER[label]
        # topic인데 매핑 없으면 기술명으로도 시도
        matched = _match_tech(label)
        return matched if matched else MODERATE

    # subtopic
    matched = _match_tech(label)
    if matched:
        return matched

    # subtopic인데 미매핑이면 topic 테이블도 확인
    if label in _TOPIC_TIER:
        return _TOPIC_TIER[label]

    return MODERATE


def is_highlighted(tier: str) -> bool:
    return tier in HIGHLIGHT_TIERS
