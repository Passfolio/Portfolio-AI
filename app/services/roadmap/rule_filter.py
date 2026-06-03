"""
rule_filter.py — 프로젝트 분석 → 로드맵 노드 매칭 (covered / partial / uncovered)

[입력 스키마] (고정)
  service_name, service_description, dev_type, frameworks, skills,
  user_role, contribute_percent, period(start/end/days),
  analysis_form: {
    core_feat:   [{feat_title, description}],
    core_perf:   [{perf_title, description, about_feat_title}],
    feedback:    [{feedback_title, feedback, about_perf_title}],
    contribute:  {contribute_titles: [...]},
  }

[신호 강도]
  strong — skills, core_feat, core_perf, contribute.contribute_titles  (구현 증거)
  weak   — feedback (검증 제안일 뿐 구현 증거 아님 → partial 전용)

[분류]  covered = strong 적중 / partial = weak만 적중 / uncovered = 미적중
[점수]  core×2, support×1 / covered=1.0, partial=0.5, uncovered=0.0

[v2 변경점]
  · 라벨 매칭을 substring → 토큰 집합 기반으로 교체 (java↔javascript, go↔mongodb 오탐 제거)
  · parent_topic은 '명시값 + curated 사전'만 사용 (위치 추론 폴백 제거: Java→Internet 오귀속 방지)
  · frontend/devops tier 테이블을 CSV 라벨과 동기화
  · 로드맵 순서 준수 점검(analyze_sequence) 추가 → per_role.sequence

[v3 변경점]
  · 한국 채용시장 기반 kr_roadmap.csv 지원 추가 (source='kr-market-2026')
  · KR_TOPIC_TIERS: 8개 직군 (backend/frontend/devops/android/ios/ai-engineer/machine-learning/data-engineer)
  · KR_SKILL_SYNONYMS: 한국 로드맵 라벨 기반 스킬 매핑
  · android/ios/data-engineer 신규 직군 지원
"""
from __future__ import annotations
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
# 1. 설정
# ══════════════════════════════════════════════════════════════════════════════

FRAMEWORK_TO_ROLES: dict[str, list[str]] = {
    "spring_boot": ["backend"], "django": ["backend"], "fastapi": ["backend"],
    "flask": ["backend"], "express": ["backend"], "nestjs": ["backend"],
    "laravel": ["backend"], "rails": ["backend"],
    "react": ["frontend"], "vue": ["frontend"], "angular": ["frontend"],
    "nextjs": ["frontend"], "next": ["frontend"],
    "flutter": ["mobile"], "react_native": ["mobile"],
    "ios": ["ios"], "swift": ["ios"], "swiftui": ["ios"],
    "android": ["android"], "kotlin": ["android"],
}

SKILL_TO_ROLES: dict[str, list[str]] = {
    "langchain": ["ai-engineer"], "langchain4j": ["ai-engineer"],
    "langgraph": ["ai-engineer"], "crewai": ["ai-engineer"],
    "tensorflow": ["machine-learning"], "pytorch": ["machine-learning"],
    "scikit": ["machine-learning"], "mlflow": ["machine-learning"],
    "docker": ["devops"], "kubernetes": ["devops"], "terraform": ["devops"],
    "airflow": ["data-engineer"], "spark": ["data-engineer"],
    "kafka": ["data-engineer"], "dbt": ["data-engineer"],
}

VALID_PRIMARY_ROLES = {"backend", "frontend", "mobile", "android", "ios", "data-engineer"}

# Role별 topic tier (정의 안 된 topic은 기본 support). CSV 라벨과 1:1로 동기화됨.
TOPIC_TIERS: dict[str, dict[str, str]] = {
    "backend": {
        "Internet": "core", "Pick a Language": "core", "Learn about APIs": "core",
        "Relational Databases": "core", "NoSQL Databases": "support",
        "More about Databases": "core", "Scaling Databases": "support",
        "Caching": "core", "Web Security": "core", "Testing": "core",
        "CI / CD": "support", "Architectural Patterns": "core",
        "Message Brokers": "support", "Containerization vs Virtualization": "core",
        "Web Servers": "support", "Version Control Systems": "support",
        "Repo Hosting Services": "support", "Design and Development Principles": "core",
        "Search Engines": "support", "Real-Time Data": "support",
        "Building For Scale": "core",
    },
    "frontend": {
        "Internet": "core", "HTML": "core", "CSS": "core", "JavaScript": "core",
        "VCS Hosting": "support", "Version Control Systems": "support",
        "Package Managers": "support", "Pick a Framework": "core",
        "Writing CSS": "support", "CSS Architecture": "support",
        "CSS Preprocessors": "support", "Build Tools": "support",
        "Linters and Formatters": "support", "Module Bundlers": "support",
        "Testing": "core", "Authentication Strategies": "core",
        "Web Security Basics": "core", "Web Components": "support",
        "Type Checkers": "support", "SSR": "support", "GraphQL": "support",
        "Static Site Generators": "support", "PWAs": "support",
        "Mobile Apps": "support", "Desktop Apps": "support",
    },
    "devops": {
        "Learn a Programming Language": "core", "Operating System": "core",
        "Terminal Knowledge": "core", "Version Control Systems": "core",
        "VCS Hosting": "support", "What is and how to setup X ?": "support",
        "Containers": "core", "Cloud Providers": "core",
        "Networking & Protocols": "core", "Serverless": "support",
        "Provisioning": "support", "Configuration Management": "support",
        "CI / CD Tools": "core", "Secret Management": "support",
        "Infrastructure Monitoring": "support", "Logs Management": "support",
        "Container Orchestration": "support", "Artifact Management": "support",
        "GitOps": "support", "Service Mesh": "support",
        "Cloud Design Patterns": "support", "Application Monitoring": "support",
    },
}

# ── 한국 채용시장 기반 로드맵 tier 정의 (kr-market-2026 소스) ──────────────────
KR_TOPIC_TIERS: dict[str, dict[str, str]] = {
    "backend": {
        "프로그래밍 언어": "core", "HTTP / 웹 기초": "core", "웹 프레임워크": "core",
        "관계형 데이터베이스": "core", "인증 / 보안": "core", "API 설계": "core",
        "테스트": "core", "컨테이너 / 배포": "core", "아키텍처 패턴": "core",
        "성능 / 스케일링": "core",
        "Git / 협업": "support", "NoSQL / 캐싱": "support", "클라우드 기초": "support",
        "메시지 브로커": "support", "AI 연동 (선택)": "support",
    },
    "frontend": {
        "HTML5 / CSS3 기초": "core", "JavaScript (ES6+)": "core", "TypeScript": "core",
        "React": "core", "상태 관리": "core", "Next.js": "core",
        "테스트": "core", "성능 최적화": "core",
        "Git / 협업": "support", "스타일링": "support", "빌드 도구": "support",
        "백엔드 연동": "support",
    },
    "devops": {
        "Linux / 서버 운영": "core", "네트워크 기초": "core", "컨테이너화": "core",
        "클라우드 - AWS": "core", "컨테이너 오케스트레이션": "core",
        "IaC (인프라 코드화)": "core", "CI/CD 파이프라인": "core",
        "스크립팅 / 자동화": "core",
        "버전 관리": "support", "클라우드 - 멀티클라우드": "support",
        "모니터링 / 관측성": "support", "보안": "support", "SRE 실무": "support",
    },
    "android": {
        "Kotlin 기초": "core", "Android 기본": "core", "Jetpack 컴포넌트": "core",
        "비동기 / 반응형": "core", "아키텍처 패턴": "core", "네트워크 연동": "core",
        "배포 / CI/CD": "support",
    },
    "ios": {
        "Swift 기초": "core", "UIKit / SwiftUI": "core", "반응형 / 비동기": "core",
        "아키텍처 패턴": "core", "네트워크 연동": "core",
        "로컬 데이터": "support", "테스트 / 배포": "support",
    },
    "ai-engineer": {
        "Python 심화": "core", "LLM API 활용": "core", "RAG 파이프라인": "core",
        "AI 에이전트": "core", "프롬프트 엔지니어링": "core", "배포 / 서빙": "core",
        "Fine-tuning": "support", "데이터 처리": "support",
    },
    "machine-learning": {
        "Python + 수학 기초": "core", "전통 ML": "core", "딥러닝 - PyTorch": "core",
        "실험 관리": "core", "모델 서빙 / 최적화": "core",
        "딥러닝 응용": "support", "ML 인프라": "support",
    },
    "data-engineer": {
        "SQL 심화": "core", "Python 데이터 처리": "core",
        "파이프라인 오케스트레이션": "core", "클라우드 데이터 플랫폼": "core",
        "이벤트 스트리밍": "support", "대규모 처리": "support",
        "데이터 변환 / 품질": "support", "컨테이너 / 인프라": "support",
    },
}

# 한국 로드맵 스킬 매핑: skill 소문자 키 → 한국 로드맵 라벨 토큰
KR_SKILL_SYNONYMS: dict[str, list[str]] = {
    # 언어
    "java": ["java"], "python": ["python"], "kotlin": ["kotlin"],
    "typescript": ["typescript"], "javascript": ["javascript"],
    "go": ["go"], "swift": ["swift"], "rust": ["rust"], "scala": ["scala"],
    # 프론트엔드
    "react": ["react"], "react.js": ["react"],
    "next.js": ["next"], "nextjs": ["next"], "next": ["next"],
    "vue": ["vue"], "vue.js": ["vue"],
    "html": ["html5"], "html5": ["html5"],
    "css": ["css3"], "css3": ["css3"],
    "flexbox": ["css3", "flexbox"], "grid": ["css3", "grid"],
    "zustand": ["zustand"],
    "react query": ["react", "query"], "tanstack": ["react", "query"],
    "redux": ["redux"], "recoil": ["recoil"],
    "vite": ["vite"],
    "tailwind": ["tailwind"], "tailwind css": ["tailwind"],
    "styled-components": ["styled", "components"],
    "storybook": ["storybook"],
    "jest": ["jest"], "cypress": ["cypress"], "playwright": ["playwright"],
    "react testing library": ["react", "testing"],
    "web vitals": ["web", "vitals"],
    # 백엔드 프레임워크
    "spring boot": ["spring boot"], "spring": ["spring boot"],
    "nestjs": ["nestjs"], "fastapi": ["fastapi"],
    "express": ["express"], "django": ["django"],
    # DB
    "mysql": ["mysql", "mariadb"], "mariadb": ["mysql", "mariadb"],
    "postgresql": ["postgresql"], "redis": ["redis"],
    "mongodb": ["mongodb"], "elasticsearch": ["elasticsearch"],
    "oracle": ["oracle"],
    # ORM
    "jpa": ["jpa", "prisma", "sqlalchemy"], "hibernate": ["jpa"],
    "prisma": ["jpa", "prisma", "sqlalchemy"],
    "sqlalchemy": ["jpa", "prisma", "sqlalchemy"],
    # 인증
    "jwt": ["jwt"], "oauth2": ["oauth2"], "oauth": ["oauth2"],
    "spring security": ["spring security"],
    # 컨테이너
    "docker": ["docker"], "kubernetes": ["kubernetes"], "k8s": ["kubernetes"],
    "helm": ["helm"], "argocd": ["argocd"],
    # 클라우드
    "aws": ["aws"], "gcp": ["gcp"], "azure": ["azure"],
    "ncp": ["ncp"], "lambda": ["lambda"],
    "terraform": ["terraform"], "ansible": ["ansible"],
    # CI/CD
    "github actions": ["github actions"], "jenkins": ["jenkins"],
    "gitlab": ["gitlab"], "gitops": ["argocd"],
    # AI/LLM
    "langchain": ["langchain"], "langgraph": ["langgraph"],
    "crewai": ["crewai"], "openai": ["openai"],
    "anthropic": ["claude"], "claude": ["claude"],
    "hugging face": ["hugging face"], "rag": ["rag"],
    "vector db": ["vector db"], "pinecone": ["pinecone"],
    "weaviate": ["weaviate"], "chroma": ["chroma"],
    # ML
    "pytorch": ["pytorch"], "tensorflow": ["tensorflow"],
    "keras": ["keras"], "scikit-learn": ["scikit-learn"],
    "scikit": ["scikit-learn"], "xgboost": ["xgboost"],
    "lightgbm": ["lightgbm"], "mlflow": ["mlflow"],
    "onnx": ["onnx"], "triton": ["triton"],
    # 데이터 엔지니어링
    "airflow": ["airflow"], "kafka": ["kafka"],
    "spark": ["spark"], "flink": ["flink"],
    "bigquery": ["bigquery"], "redshift": ["redshift"],
    "dbt": ["dbt"], "databricks": ["databricks"],
    # 모바일 Android
    "jetpack compose": ["jetpack compose"], "compose": ["jetpack compose"],
    "hilt": ["hilt"], "room": ["room"],
    "coroutines": ["coroutines"], "flow": ["flow"],
    "retrofit": ["retrofit"], "ktor": ["ktor"],
    "mvvm": ["mvvm"], "mvc": ["mvc"], "mvi": ["mvi"],
    "clean architecture": ["clean", "architecture"],
    "repository": ["repository"], "viewmodel": ["viewmodel"],
    "livedata": ["livedata"], "stateflow": ["stateflow"],
    "navigation": ["navigation"], "workmanager": ["workmanager"],
    "activity": ["activity"], "fragment": ["fragment"],
    "recyclerview": ["recyclerview"],
    "firebase": ["firebase"], "google play": ["google", "play"],
    # 모바일 iOS
    "swiftui": ["swiftui"], "uikit": ["uikit"],
    "combine": ["combine"], "rxswift": ["rxswift"],
    "alamofire": ["alamofire"],
    "coordinator": ["coordinator"],
    "core data": ["core", "data"], "keychain": ["keychain"],
    "xctest": ["xctest"], "testflight": ["testflight"],
    "async/await": ["async", "await"],
    # 공통
    "git": ["git"], "github": ["github"], "gitlab ci": ["gitlab"],
    "rest": ["rest"], "restful": ["restful"], "graphql": ["graphql"],
    "grpc": ["grpc"], "nginx": ["nginx"],
    "rabbitmq": ["rabbitmq"],
}

# 한국 로드맵 키워드 → 라벨 추론
KR_TEXT_KEYWORD_INFERENCES: dict[str, list[str]] = {
    "트랜잭션": ["트랜잭션", "acid"], "transaction": ["트랜잭션", "acid"],
    "인덱스": ["인덱스"], "index": ["인덱스"],
    "캐시": ["캐싱"], "cache": ["캐싱"], "redis": ["redis"],
    "비동기": ["비동기"], "async": ["비동기"],
    "마이크로서비스": ["마이크로서비스"], "microservice": ["마이크로서비스"],
    "이벤트 드리븐": ["이벤트 드리븐"], "event driven": ["이벤트 드리븐"],
    "단위 테스트": ["단위 테스트"], "unit test": ["단위 테스트"],
    "통합 테스트": ["통합 테스트"], "integration test": ["통합 테스트"],
    "jwt": ["jwt"], "oauth": ["oauth2"],
    "docker": ["docker"], "kubernetes": ["kubernetes"],
    "terraform": ["terraform"], "airflow": ["airflow"],
    "rag": ["rag"], "langchain": ["langchain"],
    "pagination": ["인덱스"], "페이지네이션": ["인덱스"],
    "샤딩": ["db 샤딩"], "sharding": ["db 샤딩"],
    "복제": ["db 샤딩"], "replicat": ["db 샤딩"],
    "웹소켓": ["websocket"], "websocket": ["websocket"],
    "ddd": ["ddd"], "도메인 주도": ["ddd"],
    "cqrs": ["cqrs"], "event sourcing": ["cqrs"],
}

# 빈 parent_topic 보정 사전 (위치 추론 대신 명시적으로). 필요 시 확장.
CURATED_PARENTS: dict[str, dict[str, str]] = {
    "backend": {
        "Rust": "Pick a Language", "PHP": "Pick a Language", "Go": "Pick a Language",
        "JavaScript": "Pick a Language", "Java": "Pick a Language",
        "Python": "Pick a Language", "C#": "Pick a Language", "Ruby": "Pick a Language",
        "GitHub": "Repo Hosting Services",
        "Authentication": "Web Security", "Basic Authentication": "Web Security",
        "Token Authentication": "Web Security", "JWT": "Web Security",
        "OAuth": "Web Security", "OpenID": "Web Security", "Cookie Based Auth": "Web Security",
        "SAML": "Web Security",
    },
}

# skill 문자열 → 로드맵 라벨(소문자). 매핑 값은 실제 노드 라벨이어야 토큰 매칭됨.
SKILL_SYNONYMS: dict[str, list[str]] = {
    "java": ["java"], "python": ["python"], "javascript": ["javascript"],
    "typescript": ["typescript"], "go": ["go"], "rust": ["rust"],
    "ruby": ["ruby"], "php": ["php"], "c#": ["c#"],
    "spring boot": ["java", "rest"],
    "spring security": ["authentication", "web security", "jwt"],
    "spring data jpa": ["orms", "relational databases"],
    "langchain4j": [], "langchain": [],
    "mariadb": ["mariadb", "relational databases"],
    "mysql": ["mysql", "relational databases"],
    "postgresql": ["postgresql", "relational databases"],
    "oracle": ["oracle", "relational databases"],
    "mongodb": ["mongodb", "nosql databases"],
    "redis": ["redis", "server side", "caching"],
    "sqlite": ["sqlite"],
    "jpa/hibernate": ["orms"], "hibernate": ["orms"],
    "querydsl": ["orms"],
    "caffeine cache": ["server side", "caching"], "caffeine": ["server side", "caching"],
    "memcached": ["memcached", "caching"],
    "jwt": ["jwt", "token authentication", "authentication"],
    "oauth": ["oauth", "authentication"], "oauth2": ["oauth", "authentication"],
    "docker": ["containerization vs virtualization"],
    "kubernetes": ["containerization vs virtualization", "container orchestration"],
    "git": ["git"], "github": ["github"], "gitlab": ["gitlab"],
    "rest": ["rest"], "restful": ["rest"], "graphql": ["graphql"],
    "grpc": ["grpc"], "soap": ["soap"], "websocket": ["websockets"],
    "kafka": ["kafka"], "rabbitmq": ["rabbitmq"], "nginx": ["nginx"],
}

# 구현 텍스트 키워드 → 로드맵 라벨(소문자).
TEXT_KEYWORD_INFERENCES: dict[str, list[str]] = {
    "transaction": ["transactions", "acid"], "트랜잭션": ["transactions", "acid"],
    "pessimistic": ["transactions"], "optimistic lock": ["transactions"],
    "cache": ["caching", "server side"], "캐시": ["caching", "server side"],
    "pagination": ["n+1 problem"], "페이지네이션": ["n+1 problem"],
    "n+1": ["n+1 problem"],
    "rate limit": ["throttling"], "throttl": ["throttling"], "레이트": ["throttling"],
    "circuit breaker": ["circuit breaker"],
    "backoff": ["graceful degradation"], "백오프": ["graceful degradation"],
    "retry": ["graceful degradation"], "재시도": ["graceful degradation"],
    "monitoring": ["monitoring", "observability"],
    "telemetry": ["telemetry", "observability"],
    "scheduler": ["ci / cd"], "cron": ["ci / cd"], "스케줄러": ["ci / cd"],
    "migration": ["migrations"], "마이그레이션": ["migrations"],
    "index": ["database indexes"], "인덱스": ["database indexes"],
    "shard": ["sharding strategies"], "샤딩": ["sharding strategies"],
    "replicat": ["data replication"], "복제": ["data replication"],
    "websocket": ["websockets"], "웹소켓": ["websockets"],
    "unit test": ["unit testing", "testing"], "단위 테스트": ["unit testing", "testing"],
    "integration test": ["integration testing", "testing"],
    "통합 테스트": ["integration testing", "testing"],
    "cors": ["cors"], "ssl": ["ssl/tls"], "https": ["https"],
    "microservice": ["microservices"], "마이크로서비스": ["microservices"],
    "twelve factor": ["twelve factor apps"],
    "event sourcing": ["event sourcing"], "cqrs": ["cqrs"],
    "domain driven": ["domain driven design"], "ddd": ["domain driven design"],
    "design pattern": ["gof design patterns"], "디자인 패턴": ["gof design patterns"],
    "semaphore": ["building for scale"], "세마포어": ["building for scale"],
    "virtual thread": ["building for scale"], "가상 스레드": ["building for scale"],
    "async": ["building for scale"], "비동기": ["building for scale"],
    "동시성": ["building for scale"], "concurren": ["building for scale"],
    "rest": ["rest"], "http": ["internet", "what is http?"],
    "dns": ["dns and how it works?"],
    "sse": ["server sent events", "real-time data"],
    "server sent": ["server sent events", "real-time data"],
    "server-sent": ["server sent events", "real-time data"],
    "실시간 알림": ["server sent events", "real-time data"],
    "heartbeat": ["server sent events", "real-time data"],
    "하트비트": ["server sent events", "real-time data"],
    "message broker": ["message brokers", "rabbitmq"],
    "메시지 브로커": ["message brokers"],
    "pub/sub": ["message brokers"], "pubsub": ["message brokers"],
}

_LEVEL_THRESHOLDS = [
    (0.75, "시니어 진입 단계"),
    (0.55, "중급 (실무 가능) 단계"),
    (0.35, "초중급 단계"),
    (0.0,  "입문 단계"),
]

_STATUS_SCORE = {"covered": 1.0, "partial": 0.5, "uncovered": 0.0}
_TIER_WEIGHT = {"core": 2, "support": 1}


# ══════════════════════════════════════════════════════════════════════════════
# 2. 데이터 모델
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RoadmapNode:
    node_id: str
    topic_label: str
    parent_topic: str
    node_type: str
    order_index: int
    roadmap_role: str


@dataclass
class FilterResult:
    service_name: str
    primary_roles: list[str]
    secondary_roles: list[str]
    per_role: dict[str, dict] = field(default_factory=dict)


def load_roadmap(csv_path: str | Path) -> list[RoadmapNode]:
    nodes: list[RoadmapNode] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            oi = row.get("order_index", "")
            nodes.append(RoadmapNode(
                node_id=row["node_id"],
                topic_label=row["topic_label"],
                parent_topic=row["parent_topic"],
                node_type=row["node_type"],
                order_index=int(oi) if oi.isdigit() else 0,
                roadmap_role=row["roadmap_role"],
            ))
    return nodes


# ══════════════════════════════════════════════════════════════════════════════
# 3. 토큰 기반 라벨 매칭  (substring 오탐 제거)
# ══════════════════════════════════════════════════════════════════════════════

def _tokens(s: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9#+]+", s.lower()))


def _word_in(needle: str, haystack: str) -> bool:
    """needle이 haystack 안에 완전한 단어(word boundary)로 존재하는지 확인."""
    return bool(re.search(r"(?<![a-z0-9#])" + re.escape(needle) + r"(?![a-z0-9#])",
                          haystack))


def _label_matched(node_label: str, signal_labels: set[str]) -> bool:
    """node 라벨 토큰집합이 어떤 신호 라벨과 부분집합 관계일 때만 매칭."""
    lt = _tokens(node_label)
    if not lt:
        return False
    for sig in signal_labels:
        st = _tokens(sig)
        if st and (lt <= st or st <= lt):
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# 4. 신호 수집  (고정 스키마 기준)
# ══════════════════════════════════════════════════════════════════════════════

def _infer_labels(text: str, kr_mode: bool = False) -> set[str]:
    """텍스트에서 SKILL_SYNONYMS + TEXT_KEYWORD_INFERENCES 로 로드맵 라벨 추론."""
    t = text.lower()
    out: set[str] = set()
    synonyms = KR_SKILL_SYNONYMS if kr_mode else SKILL_SYNONYMS
    keyword_map = KR_TEXT_KEYWORD_INFERENCES if kr_mode else TEXT_KEYWORD_INFERENCES
    for key, mapped in synonyms.items():
        if _word_in(key, t):
            out.update(m.lower() for m in mapped)
    for key, mapped in keyword_map.items():
        if _word_in(key, t):
            out.update(m.lower() for m in mapped)
    return out


def collect_signal_labels(project: dict[str, Any], kr_mode: bool = False) -> tuple[set[str], set[str]]:
    """Returns (strong_labels, weak_labels)."""
    form = project.get("analysis_form", {})
    synonyms = KR_SKILL_SYNONYMS if kr_mode else SKILL_SYNONYMS

    # strong ── skills (명시 기술)
    strong: set[str] = set()
    for vals in project.get("skills", {}).values():
        for v in (vals if isinstance(vals, list) else [vals]):
            sl = str(v).lower()
            for key, mapped in synonyms.items():
                if _word_in(key, sl) or _word_in(sl, key):
                    strong.update(m.lower() for m in mapped)

    # strong ── 구현 텍스트 (core_feat / core_perf / contribute_titles)
    impl: list[str] = []
    for feat in form.get("core_feat", []):
        impl.append(f"{feat.get('feat_title','')} {feat.get('description','')}")
    for perf in form.get("core_perf", []):
        impl.append(f"{perf.get('perf_title','')} {perf.get('description','')}")
    for title in form.get("contribute", {}).get("contribute_titles", []):
        impl.append(str(title))
    strong |= _infer_labels(" ".join(impl), kr_mode=kr_mode)

    # weak ── feedback (검증 제안)
    weak_parts: list[str] = []
    for fb in form.get("feedback", []):
        weak_parts.append(f"{fb.get('feedback_title','')} {fb.get('feedback','')}")
    weak = _infer_labels(" ".join(weak_parts), kr_mode=kr_mode) - strong

    return strong, weak


# ══════════════════════════════════════════════════════════════════════════════
# 5. parent_topic  (명시값 + curated 만)
# ══════════════════════════════════════════════════════════════════════════════

def build_parent_map(role_nodes: list[RoadmapNode], role: str) -> dict[str, str]:
    curated = CURATED_PARENTS.get(role, {})
    pmap: dict[str, str] = {}
    for n in role_nodes:
        if n.node_type != "subtopic":
            continue
        if n.parent_topic:
            pmap[n.topic_label] = n.parent_topic
        elif n.topic_label in curated:
            pmap[n.topic_label] = curated[n.topic_label]
        # 그 외 빈 parent subtopic은 부모 전파 제외 (과대평가 방지)
    return pmap


# ══════════════════════════════════════════════════════════════════════════════
# 6. 분류  (직접 매칭 + subtopic→parent 전파)
# ══════════════════════════════════════════════════════════════════════════════

def classify_nodes(role_nodes: list[RoadmapNode], strong: set[str], weak: set[str],
                   parent_map: dict[str, str]) -> dict[str, str]:
    status: dict[str, str] = {}
    for n in role_nodes:
        if _label_matched(n.topic_label, strong):
            status[n.topic_label] = "covered"
        elif _label_matched(n.topic_label, weak):
            status[n.topic_label] = "partial"
        else:
            status[n.topic_label] = "uncovered"

    # subtopic이 있는 topic은 모든 subtopic이 covered일 때만 covered
    parent_to_subs: dict[str, list[str]] = {}
    for sub, parent in parent_map.items():
        parent_to_subs.setdefault(parent, []).append(sub)

    for parent, subs in parent_to_subs.items():
        if not subs:
            continue
        sub_statuses = [status.get(s, "uncovered") for s in subs]
        if all(s == "covered" for s in sub_statuses):
            status[parent] = "covered"
        elif any(s in ("covered", "partial") for s in sub_statuses):
            status[parent] = "partial"
        else:
            status[parent] = "uncovered"
    return status


# ══════════════════════════════════════════════════════════════════════════════
# 7. 점수 / 레벨
# ══════════════════════════════════════════════════════════════════════════════

def _get_tier_map(role: str, kr_mode: bool) -> dict[str, str]:
    if kr_mode:
        return KR_TOPIC_TIERS.get(role, {})
    return TOPIC_TIERS.get(role, {})


def compute_score(role_nodes: list[RoadmapNode], status: dict[str, str],
                  role: str, kr_mode: bool = False) -> tuple[float, str]:
    tier_map = _get_tier_map(role, kr_mode)
    total = earned = 0.0
    for n in role_nodes:
        if n.node_type != "topic":
            continue
        w = _TIER_WEIGHT[tier_map.get(n.topic_label, "support")]
        total += w
        earned += _STATUS_SCORE[status.get(n.topic_label, "uncovered")] * w
    pct = earned / total if total else 0.0
    level = next(lbl for th, lbl in _LEVEL_THRESHOLDS if pct >= th)
    return round(pct, 3), level


# ══════════════════════════════════════════════════════════════════════════════
# 8. 로드맵 순서 준수 점검  (신규)
# ══════════════════════════════════════════════════════════════════════════════

def analyze_sequence(role_nodes: list[RoadmapNode], status: dict[str, str],
                     role: str, kr_mode: bool = False) -> dict:
    """order_index = 로드맵 의도 순서. 전선(frontier) 이전의 미학습 core를 '건너뜀'으로 표시."""
    tier_map = _get_tier_map(role, kr_mode)
    topics = sorted([n for n in role_nodes if n.node_type == "topic"],
                    key=lambda n: n.order_index)
    if not topics:
        return {}

    covered_idx = [t.order_index for t in topics if status.get(t.topic_label) == "covered"]
    frontier = max(covered_idx) if covered_idx else None

    contiguous_until = None
    for t in topics:
        if status.get(t.topic_label) == "covered":
            contiguous_until = t.topic_label
        else:
            break

    skipped_core, skipped_support = [], []
    before_total = before_covered = 0
    for t in topics:
        if frontier is None or t.order_index >= frontier:
            continue
        before_total += 1
        if status.get(t.topic_label) == "covered":
            before_covered += 1
        else:
            tier = tier_map.get(t.topic_label, "support")
            (skipped_core if tier == "core" else skipped_support).append(t.topic_label)

    in_order_ratio = round(before_covered / before_total, 3) if before_total else 1.0

    if not covered_idx:
        verdict = "아직 로드맵 진입 전 단계입니다."
    elif not skipped_core:
        verdict = "핵심 토픽을 로드맵 순서에 맞게 빠짐없이 밟아왔습니다."
    elif len(skipped_core) <= 2:
        verdict = ("앞선 핵심 토픽 일부를 건너뛴 채 뒷 단계로 진행했습니다 "
                   f"(건너뛴 core: {', '.join(skipped_core)}). 보강이 권장됩니다.")
    else:
        verdict = ("로드맵 순서를 상당 부분 건너뛰고 특정 영역만 깊게 파고든 형태입니다 "
                   f"(건너뛴 core {len(skipped_core)}개). 기초 토픽 보강이 필요합니다.")

    frontier_label = next((t.topic_label for t in topics
                           if t.order_index == frontier), None)
    return {
        "frontier_topic": frontier_label,
        "contiguous_until": contiguous_until,
        "in_order_ratio": in_order_ratio,
        "skipped_core_prereqs": skipped_core,
        "skipped_support_prereqs": skipped_support,
        "verdict": verdict,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 9. 앞으로 배울 경로
# ══════════════════════════════════════════════════════════════════════════════

def build_roadmap_path(role_nodes: list[RoadmapNode], status: dict[str, str],
                       parent_map: dict[str, str], role: str,
                       kr_mode: bool = False) -> list[dict]:
    tier_map = _get_tier_map(role, kr_mode)
    parent_to_subs: dict[str, list[RoadmapNode]] = {}
    for n in role_nodes:
        if n.node_type == "subtopic":
            p = parent_map.get(n.topic_label, n.parent_topic)
            if p:
                parent_to_subs.setdefault(p, []).append(n)

    todo = sorted([n for n in role_nodes
                   if n.node_type == "topic"
                   and status.get(n.topic_label) in ("uncovered", "partial")],
                  key=lambda n: n.order_index)

    path = []
    for i, topic in enumerate(todo, 1):
        subs = sorted(parent_to_subs.get(topic.topic_label, []), key=lambda n: n.order_index)
        path.append({
            "step": i,
            "topic": topic.topic_label,
            "tier": tier_map.get(topic.topic_label, "support"),
            "status": status.get(topic.topic_label, "uncovered"),
            "subtopics_to_learn": [s.topic_label for s in subs
                                   if status.get(s.topic_label) != "covered"],
        })
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 10. 입력 정규화 (새 CDN JSON 스키마 → 내부 스키마 변환)
# ══════════════════════════════════════════════════════════════════════════════

_KR_DEV_TYPE_MAP: dict[str, str] = {
    "백엔드": "backend", "backend": "backend",
    "프론트엔드": "frontend", "frontend": "frontend",
    "풀스택": "fullstack", "fullstack": "fullstack",
    "모바일": "mobile", "mobile": "mobile",
    "android": "android", "안드로이드": "android",
    "ios": "ios",
    "devops": "devops", "데브옵스": "devops", "클라우드": "devops",
    "ai": "ai-engineer", "ai/ml": "ai-engineer", "ai엔지니어": "ai-engineer",
    "머신러닝": "machine-learning", "machine-learning": "machine-learning",
    "데이터": "data-engineer", "data-engineer": "data-engineer",
    "데이터엔지니어": "data-engineer",
    "보안": "cyber-security", "security": "cyber-security",
    "게임": "game-developer", "game": "game-developer",
}


def normalize_project(raw: dict[str, Any]) -> dict[str, Any]:
    """
    CDN JSON 스키마(새 형식) → rule_filter 내부 스키마로 변환.

    변환 항목:
    - dev_type      : list[str] 또는 str (한글 포함) → str (영문 단일값)
    - frameworks    : list[{name, category}] → list[str]
    - skills        : list[{name, category}] → dict[category, list[str]]
    - analysis      : 'analysis' 키 → 'analysis_form' 키
    - contribute_share_percent → contribute_percent
    - analysis_period          → period
    """
    p = dict(raw)

    # dev_type: list or str → single English string
    dt = p.get("dev_type", "")
    if isinstance(dt, list):
        dt = dt[0] if dt else ""
    dt_key = str(dt).lower().strip().replace(" ", "").replace("/", "")
    p["dev_type"] = _KR_DEV_TYPE_MAP.get(dt_key, dt_key)

    # frameworks: [{name, category}] → [str]
    fws = p.get("frameworks", [])
    if fws and isinstance(fws[0], dict):
        p["frameworks"] = [f["name"] for f in fws if "name" in f]

    # skills: [{name, category}] → {category: [name, ...]}  (이미 dict이면 skip)
    skills_raw = p.get("skills", [])
    if isinstance(skills_raw, list) and skills_raw and isinstance(skills_raw[0], dict):
        skills_dict: dict[str, list[str]] = {}
        for s in skills_raw:
            cat = s.get("category", "etc")
            skills_dict.setdefault(cat, []).append(s["name"])
        p["skills"] = skills_dict

    # analysis → analysis_form
    if "analysis" in p and "analysis_form" not in p:
        p["analysis_form"] = p.pop("analysis")

    # contribute_share_percent → contribute_percent
    if "contribute_share_percent" in p and "contribute_percent" not in p:
        p["contribute_percent"] = p.pop("contribute_share_percent")

    # analysis_period → period
    if "analysis_period" in p and "period" not in p:
        p["period"] = p.pop("analysis_period")

    return p


# ══════════════════════════════════════════════════════════════════════════════
# 11. Role 감지
# ══════════════════════════════════════════════════════════════════════════════

def _flatten_skills(skills: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for v in skills.values():
        out.extend(v if isinstance(v, list) else [v])
    return [str(s) for s in out if s]


def detect_roles(project: dict[str, Any]) -> tuple[list[str], list[str]]:
    primary: list[str] = []
    secondary: list[str] = []

    dev_type = str(project.get("dev_type", "")).lower().strip()
    if dev_type in VALID_PRIMARY_ROLES:
        primary.append(dev_type)

    frameworks = project.get("frameworks", [])
    if isinstance(frameworks, str):
        frameworks = [frameworks]
    for fw in frameworks:
        key = str(fw).lower().replace("-", "_").replace(" ", "_")
        for r in FRAMEWORK_TO_ROLES.get(key, []):
            if r not in primary:
                primary.append(r)

    for skill in _flatten_skills(project.get("skills", {})):
        sl = skill.lower()
        for key, roles in SKILL_TO_ROLES.items():
            if key in sl:
                for r in roles:
                    if r not in primary and r not in secondary:
                        secondary.append(r)

    if not primary:
        primary = ["backend"]
    return primary, secondary


# ══════════════════════════════════════════════════════════════════════════════
# 11. 진입점
# ══════════════════════════════════════════════════════════════════════════════

def _is_kr_roadmap(all_nodes: list[RoadmapNode]) -> bool:
    """CSV source 필드로 한국 로드맵 여부 자동 감지."""
    # node_id 접두사로 판별 (kr_ 로 시작하면 한국 로드맵)
    return any(n.node_id.startswith("kr_") for n in all_nodes[:5])


def run(project: dict[str, Any], roadmap_csv: str | Path,
        kr_mode: bool | None = None) -> FilterResult:
    """kr_mode=None 이면 CSV에서 자동 감지 (node_id 접두사 kr_ 기준)."""
    project = normalize_project(project)

    all_nodes = load_roadmap(roadmap_csv)
    if kr_mode is None:
        kr_mode = _is_kr_roadmap(all_nodes)

    primary, secondary = detect_roles(project)
    strong, weak = collect_signal_labels(project, kr_mode=kr_mode)

    result = FilterResult(
        service_name=project.get("service_name", ""),
        primary_roles=primary,
        secondary_roles=secondary,
    )

    for role in primary + secondary:
        _seen: dict[str, RoadmapNode] = {}
        for n in sorted((n for n in all_nodes if n.roadmap_role == role),
                        key=lambda n: n.order_index):
            if n.topic_label not in _seen:
                _seen[n.topic_label] = n
        role_nodes = list(_seen.values())
        if not role_nodes:
            continue

        parent_map = build_parent_map(role_nodes, role)
        status = classify_nodes(role_nodes, strong, weak, parent_map)
        score, level = compute_score(role_nodes, status, role, kr_mode=kr_mode)
        tier_map = _get_tier_map(role, kr_mode)

        def entry(n: RoadmapNode, _tm=tier_map, _pm=parent_map, _st=status) -> dict:
            return {
                "label": n.topic_label, "type": n.node_type,
                "parent": _pm.get(n.topic_label, n.parent_topic),
                "order": n.order_index,
                "tier": _tm.get(n.topic_label, "support"),
                "status": _st.get(n.topic_label, "uncovered"),
            }

        topics = [n for n in role_nodes if n.node_type == "topic"]
        ordered = sorted(role_nodes, key=lambda n: n.order_index)

        covered_list   = [entry(n) for n in ordered if status.get(n.topic_label) == "covered"]
        partial_list   = [entry(n) for n in ordered if status.get(n.topic_label) == "partial"]
        uncovered_list = [entry(n) for n in ordered if status.get(n.topic_label) == "uncovered"]

        result.per_role[role] = {
            "total": len(role_nodes),
            "topic_coverage": f"{sum(1 for t in topics if status.get(t.topic_label)=='covered')}/{len(topics)}",
            "weighted_score": score,
            "coverage_pct": round(score * 100, 1),
            "coverage_level": level,
            "sequence": analyze_sequence(role_nodes, status, role, kr_mode=kr_mode),
            "roadmap_path": build_roadmap_path(role_nodes, status, parent_map, role, kr_mode=kr_mode),
            "covered": covered_list,
            "partial": partial_list,
            "uncovered": uncovered_list,
            "covered_count": len(covered_list),
            "partial_count": len(partial_list),
            "uncovered_count": len(uncovered_list),
        }

    return result


if __name__ == "__main__":
    import json

    # 고정 스키마 샘플 (Learning Crew 백엔드 발췌)
    sample = {
        "service_name": "Learning Crew",
        "dev_type": "backend",
        "frameworks": ["Spring Boot"],
        "skills": {
            "languages": ["Java"],
            "frameworks": ["Spring Boot", "Spring Security", "Spring Data JPA", "LangChain4j"],
            "data_stores": ["MariaDB", "JPA/Hibernate", "QueryDSL", "Caffeine Cache"],
            "auth": ["JWT"], "deployment": ["Docker"], "build": ["Gradle"],
        },
        "analysis_form": {
            "core_feat": [
                {"feat_title": "JWT 인증/토큰 관리", "description": "로그인·리프레시 토큰 순환, Caffeine 캐시 이중 저장"},
                {"feat_title": "스터디 그룹 검색", "description": "QueryDSL 동적 쿼리 키워드 검색·페이지네이션"},
            ],
            "core_perf": [
                {"perf_title": "AI 퀴즈 다층 동시성 제어", "description": "Semaphore·ReentrantLock·가상 스레드·지수 백오프 재시도",
                 "about_feat_title": "AI 퀴즈 생성"},
                {"perf_title": "QueryDSL 조인 조회", "description": "다중 엔터티 조인, N+1 문제 관련 leftJoin 패턴",
                 "about_feat_title": "스터디 그룹 검색"},
            ],
            "feedback": [
                {"feedback_title": "테스트 자동화 필요", "feedback": "JUnit 단위 테스트로 동시성 검증 권장",
                 "about_perf_title": "AI 퀴즈 다층 동시성 제어"},
            ],
            "contribute": {"contribute_titles": ["JWT 인증 모듈 구현", "스터디 그룹 도메인 구현"]},
        },
    }

    res = run(sample, "/mnt/user-data/uploads/roadmap.csv")
    be = res.per_role["backend"]
    print(f"service: {res.service_name} | primary={res.primary_roles} secondary={res.secondary_roles}")
    print(f"backend: {be['topic_coverage']} topics, score {be['coverage_pct']}% — {be['level']}")
    print()
    print("[sequence]"); print(json.dumps(be["sequence"], indent=2, ensure_ascii=False))
    print()
    print("[covered topics]",
          [e["label"] for e in be["covered"] if e["type"] == "topic"])
    print("[next path]",
          [p["topic"] for p in be["roadmap_path"]])