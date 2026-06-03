# 2026 대한민국 개발자 채용시장 기술스택 분석 DATASET

> 수집일: 2026-06-01  
> 출처: 잡코리아, 사람인(점핏), 원티드, 링크드인, 인크루트, 업계 보고서 종합  
> 분석 도구: Claude AI (Anthropic)

---

## 파일 목록

### v1 — 웹검색 기반 (간접 자료)
| 파일 | 내용 |
|------|------|
| `01_market_overview.json` | 2026 채용시장 종합 개요 (직군별 수요, 구조적 변화) |
| `02_backend_stacks.json` | 백엔드/서버 개발자 기술스택 상세 |
| `03_frontend_stacks.json` | 프론트엔드 개발자 기술스택 상세 |
| `04_devops_cloud_stacks.json` | DevOps / 클라우드 / SRE 엔지니어 스택 |
| `05_ai_ml_data_stacks.json` | AI·ML·데이터엔지니어·MLOps 스택 |
| `06_mobile_stacks.json` | Android·iOS·크로스플랫폼 모바일 스택 |
| `07_security_game_stacks.json` | 보안 엔지니어 + 게임 개발자 스택 |
| `08_deep_analysis.json` | 심층 분석 v1 (구조적 변화, 트렌드 매트릭스, 스킬갭, 미래 예측) |
| `09_stack_frequency_matrix.json` | 직군×스택 빈도 점수 매트릭스 (5점 척도, 추정값) |
| `10_salary_comprehensive.json` | 직군·연차·스택별 연봉 종합 데이터 |

### v2 — 실제 채용공고 직접 파싱 기반 (1차 소스)
| 파일 | 내용 |
|------|------|
| `11_raw_job_postings.json` | **27개 실제 공고 원본 파싱** (점핏 14건, 원티드 11건, 링크드인 2건) |
| `12_stack_frequency_realdata.json` | **실제 공고 기반 스택 빈도 집계** (직군별 세분화 포함) |
| `13_deep_analysis_v2_realdata.json` | **심층 분석 v2** — 사이트 접근성 보고, 검증된 사실, 갭 분석, 기업유형별 패턴 |
| `14_linkareer_job_postings.json` | **링커리어 18건 실제 파싱** — LG전자·네이버·AWS·마키나락스 등 대기업·AI 특화 공고 |

### v3 — 보강 데이터 (빅테크 실제 스택 + 글로벌 서베이 + 다이어그램 오버레이)
| 파일 | 내용 |
|------|------|
| `15_bigtech_company_stacks.json` | **네카라쿠배당토 실제 기술스택** — 네이버·카카오·라인·쿠팡·배민·당근·토스·카카오뱅크 |
| `16_stackoverflow_survey_2025.json` | **SO 2025 글로벌 서베이** — 전문 개발자 49,000+명 기술스택 사용률, 한국 관련성 주석 |
| `17_kr_roadmap_overlay.json` | **⭐ 다이어그램 강조 오버레이** — roadmap.csv 노드 61개에 한국시장 상태 매핑 (필수/급증/증가/유지/감소/하락) |

---

## 핵심 인사이트 요약

### 1. 가장 수요 높은 직군 TOP 5
1. 백엔드/서버 (28.1%)
2. AI/ML 엔지니어 (급증, YoY+50%)
3. 프론트엔드 (11.1%)
4. 데이터 엔지니어
5. DevOps/클라우드

### 2. 2026 핵심 기술 트렌드
- **필수화**: TypeScript, Next.js, Docker, Kubernetes, Terraform
- **급부상**: LangChain/RAG, FastAPI, dbt, Zustand
- **하락**: Angular, jQuery, PHP(신규), SVN

### 3. 연봉 최고 기술스택
C++ > Keras > TensorFlow > R > Python 순 (평균 기준)

### 4. 스킬갭 (공급 부족 = 프리미엄)
- LLM 앱 개발 (+30%)
- MLOps (+25%)
- K8s 심화 운영 (+20%)
- Go 백엔드 (+18%)

### 5. 모든 직군 공통 필수
Git/GitHub, Linux/Bash, Docker, Python(부분), AWS
