# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Passfolio** — AI 서버(FastAPI) 레포지토리. 사용자의 PDF 포트폴리오 문구를 GitHub 실제 코드와 교차 검증하여, 기술 근거가 강화된 Before/After 버전과 로드맵을 생성한다.

이 레포는 **AI 비즈니스 로직 전담 서버**만 담당한다. 메인 서버(Spring Boot)가 GitHub 소스코드와 PDF ID를 POST로 전달하면, 분석 결과를 JSON으로 반환하는 구조다.

## Architecture

```
Spring Boot (메인 서버)
    │  POST /analyze  {pdf_id, github_code}
    ▼
FastAPI (이 레포)
    ├── PDF 파싱       : Docling (Layout-aware)
    ├── 임베딩         : gemini-embedding-2-preview — 반드시 3072차원
    ├── Vector DB      : PostgreSQL + pgvector
    └── LLM 생성       : gemini-3.0-flash-preview
    │  JSON 응답  {sections[before/after/evidence], roadmap}
    ▼
Spring Boot → 최종 PDF 추출
```

### 처리 파이프라인

1. PDF 파일 획득 → Docling으로 섹션 단위 파싱
2. GitHub 소스코드 → 파일/함수 단위 청킹
3. 포트폴리오 섹션 + 코드 청크 임베딩 → pgvector 저장
4. Vector Search로 "포트폴리오 문구 ↔ 실제 코드" 교차 검증
5. gemini-3.0-flash-preview로 Before/After 문구 생성 + 로드맵 제안

### 예정 디렉토리 구조

```
app/
├── api/v1/analyze.py       # POST /analyze 엔드포인트
├── services/
│   ├── pdf_service.py      # Docling 파싱
│   ├── code_service.py     # GitHub 코드 청킹
│   ├── embedding_service.py
│   ├── vector_service.py   # pgvector CRUD
│   └── analysis_service.py # 교차검증 + LLM 생성
├── models/
│   ├── schemas.py          # Pydantic 요청/응답 모델
│   └── db.py               # SQLAlchemy + pgvector 테이블
└── core/
    ├── config.py
    └── gemini_client.py
```

## AI Stack 제약사항

- **임베딩 모델**: `gemini-embedding-2-preview` — 출력 차원은 **반드시 3072**. 다른 차원 사용 금지.
- **LLM**: `gemini-3.0-flash-preview`
- **pgvector 컬럼**: `VECTOR(3072)` 고정

## 개발 명령어

> 환경이 세팅된 이후 채워질 예정

```bash
# 서버 실행
uvicorn app.main:app --reload

# 테스트
pytest

# DB 마이그레이션
alembic upgrade head
```

## 인터페이스 스펙 (Spring Boot와의 계약)

**Request** `POST /analyze`
```json
{
  "pdf_id": "string",
  "github_code": "TBD — Spring Boot 스펙 확정 필요"
}
```

**Response**
```json
{
  "pdf_id": "string",
  "sections": [
    {
      "original": "기존 포트폴리오 문구",
      "enhanced": "근거 강화된 문구",
      "evidence_code": "파일명:라인 또는 함수명",
      "confidence": 0.91
    }
  ],
  "roadmap": ["향후 개선 제안 1", "향후 개선 제안 2"]
}
```

## 미확정 사항 (TBD)

- Spring Boot가 GitHub 소스코드를 전달하는 정확한 형태 (단일 String / 파일별 Map / 기타)
- FastAPI가 PDF 파일 자체를 받는 방식 (pdf_id로 Spring Boot API 재호출 / POST body에 포함)

## Git Workflow

모든 작업은 아래 순서를 따른다.

### 1. 작업 시작 전 — `start-work` 스킬
새 기능, 버그 수정, 리팩토링을 시작하기 **전에 반드시** `start-work` 스킬을 호출한다.
사용자에게 제안하고 승인을 받은 후 실행한다.
- GitHub Issue 생성
- `feature/[issue#]-[slug]` 브랜치 생성 및 체크아웃

### 2. 작업 중 — `smart-commit` 스킬
논리적 단위(모델, 서비스, API, 테스트 등)가 완성될 때마다 **자동으로** `smart-commit` 스킬을 실행한다.
사용자 확인 없이 커밋한다. Push는 하지 않는다.

### 3. 작업 완료 후 — 사용자가 Push
사용자가 직접 `git push origin [branch]`를 실행한다.
Push 후 GitHub Actions가 자동으로 PR을 생성한다 (`Closes #[issue#]` 포함).
