"""GitHub 데이터 기반 포트폴리오 아티클 생성 서비스."""
from __future__ import annotations

import asyncio

from google.genai import types as _genai_types
from pydantic import BaseModel

from app.schemas.generate import DevSpec, GenerateRequest, GithubRepo
from app.services._rag_utils import (
    _LLM_MODEL,
    _generate_with_retry,
    _get_gemini_client,
)

_MAX_ARTICLES = 5
_MIN_REPO_SIZE = 50  # KB 미만 레포 제외


class _ArticleSchema(BaseModel):
    title: str
    contents: str


def _select_repos(repos: list[GithubRepo]) -> list[GithubRepo]:
    """생성 대상 레포 선별 — 설명 있고 크기 있는 순으로 최대 _MAX_ARTICLES개."""
    scored = [
        (r, (1 if r.description else 0) + (1 if r.language else 0) + (1 if r.size >= _MIN_REPO_SIZE else 0))
        for r in repos
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [r for r, _ in scored[:_MAX_ARTICLES]]


def _build_dev_spec_summary(spec: DevSpec) -> str:
    lines = [f"경력: {spec.experience}년"]
    for c in spec.careers:
        if c.careerMajors:
            lines.append(f"주요 기술: {', '.join(c.careerMajors)}")
        if c.careerSkills:
            lines.append(f"보유 스킬: {', '.join(c.careerSkills)}")
        if c.careerKeywords:
            lines.append(f"직무 키워드: {', '.join(c.careerKeywords)}")
    for edu in spec.educationHistory:
        lines.append(f"학력: {edu.name} {edu.department} ({edu.degree})")
    return "\n".join(lines)


def _generate_article_sync(
    repo: GithubRepo,
    dev_spec_summary: str,
    nickname: str,
) -> dict:
    client = _get_gemini_client()
    prompt = (
        f"다음 개발자의 GitHub 프로젝트를 기반으로 포트폴리오 아티클을 작성해주세요.\n\n"
        f"[개발자 정보]\n"
        f"이름: {nickname}\n"
        f"{dev_spec_summary}\n\n"
        f"[프로젝트 정보]\n"
        f"저장소명: {repo.name}\n"
        f"설명: {repo.description or '(설명 없음)'}\n"
        f"주요 언어: {repo.language or '(미지정)'}\n"
        f"GitHub URL: {repo.htmlUrl}\n\n"
        f"[작성 요구사항]\n"
        f"- title: 프로젝트명과 핵심 역할이 드러나는 아티클 제목 (최대 60자)\n"
        f"- contents: 마크다운 형식의 포트폴리오 본문. 아래 구조를 따를 것:\n"
        f"  ## 프로젝트 개요\n"
        f"  ## 기술 스택\n"
        f"  ## 주요 기여 및 구현 내용\n"
        f"  ## 성과 및 배운 점\n"
        f"- 사실에 기반해 작성하고, 없는 내용은 추가하지 마세요.\n"
        f"- 본문은 500자 이상 1500자 이하로 작성하세요.\n"
    )
    resp = _generate_with_retry(
        client,
        model=_LLM_MODEL,
        contents=prompt,
        config=_genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_ArticleSchema,
        ),
    )
    parsed = _ArticleSchema.model_validate_json(resp.text)
    return {"title": parsed.title, "contents": parsed.contents, "fileUrls": []}


async def generate_portfolio_articles(req: GenerateRequest) -> list[dict]:
    """GitHub 레포 + DevSpec 기반으로 포트폴리오 아티클 리스트를 생성한다."""
    repos = _select_repos(req.githubRepos)
    if not repos:
        return []

    dev_spec_summary = _build_dev_spec_summary(req.devSpec)

    tasks = [
        asyncio.to_thread(
            _generate_article_sync,
            repo,
            dev_spec_summary,
            req.user.nickname,
        )
        for repo in repos
    ]
    return await asyncio.gather(*tasks)
