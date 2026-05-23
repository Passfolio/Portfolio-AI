from __future__ import annotations

import logging

from fastapi import APIRouter

from app.schemas.generate import ArticleItem, GenerateRequest, GenerateResponse
from app.services.generator import generate_portfolio_articles

router = APIRouter(tags=["generate"])
_logger = logging.getLogger(__name__)


@router.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    _logger.info("[generate] 요청 수신: user_id=%s, repos=%d개", req.user.id, len(req.githubRepos))

    articles_data = await generate_portfolio_articles(req)
    _logger.info("[generate] 아티클 생성 완료: %d개", len(articles_data))

    articles = [
        ArticleItem(
            title=a["title"],
            contents=a["contents"],
            fileUrls=a.get("fileUrls", []),
        )
        for a in articles_data
    ]
    return GenerateResponse(articles=articles)
