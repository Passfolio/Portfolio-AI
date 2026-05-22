from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from app.core.security import verify_api_key
from app.schemas.generate import ArticleItem, GenerateRequest, GenerateResponse
from app.services.be_client import be_client
from app.services.generator import generate_portfolio_articles

router = APIRouter(tags=["generate"])
_logger = logging.getLogger(__name__)


@router.post("/generate", response_model=GenerateResponse)
async def generate(
    req: GenerateRequest,
    _: None = Depends(verify_api_key),
) -> GenerateResponse:
    _logger.info("[generate] 요청 수신: user_id=%s, repos=%d개", req.user.id, len(req.githubRepos))

    articles_data = await generate_portfolio_articles(req)
    _logger.info("[generate] 아티클 생성 완료: %d개", len(articles_data))

    saved: list[ArticleItem] = []
    for a in articles_data:
        await be_client.post_article(
            title=a["title"],
            contents=a["contents"],
            file_urls=a.get("fileUrls", []),
        )
        saved.append(ArticleItem(title=a["title"], contents=a["contents"], fileUrls=a.get("fileUrls", [])))
        _logger.info("[generate] 아티클 저장 완료: %s", a["title"])

    return GenerateResponse(articles=saved)
