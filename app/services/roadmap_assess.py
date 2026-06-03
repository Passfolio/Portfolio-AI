"""로드맵 평가 태스크 — BackgroundTasks에서 호출."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import anyio

from app.services.roadmap import rule_filter, llm_analyzer, market_tier, market_loader
from app.services.roadmap_merger import run_merged
from app.services._rag_utils import _fetch_code_analysis as _fetch_project
from app.services.webhook import notify_be
from app.jobs.store import JobStatus, update_job

logger = logging.getLogger(__name__)

ROADMAP_CSV = Path(__file__).resolve().parents[2] / "DATASET" / "roadmap.csv"



_COVERAGE_TO_NUM = {
    "입문 단계": 1, "초중급 단계": 2, "중급 (실무 가능) 단계": 3, "시니어 진입 단계": 4,
}
_LLM_TO_NUM    = {"Beginner": 1, "Junior": 2, "Mid-level": 3, "Senior": 4, "Expert": 5}
_NUM_TO_FINAL  = [(4.5, "Senior"), (3.5, "Mid-level"), (1.5, "Junior"), (0.0, "Beginner")]



def _calc_reliability(project: dict) -> dict:
    pct = project.get("contribute_percent") or 0
    try:
        pct = float(pct)
    except (TypeError, ValueError):
        pct = 0.0
    if pct >= 70:
        grade, note = "high", "단독·주도 개발로 평가 대표성이 높습니다."
    elif pct >= 40:
        grade, note = "medium", "공동 개발이 포함되어 일부 신호가 협업 결과일 수 있습니다."
    else:
        grade, note = "low", "기여도가 낮아 평가 신뢰도가 제한됩니다."
    return {"grade": grade, "contribute_pct": pct, "note": note}


def _reconcile_level(assessment: dict, filter_result: Any) -> dict:
    llm_label      = assessment.get("overall_level", "")
    primary_roles  = filter_result.primary_roles
    coverage_label = filter_result.per_role.get(primary_roles[0], {}).get("coverage_level", "") \
                     if primary_roles else ""
    llm_num = _LLM_TO_NUM.get(llm_label, 0)
    cov_num = _COVERAGE_TO_NUM.get(coverage_label, 0)
    if llm_num and cov_num:
        weighted = llm_num * 0.7 + cov_num * 0.3
        final = next(lbl for th, lbl in _NUM_TO_FINAL if weighted >= th)
    else:
        final = llm_label or coverage_label
    return {
        "final_level":    final,
        "skill_level":    llm_label,
        "coverage_level": coverage_label,
        "note": "skill_level(구현 깊이 기준 70%) + coverage_level(로드맵 폭 기준 30%) 가중 합산",
    }


def _assess_one(
    project: dict[str, Any],
    filter_result: Any = None,
    filter_dict: dict | None = None,
) -> dict[str, Any]:
    project = rule_filter.normalize_project(project)

    if filter_result is None:
        filter_result = rule_filter.run(project, ROADMAP_CSV)
    if filter_dict is None:
        filter_dict = {
            "primary_roles":   filter_result.primary_roles,
            "secondary_roles": filter_result.secondary_roles,
            "per_role":        filter_result.per_role,
        }

    raw_fws   = project.get("frameworks") or []
    framework = (raw_fws[0].get("name", "") if isinstance(raw_fws[0], dict) else raw_fws[0]) \
                if raw_fws else ""
    raw_dt  = project.get("dev_type", "")
    devtype = (raw_dt[0] if raw_dt else "") if isinstance(raw_dt, list) else raw_dt

    market_ctx = market_loader.load_market_context(tuple(filter_result.primary_roles))

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_assessment = pool.submit(llm_analyzer.analyze, project, filter_dict)
        fut_market     = pool.submit(
            llm_analyzer.generate_market_brief,
            filter_dict, filter_result.primary_roles, market_ctx,
            framework, devtype, project,
        )
        assessment   = fut_assessment.result()
        market_brief = fut_market.result()

    def _with_tier(node: dict) -> dict:
        t = market_tier.get_tier(node["label"], node["type"])
        return {**node, "market_tier": t, "highlighted": market_tier.is_highlighted(t)}

    per_role_out: dict[str, Any] = {}
    for role, data in filter_result.per_role.items():
        nodes = (
            [_with_tier(dict(n, status="covered"))    for n in data["covered"]]
            + [_with_tier(dict(n, status="partial"))  for n in data["partial"]]
            + [_with_tier(dict(n, status="uncovered")) for n in data["uncovered"]]
        )
        roadmap_path = [
            {**step,
             "market_tier": market_tier.get_tier(step["topic"], "topic"),
             "highlighted":  market_tier.is_highlighted(market_tier.get_tier(step["topic"], "topic"))}
            for step in data.get("roadmap_path", [])
        ]
        per_role_out[role] = {
            "coverage_level":  data["coverage_level"],
            "coverage_pct":    data["coverage_pct"],
            "topic_coverage":  data["topic_coverage"],
            "sequence":        data.get("sequence", {}),
            "roadmap_path":    roadmap_path,
            "nodes":           nodes,
            "covered_count":   data["covered_count"],
            "partial_count":   data["partial_count"],
            "uncovered_count": data["uncovered_count"],
        }

    return {
        "service_name":           project.get("service_name"),
        "dev_type":               project.get("dev_type"),
        "primary_roles":          filter_result.primary_roles,
        "secondary_roles":        filter_result.secondary_roles,
        "assessment_reliability": _calc_reliability(project),
        "final_level":            _reconcile_level(assessment, filter_result),
        "per_role":               per_role_out,
        "llm_assessment":         assessment,
        "market_brief":           market_brief,
    }


async def run_assess_task(
    job_id: str,
    ai_job_id: str,
    code_analysis_urls: list[str],
    merge: bool,
) -> None:
    update_job(job_id, JobStatus.RUNNING)
    results: list[dict[str, Any]] | None = None
    error_message: str | None = None
    try:
        def _run() -> list[dict[str, Any]]:
            projects: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=min(len(code_analysis_urls), 4)) as pool:
                futures = {pool.submit(_fetch_project, url): url for url in code_analysis_urls}
                for future in as_completed(futures):
                    url = futures[future]
                    exc = future.exception()
                    if not exc:
                        p = future.result()
                        p["_source_url"] = url
                        projects.append(p)
                    else:
                        logger.warning("[로드맵] URL fetch 실패: %s — %s", url, exc)

            if not projects:
                raise RuntimeError("모든 URL fetch 실패")

            if merge:
                merged_project, filter_result = run_merged(projects)
                filter_dict = {
                    "primary_roles":   filter_result.primary_roles,
                    "secondary_roles": filter_result.secondary_roles,
                    "per_role":        filter_result.per_role,
                }
                r = _assess_one(merged_project, filter_result, filter_dict)
                r["source_urls"] = [p["_source_url"] for p in projects]
                return [r]
            else:
                out = []
                with ThreadPoolExecutor(max_workers=min(len(projects), 4)) as pool:
                    futures = {pool.submit(_assess_one, p): p for p in projects}
                    for future in as_completed(futures):
                        p = futures[future]
                        if not future.exception():
                            r = future.result()
                            r["source_url"] = p.get("_source_url", "")
                            out.append(r)
                        else:
                            logger.warning("[로드맵] 평가 실패: %s", future.exception())
                return out

        results = await anyio.to_thread.run_sync(_run)
        update_job(job_id, JobStatus.DONE, result=results)
    except Exception as e:
        error_message = str(e)
        logger.error("[로드맵] 평가 태스크 실패: %s", e)
        update_job(job_id, JobStatus.ERROR, message=error_message)
    finally:
        try:
            await notify_be(ai_job_id=ai_job_id, output_pdf_url=None, error_message=error_message)
        except Exception as webhook_err:
            logger.warning("[Webhook] 콜백 실패 (결과는 저장됨): %s", webhook_err)
