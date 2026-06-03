"""
market_loader.py — DATASET/webdataset에서 role별 시장 데이터를 추출·압축

LLM 프롬프트에 포함하기 위해 필요한 핵심 수치만 추출한다.
전체 JSON을 그대로 넘기면 토큰 낭비이므로 role에 맞는 데이터만 선별한다.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT       = Path(__file__).resolve().parents[3]
WEB_DS_DIR = ROOT / "DATASET" / "webdataset"


def _load(filename: str) -> dict:
    path = WEB_DS_DIR / filename
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# role → 상세 스택 파일
_ROLE_DETAIL_FILE: dict[str, str] = {
    "backend":          "02_backend_stacks.json",
    "frontend":         "03_frontend_stacks.json",
    "devops":           "04_devops_cloud_stacks.json",
    "machine-learning": "05_ai_ml_data_stacks.json",
    "ai-engineer":      "05_ai_ml_data_stacks.json",
    "data-engineer":    "05_ai_ml_data_stacks.json",
    "android":          "06_mobile_stacks.json",
    "ios":              "06_mobile_stacks.json",
    "cyber-security":   "07_security_game_stacks.json",
}


def _condense_role_detail(role: str, data: dict) -> dict:
    """role별 상세 파일에서 핵심 수치만 추출."""
    out: dict[str, Any] = {}

    if role == "backend":
        out["top_languages"]  = data.get("languages", [])[:5]
        out["top_frameworks"] = data.get("frameworks", [])[:5]
        out["top_databases"]  = data.get("databases", [])[:5]
        out["infrastructure"] = data.get("infrastructure", [])[:5]
        out["must_have"]      = data.get("common_requirements", {}).get("must", [])
        out["nice_to_have"]   = data.get("common_requirements", {}).get("preferred", [])
        out["salary"]         = data.get("salary_by_stack_万KRW", {})

    elif role == "frontend":
        out["top_frameworks"]  = data.get("frameworks_libraries", [])[:5]
        out["state_mgmt"]      = data.get("state_management", [])[:4]
        out["build_tools"]     = data.get("build_tools", [])[:3]
        out["must_have"]       = data.get("common_requirements", {}).get("must", [])
        out["nice_to_have"]    = data.get("common_requirements", {}).get("preferred", [])
        out["salary"]          = data.get("salary_by_level_万KRW", {})

    elif role == "devops":
        out["top_cloud"]       = data.get("cloud_platforms", [])[:3]
        out["container"]       = data.get("container_orchestration", [])[:4]
        out["cicd"]            = data.get("cicd_tools", [])[:4]
        out["iac"]             = data.get("iac_config_management", [])[:3]
        out["monitoring"]      = data.get("monitoring_observability", [])[:3]
        out["must_have"]       = data.get("common_requirements", {}).get("must", [])
        out["salary"]          = data.get("salary_by_level_万KRW", {})

    elif role in ("machine-learning", "ai-engineer", "data-engineer"):
        sub = data.get("sub_roles", {})
        role_key_map = {
            "machine-learning": "ML_엔지니어",
            "ai-engineer":      "AI_엔지니어",
            "data-engineer":    "데이터_엔지니어",
        }
        key = role_key_map.get(role, "")
        if key and key in sub:
            out["sub_role_stacks"] = sub[key]
        out["ml_frameworks"]    = data.get("ml_frameworks", [])[:5]
        out["llm_stacks"]       = data.get("llm_genai_stacks", [])[:5]
        out["pipeline_tools"]   = data.get("data_pipeline_tools", [])[:4]
        out["salary"]           = data.get("salary_by_role_万KRW", {})

    elif role in ("android", "ios"):
        platform = data.get(role, data.get("android" if role == "android" else "ios", {}))
        out["primary_language"] = platform.get("primary_language")
        out["key_frameworks"]   = platform.get("key_frameworks", [])[:5]
        out["architecture"]     = platform.get("architecture", [])
        out["must_have"]        = platform.get("common_requirements", {}).get("must", [])
        out["salary"]           = data.get("salary_by_level_万KRW", {})

    return out


def _get_real_postings_summary(roles: list[str], realdata: dict) -> dict:
    """12_stack_frequency_realdata에서 role별 실제 공고 스택 요약 추출."""
    by_role = realdata.get("by_role_category", {})
    result: dict[str, Any] = {}

    role_key_map = {
        "backend":          "백엔드",
        "frontend":         "프론트엔드",
        "devops":           "DevOps",
        "machine-learning": "AI_ML",
        "ai-engineer":      "AI_LLM",
        "data-engineer":    "데이터엔지니어",
        "android":          "모바일_Android",
        "ios":              "모바일_iOS",
    }

    for role in roles:
        key = role_key_map.get(role)
        if key and key in by_role:
            entry = by_role[key]
            result[role] = {
                "sample_size":  entry.get("sample_size"),
                "top_stacks":   entry.get("top_stacks", []),
                "notable":      entry.get("notable"),
            }

    return result


def _get_gap_analysis(roles: list[str], deep: dict) -> list[dict]:
    """13_deep_analysis_v2_realdata에서 gap 분석 + 기업유형별 패턴 추출."""
    gaps   = deep.get("gap_analysis_from_real_data", {}).get("gaps", [])
    action = deep.get("action_plan_updated", {})
    result = []

    role_key_map = {
        "backend":          "백엔드",
        "frontend":         "프론트엔드",
        "machine-learning": "AI_ML",
        "ai-engineer":      "AI_ML",
        "data-engineer":    "데이터엔지니어",
        "devops":           "DevOps",
    }

    for role in roles:
        key = role_key_map.get(role)
        if key and key in action:
            result.append({
                "role":        role,
                "must_have":   action[key].get("core", []),
                "growing":     action[key].get("plus", []),
            })

    return result


@lru_cache(maxsize=8)
def load_market_context(primary_roles_tuple: tuple[str, ...]) -> dict[str, Any]:
    """
    primary_roles 기반으로 관련 DATASET 데이터를 로드·압축하여 반환.
    lru_cache를 위해 list 대신 tuple로 받는다.
    """
    roles = list(primary_roles_tuple)
    ctx: dict[str, Any] = {}

    # ── 공통: 시장 개요 ──────────────────────────────────────────────
    overview = _load("01_market_overview.json")
    ctx["market_overview"] = {
        "top_demanded_roles":   overview.get("top_demanded_roles", [])[:6],
        "hiring_quality_shift": overview.get("hiring_quality_shift", {}),
        "ai_related_growth":    overview.get("ai_related_demand_growth", {}),
        "salary_overview":      overview.get("salary_overview", {}),
    }

    # ── role 상세 스택 ────────────────────────────────────────────────
    ctx["role_details"] = {}
    for role in roles:
        fname = _ROLE_DETAIL_FILE.get(role)
        if fname:
            raw = _load(fname)
            ctx["role_details"][role] = _condense_role_detail(role, raw)

    # ── 실제 공고 기반 스택 빈도 ──────────────────────────────────────
    try:
        realdata = _load("12_stack_frequency_realdata.json")
        ctx["real_postings_top_stacks"] = {
            k: v for k, v in realdata.get("total_stack_frequency", {}).items()
            if v.get("pct", 0) >= 15           # 15% 이상만
        }
        ctx["real_postings_by_role"] = _get_real_postings_summary(roles, realdata)
        ctx["key_findings"] = realdata.get("key_findings_from_real_data", [])
    except Exception:
        pass

    # ── 갭 분석 + 액션 플랜 ──────────────────────────────────────────
    try:
        deep = _load("13_deep_analysis_v2_realdata.json")
        ctx["gap_analysis"] = _get_gap_analysis(roles, deep)
        ctx["company_patterns"] = deep.get("company_size_stack_pattern", {})
        ctx["emerging_stacks"]  = deep.get("emerging_stacks_confirmed", {}).get("items", [])
    except Exception:
        pass

    # ── 빅테크 기업 스택 ──────────────────────────────────────────────
    try:
        bigtech = _load("15_bigtech_company_stacks.json")
        ctx["bigtech_common"] = bigtech.get("common_stack_patterns", {})
        ctx["bigtech_insights"] = bigtech.get("key_insights", [])[:5]
    except Exception:
        pass

    # ── 연봉 데이터 (role별) ──────────────────────────────────────────
    try:
        salary = _load("10_salary_comprehensive.json")
        ctx["salary_by_role"] = salary.get("by_role", {})
        ctx["salary_insights"] = salary.get("insights", [])[:4]
    except Exception:
        pass

    # ── 글로벌 트렌드 (한국 관련성 주석 포함) ─────────────────────────
    try:
        so = _load("16_stackoverflow_survey_2025.json")
        ctx["global_trends"] = {
            "notable": so.get("notable_global_trends", {}),
            "top_languages": so.get("programming_languages", [])[:6],
            "top_frameworks": so.get("web_frameworks", [])[:5],
        }
    except Exception:
        pass

    return ctx
