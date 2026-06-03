"""
LLM-based roadmap assessment using Google Gemini API.

인증: GEMINI_API_KEY 환경변수

수정 이력 (분석 근거):
  [FIX #1] market_brief 프롬프트에 프로젝트 evidence(user_role/core_perf/skills) 주입
           → "전문가에게 기초부터 배우라"는 조언 + 본문-시장 톤 충돌 해결
  [FIX #2] 필드명 폴백(analysis_form|analysis, contribute_percent|contribute_share_percent)
           → 조용히 빈 값이 되던 core_feat/core_perf/contribute 복구
  [FIX #3] strong/covered에 subtopic 포함 (topic 롤업 의존 제거)
           → SSE·Redis 등 subtopic 증거가 LLM에 전달되도록
  [FIX #4] declining/trending/priority를 dev_type 직군으로 한정 (프론트 스택 혼입 방지)
  [FIX #5] core_feat/core_perf를 title만이 아니라 description까지 전달
  [FIX #6] data.get("level") → coverage_level 폴백, 하드 인덱싱 → .get 방어
  [FIX #7] market_brief는 환각 위험이 큰 수치를 생성하므로 모델 분리(상향 권장)
"""

from __future__ import annotations
import json
import os
from typing import Any

from google import genai
from google.genai import types

MODEL = "gemini-3.1-flash-lite"  # "gemini-3.5-pro

# [FIX #7] market_brief는 연봉/수요율 등 수치를 생성하므로 환각 위험이 큼.
# 사용 가능한 더 강한 모델명으로 교체 권장 (예: 주석의 gemini-3.5-pro 계열).
# 기본값은 호환성을 위해 MODEL과 동일하게 둠.
MARKET_BRIEF_MODEL = MODEL

SYSTEM_PROMPT = """당신은 시니어 개발자 멘토입니다.
사용자의 GitHub 프로젝트 분석 결과와 개발자 로드맵 데이터를 바탕으로
현재 기술 수준을 진단하고, 로드맵 순서에 맞는 구체적인 학습 경로를 제시합니다.

반드시 아래 JSON 스키마 형식으로만 응답하세요. 마크다운 코드블록 없이 순수 JSON만 출력합니다.

{
  "overall_level": "Beginner | Junior | Mid-level | Senior | Expert",
  "level_rationale": "수준 판단 근거 (2-3문장)",
  "strong_areas": [
    {"area": "영역명", "reason": "근거"}
  ],
  "gap_areas": [
    {"area": "영역명", "priority": "High | Medium | Low", "reason": "왜 중요한지"}
  ],
  "roadmap_path": [
    {
      "priority": 1,
      "topic": "로드맵 토픽명 (roadmap_path에서 가져올 것)",
      "tier": "core | support",
      "subtopics_to_learn": ["배울 세부 항목"],
      "why": "현재 수준에서 이 단계가 필요한 이유",
      "how": "구체적인 학습 방법 또는 실습 아이디어"
    }
  ],
  "summary": "전체 평가 요약 (3-4문장)"
}"""


_DEVTYPE_KO: dict[str, str] = {
    "backend":          "백엔드",
    "frontend":         "프론트엔드",
    "devops":           "DevOps",
    "android":          "Android",
    "ios":              "iOS",
    "machine-learning": "ML",
    "ai-engineer":      "AI",
    "data-engineer":    "데이터",
    "fullstack":        "풀스택",
}


def _flatten_skills(skills: Any) -> list[str]:
    """skills가 list[dict], list[str], dict 어떤 형태든 이름 문자열 리스트로 반환."""
    if not skills:
        return []
    if isinstance(skills, dict):
        out = []
        for v in skills.values():
            out += v if isinstance(v, list) else [v]
        return out
    if isinstance(skills, list):
        return [s.get("name") if isinstance(s, dict) else str(s) for s in skills]
    return []


def _to_devtype_ko(devtype: str) -> str:
    """영문 role 키면 한글로, 이미 한글이면 그대로."""
    if not devtype:
        return ""
    if any("가" <= c <= "힣" for c in devtype):
        return devtype
    return _DEVTYPE_KO.get(devtype, devtype)


def _build_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)

    raise EnvironmentError("GEMINI_API_KEY 환경변수를 설정하세요.")


# ──────────────────────────────────────────────────────────────────────
# 공통 추출 헬퍼
# ──────────────────────────────────────────────────────────────────────
def _extract_project_core(project: dict[str, Any]) -> dict[str, Any]:
    """
    프로젝트 dict에서 평가에 필요한 핵심 정보를 추출한다.

    [FIX #2] 필드명 폴백:
      - analysis_form  ↔  analysis
      - contribute_percent  ↔  contribute_share_percent
      raw request.json과 정규화된 dict를 모두 지원한다.
    [FIX #5] core_feat/core_perf는 title뿐 아니라 description까지 담는다.
    """
    analysis = project.get("analysis_form") or project.get("analysis") or {}
    if not analysis:
        # 키 누락을 조용히 넘기지 않고 경고 (디버깅용)
        print("[llm_analyzer] WARNING: analysis_form/analysis 가 비어 있습니다. "
              "프로젝트 dict 필드명을 확인하세요.")

    contribute = project.get("contribute_percent")
    if contribute is None:
        contribute = project.get("contribute_share_percent")

    core_feat = [
        {"title": f.get("feat_title"), "description": f.get("description")}
        for f in analysis.get("core_feat", [])
    ]
    core_perf = [
        {"title": p.get("perf_title"), "description": p.get("description")}
        for p in analysis.get("core_perf", [])
    ]

    return {
        "service_name":        project.get("service_name"),
        "service_description": project.get("service_description"),
        "dev_type":            project.get("dev_type"),
        "frameworks":          project.get("frameworks"),
        "skills":              project.get("skills"),
        "user_role":           project.get("user_role"),
        "contribute_percent":  contribute,
        "core_feat":           core_feat,
        "core_perf":           core_perf,
        "contribute_titles":   analysis.get("contribute", {}).get("contribute_titles", []),
    }


def _role_nodes(role_data: dict[str, Any], status: str) -> list[dict[str, Any]]:
    """
    role_data에서 주어진 status의 노드를 반환.

    [FIX #6] 두 스키마를 모두 지원:
      - 신규: role_data["nodes"] = [{..., "status": "covered"}, ...]
      - 구버전: role_data["covered"|"partial"|"uncovered"] = [...]
    """
    nodes = role_data.get("nodes")
    if isinstance(nodes, list):
        return [n for n in nodes if n.get("status") == status]
    return role_data.get(status, []) or []


def _label_items(nodes: list[dict[str, Any]], topic_only: bool = False) -> list[dict[str, Any]]:
    """노드 리스트 → {label, type, tier} 리스트."""
    out = []
    for n in nodes:
        if topic_only and n.get("type") != "topic":
            continue
        out.append({
            "label": n["label"],
            "type":  n.get("type", "subtopic"),
            "tier":  n.get("tier", "support"),
        })
    return out


# ──────────────────────────────────────────────────────────────────────
# 1) 메인 평가 (overall_level / strong / gap / roadmap)
# ──────────────────────────────────────────────────────────────────────
def _build_user_prompt(project: dict[str, Any], filter_result_dict: dict[str, Any]) -> str:
    project_core = _extract_project_core(project)  # [FIX #2][FIX #5]

    roadmap_summary: dict[str, Any] = {}
    for role, data in filter_result_dict.get("per_role", {}).items():
        covered_nodes   = _role_nodes(data, "covered")    # [FIX #6]
        partial_nodes   = _role_nodes(data, "partial")
        uncovered_nodes = _role_nodes(data, "uncovered")

        roadmap_summary[role] = {
            "weighted_score_pct": data.get("coverage_pct"),
            # [FIX #6] level → coverage_level 폴백
            "level": data.get("coverage_level") or data.get("level"),
            "topic_coverage": data.get("topic_coverage"),
            # [FIX #3] covered는 topic+subtopic 모두 포함 → SSE/Redis 등 증거 손실 방지
            "covered": _label_items(covered_nodes),
            "partial_topics": _label_items(partial_nodes, topic_only=True),
            "uncovered_topics": _label_items(uncovered_nodes, topic_only=True),
            "roadmap_path": data.get("roadmap_path", []),
        }

    # primary role의 roadmap_path를 별도로 추출 (LLM 프롬프트에 강조)
    primary_role = filter_result_dict.get("primary_roles", ["backend"])[0]
    primary_path = filter_result_dict.get("per_role", {}).get(primary_role, {}).get("roadmap_path", [])

    return (
        "## 프로젝트 분석 결과\n"
        + json.dumps(project_core, ensure_ascii=False, indent=2)
        + "\n\n## 로드맵 매핑 결과\n"
        + json.dumps(roadmap_summary, ensure_ascii=False, indent=2)
        + "\n\n## 추천 학습 경로 (roadmap.csv order_index 기준 순서)\n"
        + "아래 경로는 로드맵 순서대로 정렬된 미학습 토픽입니다. "
        + "roadmap_path 필드는 반드시 이 순서와 토픽명을 그대로 사용하세요.\n"
        + json.dumps(primary_path, ensure_ascii=False, indent=2)
        + "\n\n## 작성 지침\n"
        + "- 'covered'에 있는 항목(subtopic 포함)은 이미 구현·학습한 강점입니다. "
        + "이를 '아직 모르는 것'으로 취급하지 마세요.\n"
        + "- gap_areas/roadmap_path는 partial_topics·uncovered_topics 중심으로 제시하세요.\n"
        + "\n위 데이터를 분석하여 JSON 형식으로 평가를 제공하세요."
    )


# ──────────────────────────────────────────────────────────────────────
# 2) 시장 브리핑 (market_brief)
# ──────────────────────────────────────────────────────────────────────
_MARKET_BRIEF_SCHEMA = """{
  "market_overview": "현재 한국 {role} 시장 현황 요약 (3-4문장, 수치 포함)",
  "trending_stacks": [
    {"stack": "기술명", "trend": "급증|증가|필수화", "kr_demand_pct": 숫자, "reason": "왜 뜨는지"}
  ],
  "declining_stacks": [
    {"stack": "기술명", "reason": "왜 줄어드는지"}
  ],
  "company_patterns": {
    "startup": "스타트업 주류 스택 패턴 (1-2문장)",
    "enterprise": "대기업 주류 스택 패턴 (1-2문장)"
  },
  "salary_context": "현재 역할의 연봉 맥락 (신입~시니어 범위, 고연봉 스택 언급)",
  "user_market_position": "유저의 현재 스택이 시장에서 차지하는 위치 (2-3문장, 강점과 공백 언급)",
  "strategic_direction": "시장 흐름을 고려한 나아갈 방향 전략 (3-4문장)",
  "priority_skills": [
    {
      "skill": "기술명",
      "urgency": "urgent|important|nice-to-have",
      "market_demand": "높음|중간|낮음",
      "reason": "유저 수준 + 시장 수요를 결합한 우선순위 근거"
    }
  ]
}"""


def _build_market_brief_system(framework: str, devtype: str) -> str:
    devtype_ko = _to_devtype_ko(devtype)
    return (
        f"당신은 30년 경력의 시니어 {framework} {devtype_ko} 개발자이자 아키텍트 그리고 전문 커리어 어드바이저이다.\n"
        "유저의 현재 기술 수준 평가 데이터와 한국 시장 데이터를 분석하여\n"
        "설명 탭에 보여줄 시장 흐름 분석 + 개인 맞춤 방향을 제시한다.\n\n"
        "반드시 아래 JSON 스키마 형식으로만 응답하라. 마크다운 코드블록 없이 순수 JSON만 출력한다.\n\n"
        + _MARKET_BRIEF_SCHEMA
    )


def _build_market_brief_prompt(
    filter_result_dict: dict[str, Any],
    market_ctx: dict[str, Any],
    primary_role: str,
    project: dict[str, Any] | None = None,   # [FIX #1] evidence 주입용
    devtype: str = "",                        # [FIX #4] 직군 한정용
) -> str:
    role_data = filter_result_dict.get("per_role", {}).get(primary_role, {})

    covered_nodes   = _role_nodes(role_data, "covered")    # [FIX #6]
    partial_nodes   = _role_nodes(role_data, "partial")
    uncovered_nodes = _role_nodes(role_data, "uncovered")

    # [FIX #3] strong_areas에 subtopic 포함 → 실제 구현 역량(SSE/Redis 등)이 전달되게
    strong_areas = [n["label"] for n in covered_nodes]
    gap_areas = [
        {"area": n["label"], "tier": n.get("tier", "support")}
        for n in (uncovered_nodes + partial_nodes)
        if n.get("type") == "topic"
    ]

    user_summary = {
        "role":           primary_role,
        "coverage_level": role_data.get("coverage_level") or role_data.get("level"),  # [FIX #6]
        "coverage_pct":   role_data.get("coverage_pct"),
        "strong_areas":   strong_areas,
        "gap_areas":      gap_areas[:10],
    }

    # [FIX #1] 프로젝트에서 실제 구현한 핵심 기술/경험을 evidence로 추가
    evidence_block = ""
    if project:
        pc = _extract_project_core(project)
        evidence = {
            "service_description": pc["service_description"],
            "user_role":           pc["user_role"],
            "skills":              _flatten_skills(pc["skills"]),
            "frameworks":          [f.get("name") if isinstance(f, dict) else str(f) for f in (pc["frameworks"] or [])],
            "implemented_features": [f["title"] for f in pc["core_feat"]],
            "implemented_perf":    [
                {"title": p["title"], "description": p["description"]}
                for p in pc["core_perf"]
            ],
            "contribute_percent":  pc["contribute_percent"],
        }
        evidence_block = (
            "\n\n## 유저가 이미 구현한 핵심 기술 (강점 근거)\n"
            + json.dumps(evidence, ensure_ascii=False, indent=2)
        )

    devtype_ko = _to_devtype_ko(devtype) or primary_role

    return (
        "## 유저 현재 평가 결과\n"
        + json.dumps(user_summary, ensure_ascii=False, indent=2)
        + evidence_block
        + "\n\n## 한국 시장 데이터 (2026년 기준)\n"
        + json.dumps(market_ctx, ensure_ascii=False, indent=2)
        + "\n\n## 작성 지침\n"
        # [FIX #4] 직군 한정
        + f"- 분석 대상 직군은 '{devtype_ko}'입니다. trending_stacks·declining_stacks·"
        + "priority_skills는 모두 이 직군 범위의 기술로만 한정하세요. "
        + "타 직군 전용 기술(예: 백엔드 분석에 jQuery/Angular 같은 프론트엔드 스택)을 포함하지 마세요.\n"
        # [FIX #1] 증거 우선 + 기초 권유 방지
        + "- '유저가 이미 구현한 핵심 기술'에 나타난 역량은 강점으로 인정하고, "
        + "이를 '기초부터 배우라'고 권하지 마세요. priority_skills는 유저가 아직 약한 "
        + "영역(gap_areas) 중심으로, 현재 강점 위에 쌓을 수 있는 다음 단계를 제시하세요.\n"
        + "- coverage_pct/coverage_level은 '로드맵 횡단 폭' 지표일 뿐 구현 깊이가 아닙니다. "
        + "구현 깊이는 위 evidence로 판단하세요.\n"
        + "- kr_demand_pct 등 수치는 위 '한국 시장 데이터'에 근거가 있을 때만 제시하고, "
        + "근거가 없으면 추정임을 reason에 명시하세요.\n"
        + "\n위 데이터를 바탕으로 시장 흐름 분석과 개인 맞춤 방향을 JSON으로 제시하세요."
    )


def generate_market_brief(
    filter_result_dict: dict[str, Any],
    primary_roles: list[str],
    market_ctx: dict[str, Any],
    framework: str = "",
    devtype: str = "",
    project: dict[str, Any] | None = None,   # [FIX #1] 호출부에서 프로젝트 dict 전달
) -> dict[str, Any]:
    """rule_filter 데이터 + 시장 데이터 → 설명 탭용 시장 흐름 + 개인 방향 생성.

    [FIX #1] project를 함께 받아 실제 구현 역량(evidence)을 프롬프트에 주입한다.
             호출부에서 project(=분석 대상 프로젝트 dict)를 넘기도록 업데이트할 것.
    """
    primary_role  = primary_roles[0] if primary_roles else "backend"
    system_prompt = _build_market_brief_system(
        framework or primary_role,
        devtype   or primary_role,
    )
    client = _build_client()
    prompt = _build_market_brief_prompt(
        filter_result_dict, market_ctx, primary_role,
        project=project, devtype=devtype or primary_role,
    )

    response = client.models.generate_content(
        model=MARKET_BRIEF_MODEL,   # [FIX #7]
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=4096,
            temperature=0.4,
        ),
    )

    if response.text is None:
        candidates = getattr(response, "candidates", [])
        reason = getattr(candidates[0], "finish_reason", "UNKNOWN") if candidates else "UNKNOWN"
        raise RuntimeError(f"Gemini market_brief 응답 없음. finish_reason={reason}")

    raw = response.text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        clean = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)


def analyze(project: dict[str, Any], filter_result_dict: dict[str, Any]) -> dict[str, Any]:
    client = _build_client()
    user_prompt = _build_user_prompt(project, filter_result_dict)

    response = client.models.generate_content(
        model=MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=8192,
            temperature=0.3,
        ),
    )

    # response.text가 None이면 finish_reason 출력 후 원인 안내
    if response.text is None:
        candidates = getattr(response, "candidates", [])
        if candidates:
            reason = getattr(candidates[0], "finish_reason", "UNKNOWN")
            raise RuntimeError(
                f"Gemini 응답이 비어있습니다. finish_reason={reason}\n"
                "safety filter 차단이거나 max_output_tokens 초과일 수 있습니다."
            )
        raise RuntimeError("Gemini 응답이 비어있습니다 (candidates 없음).")

    raw = response.text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        clean = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)