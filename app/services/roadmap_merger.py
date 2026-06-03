"""
여러 프로젝트 rule_filter 결과 → 합산된 단일 평가 입력 생성

합산 규칙:
- topic/subtopic status: covered > partial > uncovered (any 프로젝트에서 covered이면 covered)
- skills / core_feat / core_perf: 전 프로젝트 union
- coverage_pct: weighted average
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.roadmap import rule_filter as rf

ROADMAP_CSV = Path(__file__).resolve().parents[2] / "DATASET" / "roadmap.csv"
_RANK = {"covered": 2, "partial": 1, "uncovered": 0}


def _merge_status(status_list: list[dict[str, str]]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for status in status_list:
        for label, st in status.items():
            if _RANK.get(st, 0) > _RANK.get(merged.get(label, "uncovered"), 0):
                merged[label] = st
    return merged


def _merge_projects(projects: list[dict[str, Any]]) -> dict[str, Any]:
    skills_seen: set[str] = set()
    skills: list[dict] = []
    frameworks_seen: set[str] = set()
    frameworks: list[dict] = []
    core_feat: list[dict] = []
    core_perf: list[dict] = []
    service_names: list[str] = []

    for p in projects:
        p = rf.normalize_project(p)
        if p.get("service_name"):
            service_names.append(p["service_name"])

        for s in (p.get("skills") or []):
            name = s.get("name") if isinstance(s, dict) else str(s)
            if name and name not in skills_seen:
                skills_seen.add(name)
                skills.append(s if isinstance(s, dict) else {"name": name})

        for f in (p.get("frameworks") or []):
            name = f.get("name") if isinstance(f, dict) else str(f)
            if name and name not in frameworks_seen:
                frameworks_seen.add(name)
                frameworks.append(f if isinstance(f, dict) else {"name": name})

        analysis = p.get("analysis_form") or {}
        core_feat.extend(analysis.get("core_feat", []))
        core_perf.extend(analysis.get("core_perf", []))

    return {
        "service_name":            " / ".join(service_names) or "merged",
        "service_description":     f"{len(projects)}개 프로젝트 합산 평가",
        "dev_type":                projects[0].get("dev_type", ""),
        "frameworks":              frameworks,
        "skills":                  skills,
        "user_role":               projects[0].get("user_role", ""),
        "contribute_share_percent": 100,
        "analysis_form": {
            "core_feat":  core_feat,
            "core_perf":  core_perf,
            "feedback":   [],
            "contribute": {"contribute_titles": []},
        },
    }


def run_merged(projects: list[dict[str, Any]]) -> tuple[dict[str, Any], rf.FilterResult]:
    all_nodes = rf.load_roadmap(ROADMAP_CSV)
    kr_mode   = rf._is_kr_roadmap(all_nodes)

    merged_project = _merge_projects(projects)
    merged_project = rf.normalize_project(merged_project)

    primary, secondary = rf.detect_roles(merged_project)
    strong, weak = rf.collect_signal_labels(merged_project, kr_mode=kr_mode)

    for p in projects:
        p_norm = rf.normalize_project(p)
        s, w = rf.collect_signal_labels(p_norm, kr_mode=kr_mode)
        strong |= s
        weak   |= (w - strong)

    filter_result = rf.FilterResult(
        service_name=merged_project.get("service_name", ""),
        primary_roles=primary,
        secondary_roles=secondary,
    )

    for role in primary + secondary:
        _seen: dict[str, rf.RoadmapNode] = {}
        for n in sorted((n for n in all_nodes if n.roadmap_role == role),
                        key=lambda n: n.order_index):
            if n.topic_label not in _seen:
                _seen[n.topic_label] = n
        role_nodes = list(_seen.values())
        if not role_nodes:
            continue

        parent_map = rf.build_parent_map(role_nodes, role)
        status     = rf.classify_nodes(role_nodes, strong, weak, parent_map)
        score, level = rf.compute_score(role_nodes, status, role, kr_mode=kr_mode)
        tier_map   = rf._get_tier_map(role, kr_mode)

        def entry(n: rf.RoadmapNode, _tm=tier_map, _pm=parent_map, _st=status) -> dict:
            return {
                "label":  n.topic_label,
                "type":   n.node_type,
                "parent": _pm.get(n.topic_label, n.parent_topic),
                "order":  n.order_index,
                "tier":   _tm.get(n.topic_label, "support"),
                "status": _st.get(n.topic_label, "uncovered"),
            }

        topics  = [n for n in role_nodes if n.node_type == "topic"]
        ordered = sorted(role_nodes, key=lambda n: n.order_index)

        filter_result.per_role[role] = {
            "total":           len(role_nodes),
            "topic_coverage":  f"{sum(1 for t in topics if status.get(t.topic_label)=='covered')}/{len(topics)}",
            "weighted_score":  score,
            "coverage_pct":    round(score * 100, 1),
            "coverage_level":  level,
            "sequence":        rf.analyze_sequence(role_nodes, status, role, kr_mode=kr_mode),
            "roadmap_path":    rf.build_roadmap_path(role_nodes, status, parent_map, role, kr_mode=kr_mode),
            "covered":         [entry(n) for n in ordered if status.get(n.topic_label) == "covered"],
            "partial":         [entry(n) for n in ordered if status.get(n.topic_label) == "partial"],
            "uncovered":       [entry(n) for n in ordered if status.get(n.topic_label) == "uncovered"],
            "covered_count":   sum(1 for n in ordered if status.get(n.topic_label) == "covered"),
            "partial_count":   sum(1 for n in ordered if status.get(n.topic_label) == "partial"),
            "uncovered_count": sum(1 for n in ordered if status.get(n.topic_label) == "uncovered"),
        }

    return merged_project, filter_result
