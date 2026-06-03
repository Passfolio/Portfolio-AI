"""
portfolio_chunker.py
────────────────────────────────────────────────────────────
포트폴리오 PDF → 프로젝트 단위 청킹 (규칙 기반 1차 분할 + LLM 2차 메타 주입)

파이프라인:
  PDF
   └─ Upstage Document Parse API → elements (구조화된 요소 배열) + figure 요소
       ├─ 텍스트 요소 → 규칙 기반 알고리즘 (heading1 & page 전환점 기준 1차 청킹)
       ├─ 1차 청크 리스트 → Gemini LLM (각 청크별 병렬/순차 분석) → 메타데이터 & 섹션 확정
       └─ figure 요소 → Gemini Vision → 이미지 유형 분류 + 캡션 생성 (기존 유지)
           └─ 전체 청크 리스트 반환
"""

from __future__ import annotations

import base64
import concurrent.futures
import html
import json
import os
import re
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel
from google import genai as _genai
from google.genai import types as _types


# ═══════════════════════════════════════════════════════════════
# 설정값
# ═══════════════════════════════════════════════════════════════

MIN_CHUNK_CHARS = 80    # 청크 최소 길이 (이하면 드롭)
MAX_CHUNK_CHARS = 3000  # 임베딩 안전 한도 (한국어 ~2토큰/자 기준)
LLM_MODEL       = "gemini-3-flash-preview"  # 섹션 분리 + 메타 추출
VISION_MODEL    = "gemini-3-flash-preview"  # figure 분류 + 캡션
LLM_RETRIES     = 3
VISION_RETRIES  = 5

_UPSTAGE_FIGURE_CATEGORIES = {"figure"}


# ═══════════════════════════════════════════════════════════════
# Gemini 클라이언트
# ═══════════════════════════════════════════════════════════════

def _get_gemini_client() -> _genai.Client:
    project = os.getenv("GCP_PROJECT_ID")
    if project:
        return _genai.Client(vertexai=True, project=project, location=os.getenv("GCP_LOCATION", "global"))
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        return _genai.Client(api_key=api_key)
    raise ValueError("GCP_PROJECT_ID 또는 GEMINI_API_KEY 환경변수를 설정하세요.")


def _call_with_retry(fn, retries: int, label: str):
    """rate limit / 일시 오류에 대한 지수 백오프 재시도."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            err = str(e)
            is_rate_limit = "429" in err or "quota" in err.lower()
            is_server_err = "503" in err or "500" in err
            is_last = attempt == retries - 1
            if is_last or (not is_rate_limit and not is_server_err):
                raise
            wait = (60 if is_rate_limit else 10) * (attempt + 1)
            err_type = "429 Rate Limit" if is_rate_limit else "503 서버오류"
            print(f"  [WARN] {label} {err_type} (attempt={attempt+1}/{retries}) → {wait}초 대기")
            time.sleep(wait)


# ═══════════════════════════════════════════════════════════════
# 1. Upstage Document Parse & 규칙 기반 1차 청킹
# ═══════════════════════════════════════════════════════════════

def _parse_and_rule_chunk_with_upstage(pdf_path: str) -> tuple[list[dict], int, list[dict]]:
    """Upstage Document Parse API를 호출하고, elements 배열을 바탕으로 규칙 기반 1차 청킹을 수행합니다.

    Returns:
        intermediate_chunks: [{"suggested_title": str, "page": int, "raw_text": str}, ...]
        total_pages: 총 페이지 수
        figures: 이미지 요소 리스트
    """
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise ValueError("UPSTAGE_API_KEY 환경변수를 설정하세요.")

    with open(pdf_path, "rb") as f:
        resp = requests.post(
            "https://api.upstage.ai/v1/document-digitization",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"document": (Path(pdf_path).name, f, "application/pdf")},
            data={
                "model":           "document-parse",
                "output_formats":  '["markdown"]',
                "ocr":             "force",
                "base64_encoding": '["figure"]',
            },
        )
    if not resp.ok:
        print(f"[Upstage 오류] {resp.status_code}: {resp.text}")
        resp.raise_for_status()

    data     = resp.json()
    elements = data.get("elements", [])

    if not elements:
        # elements가 없을 경우의 Fallback 처리
        content  = data.get("content", {})
        markdown = content.get("markdown") or content.get("html") or content.get("text") or str(content)
        fallback_chunk = [{
            "suggested_title": "전체 본문 (Fallback)",
            "page": 1,
            "raw_text": markdown
        }]
        return fallback_chunk, 0, []

    total_pages = max((el.get("page", 0) for el in elements), default=0)
    
    intermediate_chunks: list[dict] = []
    figures: list[dict] = []
    
    current_title = "시작"
    current_page = 1
    current_parts = []
    fig_count = 0

    for el in elements:
        category = el.get("category")
        page = el.get("page", 1)
        
        # 이미지 요소 분리
        if category in _UPSTAGE_FIGURE_CATEGORIES:
            fig_count += 1
            image_b64 = el.get("base64_encoding")
            if image_b64:
                figures.append({
                    "id":        el.get("id", f"fig_{fig_count}"),
                    "page":      page,
                    "image_b64": image_b64,
                })
            continue

        md = el.get("content", {}).get("markdown", "").strip()
        if not md:
            continue

        # [규칙 기반 분할 트리거]
        # 1. heading1(대제목)을 만나거나 
        # 2. PPT 슬라이드가 바뀔 때 (포트폴리오는 페이지 전환이 문맥 전환인 경우가 많음)
        is_new_heading = category in ("heading1", "heading2") and len(md) < 100
        is_page_changed = page != current_page

        if (is_new_heading or is_page_changed) and current_parts:
            # 기존까지 모인 텍스트를 하나의 중간 청크로 빌드
            combined_text = "\n\n".join(current_parts).strip()
            if len(combined_text) >= MIN_CHUNK_CHARS:
                intermediate_chunks.append({
                    "suggested_title": current_title,
                    "page": current_page,
                    "raw_text": combined_text
                })
            current_parts = []
            if is_new_heading:
                current_title = md

        current_page = page
        current_parts.append(md)

    # 마지막 잔여 블록 처리
    if current_parts:
        combined_text = "\n\n".join(current_parts).strip()
        if len(combined_text) >= MIN_CHUNK_CHARS:
            intermediate_chunks.append({
                "suggested_title": current_title,
                "page": current_page,
                "raw_text": combined_text
            })

    if fig_count:
        print(f"  [Upstage] figure {fig_count}개 감지 → base64 확보 {len(figures)}개")

    return intermediate_chunks, total_pages, figures


# ═══════════════════════════════════════════════════════════════
# 2. 텍스트 후처리
# ═══════════════════════════════════════════════════════════════

_RE_IMAGE_TAG    = re.compile(r"", re.I)
_RE_BLANK_LINES  = re.compile(r"\n{3,}")
_RE_MD_IMAGE     = re.compile(r'!\[.*?\]\([^)]*\)')
_RE_HTML_IMG     = re.compile(r'<img\b[^>]*/?>', re.I)
_RE_FIGURE_TAG   = re.compile(r'<figure\b[^>]*>.*?</figure>', re.I | re.S)
_RE_CTRL_CHARS   = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')
_RE_CHART_META   = re.compile(r'^(Chart Type|Chart Title|X-Axis|Y-Axis)\s*:.*$', re.M)
_RE_SPACED_CHARS = re.compile(r'(?<=[A-Za-z가-힣])\s{2,}(?=[A-Za-z가-힣])')


def _clean_text(text: str) -> str:
    text = _RE_IMAGE_TAG.sub("", text)
    text = _RE_FIGURE_TAG.sub("", text)
    text = _RE_MD_IMAGE.sub("", text)
    text = _RE_HTML_IMG.sub("", text)
    text = _RE_CTRL_CHARS.sub("", text)
    text = _RE_CHART_META.sub("", text)
    text = _RE_SPACED_CHARS.sub(" ", text)
    text = _RE_BLANK_LINES.sub("\n\n", text)
    text = html.unescape(text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════
# 3. 사이즈 초과 청크 재분할
# ═══════════════════════════════════════════════════════════════

def _split_oversized(chunk: dict) -> list[dict]:
    """MAX_CHUNK_CHARS 초과 청크를 단락 단위로 재분할.

    분할 시 sub_section에 _0, _1, ... 접미사를 붙인다.
    """
    text = chunk["text"]
    if len(text) <= MAX_CHUNK_CHARS:
        return [chunk]

    segments: list[str] = [text]
    for sep in ("\n\n", "\n", "。", ". "):
        if max(len(s) for s in segments) <= MAX_CHUNK_CHARS:
            break
        new_segs: list[str] = []
        for s in segments:
            new_segs.extend(s.split(sep) if len(s) > MAX_CHUNK_CHARS else [s])
        segments = [s.strip() for s in new_segs if s.strip()]

    groups: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for seg in segments:
        if buf_len + len(seg) > MAX_CHUNK_CHARS and buf:
            groups.append("\n\n".join(buf))
            buf, buf_len = [seg], len(seg)
        else:
            buf.append(seg)
            buf_len += len(seg)
    if buf:
        groups.append("\n\n".join(buf))

    if len(groups) <= 1:
        return [chunk]

    base_sub = chunk.get("sub_section", "")
    result: list[dict] = []
    for i, g in enumerate(groups):
        if len(g) < MIN_CHUNK_CHARS:
            continue
        result.append({
            **chunk,
            "sub_section": f"{base_sub}_{i}" if base_sub else str(i),
            "text":        g,
            "char_count":  len(g),
        })

    print(f"  [SPLIT] '{chunk.get('project') or chunk['section']}' {len(text)}자 → {len(result)}개 분할")
    return result


# ═══════════════════════════════════════════════════════════════
# 4. Gemini Vision — figure 분류 + 캡션 (기존 유지)
# ═══════════════════════════════════════════════════════════════

_FIGURE_PROMPT = """\
이 이미지를 분석하세요.

[STEP 1] 아래 중 가장 적합한 유형 하나를 선택:
- architecture : 시스템/서비스 구성도, 인프라 다이어그램, 플로우차트
- erd          : DB 스키마, 테이블 관계도, 클래스 다이어그램
- ui           : 앱/웹 화면 캡처, 와이어프레임
- chart        : 성능 그래프, 지표 차트, 비교표
- code_image   : 코드가 이미지로 캡처된 것
- other        : 위에 해당 없음

[STEP 2] 유형에 맞는 2~3문장 캡션 작성:
- architecture : 구조 패턴(MSA/모놀리식 등), 주요 컴포넌트와 역할, 데이터 흐름, 기술스택
- erd          : 주요 엔티티(테이블) 목록, 핵심 관계(1:N/N:M), 도메인 목적
- ui           : 기능 화면명, 주요 UI 요소, 사용자 흐름
- chart        : 지표 종류(TPS/응답시간 등), 핵심 수치, 개선 결과(before/after)
- code_image   : 코드 목적, 핵심 구현 방식, 사용 기술/라이브러리
- other        : 이미지 내용 요약

반드시 아래 형식으로만 출력하세요 (다른 텍스트 금지):
TYPE: {유형}
CAPTION: {캡션}
"""


def _detect_mime_type(image_b64: str) -> str:
    if image_b64.startswith("/9j/"):   return "image/jpeg"
    if image_b64.startswith("iVBOR"): return "image/png"
    if image_b64.startswith("R0lG"):  return "image/gif"
    if image_b64.startswith("UklG"):  return "image/webp"
    return "image/jpeg"


def _parse_figure_response(raw: str) -> tuple[str, str]:
    img_type = "other"
    caption  = raw
    for line in raw.splitlines():
        if line.startswith("TYPE:"):
            img_type = line.split(":", 1)[1].strip().lower()
        elif line.startswith("CAPTION:"):
            caption = line.split(":", 1)[1].strip()
    return img_type, caption


def _caption_figure(image_b64: str, client: _genai.Client) -> tuple[str, str]:
    mime_type = _detect_mime_type(image_b64)

    def _call():
        resp = client.models.generate_content(
            model=VISION_MODEL,
            contents=[
                _types.Part.from_bytes(
                    data=base64.b64decode(image_b64),
                    mime_type=mime_type,
                ),
                _FIGURE_PROMPT,
            ],
        )
        return _parse_figure_response(resp.text.strip())

    return _call_with_retry(_call, retries=VISION_RETRIES, label="Vision 캡션")


def _resolve_figure_project(fig_page: int, text_chunks: list[dict]) -> str:
    best_project = ""
    best_dist    = float("inf")
    for c in text_chunks:
        chunk_page = c.get("page", 0)
        if chunk_page and c.get("project"):
            dist = abs(chunk_page - fig_page)
            if dist < best_dist:
                best_dist    = dist
                best_project = c["project"]
    return best_project


_CAPTION_CACHE_DIR = Path(__file__).parent.parent / "output" / "caption_cache"


def _cache_path(source: str) -> Path:
    return _CAPTION_CACHE_DIR / f"{source}_image_captions.json"


def _load_caption_cache(source: str) -> dict[str, dict] | None:
    path = _cache_path(source)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_caption_cache(source: str, cache: dict[str, dict]) -> None:
    _CAPTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_cache_path(source), "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _build_figure_chunks(
    figures: list[dict],
    source: str,
    client: _genai.Client,
    text_chunks: list[dict],
) -> list[dict]:
    img_dir = Path(__file__).parent.parent / "output" / "images" / source
    img_dir.mkdir(parents=True, exist_ok=True)

    cache     = _load_caption_cache(source) or {}
    cache_hit = sum(1 for fig in figures if fig["id"] in cache)
    if cache_hit:
        print(f"  [캡션 캐시] {cache_hit}/{len(figures)}개 캐시 적중 → Vision API 생략")

    chunks: list[dict] = []
    total    = len(figures)
    modified = False

    for i, fig in enumerate(figures, 1):
        try:
            img_bytes = base64.b64decode(fig["image_b64"])
            img_path  = img_dir / f"{fig['id']}.jpg"
            img_path.write_bytes(img_bytes)

            if fig["id"] in cache:
                entry    = cache[fig["id"]]
                img_type = entry["img_type"]
                caption  = entry["caption"]
                project  = entry["project"]
            else:
                img_type, caption = _caption_figure(fig["image_b64"], client)
                project = _resolve_figure_project(fig["page"], text_chunks)
                cache[fig["id"]] = {
                    "img_type":   img_type,
                    "caption":    caption,
                    "page":       fig["page"],
                    "project":    project,
                    "image_path": str(img_path),
                }
                modified = True
                print(f"  [figure {i}/{total}] page={fig['page']} type={img_type} project={project or '?'} → {len(caption)}자")

            if len(caption) < MIN_CHUNK_CHARS:
                continue

            base_chunk = {
                "source":            source,
                "doc_type":          "portfolio",
                "section":           "프로젝트경험",
                "project":           project,
                "sub_section":       "이미지",
                "content_type":      img_type,
                "page":              fig["page"],
                "fig_id":            fig["id"],
                "image_path":        str(img_path),
                "text":              caption,
                "context":           "",
                "text_with_context": "",
                "meta":              {},
                "char_count":        len(caption),
            }
            chunks.extend(_split_oversized(base_chunk))

        except Exception as e:
            print(f"  [figure {i}/{total}] page={fig['page']} 캡션 생성 실패: {e}")

    if modified:
        _save_caption_cache(source, cache)

    return chunks


# ═══════════════════════════════════════════════════════════════
# 5. Gemini LLM — 1차 청크에 대한 메타데이터 주입 (Sequential Map)
# ═══════════════════════════════════════════════════════════════

class _Meta(BaseModel):
    period:        str
    role:          str
    team:          str
    tech_stack:    list[str]
    contributions: list[str]
    achievements:  list[str]
    keywords:      list[str]


class _ChunkAnalysis(BaseModel):
    section:     str  # 프로젝트경험, 기술스택, 자기소개, 경력, 기타
    project:     str  # 프로젝트 이름 (없으면 "")
    sub_section: str  # 프로젝트경험일 때: 개요, 개발, 이슈, 성과 / 나머지: ""
    meta:        _Meta


_CHUNK_ANALYSIS_PROMPT = """\
주어진 텍스트 청크를 분석하여 문서 구조 정보와 핵심 메타데이터를 추출해주세요.

[분석 대상 청크 정보]
- 추정 제목: {suggested_title}
- 해당 페이지: {page}
- 본문 내용:
{text}

[section 선택 기준]
- 프로젝트경험, 기술스택, 자기소개, 경력, 기타 중 하나 선택

[프로젝트경험 sub_section 분류 기준]
- "개요": 프로젝트 개요, 기술스택, 아키텍처, 핵심 요구사항
- "개발": 핵심 개발사항, 주요 구현 내용
- "이슈": 주요 이슈 & 해결, 트러블슈팅 경험
- "성과": 성과, 설계 원칙, 배운 점
- 프로젝트경험 외 section은 반드시 빈 문자열("")로 설정

[meta 추출 기준] — 프로젝트경험일 때 기술하고, 나머지는 str->"" / list->[] 처리
- period:        "YYYY.MM ~ YYYY.MM" 형식으로 정제. 없으면 ""
- role:          역할/담당 설명을 간결하게. 없으면 ""
- team:          명시된 경우만 기재 (예: "4인 팀"). 없으면 ""
- tech_stack:    언어·프레임워크·라이브러리·툴 목록. 없으면 []
- contributions: 본인의 구현·설계 내용을 "~구현", "~개발" 형태로. 없으면 []
- achievements:  수치 포함 성과 또는 명확한 결과물. 없으면 []
- keywords:      핵심 기술/도메인 키워드. 없으면 []
"""

_CHUNK_ANALYSIS_SYSTEM = (
    "당신은 IT 포트폴리오 구조화 전문가입니다. "
    "주어진 단일 청크의 텍스트를 파악해 올바른 섹션 분류와 JSON 메타데이터를 채워 반환해야 합니다. "
    "본문에 명시되지 않은 정보는 지어내지 말고 공백이나 빈 배열로 두십시오."
)


def _analyze_chunk_meta(chunk_info: dict, client: _genai.Client) -> dict | None:
    """규칙 파싱된 단일 청크를 LLM에 전달하여 고도로 구조화된 메타데이터를 동적으로 주입합니다."""
    text_content = chunk_info["raw_text"]
    
    prompt = _CHUNK_ANALYSIS_PROMPT.format(
        suggested_title=chunk_info["suggested_title"],
        page=chunk_info["page"],
        text=text_content
    )

    def _call():
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config=_types.GenerateContentConfig(
                system_instruction=_CHUNK_ANALYSIS_SYSTEM,
                response_mime_type="application/json",
                response_schema=_ChunkAnalysis,
                temperature=0,
            ),
        )
        return json.loads(response.text)

    try:
        analysis = _call_with_retry(_call, retries=LLM_RETRIES, label=f"청크 분석(p.{chunk_info['page']})")
    except Exception as e:
        print(f"  [ERROR] 청크 메타 주입 실패 (p.{chunk_info['page']}): {e}")
        return None

    cleaned_text = _clean_text(text_content)
    if len(cleaned_text) < MIN_CHUNK_CHARS:
        return None

    meta_raw: dict = analysis.get("meta", {})
    meta = {
        k: v for k, v in meta_raw.items()
        if (isinstance(v, str) and v.strip()) or (isinstance(v, list) and v)
    }

    return {
        "section":           analysis.get("section", "기타"),
        "project":           analysis.get("project", ""),
        "sub_section":       analysis.get("sub_section", ""),
        "page":              chunk_info["page"],
        "text":              cleaned_text,
        "context":           "",
        "text_with_context": "",
        "meta":              meta,
        "char_count":        len(cleaned_text),
    }


# ═══════════════════════════════════════════════════════════════
# 6. 공개 API
# ═══════════════════════════════════════════════════════════════

def chunk(pdf_path: str) -> list[dict]:
    """PDF → Upstage 규칙 기반 1차 분할 → 개별 청크 LLM 메타 주입 → Figure 결합 파이프라인."""
    source = Path(pdf_path).stem
    print(f"[시작] 하이브리드 파이프라인 프로세싱: {Path(pdf_path).name}")

    # 1. 규칙 기반 1차 분할 수행
    t0 = time.time()
    intermediate_chunks, total_pages, figures = _parse_and_rule_chunk_with_upstage(pdf_path)
    print(f"  [완료] 1차 규칙 분할 ({time.time() - t0:.1f}s) -> 생성된 중간 블록: {len(intermediate_chunks)}개")

    if not intermediate_chunks and not figures:
        return []

    client = _get_gemini_client()
    text_chunks: list[dict] = []

    # 2. 중간 청크마다 2차 병렬 LLM 메타 주입
    if intermediate_chunks:
        print(f"  → 개별 청크별 메타데이터 병렬 주입 시작... ({len(intermediate_chunks)}개)")
        t1 = time.time()

        total = len(intermediate_chunks)

        def _analyze(args: tuple) -> dict | None:
            idx, inter_chunk = args
            print(f"  [LLM {idx}/{total}] p.{inter_chunk['page']} 「{inter_chunk['suggested_title'][:30]}」 분석 중...")
            analyzed = _analyze_chunk_meta(inter_chunk, client)
            if analyzed:
                analyzed["source"] = source
                analyzed["doc_type"] = "portfolio"
                print(f"  [LLM {idx}/{total}] 완료 → section={analyzed['section']} project={analyzed['project'][:20] or '-'}")
            return analyzed

        max_workers = min(3, len(intermediate_chunks))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_analyze, (i, c)) for i, c in enumerate(intermediate_chunks, 1)]
            results = [f.result() for f in futures]

        for analyzed in results:
            if analyzed:
                text_chunks.extend(_split_oversized(analyzed))

        print(f"  [완료] LLM 메타 주입 완료 ({time.time() - t1:.1f}s) -> 텍스트 청크 {len(text_chunks)}개 확정")

    # 4. 이미지 캡션 및 인접 프로젝트 바인딩 (기존 메커니즘 유지)
    img_chunks: list[dict] = []
    if figures:
        print(f"  → Gemini Vision figure 캡션 프로세싱 ({len(figures)}개)...")
        t2 = time.time()
        img_chunks = _build_figure_chunks(figures, source, client, text_chunks)
        print(f"  [완료] Vision 처리 완료 ({time.time() - t2:.1f}s) -> 이미지 청크 {len(img_chunks)}개 생성")

    # 최종 병합 및 고유 ID 부여
    total = text_chunks + img_chunks
    for i, c in enumerate(total):
        c["id"] = f"{source}_{i:03d}"

    print(f"  [최종] 파이프라인 종료: 총 {len(total)}개 청크 (텍스트 {len(text_chunks)} + 이미지 {len(img_chunks)})")
    return total


def get_markdown(pdf_path: str) -> str:
    """기존 raw 디버깅 호환성 유지용"""
    api_key = os.getenv("UPSTAGE_API_KEY")
    with open(pdf_path, "rb") as f:
        resp = requests.post(
            "https://api.upstage.ai/v1/document-digitization",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"document": (Path(pdf_path).name, f, "application/pdf")},
            data={"model": "document-parse", "output_formats": '["markdown"]'},
        )
    return resp.json().get("content", {}).get("markdown", "")


# ═══════════════════════════════════════════════════════════════
# 7. CLI Output 뷰어 (기존 유지)
# ═══════════════════════════════════════════════════════════════

def _write_chunks_md(results: list[dict], pdf_stem: str) -> Path:
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    md_path = output_dir / f"{pdf_stem}_chunks.md"

    lines = [f"# {pdf_stem} 포트폴리오 청킹 결과\n\n총 {len(results)}개 청크\n"]
    for i, c in enumerate(results, 1):
        content_type = f"  |  content_type: {c['content_type']}" if c.get("content_type") else ""
        lines.append(
            f"---\n\n### [{i}/{len(results)}] "
            f"section: {c['section']}  |  project: {c['project']}  |  "
            f"sub: {c['sub_section']}{content_type}  |  {c['char_count']}자\n"
        )
        if c.get("meta"):
            for k, v in c["meta"].items():
                lines.append(f"> **{k}**: {v}  ")
            lines.append("")
        lines.append(c["text"])
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


if __name__ == "__main__":
    import sys

    pdf = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).parent.parent / "portfoliosample" / "output예시 포폴.pdf"
    )
    mode = sys.argv[2] if len(sys.argv) > 2 else "chunk"

    if mode == "raw":
        print(get_markdown(str(pdf)))
    else:
        results = chunk(str(pdf))
        print(f"\n총 {len(results)}개 청크 생성\n{'=' * 60}")
        md_path = _write_chunks_md(results, pdf.stem)
        print(f"MD 저장 완료: {md_path}")