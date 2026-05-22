"""
portfolio_chunker.py
────────────────────────────────────────────────────────────
포트폴리오 PDF → 프로젝트 단위 청킹

파이프라인:
  PDF
   └─ Upstage Document Parse API → 마크다운 + figure 요소(base64)
       ├─ 텍스트 요소 → Gemini LLM (1-pass) → 섹션 분리 + 메타 추출
       └─ figure 요소 → Gemini Vision → 이미지 유형 분류 + 캡션 생성
           └─ 전체 청크 리스트 반환

필요 환경변수:
  UPSTAGE_API_KEY  — Document Parse (텍스트 추출)
  GCP_PROJECT_ID   — Vertex AI (Gemini)
  GCP_LOCATION     — Vertex AI 리전 (기본값: global)

설치:
  pip install requests google-genai pydantic python-dotenv
"""

from __future__ import annotations

import base64
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
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        return _genai.Client(api_key=api_key)
    project = os.getenv("GCP_PROJECT_ID")
    if project:
        return _genai.Client(vertexai=True, project=project, location=os.getenv("GCP_LOCATION", "global"))
    raise ValueError("GEMINI_API_KEY 또는 GCP_PROJECT_ID 환경변수를 설정하세요.")


def _call_with_retry(fn, retries: int, label: str):
    """rate limit / 일시 오류에 대한 지수 백오프 재시도."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            err = str(e)
            is_rate_limit = "429" in err or "quota" in err.lower()
            is_last = attempt == retries - 1
            if is_last or not is_rate_limit:
                raise
            wait = (60 if is_rate_limit else 5) * (attempt + 1)
            print(f"  [WARN] {label} 실패 ({err[:60]}). {wait}초 후 재시도...")
            time.sleep(wait)


# ═══════════════════════════════════════════════════════════════
# 1. Upstage Document Parse
# ═══════════════════════════════════════════════════════════════

def _parse_with_upstage(pdf_path: str) -> tuple[str, int, list[dict], dict[int, int]]:
    """Upstage Document Parse API → (마크다운, 총 페이지 수, figure 목록, line_to_page).

    line_to_page: {줄 번호 → 페이지 번호} — 텍스트 청크의 page 필드 채우는 데 사용.
    figure 목록: [{"id": str, "page": int, "image_b64": str}, ...]
    elements 없으면 전체 markdown으로 폴백, figure/line_to_page는 빈값.
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
        content  = data.get("content", {})
        markdown = (
            content.get("markdown")
            or content.get("html")
            or content.get("text")
            or str(content)
        )
        return markdown, 0, [], {}

    total_pages = max((el.get("page", 0) for el in elements), default=0)
    parts: list[str]    = []
    figures: list[dict] = []
    fig_count  = 0
    cur_line   = 1
    line_to_page: dict[int, int] = {}

    for el in elements:
        if el.get("category") not in _UPSTAGE_FIGURE_CATEGORIES:
            md = el.get("content", {}).get("markdown", "").strip()
            if md:
                page       = el.get("page", 0)
                el_lines   = md.count("\n") + 1
                for ln in range(cur_line, cur_line + el_lines):
                    line_to_page[ln] = page
                cur_line += el_lines + 1  # +1: \n\n 구분자로 생기는 빈 줄
                parts.append(md)
        else:
            fig_count += 1
            image_b64 = el.get("base64_encoding")
            if image_b64:
                figures.append({
                    "id":        el.get("id", f"fig_{fig_count}"),
                    "page":      el.get("page", 0),
                    "image_b64": image_b64,
                })

    if fig_count:
        print(f"  [Upstage] figure {fig_count}개 감지 → base64 확보 {len(figures)}개")

    return "\n\n".join(parts), total_pages, figures, line_to_page


# ═══════════════════════════════════════════════════════════════
# 2. 텍스트 후처리
# ═══════════════════════════════════════════════════════════════

_RE_IMAGE_TAG    = re.compile(r"<!--\s*image\s*-->", re.I)
_RE_BLANK_LINES  = re.compile(r"\n{3,}")
_RE_MD_IMAGE     = re.compile(r'!\[.*?\]\([^)]*\)')
_RE_HTML_IMG     = re.compile(r'<img\b[^>]*/?>', re.I)
_RE_FIGURE_TAG   = re.compile(r'<figure\b[^>]*>.*?</figure>', re.I | re.S)
_RE_CTRL_CHARS   = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')
# Upstage가 차트 이미지를 텍스트로 변환할 때 생기는 노이즈
_RE_CHART_META   = re.compile(r'^(Chart Type|Chart Title|X-Axis|Y-Axis)\s*:.*$', re.M)
# OCR 레이아웃 노이즈: 글자 사이 과도한 공백 (예: "P A R K")
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

    \\n\\n → \\n → 문장 끝 순으로 경계를 찾고 MAX_CHUNK_CHARS 이하로 묶는다.
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
# 4. Gemini Vision — figure 분류 + 캡션 (1-pass)
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
    """base64 앞 4자로 이미지 포맷 판별 (magic bytes)."""
    if image_b64.startswith("/9j/"):   return "image/jpeg"
    if image_b64.startswith("iVBOR"): return "image/png"
    if image_b64.startswith("R0lG"):  return "image/gif"
    if image_b64.startswith("UklG"):  return "image/webp"
    return "image/jpeg"


def _parse_figure_response(raw: str) -> tuple[str, str]:
    """Gemini Vision 응답 텍스트 → (img_type, caption) 파싱."""
    img_type = "other"
    caption  = raw  # 파싱 실패 시 전체를 캡션으로
    for line in raw.splitlines():
        if line.startswith("TYPE:"):
            img_type = line.split(":", 1)[1].strip().lower()
        elif line.startswith("CAPTION:"):
            caption = line.split(":", 1)[1].strip()
    return img_type, caption


def _caption_figure(image_b64: str, client: _genai.Client) -> tuple[str, str]:
    """Gemini Vision으로 이미지 유형 분류 + 캡션 생성.

    Returns:
        (image_type, caption)
        image_type: architecture | erd | ui | chart | code_image | other
    """
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
    """figure의 페이지 번호로 가장 가까운 텍스트 청크의 project 값을 매핑."""
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
    """캐시 파일이 있으면 {fig_id: {img_type, caption, page, project, image_path}} 반환."""
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
    """figure 요소 리스트 → 이미지 저장 + Gemini Vision 캡션 생성 → 청크 리스트 반환.

    캡션은 output/caption_cache/{source}_image_captions.json에 캐싱되며,
    재실행 시 캐시가 있는 figure는 Vision API 호출을 건너뜁니다.
    """
    img_dir = Path(__file__).parent.parent / "output" / "images" / source
    img_dir.mkdir(parents=True, exist_ok=True)

    cache     = _load_caption_cache(source) or {}
    cache_hit = sum(1 for fig in figures if fig["id"] in cache)
    if cache_hit:
        print(f"  [캡션 캐시] {cache_hit}/{len(figures)}개 캐시 적중 → Vision API 생략")

    chunks: list[dict] = []
    total    = len(figures)
    modified = False  # 새 캡션이 생성됐을 때만 캐시 저장

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
                print(f"  [figure {i}/{total}] page={fig['page']} → 캡션 너무 짧음, 스킵")
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
        print(f"  [캡션 캐시] 저장 완료 → {_cache_path(source)}")

    return chunks


# ═══════════════════════════════════════════════════════════════
# 5. Gemini LLM — 섹션 분리 + 메타 추출 (1-pass)
# ═══════════════════════════════════════════════════════════════

class _Meta(BaseModel):
    period:        str
    role:          str
    team:          str
    tech_stack:    list[str]
    contributions: list[str]
    achievements:  list[str]
    keywords:      list[str]


class _Section(BaseModel):
    section:     str
    project:     str
    sub_section: str  # 프로젝트경험: "개요"/"개발"/"이슈"/"성과" / 나머지: ""
    start_line:  int
    end_line:    int
    meta:        _Meta


class _SectionList(BaseModel):
    sections: list[_Section]


_SECTION_SPLIT_PROMPT = """\
다음은 포트폴리오 PDF에서 추출한 마크다운입니다. 각 줄 앞에 줄 번호가 붙어 있습니다.
이 텍스트를 읽고 섹션/서브섹션 단위로 분리하고, 각 섹션의 메타데이터도 함께 추출해주세요.

[section 선택 기준]
- 프로젝트경험: 개발/기획/디자인 등 프로젝트 경험
- 기술스택: 보유 기술, 언어, 프레임워크 목록
- 자기소개: 프로필, 소개, 목표
- 경력: 인턴, 직장, 대외활동, 수상
- 기타: 위에 해당하지 않는 내용

[분리 기준]
- section이 프로젝트경험이면 각 프로젝트를 sub_section 단위로 세분화 (아래 참고)
- 요약 슬라이드와 상세 슬라이드가 같은 프로젝트면 상세 내용을 기준으로 sub_section 분리
- start_line과 end_line은 실제 줄 번호(1~{total_lines})여야 하며 누락 없이 커버
- 모든 줄은 정확히 하나의 섹션에 속해야 합니다

[프로젝트경험 sub_section 분리 기준]
- "개요": 프로젝트 개요, 기술스택, 아키텍처, 핵심 요구사항
- "개발": 핵심 개발사항, 주요 구현 내용
- "이슈": 주요 이슈 & 해결, 트러블슈팅 경험
- "성과": 성과, 설계 원칙, 배운 점
- 위 내용이 없는 sub_section은 생략
- 프로젝트경험 외 section은 sub_section을 빈 문자열("")로 설정

[meta 추출 기준] — 프로젝트경험일 때 모든 sub_section에 동일 값, 나머지는 str→"" / list→[]
- period:        "YYYY.MM ~ YYYY.MM" 형식으로 정제. 없으면 ""
- role:          역할/담당 설명을 간결하게. 없으면 ""
- team:          명시된 경우만 기재 (예: "4인 팀"). 없으면 반드시 ""
- tech_stack:    언어·프레임워크·라이브러리·인프라·툴. 없으면 []
- contributions: 수치 없이 본인 역할·구현·설계 내용을 "~구현", "~개발" 형태로. 없으면 []
- achievements:  수치 포함 성과 또는 명확한 결과. contributions와 중복 금지. 없으면 []
- keywords:      기술·직무·도메인 키워드 중복 없이. 없으면 []

마크다운:
{numbered_md}
"""

_SECTION_SPLIT_SYSTEM = (
    "당신은 포트폴리오 문서 구조 분석 전문가입니다. "
    "줄 번호가 붙은 마크다운을 읽고 섹션 경계(start_line, end_line)와 "
    "메타데이터를 정확히 추출합니다. "
    "텍스트를 복사하지 말고 줄 번호와 정제된 메타값만 반환하세요."
)


def _number_lines(markdown: str) -> tuple[str, list[str]]:
    lines    = markdown.split("\n")
    numbered = "\n".join(f"{i + 1:04d} | {line}" for i, line in enumerate(lines))
    return numbered, lines


def _slice_lines(lines: list[str], start: int, end: int) -> str:
    return "\n".join(lines[max(0, start - 1):min(len(lines), end)]).strip()


def _gemini_split_sections(
    markdown: str,
    source: str,
    client: _genai.Client,
    line_to_page: dict[int, int],
) -> list[dict]:
    """Gemini로 포트폴리오 마크다운을 섹션 단위로 분리하고 메타 추출 (1-pass)."""
    numbered_md, raw_lines = _number_lines(markdown)
    prompt = _SECTION_SPLIT_PROMPT.format(
        total_lines=len(raw_lines),
        numbered_md=numbered_md,
    )

    def _call():
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config=_types.GenerateContentConfig(
                system_instruction=_SECTION_SPLIT_SYSTEM,
                response_mime_type="application/json",
                response_schema=_SectionList,
            ),
        )
        return json.loads(response.text)

    data = _call_with_retry(_call, retries=LLM_RETRIES, label="LLM 섹션 분리")

    chunks: list[dict] = []
    for item in data["sections"]:
        start = int(item["start_line"])
        raw   = _slice_lines(raw_lines, start, int(item["end_line"]))
        text  = _clean_text(raw)

        if len(text) < MIN_CHUNK_CHARS:
            continue

        meta_raw: dict = item.get("meta", {})
        meta = {
            k: v for k, v in meta_raw.items()
            if (isinstance(v, str) and v.strip()) or (isinstance(v, list) and v)
        }

        base_chunk = {
            "source":            source,
            "doc_type":          "portfolio",
            "section":           item["section"],
            "project":           item["project"],
            "sub_section":       item.get("sub_section", ""),
            "page":              line_to_page.get(start, 0),
            "text":              text,
            "context":           "",
            "text_with_context": "",
            "meta":              meta,
            "char_count":        len(text),
        }
        chunks.extend(_split_oversized(base_chunk))

    return chunks


# ═══════════════════════════════════════════════════════════════
# 6. 공개 API
# ═══════════════════════════════════════════════════════════════

def chunk(pdf_path: str) -> list[dict]:
    """PDF → Upstage Document Parse → 텍스트 청크 + figure 캡션 청크 반환."""
    source = Path(pdf_path).stem
    print(f"[Upstage] Document Parse: {Path(pdf_path).name}")

    markdown, total_pages, figures, line_to_page = _parse_with_upstage(pdf_path)

    if not markdown.strip() and not figures:
        return []

    page_info = f"총 {total_pages}페이지 / " if total_pages else ""
    print(f"  마크다운 추출 완료 ({page_info}{len(markdown)}자)")

    client = _get_gemini_client()

    # ── 텍스트 청크
    text_chunks: list[dict] = []
    if markdown.strip():
        print("  → Gemini LLM 섹션 분리 중...")
        text_chunks = _gemini_split_sections(markdown, source, client, line_to_page)
        print(f"  텍스트 청크 {len(text_chunks)}개 생성")

    # ── figure 캡션 청크
    img_chunks: list[dict] = []
    if figures:
        print(f"  → Gemini Vision figure 캡션 생성 중... ({len(figures)}개)")
        img_chunks = _build_figure_chunks(figures, source, client, text_chunks)
        print(f"  이미지 청크 {len(img_chunks)}개 생성")

    total = text_chunks + img_chunks
    for i, c in enumerate(total):
        c["id"] = f"{source}_{i:03d}"

    print(f"  최종 청크 합계: {len(total)}개 (텍스트 {len(text_chunks)} + 이미지 {len(img_chunks)})")
    return total


def get_markdown(pdf_path: str) -> str:
    """PDF를 마크다운 문자열로 변환 (디버깅용)."""
    markdown, total_pages, figures, _ = _parse_with_upstage(pdf_path)
    if total_pages:
        print(f"총 {total_pages}페이지")
    if figures:
        print(f"figure {len(figures)}개 감지 (캡션 생성 안 함 — raw 모드)")
    return markdown


# ═══════════════════════════════════════════════════════════════
# 7. CLI
# ═══════════════════════════════════════════════════════════════

def _write_chunks_md(results: list[dict], pdf_stem: str) -> Path:
    """청크 결과를 마크다운 파일로 저장하고 경로 반환."""
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