"""
contextual_retrieval.py
─────────────────────────────────────────────────────────────────
Anthropic Contextual Retrieval 기법 적용

각 청크에 전체 문서 기반 짧은 문맥을 추가해 검색 품질 향상
  - context          : Gemini가 생성한 2~3문장 문맥
  - text_with_context: "문맥 + 원본 텍스트" (임베딩에 사용)

사용법:
    python contextual_retrieval.py
    python contextual_retrieval.py --input output/mock_cover_letters.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import time
from collections import defaultdict
from pathlib import Path

def _prevent_sleep() -> None:
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)

def _allow_sleep() -> None:
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

from dotenv import load_dotenv
load_dotenv()

from google import genai as _genai
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════
# 상수
# ═══════════════════════════════════════════════════════════════

_LLM_MODEL = "gemini-2.5-flash"

_PROMPT_TEMPLATE = """\
<document>
{whole_doc}
</document>
전체 문서 내에 배치하려는 청크는 다음과 같습니다.
<chunk>
{chunk_text}
</chunk>
이 청크가 검색될 때 더 잘 찾아지도록, 아래 두 가지를 포함한 2~3문장의 문맥을 작성하세요:
1. 이 청크가 문서에서 어떤 주제/섹션에 속하는지
2. 청크의 핵심 내용 (어떤 경험, 역량, 목표를 다루는지)
문맥만 작성하고 다른 내용은 입력하지 마세요."""
# ═══════════════════════════════════════════════════════════════
# 클라이언트
# ═══════════════════════════════════════════════════════════════

def _get_gemini_client() -> _genai.Client:
    project = os.getenv("GCP_PROJECT_ID")
    if not project:
        raise ValueError("GCP_PROJECT_ID 환경변수를 설정하세요.")
    return _genai.Client(vertexai=True, project=project, location="global")


# ═══════════════════════════════════════════════════════════════
# 유틸리티
# ═══════════════════════════════════════════════════════════════

def _build_whole_doc(group: list[dict]) -> str:
    """source 내 모든 청크를 이어 붙여 전체 문서 구성."""
    parts = []
    for c in group:
        sub = c.get("sub_section", "")
        header = f"[{c['section']} > {sub}]" if sub else f"[{c['section']}]"
        parts.append(f"{header}\n{c['text']}")
    return "\n\n".join(parts)


def _generate_context(chunk: dict, whole_doc: str, client: _genai.Client) -> str:
    """Gemini로 청크 문맥 생성 (실패 시 지수 백오프 재시도)."""
    prompt = _PROMPT_TEMPLATE.format(
        whole_doc=whole_doc,
        chunk_text=chunk["text"],
    )
    for attempt in range(5):
        try:
            resp = client.models.generate_content(
                model=_LLM_MODEL,
                contents=prompt,
            )
            return resp.text.strip()
        except Exception as e:
            err = str(e)
            if attempt < 4:
                wait = 60 if "429" in err or "rate" in err.lower() else 10 * (attempt + 1)
                tqdm.write(f"  [재시도 {attempt+1}/4] {err[:60]} → {wait}초 대기...")
                time.sleep(wait)
            else:
                raise


# ═══════════════════════════════════════════════════════════════
# 메인 파이프라인
# ═══════════════════════════════════════════════════════════════

def run(input_path: Path, output_path: Path, delay: float = 0.3) -> None:
    _prevent_sleep()
    client = _get_gemini_client()

    with open(input_path, encoding="utf-8") as f:
        chunks: list[dict] = json.load(f)

    # 이어쓰기: 이미 처리된 id 건너뜀
    result: list[dict] = []
    done_ids: set[str] = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            result = json.load(f)
        done_ids = {c["id"] for c in result}
        print(f"기존 {len(result)}개 로드 — 이어서 처리합니다.")

    # source별 그룹화 및 전체 문서 캐시
    groups: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        groups[c["source"]].append(c)
    doc_cache: dict[str, str] = {
        src: _build_whole_doc(grp) for src, grp in groups.items()
    }

    todo = [c for c in chunks if c["id"] not in done_ids]
    print(f"처리 대상: {len(todo)}개 청크 ({len(chunks) - len(todo)}개 이미 완료)")

    with tqdm(total=len(todo), desc="문맥 생성 중") as bar:
        for chunk in todo:
            try:
                ctx = _generate_context(chunk, doc_cache[chunk["source"]], client)
                chunk = {**chunk, "context": ctx, "text_with_context": f"{ctx}\n\n{chunk['text']}"}
                result.append(chunk)
                # 매 청크마다 저장 (중단 시 복구 가능)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
            except Exception as e:
                tqdm.write(f"  [ERROR] {chunk['id']} 실패: {str(e)[:80]}, 건너뜀")
            finally:
                bar.update(1)
                time.sleep(delay)

    _allow_sleep()
    print(f"\n완료: {len(result)}개 → {output_path}")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contextual Retrieval 문맥 추가 (Gemini on Vertex AI)")
    parser.add_argument("--type",   default="coverletter",
                        choices=["coverletter", "portfolio"],
                        help="처리할 문서 타입 (기본: coverletter)")
    parser.add_argument("--input",  default=None,
                        help="입력 JSON 파일 (미지정 시 type에 따라 자동 설정)")
    parser.add_argument("--output", default=None,
                        help="출력 JSON 파일 (미지정 시 type에 따라 자동 설정)")
    parser.add_argument("--delay",  type=float, default=0.3,
                        help="요청 간격(초), 기본 0.3")
    args = parser.parse_args()

    defaults = {
        "coverletter": ("output/mock_cover_letters.json",  "output/mock_cover_letters_ctx.json"),
        "portfolio":   ("output/mock_portfolios.json",     "output/mock_portfolios_ctx.json"),
    }
    in_path  = Path(args.input  or defaults[args.type][0])
    out_path = Path(args.output or defaults[args.type][1])

    run(in_path, out_path, delay=args.delay)
