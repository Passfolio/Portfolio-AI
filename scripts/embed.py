import os
import json
import pickle
import time
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi
from google import genai as _genai
from google.genai import types as _types
import pg8000

_kiwi = Kiwi()

def _tokenize_ko(text: str) -> list[str]:
    keep = {"NNG", "NNP", "NNB", "NR", "NP", "VV", "VA", "SL"}
    return [t.form for t in _kiwi.tokenize(text) if t.tag in keep]


# ── 설정 ──────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent.parent / "output"

CL_CHUNKS_PATH      = OUTPUT_DIR / "mock_cover_letters_ctx.json"
CL_BM25_PATH        = OUTPUT_DIR / "bm25_cover_letters.pkl"

PORTFOLIO_CHUNKS_PATH = OUTPUT_DIR / "mock_portfolios_ctx.json"
PORTFOLIO_BM25_PATH   = OUTPUT_DIR / "bm25_portfolios.pkl"

DB_CONFIG = {
    "host":     os.getenv("POSTGRES_HOST", "localhost"),
    "port":     int(os.getenv("POSTGRES_PORT", 5432)),
    "database": os.getenv("POSTGRES_DB", "passfolio_db"),
    "user":     os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
}


# ═══════════════════════════════════════════════════════════════
# 임베딩
# ═══════════════════════════════════════════════════════════════

_GEMINI_EMBED_MODEL  = "gemini-embedding-2"
_GEMINI_EMBED_DIM    = 1024


def _embed_gemini(chunks: list[dict], texts: list[str], log_interval: int = 50) -> list[dict]:
    """Gemini Embedding 2 임베딩 (포트폴리오 전용, 1024차원)."""
    client = _genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    total  = len(texts)
    print(f"임베딩 시작 [{_GEMINI_EMBED_MODEL}]: {total}개 청크")
    for i, (chunk, text) in enumerate(zip(chunks, texts)):
        for attempt in range(6):
            try:
                resp = client.models.embed_content(
                    model=_GEMINI_EMBED_MODEL,
                    contents=text,
                    config=_types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=_GEMINI_EMBED_DIM,
                    ),
                )
                chunk["embedding"] = resp.embeddings[0].values
                time.sleep(0.1)
                break
            except Exception as e:
                err = str(e)
                if attempt < 5 and ("429" in err or "quota" in err.lower()):
                    wait = 60 * (attempt + 1)
                    print(f"  [rate limit] {wait}초 대기 후 재시도...")
                    time.sleep(wait)
                else:
                    raise
        if (i + 1) % log_interval == 0 or (i + 1) == total:
            print(f"  {i + 1}/{total} 완료")
    print("임베딩 완료")
    return chunks


# ═══════════════════════════════════════════════════════════════
# BM25 인덱스
# ═══════════════════════════════════════════════════════════════

def _build_bm25_index(chunks: list[dict], index_path: Path) -> None:
    print(f"BM25 인덱스 빌드 중: {len(chunks)}개 청크")
    tokenized = [_tokenize_ko(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    index_data = {
        "bm25":         bm25,
        "ids":          [c["id"] for c in chunks],
        "sub_sections": [c.get("sub_section", "") for c in chunks],
    }
    with open(index_path, "wb") as f:
        pickle.dump(index_data, f)
    print(f"BM25 인덱스 저장 완료 → {index_path}")


# ═══════════════════════════════════════════════════════════════
# 이어쓰기 유틸
# ═══════════════════════════════════════════════════════════════

def _fetch_embedded_ids(table: str) -> set[str]:
    try:
        conn = pg8000.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(f"SELECT id FROM {table} WHERE embedding IS NOT NULL")
        ids = {row[0] for row in cur.fetchall()}
        cur.close()
        conn.close()
        return ids
    except Exception:
        return set()


# ═══════════════════════════════════════════════════════════════
# DB INSERT
# ═══════════════════════════════════════════════════════════════

def insert_cover_letter_to_db(chunks: list[dict]):
    conn = pg8000.connect(**DB_CONFIG)
    cur = conn.cursor()
    for c in chunks:
        cur.execute(
            """
            INSERT INTO cover_letter_chunks
                (id, source, doc_type, category, sub_section, keywords_str,
                 text, achievements, keywords, sub_categories, key_points,
                 context, char_count, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                embedding       = EXCLUDED.embedding,
                category        = EXCLUDED.category,
                keywords_str    = EXCLUDED.keywords_str,
                achievements    = EXCLUDED.achievements,
                keywords        = EXCLUDED.keywords,
                sub_categories  = EXCLUDED.sub_categories,
                key_points      = EXCLUDED.key_points,
                context         = EXCLUDED.context
            """,
            (
                c["id"], c["source"], c["doc_type"],
                c["category"], c["section"], c["keywords_str"],
                c["text"],
                json.dumps(c.get("achievements",   []), ensure_ascii=False),
                json.dumps(c.get("keywords",       []), ensure_ascii=False),
                json.dumps(c.get("sub_categories", []), ensure_ascii=False),
                json.dumps(c.get("key_points",     []), ensure_ascii=False),
                c.get("context", ""),
                c["char_count"],
                str(c["embedding"]),
            ),
        )
    conn.commit()
    cur.close()
    conn.close()
    print(f"cover_letter_chunks 저장 완료: {len(chunks)}개")


def insert_portfolio_to_db(chunks: list[dict]):
    conn = pg8000.connect(**DB_CONFIG)
    cur = conn.cursor()
    for c in chunks:
        meta    = c.get("meta", {})
        company = c.get("company", {})
        cur.execute(
            """
            INSERT INTO portfolio_chunks
                (id, source, doc_type, job, career,
                 company_name, company_type, company_domain,
                 section, sub_section, project,
                 period, role, team, tech_stack,
                 contributions, achievements, keywords,
                 text, context, text_with_context, char_count,
                 content_type, fig_id, image_path, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                embedding          = EXCLUDED.embedding,
                job                = EXCLUDED.job,
                career             = EXCLUDED.career,
                company_name       = EXCLUDED.company_name,
                company_type       = EXCLUDED.company_type,
                company_domain     = EXCLUDED.company_domain,
                sub_section        = EXCLUDED.sub_section,
                tech_stack         = EXCLUDED.tech_stack,
                contributions      = EXCLUDED.contributions,
                achievements       = EXCLUDED.achievements,
                keywords           = EXCLUDED.keywords,
                context            = EXCLUDED.context,
                text_with_context  = EXCLUDED.text_with_context,
                content_type       = EXCLUDED.content_type,
                fig_id             = EXCLUDED.fig_id,
                image_path         = EXCLUDED.image_path
            """,
            (
                c["id"], c["source"], c["doc_type"],
                c.get("job", ""),
                c.get("career", ""),
                company.get("name", ""),
                company.get("type", ""),
                company.get("domain", ""),
                c["section"], c.get("sub_section", ""),
                c.get("project", c["section"]),
                meta.get("period", ""),
                meta.get("role", ""),
                meta.get("team", ""),
                json.dumps(meta.get("tech_stack",    []), ensure_ascii=False),
                json.dumps(meta.get("contributions", []), ensure_ascii=False),
                json.dumps(meta.get("achievements",  []), ensure_ascii=False),
                json.dumps(meta.get("keywords",      []), ensure_ascii=False),
                c["text"],
                c.get("context", ""),
                c.get("text_with_context", c["text"]),
                c["char_count"],
                c.get("content_type", ""),
                c.get("fig_id", ""),
                c.get("image_path", ""),
                str(c["embedding"]),
            ),
        )
    conn.commit()
    cur.close()
    conn.close()
    print(f"portfolio_chunks 저장 완료: {len(chunks)}개")


# ═══════════════════════════════════════════════════════════════
# 실행
# ═══════════════════════════════════════════════════════════════

def run():
    # ── 자소서 ───────────────────────────────────────────────────
    if CL_CHUNKS_PATH.exists():
        with open(CL_CHUNKS_PATH, encoding="utf-8") as f:
            cl_chunks = json.load(f)
        print(f"\n{CL_CHUNKS_PATH.name} 로드: {len(cl_chunks)}개")

        done_ids = _fetch_embedded_ids("cover_letter_chunks")
        todo = [c for c in cl_chunks if c["id"] not in done_ids]
        print(f"  임베딩 대상: {len(todo)}개 ({len(done_ids)}개 이미 완료)")

        if todo:
            todo = _embed_gemini(todo, [c["text"] for c in todo])
            insert_cover_letter_to_db(todo)

        _build_bm25_index(cl_chunks, CL_BM25_PATH)
    else:
        print("⚠ 자소서 청크 파일 없음, 건너뜀")

    # ── 포트폴리오 ────────────────────────────────────────────────
    run_portfolio()


def run_portfolio():
    if not PORTFOLIO_CHUNKS_PATH.exists():
        print("⚠ 포트폴리오 청크 파일 없음, 포트폴리오 임베딩 건너뜀")
        return

    with open(PORTFOLIO_CHUNKS_PATH, encoding="utf-8") as f:
        portfolio_chunks = json.load(f)
    print(f"\n{PORTFOLIO_CHUNKS_PATH.name} 로드: {len(portfolio_chunks)}개")

    done_ids = _fetch_embedded_ids("portfolio_chunks")
    todo = [c for c in portfolio_chunks if c["id"] not in done_ids]
    print(f"  임베딩 대상: {len(todo)}개 ({len(done_ids)}개 이미 완료)")

    if todo:
        todo = _embed_gemini(todo, [c.get("text_with_context", c["text"]) for c in todo])
        insert_portfolio_to_db(todo)

    _build_bm25_index(portfolio_chunks, PORTFOLIO_BM25_PATH)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "portfolio":
        run_portfolio()
    else:
        run()
