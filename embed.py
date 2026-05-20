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
    """한국어 형태소 분석 기반 토크나이징. 명사/동사/형용사/영문만 추출."""
    keep = {"NNG", "NNP", "NNB", "NR", "NP", "VV", "VA", "SL"}
    return [t.form for t in _kiwi.tokenize(text) if t.tag in keep]

# ── 설정 ──────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent / "output"

# contextual retrieval 적용본 우선, 없으면 원본 사용
CL_CTX_PATH       = OUTPUT_DIR / "mock_cover_letters_ctx.json"
CL_CHUNKS_PATH    = OUTPUT_DIR / "mock_cover_letters.json"
CL_BM25_PATH      = OUTPUT_DIR / "bm25_cover_letters.pkl"

RESUME_CHUNKS_PATH        = OUTPUT_DIR / "resume_chunks.json"
PORTFOLIO_CHUNKS_PATH     = OUTPUT_DIR / "portfolio_chunks.json"
MOCK_PORTFOLIO_CTX_PATH   = OUTPUT_DIR / "mock_portfolios_ctx.json"
MOCK_PORTFOLIO_PATH       = OUTPUT_DIR / "mock_portfolios.json"
PORTFOLIO_BM25_PATH       = OUTPUT_DIR / "bm25_portfolios.pkl"

DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "database": "postgres",
    "user":     os.getenv("PG_USER", "parkjunwoo"),
    "password": os.getenv("PG_PASSWORD"),
}

# ═══════════════════════════════════════════════════════════════
# 임베딩
# ═══════════════════════════════════════════════════════════════

_EMBED_MODEL = "gemini-embedding-2"
_EMBED_DIM   = 1024


def _get_embed_client() -> _genai.Client:
    project = os.getenv("GCP_PROJECT_ID")
    if not project:
        raise ValueError("GCP_PROJECT_ID 환경변수를 설정하세요.")
    return _genai.Client(vertexai=True, project=project, location="global")


def _embed_gemini(chunks: list[dict], texts: list[str], log_interval: int = 50) -> list[dict]:
    """Gemini Embedding 2 임베딩 (Vertex AI, 1024차원).

    Vertex AI SDK는 contents에 리스트를 전달해도 임베딩을 1개만 반환하므로
    1건씩 순차 호출합니다.
    """
    client = _get_embed_client()
    total  = len(texts)
    print(f"임베딩 시작 [{_EMBED_MODEL}]: {total}개 청크")

    for i, (chunk, text) in enumerate(zip(chunks, texts)):
        for attempt in range(6):
            try:
                response = client.models.embed_content(
                    model=_EMBED_MODEL,
                    contents=text,
                    config=_types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=_EMBED_DIM,
                    ),
                )
                chunk["embedding"] = response.embeddings[0].values
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
    """BM25Okapi 인덱스를 빌드하고 pkl로 저장.

    검색 시 로드해서 쿼리 토큰과 매칭.
    corpus는 text_with_context(있을 때) 또는 text 기준으로 토크나이즈.
    """
    print(f"BM25 인덱스 빌드 중: {len(chunks)}개 청크")
    corpus_texts = [c.get("text_with_context", c["text"]) for c in chunks]
    tokenized    = [_tokenize_ko(text) for text in corpus_texts]

    bm25 = BM25Okapi(tokenized)
    index_data = {
        "bm25": bm25,
        "ids":  [c["id"] for c in chunks],
    }
    with open(index_path, "wb") as f:
        pickle.dump(index_data, f)
    print(f"BM25 인덱스 저장 완료 → {index_path}")


# ═══════════════════════════════════════════════════════════════
# 이어쓰기 유틸
# ═══════════════════════════════════════════════════════════════

def _fetch_embedded_ids(table: str) -> set[str]:
    """DB에서 이미 embedding이 있는 id 목록 조회."""
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


def insert_resume_to_db(chunks: list[dict]):
    conn = pg8000.connect(**DB_CONFIG)
    cur = conn.cursor()
    for c in chunks:
        cur.execute(
            """
            INSERT INTO resume_chunks
                (id, source, doc_type, section, sub_section,
                 facts, skills, period, char_count, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                facts     = EXCLUDED.facts,
                skills    = EXCLUDED.skills,
                period    = EXCLUDED.period
            """,
            (
                c["id"], c["source"], c["doc_type"],
                c["section"], c["sub_section"],
                json.dumps(c["facts"],  ensure_ascii=False),
                json.dumps(c["skills"], ensure_ascii=False),
                c["period"],
                c["char_count"],
                str(c["embedding"]),
            ),
        )
    conn.commit()
    cur.close()
    conn.close()
    print(f"resume_chunks 저장 완료: {len(chunks)}개")


# ═══════════════════════════════════════════════════════════════
# 실행
# ═══════════════════════════════════════════════════════════════

def run():
    # ── 자소서: contextual retrieval 적용본 우선 ──────────────────
    cl_path = CL_CTX_PATH if CL_CTX_PATH.exists() else CL_CHUNKS_PATH
    if cl_path.exists():
        with open(cl_path, encoding="utf-8") as f:
            cl_chunks = json.load(f)
        print(f"\n{cl_path.name} 로드: {len(cl_chunks)}개")
        if CL_CTX_PATH.exists():
            print("  → contextual retrieval 적용본 사용 (text_with_context로 임베딩)")

        # 이미 임베딩된 ID 건너뜀
        done_ids = _fetch_embedded_ids("cover_letter_chunks")
        todo = [c for c in cl_chunks if c["id"] not in done_ids]
        print(f"  임베딩 대상: {len(todo)}개 ({len(done_ids)}개 이미 완료)")

        if todo:
            embed_texts = [c.get("text_with_context", c["text"]) for c in todo]
            todo = _embed_gemini(todo, embed_texts)
            insert_cover_letter_to_db(todo)

        _build_bm25_index(cl_chunks, CL_BM25_PATH)
    else:
        print(f"⚠ 자소서 청크 파일 없음, 건너뜀")

    # ── mock 포트폴리오: contextual retrieval 적용본 우선 ─────────
    run_portfolio()


def run_portfolio():
    pf_path = MOCK_PORTFOLIO_CTX_PATH if MOCK_PORTFOLIO_CTX_PATH.exists() else MOCK_PORTFOLIO_PATH
    if not pf_path.exists():
        print(f"⚠ {MOCK_PORTFOLIO_PATH.name} 없음, 포트폴리오 임베딩 건너뜀")
        return

    with open(pf_path, encoding="utf-8") as f:
        portfolio_chunks = json.load(f)
    print(f"\n{pf_path.name} 로드: {len(portfolio_chunks)}개")
    if MOCK_PORTFOLIO_CTX_PATH.exists():
        print("  → contextual retrieval 적용본 사용 (text_with_context로 임베딩)")

    done_ids = _fetch_embedded_ids("portfolio_chunks")
    todo = [c for c in portfolio_chunks if c["id"] not in done_ids]
    print(f"  임베딩 대상: {len(todo)}개 ({len(done_ids)}개 이미 완료)")

    if todo:
        texts = [c.get("text_with_context", c["text"]) for c in todo]
        todo = _embed_gemini(todo, texts)
        insert_portfolio_to_db(todo)

    _build_bm25_index(portfolio_chunks, PORTFOLIO_BM25_PATH)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "portfolio":
        run_portfolio()
    else:
        run()
