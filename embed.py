import os
import json
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from sentence_transformers import SentenceTransformer
import pg8000

# ── 설정 ──────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent / "output"
CL_CHUNKS_PATH        = OUTPUT_DIR / "coverletter_chunks.json"
RESUME_CHUNKS_PATH    = OUTPUT_DIR / "resume_chunks.json"
PORTFOLIO_CHUNKS_PATH = OUTPUT_DIR / "portfolio_chunks.json"

DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "database": "postgres",
    "user":     "postgres",
    "password": os.getenv("PG_PASSWORD"),
}

# ── 모델 로드 (최초 실행 시 약 2.2GB 다운로드) ─────────────────────────
print("bge-m3 모델 로드 중...")
model = SentenceTransformer("BAAI/bge-m3")
print("모델 로드 완료")


def _embed(chunks: list[dict], texts: list[str]) -> list[dict]:
    print(f"임베딩 시작: {len(texts)}개 청크")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    print("임베딩 완료")
    return chunks


def insert_cover_letter_to_db(chunks: list[dict]):
    conn = pg8000.connect(**DB_CONFIG)
    cur = conn.cursor()
    for c in chunks:
        cur.execute(
            """
            INSERT INTO cover_letter_chunks
                (id, source, doc_type, category, sub_section, keywords_str,
                 text, achievements, keywords, char_count, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                embedding    = EXCLUDED.embedding,
                category     = EXCLUDED.category,
                keywords_str = EXCLUDED.keywords_str,
                achievements = EXCLUDED.achievements,
                keywords     = EXCLUDED.keywords
            """,
            (
                c["id"], c["source"], c["doc_type"],
                c["category"], c["section"], c["keywords_str"],
                c["text"],
                json.dumps(c["achievements"], ensure_ascii=False),
                json.dumps(c["keywords"], ensure_ascii=False),
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
        meta = c.get("meta", {})
        cur.execute(
            """
            INSERT INTO portfolio_chunks
                (id, source, doc_type, section, project,
                 period, role, team, tech_stack,
                 contributions, achievements, keywords,
                 text, char_count, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                embedding     = EXCLUDED.embedding,
                tech_stack    = EXCLUDED.tech_stack,
                contributions = EXCLUDED.contributions,
                achievements  = EXCLUDED.achievements,
                keywords      = EXCLUDED.keywords
            """,
            (
                c["id"], c["source"], c["doc_type"],
                c["section"], c.get("project", c["section"]),
                meta.get("period", ""),
                meta.get("role", ""),
                meta.get("team", ""),
                json.dumps(meta.get("tech_stack", []),    ensure_ascii=False),
                json.dumps(meta.get("contributions", []), ensure_ascii=False),
                json.dumps(meta.get("achievements", []),  ensure_ascii=False),
                json.dumps(meta.get("keywords", []),      ensure_ascii=False),
                c["text"],
                c["char_count"],
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
                json.dumps(c["facts"], ensure_ascii=False),
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


def run():
    if CL_CHUNKS_PATH.exists():
        with open(CL_CHUNKS_PATH, encoding="utf-8") as f:
            cl_chunks = json.load(f)
        print(f"\ncoverletter_chunks.json 로드: {len(cl_chunks)}개")
        cl_chunks = _embed(cl_chunks, [c["text"] for c in cl_chunks])
        insert_cover_letter_to_db(cl_chunks)
    else:
        print(f"⚠ {CL_CHUNKS_PATH.name} 없음, 자소서 임베딩 건너뜀")

    if PORTFOLIO_CHUNKS_PATH.exists():
        with open(PORTFOLIO_CHUNKS_PATH, encoding="utf-8") as f:
            portfolio_chunks = json.load(f)
        print(f"\nportfolio_chunks.json 로드: {len(portfolio_chunks)}개")
        texts = [
            c["section"] + "\n" + c.get("project", "") + "\n" + c["text"]
            for c in portfolio_chunks
        ]
        portfolio_chunks = _embed(portfolio_chunks, texts)
        insert_portfolio_to_db(portfolio_chunks)
    else:
        print(f"⚠ {PORTFOLIO_CHUNKS_PATH.name} 없음, 포트폴리오 임베딩 건너뜀")

    if RESUME_CHUNKS_PATH.exists():
        with open(RESUME_CHUNKS_PATH, encoding="utf-8") as f:
            resume_chunks = json.load(f)
        print(f"\nresume_chunks.json 로드: {len(resume_chunks)}개")
        texts = [
            c["section"] + "\n" + "\n".join(c["facts"]) + "\n" + " ".join(c["skills"])
            for c in resume_chunks
        ]
        resume_chunks = _embed(resume_chunks, texts)
        insert_resume_to_db(resume_chunks)
    else:
        print(f"⚠ {RESUME_CHUNKS_PATH.name} 없음, 이력서 임베딩 건너뜀")


if __name__ == "__main__":
    run()
