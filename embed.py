import os
import json
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from sentence_transformers import SentenceTransformer
import pg8000

# ── 설정 ──────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent / "output"
CHUNKS_PATH = OUTPUT_DIR / "chunks.json"

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


def embed_chunks(chunks: list[dict]) -> list[dict]:
    texts = [c["text"] for c in chunks]
    print(f"\n임베딩 시작: {len(texts)}개 청크")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    print("임베딩 완료")
    return chunks


def insert_to_db(chunks: list[dict]):
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
                c["id"],
                c["source"],
                c["doc_type"],
                c["category"],
                c["sub_section"],
                c["keywords_str"],
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
    print(f"DB 저장 완료: {len(chunks)}개 청크")


def run():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"chunks.json 로드: {len(chunks)}개")

    chunks = embed_chunks(chunks)
    insert_to_db(chunks)


if __name__ == "__main__":
    run()
