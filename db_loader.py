import os
import re
import json
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from FlagEmbedding import FlagModel
import torch

class ResumeDataLoader:
    def __init__(self, db_config):
        self.db_config = db_config
        # 1. BGE-M3 모델 초기화 (Mac GPU 가속 사용)
        print("🚀 BGE-M3 모델 로드 중 (MPS 가속 적용)...")
        # Apple Silicon GPU 가속을 위해 device='mps' 설정
        self.model = FlagModel(
            'BAAI/bge-m3', 
            device='mps' if torch.backends.mps.is_available() else 'cpu',
            use_fp16=True 
        )
        self.conn = None

    def connect_db(self):
        self.conn = psycopg2.connect(**self.db_config)
        # 1. 먼저 테이블 및 익스텐션을 생성 (중요!)
        self.create_table()
        # 2. 익스텐션이 생성된 후 벡터 타입을 등록
        register_vector(self.conn)

    def create_table(self):
        with self.conn.cursor() as cur:
            # 익스텐션을 먼저 설치하고 즉시 커밋될 수 있도록 함
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_data (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector(1024),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
        self.conn.commit()

    def mask_sensitive_info(self, data):
        """보안 주의: 주민등록번호(RRN) 마스킹"""
        # RRN 마스킹 (예: 000209-3******)
        if 'personalInfo' in data and data['personalInfo'].get('RRN'):
            rrn = data['personalInfo']['RRN']
            if '-' in rrn:
                parts = rrn.split('-')
                # 뒷자리 첫 번째 숫자만 남기고 나머지 마스킹
                data['personalInfo']['RRN'] = f"{parts[0]}-{parts[1][0]}{'*' * (len(parts[1])-1)}"
        
        # 연락처 마스킹 (010-****-1234)
        if 'personalInfo' in data and data['personalInfo'].get('contact'):
            contact = data['personalInfo']['contact']
            data['personalInfo']['contact'] = re.sub(r'(\d{3})-(\d{3,4})-(\d{4})', r'\1-****-\3', contact)
            
        return data

    def prepare_data(self, json_data):
        """임베딩 타겟 텍스트 추출 및 메타데이터 준비"""
        # 마스킹 적용
        masked_data = self.mask_sensitive_info(json_data)
        
        # 공통 메타데이터
        common_meta = {
            "personalInfo": masked_data.get("personalInfo"),
            "education": masked_data.get("education"),
            "military": masked_data.get("military"),
            "certificates": masked_data.get("certificates")
        }

        chunks = []
        
        # 1. Experience 임베딩 (category + content)
        for exp in masked_data.get("experience", []):
            content = f"[{exp.get('category', '경력')}] {exp.get('content', '')}"
            meta = {**common_meta, "type": "experience", "detail": exp}
            chunks.append({"content": content, "metadata": meta})

        # 2. SelfIntroduction 임베딩 (category + key_points + achievements)
        for intro in masked_data.get("selfIntroduction", []):
            points = " ".join(intro.get("key_points", []))
            achievements = " ".join(intro.get("achievements", []))
            full_text = f"[{intro.get('category', '자기소개서')}] {points} {achievements}"
            meta = {**common_meta, "type": "selfIntroduction", "detail": intro}
            chunks.append({"content": full_text, "metadata": meta})

        return chunks

    def insert_data(self, chunks):
        """Batch Insert 처리"""
        if not chunks: 
            print("⚠ 적재할 데이터가 없습니다.")
            return

        print(f"🧠 {len(chunks)}개 데이터 임베딩 생성 중 (BGE-M3)...")
        texts = [c['content'] for c in chunks]
        embeddings = self.model.encode(texts)

        data_to_insert = [
            (chunks[i]['content'], embeddings[i].tolist(), json.dumps(chunks[i]['metadata'], ensure_ascii=False))
            for i in range(len(chunks))
        ]

        print(f"🐘 PostgreSQL 적재 시작 (Table: portfolio_data)...")
        with self.conn.cursor() as cur:
            query = "INSERT INTO portfolio_data (content, embedding, metadata) VALUES %s"
            execute_values(cur, query, data_to_insert)
        self.conn.commit()
        print("✅ DB 적재 완료!")

    def close(self):
        if self.conn:
            self.conn.close()
