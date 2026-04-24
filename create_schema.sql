-- 1. 벡터 데이터 타입을 위한 확장 설치
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. 지원자 마스터 테이블 (인적사항)
CREATE TABLE IF NOT EXISTS resumes (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE,
    contact TEXT,
    address TEXT,
    military_info JSONB,
    full_json JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. 벡터 및 상세 데이터 테이블 (경력, 자소서 등)
CREATE TABLE IF NOT EXISTS resume_sections (
    id SERIAL PRIMARY KEY,
    resume_id INTEGER REFERENCES resumes(id) ON DELETE CASCADE,
    category TEXT, -- education, experience, selfIntroduction 등
    sub_category TEXT, -- 팀프로젝트, 성격의장단점 등
    content TEXT NOT NULL,
    embedding vector(1024), -- Gemini Embedding 004 등을 고려한 1024차원
    metadata JSONB, -- 기간, 장소, 키워드 등
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. 검색 최적화를 위한 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_resume_sections_resume_id ON resume_sections(resume_id);
CREATE INDEX IF NOT EXISTS idx_resume_sections_embedding ON resume_sections USING hnsw (embedding vector_cosine_ops);
