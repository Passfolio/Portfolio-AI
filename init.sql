CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS cover_letter_chunks (
    id             TEXT PRIMARY KEY,
    source         TEXT,
    doc_type       TEXT,
    category       TEXT,
    sub_section    TEXT,
    keywords_str   TEXT,
    text           TEXT,
    achievements   JSONB,
    keywords       JSONB,
    sub_categories JSONB,
    key_points     JSONB,
    context        TEXT,
    char_count     INT,
    embedding      vector(1024)
);

CREATE TABLE IF NOT EXISTS portfolio_chunks (
    id                TEXT PRIMARY KEY,
    source            TEXT,
    doc_type          TEXT,
    job               TEXT,
    career            TEXT,
    company_name      TEXT,
    company_type      TEXT,
    company_domain    TEXT,
    section           TEXT,
    sub_section       TEXT,
    project           TEXT,
    period            TEXT,
    role              TEXT,
    team              TEXT,
    tech_stack        JSONB,
    contributions     JSONB,
    achievements      JSONB,
    keywords          JSONB,
    text              TEXT,
    context           TEXT,
    text_with_context TEXT,
    char_count        INT,
    content_type      TEXT,
    fig_id            TEXT,
    image_path        TEXT,
    embedding         vector(1024)
);

CREATE TABLE IF NOT EXISTS resume_chunks (
    id          TEXT PRIMARY KEY,
    source      TEXT,
    doc_type    TEXT,
    section     TEXT,
    sub_section TEXT,
    facts       JSONB,
    skills      JSONB,
    period      TEXT,
    char_count  INT,
    embedding   vector(1024)
);

CREATE INDEX IF NOT EXISTS idx_cl_embedding  ON cover_letter_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_pf_embedding  ON portfolio_chunks    USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_res_embedding ON resume_chunks       USING hnsw (embedding vector_cosine_ops);
