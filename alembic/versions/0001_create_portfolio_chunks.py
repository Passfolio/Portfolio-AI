"""create portfolio_chunks table

Revision ID: 0001
Revises:
Create Date: 2026-05-16
"""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "portfolio_chunks",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("pdf_id", sa.String(), nullable=False),
        sa.Column("section_title", sa.String(), nullable=False),
        sa.Column("content", sa.String(), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.Column("section_type", sa.String(), nullable=False),
        sa.Column("page", sa.Integer(), nullable=False),
        sa.Column("order", sa.Integer(), nullable=False),
        sa.Column("parent_title", sa.String(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_portfolio_chunks_pdf_id", "portfolio_chunks", ["pdf_id"])
    op.execute(
        "CREATE INDEX ON portfolio_chunks USING hnsw (embedding vector_cosine_ops)"
    )
    op.execute(
        "CREATE INDEX ON portfolio_chunks USING gin (to_tsvector('english', content))"
    )


def downgrade() -> None:
    op.drop_table("portfolio_chunks")
