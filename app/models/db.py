from sqlalchemy.orm import DeclarativeBase
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, Text

EMBEDDING_DIM = 3072


class Base(DeclarativeBase):
    pass
