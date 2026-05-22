from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    postgres_user: str
    postgres_password: str
    postgres_db: str = "postgres"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    gemini_api_key: str = ""

    @property
    def db_config(self) -> dict:
        return {
            "host":     self.postgres_host,
            "port":     self.postgres_port,
            "database": self.postgres_db,
            "user":     self.postgres_user,
            "password": self.postgres_password,
            "timeout":  10,
        }

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
