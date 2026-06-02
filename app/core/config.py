from __future__ import annotations

import ssl
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    postgres_user: str
    postgres_password: str
    postgres_db: str = "postgres"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    gemini_api_key: str = ""
    openai_api_key: str = ""
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "ap-northeast-2"
    s3_bucket_name: str = ""
    be_base_url: str = "http://localhost:8080"
    passfolio_internal_api_key: str = ""

    @property
    def db_config(self) -> dict:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return {
            "host":        self.postgres_host,
            "port":        self.postgres_port,
            "database":    self.postgres_db,
            "user":        self.postgres_user,
            "password":    self.postgres_password,
            "timeout":     10,
            "ssl_context": ctx,
        }

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
