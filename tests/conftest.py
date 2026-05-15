import pytest
from app.core.config import Settings


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        postgres_user="passfolio",
        postgres_password="passfolio",
        postgres_db="passfolio_db",
        postgres_host="localhost",
        postgres_port=5432,
        openai_api_key="sk-test",
        gemini_api_key="gm-test",
    )
