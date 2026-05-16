import pytest
from app.core.config import get_settings, Settings


@pytest.fixture
def test_settings() -> Settings:
    real = get_settings()
    return Settings(
        postgres_user=real.postgres_user,
        postgres_password=real.postgres_password,
        postgres_db=real.postgres_db,
        postgres_host=real.postgres_host,
        postgres_port=real.postgres_port,
        openai_api_key="sk-test",
        gemini_api_key="gm-test",
    )
