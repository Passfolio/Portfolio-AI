import pytest

from app.core.config import Settings, get_settings
from app.domain.pdf.loader import load_mock_json
from app.domain.pdf.schemas import PdfSection

_MOCK_JSON_PATH = "tests/assets/mock_data.json"


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


@pytest.fixture(scope="session")
async def mock_sections() -> list[PdfSection]:
    return await load_mock_json(_MOCK_JSON_PATH)
