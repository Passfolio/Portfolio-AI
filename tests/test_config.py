import pytest
from pydantic import ValidationError


def test_settings_default_values():
    from app.core.config import Settings

    s = Settings(
        postgres_user="u",
        postgres_password="p",
        postgres_db="db",
        openai_api_key="sk-test",
        gemini_api_key="gm-test",
    )
    assert s.postgres_host == "localhost"
    assert s.postgres_port == 5432
    assert s.app_env == "development"
    assert s.app_debug is True


def test_database_url_format():
    from app.core.config import Settings

    s = Settings(
        postgres_user="passfolio",
        postgres_password="secret",
        postgres_db="passfolio_db",
        postgres_host="localhost",
        postgres_port=5432,
        openai_api_key="sk-test",
        gemini_api_key="gm-test",
    )
    assert s.database_url == (
        "postgresql+asyncpg://passfolio:secret@localhost:5432/passfolio_db"
    )


def test_settings_requires_postgres_user():
    from app.core.config import Settings

    with pytest.raises(ValidationError):
        Settings(
            postgres_password="p",
            postgres_db="db",
            openai_api_key="sk-test",
            gemini_api_key="gm-test",
        )


def test_settings_requires_openai_key():
    from app.core.config import Settings

    with pytest.raises(ValidationError):
        Settings(
            postgres_user="u",
            postgres_password="p",
            postgres_db="db",
            gemini_api_key="gm-test",
        )
