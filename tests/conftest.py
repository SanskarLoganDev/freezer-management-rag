"""All tests bypass dotenv loading; accidentally opening it is a test failure."""

import pytest
from pydantic_settings.sources import DotEnvSettingsSource


@pytest.fixture(autouse=True)
def forbid_dotenv(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Tests must not read any .env file; use Settings(_env_file=None)")

    monkeypatch.setattr(DotEnvSettingsSource, "_read_env_file", forbidden)
