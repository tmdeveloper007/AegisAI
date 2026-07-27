"""
Unit tests for Settings validators in config.py.

Uses environment variable patching to control which validators fire, since
Settings loads from .env at import time.  Each test sets the relevant env
vars, triggers Settings construction, and verifies the expected outcome.
"""

import os
import pytest


class TestValidateRedisUrl:
    """Tests for the validate_redis_url field validator."""

    def test_redis_url_empty_raises_in_production(self, monkeypatch):
        """When DEBUG=False and REDIS_URL is empty, ValueError is raised."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/test")
        monkeypatch.setenv("SECRET_KEY", "a" * 32)
        monkeypatch.setenv("DEBUG", "false")
        monkeypatch.delenv("REDIS_URL", raising=False)

        # Force reimport of the module so Settings picks up the env vars.
        import importlib
        import app.core.config as config_module
        importlib.reload(config_module)

        from app.core.config import Settings

        with pytest.raises(ValueError, match="REDIS_URL is required in production"):
            Settings()

    def test_redis_url_empty_allowed_in_debug(self, monkeypatch):
        """When DEBUG=True and REDIS_URL is empty, no error is raised."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/test")
        monkeypatch.setenv("SECRET_KEY", "a" * 32)
        monkeypatch.setenv("DEBUG", "true")
        monkeypatch.delenv("REDIS_URL", raising=False)

        import importlib
        import app.core.config as config_module
        importlib.reload(config_module)

        from app.core.config import Settings

        # Should not raise — DEBUG=True skips the REDIS_URL requirement.
        settings = Settings()
        assert settings.REDIS_URL == ""

    def test_redis_url_set_passes_in_production(self, monkeypatch):
        """When REDIS_URL is non-empty, the validator passes even in production."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/test")
        monkeypatch.setenv("SECRET_KEY", "a" * 32)
        monkeypatch.setenv("DEBUG", "false")
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")

        import importlib
        import app.core.config as config_module
        importlib.reload(config_module)

        from app.core.config import Settings

        settings = Settings()
        assert settings.REDIS_URL == "redis://localhost:6379"


class TestValidateSecretKey:
    """Tests for the validate_secret_key field validator."""

    def _make_settings(self, monkeypatch, secret_key: str, debug: str = "false"):
        """Construct a Settings instance with the given secret key and debug flag."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/test")
        monkeypatch.setenv("SECRET_KEY", secret_key)
        monkeypatch.setenv("DEBUG", debug)
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")

        import importlib
        import app.core.config as config_module
        importlib.reload(config_module)

        from app.core.config import Settings
        return Settings()

    @pytest.mark.parametrize("weak_secret", [
        "secret",
        "changeme",
        "password",
        "change_me",
        "insecure",
        "test",
        "your-secret-key",
        "your-secret-key-here",
    ])
    def test_weak_known_secrets_rejected(self, monkeypatch, weak_secret):
        """Well-known weak secrets are rejected regardless of length."""
        with pytest.raises(ValueError, match="insecure"):
            self._make_settings(monkeypatch, weak_secret)

    def test_short_secret_rejected_in_production(self, monkeypatch):
        """Secrets shorter than 32 characters are rejected when DEBUG=False."""
        short_secret = "a" * 31  # 31 chars, below 32-char minimum
        with pytest.raises(ValueError, match="at least 32 characters"):
            self._make_settings(monkeypatch, short_secret, debug="false")

    def test_short_secret_allowed_in_debug(self, monkeypatch):
        """Secrets shorter than 32 characters are allowed when DEBUG=True."""
        short_secret = "a" * 8
        settings = self._make_settings(monkeypatch, short_secret, debug="true")
        assert settings.SECRET_KEY == short_secret

    def test_valid_secret_passes(self, monkeypatch):
        """A non-weak secret of 32+ characters passes in production."""
        valid_secret = "a" * 32
        settings = self._make_settings(monkeypatch, valid_secret, debug="false")
        assert settings.SECRET_KEY == valid_secret
