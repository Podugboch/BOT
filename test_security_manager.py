"""Regression tests for secret-manager security guarantees."""
import importlib
from pathlib import Path

import pytest
from cryptography.fernet import Fernet


@pytest.fixture
def secret_manager(monkeypatch):
    """Reload with an explicit test key and no deployment fallback variables."""
    for name in (
        "VERCEL",
        "AWS_LAMBDA_FUNCTION_NAME",
        "ENVIRONMENT",
        "ENV",
        "BOT_ENV",
        "BOT_SECRET_KEY",
        "SECRET_KEY",
        "APP_SECRET_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv("BOT_SECRET_KEY", Fernet.generate_key().decode())
    import secret_manager

    module = importlib.reload(secret_manager)
    module.reset_cipher()
    return module


def test_encrypt_decrypt_round_trip(secret_manager):
    plaintext = "exchange-secret-value"
    encrypted = secret_manager.encrypt_secret(plaintext)

    assert encrypted.startswith(secret_manager.ENC_PREFIX)
    assert encrypted != plaintext
    assert secret_manager.decrypt_secret(encrypted) == plaintext
    assert secret_manager.encrypt_secret(encrypted) == encrypted


def test_serverless_without_external_key_fails_closed(monkeypatch):
    for name in ("BOT_SECRET_KEY", "SECRET_KEY", "APP_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME", raising=False)

    import secret_manager

    module = importlib.reload(secret_manager)
    module.reset_cipher()
    with pytest.raises(RuntimeError, match="No master encryption key configured"):
        module.get_master_key()


def test_sensitive_dotenv_values_are_not_loaded(secret_manager, monkeypatch, tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "EXCHANGE_API_SECRET=must-not-enter-process\n"
        "BOT_SECRET_KEY=must-not-enter-process\n"
        "PUBLIC_SETTING=allowed\n"
    )

    monkeypatch.delenv("EXCHANGE_API_SECRET", raising=False)
    monkeypatch.delenv("BOT_SECRET_KEY", raising=False)
    monkeypatch.delenv("PUBLIC_SETTING", raising=False)

    secret_manager.safe_load_dotenv(env_file)

    assert "EXCHANGE_API_SECRET" not in __import__("os").environ
    assert "BOT_SECRET_KEY" not in __import__("os").environ
    assert __import__("os").environ["PUBLIC_SETTING"] == "allowed"


def test_sensitive_values_already_in_environment_are_removed(secret_manager, monkeypatch, tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("PUBLIC_SETTING=allowed\n")
    monkeypatch.setenv("EXCHANGE_API_KEY", "runtime-secret")
    monkeypatch.setenv("JWT_SECRET", "runtime-jwt-secret")

    secret_manager.safe_load_dotenv(env_file)

    assert "EXCHANGE_API_KEY" not in __import__("os").environ
    assert "JWT_SECRET" not in __import__("os").environ


def test_sensitive_key_names_are_blocked(secret_manager):
    expected = {
        "BOT_SECRET_KEY",
        "SECRET_KEY",
        "APP_SECRET_KEY",
        "BOT_AUTH_TOKEN",
        "JWT_SECRET",
        "COOKIE_SECRET",
        "EXCHANGE_API_SECRET",
    }
    assert expected.issubset(secret_manager.SENSITIVE_ENV_KEYS)


def test_local_key_file_is_owner_only(monkeypatch, tmp_path: Path):
    for name in ("VERCEL", "AWS_LAMBDA_FUNCTION_NAME", "ENVIRONMENT", "ENV", "BOT_ENV"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("BOT_SECRET_KEY", raising=False)
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.delenv("APP_SECRET_KEY", raising=False)
    key_path = tmp_path / ".secret_key"
    monkeypatch.setenv("BOT_SECRET_KEY_PATH", str(key_path))

    import secret_manager

    module = importlib.reload(secret_manager)
    module.reset_cipher()
    key = module.get_master_key()

    assert Fernet(key)
    assert key_path.exists()
    assert key_path.stat().st_mode & 0o777 == 0o600
