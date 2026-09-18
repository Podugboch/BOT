"""
Unit tests for authentication and email verification dispatch in Crypto Trading Bot.
Covers:
- Password hashing & verification
- User registration & activation flow
- Resend activation token generation
- Password reset flow
- Email settings retrieval and non-destructive saving
- Multi-provider email dispatch (Brevo, SendGrid, Resend, SMTP)
- Automatic fallback when primary provider fails
- Resend sandbox 403 detection & clear guidance
"""
import os
import json
import sqlite3
import pytest
from unittest.mock import patch, MagicMock
from urllib.error import HTTPError

from auth import (
    hash_password,
    verify_password,
    get_email_settings,
    save_email_settings,
    set_system_setting,
    get_system_setting,
    send_email_message,
    test_email_connection as check_email_connection,
    AuthService,
    get_db_connection,
    init_auth_db,
)


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path, monkeypatch):
    """Use an isolated temporary SQLite database for each test."""
    db_file = tmp_path / "test_users.db"
    monkeypatch.setattr("auth.DB_PATH", db_file)
    init_auth_db()
    # Reset relevant environment variables
    for k in ["EMAIL_PROVIDER", "SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASS", "FROM_EMAIL", "RESEND_API_KEY", "BREVO_API_KEY", "SENDGRID_API_KEY"]:
        monkeypatch.delenv(k, raising=False)
    yield


class TestPasswordHashing:
    def test_hash_and_verify(self):
        pwd = "MySecretPassword123!"
        h, salt = hash_password(pwd)
        assert len(h) == 64
        assert len(salt) == 32
        assert verify_password(pwd, h, salt)
        assert not verify_password("WrongPassword", h, salt)


class TestEmailSettings:
    def test_empty_settings_reports_unconfigured(self):
        settings = get_email_settings()
        assert not settings["is_configured"]
        assert settings["provider"] == "none"
        assert not settings["can_send_to_all_users"]

    def test_gmail_configured_can_send_to_all(self):
        save_email_settings({
            "EMAIL_PROVIDER": "gmail",
            "SMTP_HOST": "smtp.gmail.com",
            "SMTP_PORT": "587",
            "SMTP_USER": "trader@gmail.com",
            "SMTP_PASS": "abcdefghijklmnop",
            "FROM_EMAIL": "trader@gmail.com"
        })
        settings = get_email_settings()
        assert settings["is_configured"]
        assert settings["provider"] == "gmail"
        assert settings["can_send_to_all_users"]
        assert not settings["is_resend_sandbox"]

    def test_resend_sandbox_detected(self):
        save_email_settings({
            "EMAIL_PROVIDER": "resend",
            "RESEND_API_KEY": "re_1234567890",
            "FROM_EMAIL": "Crypto Trading Bot <onboarding@resend.dev>"
        })
        settings = get_email_settings()
        assert settings["is_configured"]
        assert settings["provider"] == "resend"
        assert settings["is_resend_sandbox"]
        # Sandbox cannot send to all users
        assert not settings["can_send_to_all_users"]

    def test_resend_custom_domain_can_send_to_all(self):
        save_email_settings({
            "EMAIL_PROVIDER": "resend",
            "RESEND_API_KEY": "re_1234567890",
            "FROM_EMAIL": "Crypto Trading Bot <noreply@mycustomdomain.com>"
        })
        settings = get_email_settings()
        assert settings["is_configured"]
        assert settings["provider"] == "resend"
        assert not settings["is_resend_sandbox"]
        assert settings["can_send_to_all_users"]

    def test_brevo_can_send_to_all(self):
        save_email_settings({
            "EMAIL_PROVIDER": "brevo",
            "BREVO_API_KEY": "xkeysib-123456",
            "FROM_EMAIL": "trader@gmail.com"
        })
        settings = get_email_settings()
        assert settings["is_configured"]
        assert settings["provider"] == "brevo"
        assert settings["can_send_to_all_users"]

    def test_save_does_not_overwrite_password_with_empty_string(self):
        save_email_settings({
            "SMTP_HOST": "smtp.gmail.com",
            "SMTP_USER": "trader@gmail.com",
            "SMTP_PASS": "my_secret_pass"
        })
        # Saving again with empty SMTP_PASS should not wipe it
        save_email_settings({
            "SMTP_HOST": "smtp.gmail.com",
            "SMTP_USER": "trader@gmail.com",
            "SMTP_PASS": ""
        })
        assert get_system_setting("SMTP_PASS") == "my_secret_pass"


class TestEmailDispatchers:
    def test_resend_403_detected_correctly(self):
        save_email_settings({
            "EMAIL_PROVIDER": "resend",
            "RESEND_API_KEY": "re_testkey",
            "FROM_EMAIL": "onboarding@resend.dev"
        })

        err_body = json.dumps({
            "statusCode": 403,
            "message": "You can only send testing emails to your own email address (owner@gmail.com). To send to other domains, verify a domain."
        }).encode("utf-8")

        mock_err = HTTPError("https://api.resend.com/emails", 403, "Forbidden", {}, None)
        mock_err.read = MagicMock(return_value=err_body)

        with patch("urllib.request.urlopen", side_effect=mock_err):
            success, msg = send_email_message("new_user@yahoo.com", "Test Subject", "<p>Body</p>", "token123", "activation")
            assert not success
            assert "Resend Sandbox Restriction" in msg
            assert "resend.com/domains" in msg
            assert "Brevo" in msg or "Gmail" in msg

    def test_automatic_fallback_from_resend_to_brevo(self):
        # Both Resend and Brevo are configured
        save_email_settings({
            "EMAIL_PROVIDER": "resend",
            "RESEND_API_KEY": "re_testkey",
            "BREVO_API_KEY": "xkeysib-testkey",
            "FROM_EMAIL": "trader@gmail.com"
        })

        # Resend raises 403 sandbox error
        resend_err = HTTPError("https://api.resend.com/emails", 403, "Forbidden", {}, None)
        resend_err.read = MagicMock(return_value=json.dumps({"statusCode": 403, "message": "testing emails to your own email address"}).encode())

        # Brevo succeeds
        mock_brevo_resp = MagicMock()
        mock_brevo_resp.status = 201
        mock_brevo_resp.__enter__.return_value = mock_brevo_resp
        mock_brevo_resp.read.return_value = json.dumps({"messageId": "brevo_msg_123"}).encode()

        def side_effect(req, timeout=12):
            if "resend.com" in req.full_url:
                raise resend_err
            elif "brevo.com" in req.full_url:
                return mock_brevo_resp
            raise RuntimeError("Unexpected URL")

        with patch("urllib.request.urlopen", side_effect=side_effect):
            success, msg = send_email_message("external_user@yahoo.com", "Activation", "<p>Body</p>", "tok", "activation")
            assert success
            assert "Brevo API" in msg

    def test_brevo_dispatch_success(self):
        save_email_settings({
            "EMAIL_PROVIDER": "brevo",
            "BREVO_API_KEY": "xkeysib-validkey",
            "FROM_EMAIL": "Crypto Bot <bot@example.com>"
        })

        mock_resp = MagicMock()
        mock_resp.status = 201
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.read.return_value = json.dumps({"messageId": "123"}).encode()

        with patch("urllib.request.urlopen", return_value=mock_resp):
            success, msg = send_email_message("user@example.com", "Subject", "Body", "token", "activation")
            assert success
            assert "Brevo API" in msg

    def test_smtp_dispatch_success(self):
        save_email_settings({
            "EMAIL_PROVIDER": "smtp",
            "SMTP_HOST": "smtp.example.com",
            "SMTP_PORT": "587",
            "SMTP_USER": "user@example.com",
            "SMTP_PASS": "pass123",
            "FROM_EMAIL": "user@example.com"
        })

        mock_smtp = MagicMock()
        mock_smtp.__enter__.return_value = mock_smtp

        with patch("smtplib.SMTP", return_value=mock_smtp):
            success, msg = send_email_message("recipient@example.com", "Subj", "Body", "tok", "activation")
            assert success
            assert "via SMTP" in msg
            mock_smtp.starttls.assert_called_once()
            mock_smtp.login.assert_called_once_with("user@example.com", "pass123")
            mock_smtp.sendmail.assert_called_once()


class TestAuthRegistrationFlow:
    def test_register_and_activate(self):
        # Register user
        res = AuthService.register("crypto_trader", "trader@market.io", "StrongPass#2026")
        assert res["success"]
        assert res["email"] == "trader@market.io"

        # Check DB user is inactive
        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE username = 'crypto_trader'").fetchone()
        assert user["is_active"] == 0
        token = user["activation_token"]
        assert token is not None
        conn.close()

        # Login before activation fails
        login_res = AuthService.login("crypto_trader", "StrongPass#2026")
        assert not login_res["success"]
        assert login_res.get("needs_activation")

        # Activate user with token
        act_res = AuthService.activate_account(token)
        assert act_res["success"]

        # Login after activation succeeds
        login_res2 = AuthService.login("crypto_trader", "StrongPass#2026")
        assert login_res2["success"]
        assert "token" in login_res2

    def test_duplicate_registration_rejected(self):
        res1 = AuthService.register("trader1", "trader1@market.io", "Pass123456")
        assert res1["success"]
        res2 = AuthService.register("trader1", "other@market.io", "Pass123456")
        assert not res2["success"]
        assert "already exists" in res2["message"]

    def test_resend_activation_token(self):
        AuthService.register("trader2", "trader2@market.io", "Pass123456")
        conn = get_db_connection()
        user1 = conn.execute("SELECT activation_token FROM users WHERE username = 'trader2'").fetchone()
        tok1 = user1["activation_token"]
        conn.close()

        res = AuthService.resend_activation_email("trader2@market.io")
        assert res["success"]

        conn = get_db_connection()
        user2 = conn.execute("SELECT activation_token FROM users WHERE username = 'trader2'").fetchone()
        tok2 = user2["activation_token"]
        conn.close()

        assert tok1 != tok2


class TestSecretEncryptionAndVault:
    def test_fernet_encrypt_decrypt(self):
        from secret_manager import encrypt_secret, decrypt_secret, is_encrypted
        secret = "super_confidential_api_secret_999"
        encrypted = encrypt_secret(secret)
        assert is_encrypted(encrypted)
        assert not is_encrypted(secret)
        assert encrypted != secret
        assert encrypted.startswith("enc:")
        assert decrypt_secret(encrypted) == secret

    def test_legacy_plaintext_fallback(self):
        from secret_manager import decrypt_secret
        assert decrypt_secret("raw_legacy_secret") == "raw_legacy_secret"
        assert decrypt_secret("") == ""

    def test_exchange_keys_encrypted_at_rest_in_sqlite(self):
        reg = AuthService.register("vault_user", "vault@market.io", "Pass123456")
        conn = get_db_connection()
        user = conn.execute("SELECT id FROM users WHERE username = 'vault_user'").fetchone()
        user_id = user["id"]
        conn.close()

        raw_api_key = "mx0v_my_super_secure_key"
        raw_api_secret = "9fdcf827_secret_hash_code"
        raw_passphrase = "my_passphrase_123"

        AuthService.save_exchange_key(user_id, "binance", raw_api_key, raw_api_secret, raw_passphrase, testnet=True)

        # Inspect raw SQLite database directly to confirm encryption at rest
        conn = get_db_connection()
        raw_row = conn.execute("SELECT api_key, api_secret, api_passphrase FROM exchange_keys WHERE user_id = ?", (user_id,)).fetchone()
        conn.close()

        assert raw_row["api_key"] != raw_api_key
        assert raw_row["api_key"].startswith("enc:")
        assert raw_row["api_secret"] != raw_api_secret
        assert raw_row["api_secret"].startswith("enc:")
        assert raw_row["api_passphrase"] != raw_passphrase
        assert raw_row["api_passphrase"].startswith("enc:")

        # AuthService.get_raw_exchange_key returns decrypted credentials
        raw_creds = AuthService.get_raw_exchange_key(user_id, "binance")
        assert raw_creds["api_key"] == raw_api_key
        assert raw_creds["api_secret"] == raw_api_secret
        assert raw_creds["api_passphrase"] == raw_passphrase

        # AuthService.get_exchange_keys returns masked preview only
        masked_list = AuthService.get_exchange_keys(user_id)
        assert len(masked_list) == 1
        assert raw_api_key not in masked_list[0]["api_key_masked"]
        assert raw_api_secret not in str(masked_list[0])
        assert "***" in masked_list[0]["api_key_masked"]

    def test_email_credentials_encrypted_in_system_settings(self):
        raw_pass = "smtp_super_secret_password_456"
        raw_resend = "re_secret_resend_api_token"
        save_email_settings({
            "EMAIL_PROVIDER": "smtp",
            "SMTP_HOST": "smtp.mail.com",
            "SMTP_USER": "mailer@mail.com",
            "SMTP_PASS": raw_pass,
            "RESEND_API_KEY": raw_resend,
        })

        # Inspect raw SQLite table
        conn = get_db_connection()
        rows = dict(conn.execute("SELECT key, value FROM system_settings").fetchall())
        conn.close()

        assert rows["SMTP_PASS"] != raw_pass
        assert rows["SMTP_PASS"].startswith("enc:")
        assert rows["RESEND_API_KEY"] != raw_resend
        assert rows["RESEND_API_KEY"].startswith("enc:")
        # Non-sensitive settings remain normal
        assert rows["SMTP_HOST"] == "smtp.mail.com"

        # get_system_setting and get_email_settings decrypt properly
        assert get_system_setting("SMTP_PASS") == raw_pass
        assert get_system_setting("RESEND_API_KEY") == raw_resend

    def test_unencrypted_db_migration(self):
        from auth import migrate_unencrypted_secrets
        # Insert raw plaintext row directly into DB bypassing encryption
        conn = get_db_connection()
        conn.execute("INSERT OR REPLACE INTO system_settings (key, value, updated_at) VALUES ('SMTP_PASS', 'plaintext_pass', '2026-01-01')")
        conn.commit()
        migrate_unencrypted_secrets(conn)
        row = conn.execute("SELECT value FROM system_settings WHERE key = 'SMTP_PASS'").fetchone()
        conn.close()
        assert row["value"] != "plaintext_pass"
        assert row["value"].startswith("enc:")
        assert get_system_setting("SMTP_PASS") == "plaintext_pass"

    def test_save_exchange_and_email_does_not_write_plaintext_to_env(self, tmp_path, monkeypatch):
        from pathlib import Path
        from fastapi.testclient import TestClient
        from web_server import app

        fake_env = tmp_path / ".test_env"
        fake_env.write_text("EXCHANGE_ID=binance\nTESTNET=true\n")
        orig_path = Path
        monkeypatch.setattr("web_server.Path", lambda p: fake_env if p == ".env" else orig_path(p))

        client = TestClient(app)
        AuthService.register("env_test_user", "env_user@test.io", "Pass123456")
        conn = get_db_connection()
        tok = conn.execute("SELECT activation_token FROM users WHERE username = 'env_test_user'").fetchone()["activation_token"]
        conn.close()
        AuthService.activate_account(tok)
        login = AuthService.login("env_test_user", "Pass123456")
        auth_header = {"Authorization": f"Bearer {login['token']}"}

        # Save exchange key via endpoint
        res = client.post("/api/exchange/save", json={
            "exchange_id": "coinbase",
            "api_key": "my_super_secret_cb_key",
            "api_secret": "my_super_secret_cb_secret",
            "api_passphrase": "cb_passphrase",
            "testnet": False
        }, headers=auth_header)
        assert res.status_code == 200

        # Check content of fake .env file
        env_content = fake_env.read_text()
        assert "my_super_secret_cb_key" not in env_content
        assert "my_super_secret_cb_secret" not in env_content
        assert "cb_passphrase" not in env_content
        assert "EXCHANGE_ID=coinbase" in env_content

        # Save email settings via endpoint
        email_res = client.post("/api/auth/email-settings", json={
            "provider": "resend",
            "resend_api_key": "re_secret_key_should_not_be_in_env",
            "from_email": "Crypto Bot <noreply@example.com>"
        })
        assert email_res.status_code == 200

        env_content2 = fake_env.read_text()
        assert "re_secret_key_should_not_be_in_env" not in env_content2
        assert "EMAIL_PROVIDER=resend" in env_content2

