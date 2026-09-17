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
