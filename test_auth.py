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
from pathlib import Path
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

    def test_delete_account_and_delete_all_accounts(self):
        AuthService.register("user_del_1", "del1@market.io", "Pass123456")
        AuthService.register("user_del_2", "del2@market.io", "Pass123456")

        conn = get_db_connection()
        u1 = conn.execute("SELECT id FROM users WHERE username = 'user_del_1'").fetchone()
        conn.close()

        # Test delete single account
        res = AuthService.delete_account(u1["id"])
        assert res["success"]

        conn = get_db_connection()
        assert conn.execute("SELECT count(*) FROM users WHERE username = 'user_del_1'").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM users WHERE username = 'user_del_2'").fetchone()[0] == 1
        conn.close()

        # Test delete all accounts
        res_all = AuthService.delete_all_accounts()
        assert res_all["success"]

        conn = get_db_connection()
        assert conn.execute("SELECT count(*) FROM users").fetchone()[0] == 0
        conn.close()

    def test_cross_container_serverless_activation_flow(self, tmp_path, monkeypatch):
        """
        Simulates Vercel multi-container serverless environment:
        Container A processes registration into DB A.
        Container B (cold start, completely empty DB B) processes activation.
        Verifies that Container B verifies the self-sovereign token, creates the user,
        marks them active, and returns an active session token.
        """
        db_a = tmp_path / "container_a.db"
        db_b = tmp_path / "container_b.db"

        # 1. Container A handles registration
        monkeypatch.setattr("auth.DB_PATH", db_a)
        init_auth_db()

        reg_res = AuthService.register("serverless_trader", "serverless@trader.io", "PassSecure123!")
        assert reg_res["success"] is True

        conn_a = sqlite3.connect(str(db_a))
        conn_a.row_factory = sqlite3.Row
        row_a = conn_a.execute("SELECT activation_token FROM users WHERE username = 'serverless_trader'").fetchone()
        token = row_a["activation_token"]
        conn_a.close()

        # 2. Container B handles activation with blank DB
        monkeypatch.setattr("auth.DB_PATH", db_b)
        init_auth_db()

        # Before activation, Container B has no users
        conn_b_pre = sqlite3.connect(str(db_b))
        assert conn_b_pre.execute("SELECT count(*) FROM users").fetchone()[0] == 0
        conn_b_pre.close()

        # Activation on Container B
        act_res = AuthService.activate_account(token)
        assert act_res["success"] is True
        assert "serverless_trader" in act_res["message"]
        assert "session_token" in act_res
        assert act_res["session_token"] is not None

        # Verify Container B now has the active user
        conn_b = sqlite3.connect(str(db_b))
        conn_b.row_factory = sqlite3.Row
        user_b = conn_b.execute("SELECT * FROM users WHERE username = 'serverless_trader'").fetchone()
        conn_b.close()

        assert user_b is not None
        assert user_b["is_active"] == 1
        assert user_b["email"] == "serverless@trader.io"

    def test_cross_container_session_resolution(self, tmp_path, monkeypatch):
        """
        Verifies that a stateless session token issued by one container can be
        authenticated on a brand-new, cold container with an empty sessions table.
        """
        from auth import create_stateless_session_token
        db_cold = tmp_path / "container_cold.db"
        monkeypatch.setattr("auth.DB_PATH", db_cold)
        init_auth_db()

        token = create_stateless_session_token(user_id=42, username="cloud_trader", email="cloud@trader.io")
        user = AuthService.get_user_from_token(token)

        assert user is not None
        assert user["id"] == 42
        assert user["username"] == "cloud_trader"
        assert user["email"] == "cloud@trader.io"

    def test_client_token_rehydration_on_cold_container(self, tmp_path, monkeypatch):
        """
        Tests that when a cold container with an empty SQLite database receives a login
        request, it checks remote storage, verifies credentials, hydrates the user into SQLite,
        and returns a valid session token.
        """
        from auth import hash_password
        db_cold = tmp_path / "container_login_cold.db"
        monkeypatch.setattr("auth.DB_PATH", db_cold)
        init_auth_db()

        # Generate credentials and mock remote storage
        pwd_hash, salt = hash_password("MySuperPassword123!")
        remote_db = {
            "rehydrate_user": {
                "id": 88,
                "username": "rehydrate_user",
                "email": "rehydrate@market.io",
                "password_hash": pwd_hash,
                "salt": salt,
                "is_active": 1,
            }
        }
        monkeypatch.setattr("auth.remote_get_user", lambda k: remote_db.get(k.lower()) if isinstance(k, str) else None)

        # Container has 0 users initially
        conn = sqlite3.connect(str(db_cold))
        assert conn.execute("SELECT count(*) FROM users").fetchone()[0] == 0
        conn.close()

        # Login with wrong password fails
        fail_res = AuthService.login("rehydrate_user", "WrongPassword!")
        assert fail_res["success"] is False

        # Login with correct password succeeds and rehydrates
        succ_res = AuthService.login("rehydrate_user", "MySuperPassword123!")
        assert succ_res["success"] is True
        assert succ_res["user"]["username"] == "rehydrate_user"
        assert "token" in succ_res

        # Verify user is now in local database
        conn = sqlite3.connect(str(db_cold))
        conn.row_factory = sqlite3.Row
        u = conn.execute("SELECT * FROM users WHERE username = 'rehydrate_user'").fetchone()
        assert u is not None
        assert u["is_active"] == 1
        assert u["email"] == "rehydrate@market.io"
        conn.close()

    def test_magic_login_token_flow(self, tmp_path, monkeypatch):
        """
        Tests the 1-click magic login token generation, verification, and instant sign-in.
        """
        db = tmp_path / "magic_test.db"
        monkeypatch.setattr("auth.DB_PATH", db)
        init_auth_db()

        # Magic login on unregistered user fails
        from auth import generate_magic_login_token
        unregistered_tok = generate_magic_login_token("nonexistent@trader.io", username="ghost")
        assert AuthService.magic_login(unregistered_tok)["success"] is False

        # Register user first
        reg_res = AuthService.register("magic_user", "magic_user@trader.io", "Pass12345678")
        assert reg_res["success"] is True

        # Request magic login link
        res = AuthService.send_magic_login_link("magic_user@trader.io")
        assert res["success"] is True

        # Extract token from dev link if email was mock/unconfigured
        magic_tok = res.get("dev_magic_link", "").split("token=")[-1]
        if not magic_tok:
            magic_tok = generate_magic_login_token("magic_user@trader.io", username="magic_user")

        # Magic login succeeds
        login_res = AuthService.magic_login(magic_tok)
        assert login_res["success"] is True
        assert login_res["email"] == "magic_user@trader.io"
        assert "session_token" in login_res

    def test_serverless_auto_activate_flow(self, tmp_path, monkeypatch):
        """
        Tests that when auto_activate is enabled (as on Vercel), registration immediately
        marks user active and issues a session token.
        """
        db = tmp_path / "auto_act.db"
        monkeypatch.setattr("auth.DB_PATH", db)
        init_auth_db()

        reg_res = AuthService.register("instant_user", "instant@trader.io", "PassInstant123!", auto_activate=True)
        assert reg_res["success"] is True
        assert reg_res["is_active"] == 1
        assert "token" in reg_res
        assert reg_res["user"]["username"] == "instant_user"

        # Login succeeds immediately without clicking email
        login_res = AuthService.login("instant_user", "PassInstant123!")
        assert login_res["success"] is True

    def test_expired_activation_token_rejected(self, tmp_path, monkeypatch):
        from auth import generate_activation_token
        db = tmp_path / "test_exp.db"
        monkeypatch.setattr("auth.DB_PATH", db)
        init_auth_db()

        # Token with negative expiration (-2 hours)
        token = generate_activation_token(1, "expired_user", "exp@test.com", "hash", "salt", expires_hours=-2)
        res = AuthService.activate_account(token)
        assert res["success"] is False
        assert "expired" in res["message"].lower()

    def test_tampered_activation_token_rejected(self, tmp_path, monkeypatch):
        db = tmp_path / "test_tamp.db"
        monkeypatch.setattr("auth.DB_PATH", db)
        init_auth_db()

        reg = AuthService.register("valid_user", "valid@test.com", "ValidPass123!")
        conn = get_db_connection()
        tok = conn.execute("SELECT activation_token FROM users WHERE username = 'valid_user'").fetchone()["activation_token"]
        conn.close()

        tampered = tok[:-6] + "xxxxxx"
        res = AuthService.activate_account(tampered)
        assert res["success"] is False
        assert "invalid" in res["message"].lower()

    def test_storage_adapter_mock_kv(self, monkeypatch):
        from storage_adapter import (
            remote_save_user,
            remote_get_user,
            remote_save_exchange_key,
            remote_get_exchange_keys,
            remote_delete_account,
            remote_delete_all_accounts
        )
        fake_redis = {}

        def mock_kv_cmd(*args):
            cmd = args[0]
            if cmd == "SET":
                fake_redis[args[1]] = args[2]
                return "OK"
            elif cmd == "GET":
                return fake_redis.get(args[1])
            elif cmd == "DEL":
                fake_redis.pop(args[1], None)
                return 1
            elif cmd == "SADD":
                s = fake_redis.setdefault(args[1], set())
                s.add(args[2])
                return 1
            elif cmd == "SMEMBERS":
                return list(fake_redis.get(args[1], []))
            elif cmd == "SREM":
                s = fake_redis.get(args[1], set())
                s.discard(args[2])
                return 1
            return None

        monkeypatch.setattr("storage_adapter.KV_URL", "https://fake-kv.upstash.io")
        monkeypatch.setattr("storage_adapter.KV_TOKEN", "fake_token")
        monkeypatch.setattr("storage_adapter._kv_command", mock_kv_cmd)

        assert remote_save_user(10, "remote_user", "remote@test.com", "h123", "s123", is_active=1)
        u = remote_get_user("remote_user")
        assert u is not None
        assert u["email"] == "remote@test.com"

        assert remote_save_exchange_key(10, "binance", "enc_k", "enc_s", "enc_p", testnet=True)
        keys = remote_get_exchange_keys(10)
        assert len(keys) == 1
        assert keys[0]["exchange_id"] == "binance"

        assert remote_delete_account(10, "remote_user", "remote@test.com")
        assert remote_get_user("remote_user") is None

    def test_storage_adapter_mock_blob(self, monkeypatch):
        from storage_adapter import (
            remote_save_user,
            remote_get_user,
            remote_save_exchange_key,
            remote_get_exchange_keys,
            remote_delete_account,
            remote_delete_all_accounts,
            get_remote_storage_type
        )
        fake_blob = {}

        def mock_blob_put(pathname, data_bytes, content_type="application/json"):
            fake_blob[pathname] = data_bytes
            return True

        def mock_blob_get(pathname):
            return fake_blob.get(pathname)

        def mock_blob_del(pathname):
            fake_blob.pop(pathname, None)
            return True

        monkeypatch.setattr("storage_adapter.KV_URL", "")
        monkeypatch.setattr("storage_adapter.KV_TOKEN", "")
        monkeypatch.setattr("storage_adapter.BLOB_TOKEN", "mock_blob_token_123")
        monkeypatch.setattr("storage_adapter.BLOB_STORE_ID", "store_mock123")
        monkeypatch.setattr("storage_adapter._blob_put", mock_blob_put)
        monkeypatch.setattr("storage_adapter._blob_get", mock_blob_get)
        monkeypatch.setattr("storage_adapter._blob_del", mock_blob_del)

        assert get_remote_storage_type() == "blob"
        assert remote_save_user(55, "blobby", "blobby@test.com", "h456", "s456", is_active=1)
        u = remote_get_user("blobby")
        assert u is not None
        assert u["email"] == "blobby@test.com"
        assert remote_get_user("blobby@test.com")["username"] == "blobby"

        assert remote_save_exchange_key(55, "bybit", "enc_k", "enc_s")
        keys = remote_get_exchange_keys(55)
        assert len(keys) == 1
        assert keys[0]["exchange_id"] == "bybit"

        assert remote_delete_account(55, "blobby", "blobby@test.com")
        assert remote_get_user("blobby") is None


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
        assert "••••" in masked_list[0]["api_key_masked"] or "***" in masked_list[0]["api_key_masked"]

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

    def test_production_mode_requires_external_key(self, monkeypatch):
        from secret_manager import get_master_key, reset_cipher
        reset_cipher()
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.delenv("BOT_SECRET_KEY", raising=False)
        monkeypatch.delenv("SECRET_KEY", raising=False)
        monkeypatch.delenv("APP_SECRET_KEY", raising=False)

        with pytest.raises(RuntimeError) as exc_info:
            get_master_key()
        assert "Production / Live mode requires an externally managed master encryption key" in str(exc_info.value)
        reset_cipher()

    def test_atomic_env_migration_preserves_secrets(self, tmp_path, monkeypatch):
        from secret_manager import migrate_and_sanitize_env_file
        # Create legacy .env with raw credentials
        legacy_env = tmp_path / ".legacy_env"
        legacy_env.write_text(
            "EXCHANGE_ID=binance\n"
            "EXCHANGE_API_KEY=legacy_binance_key_999\n"
            "EXCHANGE_API_SECRET=legacy_binance_secret_888\n"
            "SMTP_PASS=legacy_smtp_secret_777\n"
            "RESEND_API_KEY=re_legacy_token_666\n"
        )

        ok = migrate_and_sanitize_env_file(legacy_env)
        assert ok is True

        # Verify credentials were safely migrated into encrypted database vault
        latest = AuthService.get_latest_raw_exchange_key()
        assert latest is not None
        assert latest["api_key"] == "legacy_binance_key_999"
        assert latest["api_secret"] == "legacy_binance_secret_888"
        assert get_system_setting("SMTP_PASS") == "legacy_smtp_secret_777"
        assert get_system_setting("RESEND_API_KEY") == "re_legacy_token_666"

        # Verify plaintext credentials removed from .env
        sanitized_content = legacy_env.read_text()
        assert "legacy_binance_key_999" not in sanitized_content
        assert "legacy_binance_secret_888" not in sanitized_content
        assert "legacy_smtp_secret_777" not in sanitized_content
        assert "re_legacy_token_666" not in sanitized_content

        # Verify sensitive credentials removed from process environment
        assert "EXCHANGE_API_KEY" not in os.environ
        assert "EXCHANGE_API_SECRET" not in os.environ
        assert "SMTP_PASS" not in os.environ
        assert "RESEND_API_KEY" not in os.environ

    def test_runtime_os_environ_does_not_contain_secrets(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient
        from web_server import app, runner
        from config import BotConfig

        fake_env = tmp_path / ".test_env"
        fake_env.write_text("EXCHANGE_ID=binance\nTESTNET=true\n")
        orig_path = Path
        monkeypatch.setattr("web_server.Path", lambda p: fake_env if p == ".env" else orig_path(p))

        client = TestClient(app)
        AuthService.register("runtime_user", "runtime@market.io", "Pass123456")
        conn = get_db_connection()
        tok = conn.execute("SELECT activation_token FROM users WHERE username = 'runtime_user'").fetchone()["activation_token"]
        conn.close()
        AuthService.activate_account(tok)
        login = AuthService.login("runtime_user", "Pass123456")
        auth_header = {"Authorization": f"Bearer {login['token']}"}

        # Save exchange credentials via endpoint
        client.post("/api/exchange/save", json={
            "exchange_id": "kraken",
            "api_key": "kraken_secret_key_123",
            "api_secret": "kraken_secret_sec_456",
            "api_passphrase": "",
            "testnet": False
        }, headers=auth_header)

        # Confirm sensitive exchange keys are NOT in os.environ
        assert "EXCHANGE_API_KEY" not in os.environ
        assert "EXCHANGE_API_SECRET" not in os.environ

        # But runner.cfg and new BotConfig have them directly in memory
        assert runner.cfg.exchange.api_key == "kraken_secret_key_123"
        assert runner.cfg.exchange.api_secret == "kraken_secret_sec_456"

        new_cfg = BotConfig()
        assert new_cfg.exchange.api_key == "kraken_secret_key_123"
        assert new_cfg.exchange.api_secret == "kraken_secret_sec_456"

    def test_safe_load_dotenv_blocks_sensitive_keys_from_os_environ(self, tmp_path, monkeypatch):
        from secret_manager import safe_load_dotenv, SENSITIVE_ENV_KEYS

        dirty_env = tmp_path / ".dirty_env"
        dirty_env.write_text(
            "TEST_PORT=9999\n"
            "TEST_APP_NAME=CryptoBot\n"
            "EXCHANGE_API_KEY=exposed_key_111\n"
            "EXCHANGE_API_SECRET=exposed_secret_222\n"
            "SMTP_PASS=exposed_smtp_333\n"
            "RESEND_API_KEY=exposed_resend_444\n"
        )
        for k in ["TEST_PORT", "TEST_APP_NAME"] + list(SENSITIVE_ENV_KEYS):
            monkeypatch.delenv(k, raising=False)

        safe_load_dotenv(dirty_env, override=True)

        # Non-sensitive keys must be loaded
        assert os.environ.get("TEST_PORT") == "9999"
        assert os.environ.get("TEST_APP_NAME") == "CryptoBot"

        # Sensitive keys MUST NEVER be in os.environ
        assert "EXCHANGE_API_KEY" not in os.environ
        assert "EXCHANGE_API_SECRET" not in os.environ
        assert "SMTP_PASS" not in os.environ
        assert "RESEND_API_KEY" not in os.environ

    def test_migration_aborts_and_rolls_back_on_db_error_preserving_env(self, tmp_path, monkeypatch):
        from secret_manager import migrate_and_sanitize_env_file
        dirty_env = tmp_path / ".env_fail_test"
        orig_content = (
            "EXCHANGE_ID=coinbase\n"
            "EXCHANGE_API_KEY=my_cb_key_555\n"
            "EXCHANGE_API_SECRET=my_cb_secret_666\n"
            "SMTP_PASS=my_smtp_pass_777\n"
        )
        dirty_env.write_text(orig_content)

        def mock_failing_set_system_setting(*args, **kwargs):
            raise sqlite3.OperationalError("Simulated disk I/O error during insert")

        monkeypatch.setattr("auth.set_system_setting", mock_failing_set_system_setting)

        res = migrate_and_sanitize_env_file(dirty_env)
        assert res is False

        # .env must be 100% INTACT! Not modified, not deleted
        assert dirty_env.read_text() == orig_content

        # Verify transaction rolled back in DB: exchange_keys must NOT contain the credentials
        conn = get_db_connection()
        row = conn.execute("SELECT * FROM exchange_keys WHERE api_key LIKE '%my_cb_key_555%'").fetchone()
        conn.close()
        assert row is None

    def test_migration_atomic_rename_failure_preserves_env(self, tmp_path, monkeypatch):
        from secret_manager import migrate_and_sanitize_env_file
        dirty_env = tmp_path / ".env_rename_fail"
        orig_content = (
            "EXCHANGE_ID=kraken\n"
            "EXCHANGE_API_KEY=kraken_key_test\n"
            "EXCHANGE_API_SECRET=kraken_sec_test\n"
        )
        dirty_env.write_text(orig_content)

        def failing_replace(self, target):
            raise OSError("Simulated cross-device link or permission error on rename")

        monkeypatch.setattr(Path, "replace", failing_replace)

        res = migrate_and_sanitize_env_file(dirty_env)
        assert res is False
        assert dirty_env.read_text() == orig_content

    def test_set_system_setting_raises_explicit_error_on_failure(self):
        closed_conn = sqlite3.connect(":memory:")
        closed_conn.close()
        with pytest.raises(RuntimeError) as exc_info:
            set_system_setting("SOME_KEY", "SOME_VAL", conn=closed_conn)
        assert "Failed to persist system setting" in str(exc_info.value)

    def test_save_exchange_key_raises_explicit_error_on_failure(self):
        closed_conn = sqlite3.connect(":memory:")
        closed_conn.close()
        with pytest.raises(RuntimeError) as exc_info:
            AuthService.save_exchange_key(1, "binance", "k", "s", conn=closed_conn)
        assert "Failed to save exchange key" in str(exc_info.value)

    def test_save_exchange_key_survives_remote_sync_failure(self, monkeypatch):
        """A remote store outage must not undo the encrypted local vault write."""
        AuthService.register("remote_fail_user", "remote_fail@market.io", "Pass123456")
        conn = get_db_connection()
        user_id = conn.execute("SELECT id FROM users WHERE username = 'remote_fail_user'").fetchone()["id"]
        conn.close()

        def raise_remote_unavailable(*args, **kwargs):
            raise RuntimeError("remote store unavailable")

        monkeypatch.setattr("auth.is_remote_storage_available", lambda: True)
        monkeypatch.setattr("auth.remote_save_exchange_key", raise_remote_unavailable)

        result = AuthService.save_exchange_key(user_id, "binance", "remote_fail_key", "remote_fail_secret")
        assert result["success"] is True

        conn = get_db_connection()
        row = conn.execute(
            "SELECT api_key FROM exchange_keys WHERE user_id = ? AND exchange_id = 'binance'", (user_id,)
        ).fetchone()
        conn.close()
        assert row is not None
        assert row["api_key"].startswith("enc:")

    def test_save_exchange_key_tolerates_false_sync_and_misconfigured_driver(self, monkeypatch):
        """A refused sync or an unusable remote driver must not fail the local save."""
        AuthService.register("remote_false_user", "remote_false@market.io", "Pass123456")
        conn = get_db_connection()
        user_id = conn.execute("SELECT id FROM users WHERE username = 'remote_false_user'").fetchone()["id"]
        conn.close()

        monkeypatch.setattr("auth.is_remote_storage_available", lambda: True)
        monkeypatch.setattr("auth.remote_save_exchange_key", lambda *args, **kwargs: False)
        assert AuthService.save_exchange_key(user_id, "kraken", "k", "s")["success"] is True

        def raise_misconfigured_driver():
            raise RuntimeError("REMOTE_STORAGE_DRIVER='blob' is not fully configured")

        monkeypatch.setattr("auth.is_remote_storage_available", raise_misconfigured_driver)
        assert AuthService.save_exchange_key(user_id, "okx", "k", "s", "pass")["success"] is True

        conn = get_db_connection()
        rows = conn.execute(
            "SELECT exchange_id FROM exchange_keys WHERE user_id = ? ORDER BY exchange_id", (user_id,)
        ).fetchall()
        conn.close()
        assert [row["exchange_id"] for row in rows] == ["kraken", "okx"]


