"""
Tests for security sanitization, input validation, math edge cases, and vulnerability mitigations.
"""
import pytest
import numpy as np
import pandas as pd
from starlette.testclient import TestClient

from auth import (
    AuthService,
    is_valid_username,
    is_valid_email,
    save_email_settings,
    get_system_setting,
    get_db_connection,
    init_auth_db,
)
from indicators import ema, rsi, atr, adx
from risk_manager import RiskManager
from config import RiskConfig, BotConfig
from backtester import summarize
from web_server import app, get_request_base_url


@pytest.fixture(autouse=True)
def init_db(tmp_path, monkeypatch):
    test_db = tmp_path / "test_users.db"
    monkeypatch.setattr("auth.DB_PATH", test_db)
    init_auth_db()
    yield


class TestInputSanitization:
    def test_username_validation(self):
        # Valid usernames
        assert is_valid_username("trader_queen")
        assert is_valid_username("bot-user.1")
        assert is_valid_username("Alice123")

        # Invalid usernames
        assert not is_valid_username("")
        assert not is_valid_username("ab")  # Too short (<3)
        assert not is_valid_username("a" * 33)  # Too long (>32)
        assert not is_valid_username("user\nname")  # CRLF injection
        assert not is_valid_username("user name")  # Space
        assert not is_valid_username("<script>alert(1)</script>")  # XSS
        assert not is_valid_username("user;DROP TABLE users;--")  # SQL injection syntax

    def test_email_validation(self):
        # Valid emails
        assert is_valid_email("alice@example.com")
        assert is_valid_email("bot+alerts@sub.domain.org")

        # Invalid emails
        assert not is_valid_email("")
        assert not is_valid_email("plainaddress")
        assert not is_valid_email("@missinguser.com")
        assert not is_valid_email("user@example.com\r\nBcc: victim@target.com")  # CRLF Header Injection
        assert not is_valid_email("user\nname@example.com")
        assert not is_valid_email("a" * 250 + "@example.com")  # > 254 chars

    def test_registration_password_length_bounds(self):
        # Password too short (<10)
        res = AuthService.register("valid_user", "valid@example.com", "Short1!")
        assert not res["success"]
        assert "at least 10" in res["message"]

        # Password too long (>128)
        res2 = AuthService.register("valid_user2", "valid2@example.com", "A" * 129)
        assert not res2["success"]
        assert "maximum 128" in res2["message"]

        # Valid password
        res3 = AuthService.register("valid_user3", "valid3@example.com", "CorrectHorseBatteryStaple#1")
        assert res3["success"]

    def test_email_crlf_stripping_in_save_settings(self):
        save_email_settings({
            "EMAIL_PROVIDER": "smtp\r\nINJECTED_VAR=true\nANOTHER_VAR=hacked",
            "SMTP_HOST": "smtp.mail.com\nEVIL=yes",
            "FROM_EMAIL": "test@domain.com\r\n",
        })

        val = get_system_setting("EMAIL_PROVIDER")
        assert "\r" not in val
        assert "\n" not in val
        assert val == "smtpINJECTED_VAR=trueANOTHER_VAR=hacked"

    def test_reset_password_revokes_sessions(self):
        # Register and activate
        AuthService.register("sessionuser", "session@example.com", "Password#12345")
        with get_db_connection() as conn:
            token = conn.execute("SELECT activation_token FROM users WHERE username='sessionuser'").fetchone()[0]
            user_id = conn.execute("SELECT id FROM users WHERE username='sessionuser'").fetchone()[0]
        AuthService.activate_account(token)

        # Login creates a recorded session
        login_res = AuthService.login("sessionuser", "Password#12345")
        assert login_res["success"]
        with get_db_connection() as conn:
            sessions_before = conn.execute("SELECT * FROM sessions WHERE user_id = ?", (user_id,)).fetchall()
            assert len(sessions_before) >= 1

        # Request reset
        reset_res = AuthService.request_password_reset("session@example.com")
        assert reset_res["success"]
        with get_db_connection() as conn:
            reset_token = conn.execute("SELECT reset_token FROM users WHERE id = ?", (user_id,)).fetchone()[0]

        # Complete reset
        succ = AuthService.reset_password(reset_token, "NewPassword#12345")
        assert succ["success"]

        # Verify active sessions in DB for user_id were wiped upon password reset
        with get_db_connection() as conn:
            sessions_after = conn.execute("SELECT * FROM sessions WHERE user_id = ?", (user_id,)).fetchall()
            assert len(sessions_after) == 0


class TestMathAndIndicatorEdgeCases:
    def test_adx_flat_market_no_division_by_zero(self):
        # Perfectly flat market where high == low == close
        df = pd.DataFrame({
            "high": [100.0] * 50,
            "low": [100.0] * 50,
            "close": [100.0] * 50
        })
        adx_series = adx(df, period=14)
        # Should not throw ZeroDivisionError and should not produce infinite values
        assert not np.isinf(adx_series.values).any()
        assert len(adx_series) == 50

    def test_indicator_period_bounds(self):
        s = pd.Series([10.0, 11.0, 12.0, 11.5, 13.0, 12.0])
        # EMA with period <= 0 should safely default to >= 1
        e0 = ema(s, period=0)
        assert len(e0) == len(s)
        assert not np.isnan(e0.iloc[-1])

        # RSI with period <= 0 should safely default to >= 1
        r0 = rsi(s, period=-5)
        assert len(r0) == len(s)

        df = pd.DataFrame({"high": s + 1, "low": s - 1, "close": s})
        a0 = atr(df, period=0)
        assert len(a0) == len(s)

    def test_risk_manager_zero_and_negative_equity(self):
        # Zero equity
        rm_zero = RiskManager(RiskConfig(risk_per_trade_pct=0.02), starting_equity=0.0)
        size_zero = rm_zero.position_size(entry_price=100.0, stop_price=95.0)
        assert size_zero == 0.0
        can_trade_0, _ = rm_zero.can_trade()
        assert not can_trade_0

        # Negative equity (deep drawdown)
        rm_neg = RiskManager(RiskConfig(risk_per_trade_pct=0.02), starting_equity=-500.0)
        size_neg = rm_neg.position_size(entry_price=100.0, stop_price=95.0)
        assert size_neg == 0.0
        can_trade_neg, _ = rm_neg.can_trade()
        assert not can_trade_neg

    def test_backtest_summarize_zero_equity(self):
        res = summarize([], [], starting_equity=0.0, timeframe="1h")
        assert res["total_return_pct"] == 0.0
        assert res["max_drawdown_pct"] == 0.0


class TestWebServerSecurityAndSanitization:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_host_header_poisoning_prevention(self, monkeypatch):
        # 1. Invalid characters and illegal scheme
        class DummyRequest:
            headers = {"x-forwarded-host": "evil.com\r\nX-Injected: 1", "x-forwarded-proto": "javascript"}
            url = type("URL", (), {"scheme": "http", "netloc": "localhost:8000"})()

        url = get_request_base_url(DummyRequest())
        assert "evil.com" not in url
        assert "javascript" not in url
        assert "http://localhost:8000" in url

        # 2. ALLOWED_HOSTS enforcement
        monkeypatch.setenv("ALLOWED_HOSTS", "localhost:8000,mydomain.com")
        class SpoofedRequest:
            headers = {"x-forwarded-host": "attacker-phishing.com", "x-forwarded-proto": "https"}
            url = type("URL", (), {"scheme": "http", "netloc": "localhost:8000"})()

        url2 = get_request_base_url(SpoofedRequest())
        assert "attacker-phishing.com" not in url2
        assert "localhost:8000" in url2

    def test_unauthenticated_delete_all_accounts_blocked(self, client):
        # Attempting to wipe accounts without authorization should return 401
        res = client.post("/api/auth/delete-all-accounts")
        assert res.status_code == 401

    def test_bot_settings_symbol_and_bounds_validation(self, client):
        # Invalid trading symbol
        res = client.post("/api/bot/settings", json={
            "symbol": "BTC/USDT\nEVIL_VAR=1",
            "timeframe": "1h",
            "fast_ema": 9,
            "slow_ema": 21,
            "rsi_period": 14,
            "risk_per_trade_pct": 0.02
        })
        assert res.status_code == 400
        assert "Invalid trading pair symbol" in res.json()["detail"]

        # Fast EMA >= Slow EMA
        res2 = client.post("/api/bot/settings", json={
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "fast_ema": 50,
            "slow_ema": 20,
            "rsi_period": 14,
            "risk_per_trade_pct": 0.02
        })
        assert res2.status_code == 400
        assert "Fast EMA" in res2.json()["detail"]

    def test_backtest_invalid_exchange_rejected(self, client):
        res = client.post("/api/backtest", json={
            "exchange": "not_a_real_exchange_xyz",
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "days": 30,
            "starting_equity": 1000.0,
            "risk_per_trade_pct": 2.0,
            "use_fib_filter": True
        })
        assert res.status_code == 400
        assert "Unsupported exchange" in res.json()["detail"]

    def test_get_logs_clamping(self, client):
        # Negative lines should clamp to at least 1, not error or reverse slice
        res = client.get("/api/logs?lines=-50")
        assert res.status_code == 200
        assert "logs" in res.json()

    def test_xss_prevention_in_html_endpoints(self, client):
        payload = '<script>alert("XSS")</script>'

        # Reset password endpoint
        res = client.get(f"/reset-password?token={payload}")
        assert res.status_code == 200
        assert payload not in res.text
        assert '&lt;script&gt;alert(&quot;XSS&quot;)&lt;/script&gt;' in res.text or 'alert(&quot;XSS&quot;)' in res.text

        # Activate endpoint
        res2 = client.get(f"/activate?token={payload}")
        assert res2.status_code == 200
        assert '<script>alert("XSS")</script>' not in res2.text
