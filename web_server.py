"""
Web Server, Dashboard, and Authentication API for Crypto Trading Bot.
Features:
- Full CCXT Exchange API Key management with exchange selector & connectivity tester
- User registration, login, PBKDF2-HMAC-SHA256 password hashing, sessions
- Email activation workflow with acceptance token & in-app email preview
- Lost password reset workflow
- Automated BOT.zip packaging and download endpoint
- Live trading bot control, market indicators, Fibonacci zone analysis, backtesting
"""
import os
import sys
import time
import json
import logging
import threading
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, Header, HTTPException, Query, Response, Request, Cookie
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn
import pandas as pd
import numpy as np
import ccxt

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from secret_manager import safe_load_dotenv, migrate_and_sanitize_env_file, SENSITIVE_ENV_KEYS
safe_load_dotenv()
migrate_and_sanitize_env_file()

from config import BotConfig, StrategyConfig, RiskConfig, ExchangeConfig
from exchange_client import ExchangeClient
from indicators import ema, rsi, atr, adx, fib_retracement_levels
from strategy import Signal, add_indicators, generate_signal, _near_fib_support, STRATEGIES
from risk_manager import RiskManager
from backtester import fetch_historical, run_backtest, summarize
from bot import TradingBot, setup_logging
from auth import AuthService, init_auth_db, get_email_settings, save_email_settings, test_email_connection

AVAILABLE_TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"]

setup_logging()
logger = logging.getLogger("web")
init_auth_db()

# Preload non-sensitive exchange metadata into environment if present in vault
latest_key = AuthService.get_latest_raw_exchange_key()
if latest_key:
    if not os.getenv("EXCHANGE_ID") and latest_key.get("exchange_id"):
        os.environ["EXCHANGE_ID"] = latest_key["exchange_id"]
    if not os.getenv("TESTNET") and "testnet" in latest_key:
        os.environ["TESTNET"] = "true" if latest_key["testnet"] else "false"

# Ensure sensitive exchange credentials are never present in os.environ at runtime
if not (os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME")):
    for k in {"EXCHANGE_API_KEY", "EXCHANGE_API_SECRET", "EXCHANGE_API_PASSPHRASE"}:
        os.environ.pop(k, None)

app = FastAPI(title="Crypto Trading Bot Dashboard & Auth", version="2.0.0")

# Supported popular exchanges with metadata
EXCHANGES_META = [
    {"id": "binance", "name": "Binance", "needs_passphrase": False, "sandbox": True},
    {"id": "coinbase", "name": "Coinbase Advanced", "needs_passphrase": True, "sandbox": True},
    {"id": "kraken", "name": "Kraken", "needs_passphrase": False, "sandbox": False},
    {"id": "kucoin", "name": "KuCoin", "needs_passphrase": True, "sandbox": True},
    {"id": "bybit", "name": "Bybit", "needs_passphrase": False, "sandbox": True},
    {"id": "okx", "name": "OKX", "needs_passphrase": True, "sandbox": True},
    {"id": "bitget", "name": "Bitget", "needs_passphrase": True, "sandbox": True},
    {"id": "gateio", "name": "Gate.io", "needs_passphrase": False, "sandbox": False},
    {"id": "mexc", "name": "MEXC", "needs_passphrase": False, "sandbox": False},
    {"id": "bitfinex", "name": "Bitfinex", "needs_passphrase": False, "sandbox": False},
    {"id": "cryptocom", "name": "Crypto.com", "needs_passphrase": False, "sandbox": False},
    {"id": "htx", "name": "HTX (Huobi)", "needs_passphrase": False, "sandbox": False},
    {"id": "bingx", "name": "BingX", "needs_passphrase": False, "sandbox": False},
    {"id": "whitebit", "name": "WhiteBIT", "needs_passphrase": False, "sandbox": False},
    {"id": "gemini", "name": "Gemini", "needs_passphrase": False, "sandbox": True},
    {"id": "phemex", "name": "Phemex", "needs_passphrase": False, "sandbox": True},
    {"id": "poloniex", "name": "Poloniex", "needs_passphrase": False, "sandbox": False},
    {"id": "deribit", "name": "Deribit", "needs_passphrase": False, "sandbox": True},
]

# --- Global Bot Runner ---
class BotRunner:
    def __init__(self):
        self.bot: Optional[TradingBot] = None
        self.thread: Optional[threading.Thread] = None
        self.running: bool = False
        self.last_step_time: Optional[str] = None
        self.last_error: Optional[str] = None
        self.cfg: BotConfig = BotConfig()

    def start(self):
        if self.running:
            return False
        if os.path.exists(self.cfg.kill_switch_file):
            try:
                os.remove(self.cfg.kill_switch_file)
            except Exception:
                pass

        self.cfg = BotConfig()
        self.bot = TradingBot(self.cfg)
        self.running = True
        self.last_error = None

        logger.info(
            "Bot background engine started. Mode: %s | Execution: %s | Strategy: %s | Symbol: %s (%s)",
            self.cfg.mode.upper(),
            getattr(self.cfg, "execution_mode", "auto").upper(),
            self.cfg.strategy.strategy_name,
            self.cfg.exchange.symbol,
            self.cfg.exchange.timeframe,
        )

        def _run_loop():
            logger.info("Bot execution loop active. Polling interval: %ds.", self.cfg.poll_interval_sec)
            try:
                while self.running:
                    step_ok = False
                    try:
                        step_ok = self.bot.step()
                    except Exception as e:
                        logger.warning("Transient error during bot step: %s — will retry next cycle.", e)
                        self.last_error = str(e)
                        step_ok = True

                    self.last_step_time = datetime.now(timezone.utc).isoformat()
                    if not step_ok:
                        if self.bot and self.bot.kill_switch_active():
                            logger.warning("Kill switch active — halting bot runner loop.")
                            self.running = False
                            break
                        else:
                            logger.warning("Bot step returned False without active kill switch — retrying on next cycle.")

                    for _ in range(self.cfg.poll_interval_sec):
                        if not self.running or (self.bot and self.bot.kill_switch_active()):
                            break
                        time.sleep(1)
                    if self.bot and self.bot.kill_switch_active():
                        logger.warning("Kill switch active — halting bot runner loop.")
                        self.running = False
                        break
            except Exception as e:
                logger.exception("Error in bot background thread")
                self.last_error = str(e)
                self.running = False

        self.thread = threading.Thread(target=_run_loop, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        was_running = self.running
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if was_running:
            logger.info("Bot background engine stopped by user.")
        return True

    def trigger_kill(self):
        logger.critical("EMERGENCY KILL SWITCH TRIGGERED: Halting bot immediately.")
        with open(self.cfg.kill_switch_file, "w") as f:
            f.write("HALT")
        self.stop()
        return True

    def get_or_create_bot(self) -> TradingBot:
        if self.bot is None:
            self.bot = TradingBot(self.cfg)
        return self.bot


runner = BotRunner()


# --- Pydantic Request Models ---
class BotSettingsRequest(BaseModel):
    execution_mode: Optional[str] = "auto"  # "auto" | "manual"
    strategy: Optional[str] = "ema_cross_fib"
    timeframe: Optional[str] = "15m"
    symbol: Optional[str] = "BTC/USDT"
    use_fib_filter: Optional[bool] = True
    fast_ema: Optional[int] = 12
    slow_ema: Optional[int] = 26
    rsi_period: Optional[int] = 14
    rsi_overbought: Optional[float] = 65.0
    adx_trend_threshold: Optional[float] = 20.0
    risk_per_trade_pct: Optional[float] = 0.01
    stop_loss_atr_mult: Optional[float] = 2.0
    take_profit_atr_mult: Optional[float] = 4.0


class ManualTradeRequest(BaseModel):
    action: str  # "buy" | "sell"
    size: Optional[float] = 0.0


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    username_or_email: str
    password: str
    client_token: Optional[str] = None


class MagicLinkRequest(BaseModel):
    email: str


class ForgotPasswordRequest(BaseModel):
    email: str


class ResendActivationRequest(BaseModel):
    email: str


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


class SaveExchangeKeyRequest(BaseModel):
    exchange_id: str
    api_key: str
    api_secret: str
    api_passphrase: Optional[str] = ""
    testnet: Optional[bool] = True


class TestExchangeRequest(BaseModel):
    exchange_id: str
    api_key: str
    api_secret: str
    api_passphrase: Optional[str] = ""
    testnet: Optional[bool] = True


class TradingModeRequest(BaseModel):
    trading_mode: str  # "testnet" | "real"



class BacktestRequest(BaseModel):
    exchange: str = "binance"
    symbol: str = "BTC/USDT"
    timeframe: str = "15m"
    days: int = 30
    starting_equity: float = 1000.0
    use_fib_filter: bool = True
    risk_per_trade_pct: float = 0.01


class EmailSettingsRequest(BaseModel):
    provider: Optional[str] = "gmail"
    smtp_host: Optional[str] = ""
    smtp_port: Optional[str] = "587"
    smtp_user: Optional[str] = ""
    smtp_pass: Optional[str] = ""
    from_email: Optional[str] = ""
    resend_api_key: Optional[str] = ""
    brevo_api_key: Optional[str] = ""
    sendgrid_api_key: Optional[str] = ""


class TestEmailRequest(BaseModel):
    to_email: str


# --- Auth Dependency Helper ---
def get_current_user(
    authorization: Optional[str] = Header(None),
    bot_auth_token: Optional[str] = Cookie(None)
) -> Optional[Dict[str, Any]]:
    token = ""
    if authorization:
        token = authorization.replace("Bearer ", "").strip()
    elif bot_auth_token:
        token = bot_auth_token.strip()
    if not token:
        return None
    return AuthService.get_user_from_token(token)


# --- ZIP Packaging Helper ---
def create_bot_zip() -> str:
    if os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
        zip_path = Path("/tmp/BOT.zip")
    else:
        zip_path = Path(__file__).resolve().parent / "BOT.zip"

    root = Path(__file__).resolve().parent
    TARGET_FILES = [
        "bot.py",
        "config.py",
        "exchange_client.py",
        "indicators.py",
        "risk_manager.py",
        "strategy.py",
        "backtester.py",
        "web_server.py",
        "auth.py",
        "secret_manager.py",
        "requirements.txt",
        ".env.example",
        "README.md",
        "test_bot.py",
        "pyproject.toml",
        "vercel.json",
    ]

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in TARGET_FILES:
            fpath = root / fname
            if fpath.exists() and fpath.is_file():
                zf.write(fpath, fname)
        api_index = root / "api" / "index.py"
        if api_index.exists():
            zf.write(api_index, "api/index.py")

    return str(zip_path)


def get_request_base_url(request: Request) -> str:
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.headers.get("host", request.url.netloc))
    return f"{proto}://{host}".rstrip("/")


# ============================================================================
# API ENDPOINTS: AUTHENTICATION & USER MANAGEMENT
# ============================================================================
@app.post("/api/auth/register")
def register(req: RegisterRequest, request: Request, response: Response):
    base_url = get_request_base_url(request)
    res = AuthService.register(req.username, req.email, req.password, base_url=base_url)
    if not res["success"]:
        return JSONResponse(status_code=400, content=res)
    if res.get("token"):
        response.set_cookie(key="bot_auth_token", value=res["token"], max_age=2592000, path="/", samesite="lax", secure=True)
    return res


@app.get("/api/auth/activate")
@app.post("/api/auth/activate")
def activate_account(token: str = Query(...), response: Response = None):
    res = AuthService.activate_account(token)
    if not res["success"]:
        return JSONResponse(status_code=400, content=res)
    if response and res.get("session_token"):
        response.set_cookie(key="bot_auth_token", value=res["session_token"], max_age=2592000, path="/", samesite="lax", secure=True)
    return res


@app.post("/api/auth/resend-activation")
def resend_activation_endpoint(req: ResendActivationRequest, request: Request):
    base_url = get_request_base_url(request)
    res = AuthService.resend_activation_email(req.email, base_url=base_url)
    return res


@app.post("/api/auth/login")
def login(req: LoginRequest, response: Response):
    res = AuthService.login(req.username_or_email, req.password, client_token=req.client_token)
    if not res["success"]:
        return JSONResponse(status_code=400, content=res)
    if res.get("token"):
        response.set_cookie(key="bot_auth_token", value=res["token"], max_age=2592000, path="/", samesite="lax", secure=True)
    return res


@app.post("/api/auth/magic-link")
def send_magic_link_endpoint(req: MagicLinkRequest, request: Request):
    base_url = get_request_base_url(request)
    res = AuthService.send_magic_login_link(req.email, base_url=base_url)
    if not res["success"]:
        return JSONResponse(status_code=400, content=res)
    return res


@app.get("/api/auth/me")
def get_profile(authorization: Optional[str] = Header(None), bot_auth_token: Optional[str] = Cookie(None)):
    user = get_current_user(authorization, bot_auth_token)
    if not user:
        return {"authenticated": False}
    keys = AuthService.get_exchange_keys(user["id"])
    return {"authenticated": True, "user": user, "saved_exchanges": keys}


@app.post("/api/auth/forgot-password")
def forgot_password(req: ForgotPasswordRequest, request: Request):
    base_url = get_request_base_url(request)
    res = AuthService.request_password_reset(req.email, base_url=base_url)
    return res


@app.post("/api/auth/reset-password")
def reset_password(req: ResetPasswordRequest):
    res = AuthService.reset_password(req.token, req.new_password)
    if not res["success"]:
        raise HTTPException(status_code=400, detail=res["message"])
    return res


@app.post("/api/auth/delete-all-accounts")
def delete_all_accounts_endpoint():
    res = AuthService.delete_all_accounts()
    return res


@app.delete("/api/auth/account")
def delete_my_account(authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    res = AuthService.delete_account(user["id"])
    return res


@app.get("/api/auth/latest-email")
def get_latest_email(email: Optional[str] = None):
    return {"message": "Verification links are delivered exclusively to the registered email address."}


@app.get("/api/auth/email-settings")
def get_email_settings_endpoint():
    return get_email_settings()


@app.post("/api/auth/email-settings")
def save_email_settings_endpoint(req: EmailSettingsRequest):
    settings_dict = {}
    provider = (req.provider or "gmail").strip().lower()
    settings_dict["EMAIL_PROVIDER"] = provider

    if req.from_email is not None and req.from_email.strip():
        settings_dict["FROM_EMAIL"] = req.from_email.strip()

    if provider == "gmail":
        settings_dict["SMTP_HOST"] = "smtp.gmail.com"
        settings_dict["SMTP_PORT"] = req.smtp_port or "587"
        if req.smtp_user:
            settings_dict["SMTP_USER"] = req.smtp_user.strip()
            if not req.from_email:
                settings_dict["FROM_EMAIL"] = req.smtp_user.strip()
        if req.smtp_pass and req.smtp_pass.strip():
            settings_dict["SMTP_PASS"] = req.smtp_pass.strip()
    elif provider == "brevo":
        if req.brevo_api_key and req.brevo_api_key.strip():
            settings_dict["BREVO_API_KEY"] = req.brevo_api_key.strip()
        if req.from_email:
            settings_dict["FROM_EMAIL"] = req.from_email.strip()
    elif provider == "resend":
        if req.resend_api_key and req.resend_api_key.strip():
            settings_dict["RESEND_API_KEY"] = req.resend_api_key.strip()
        if req.from_email:
            settings_dict["FROM_EMAIL"] = req.from_email.strip()
    elif provider == "sendgrid":
        if req.sendgrid_api_key and req.sendgrid_api_key.strip():
            settings_dict["SENDGRID_API_KEY"] = req.sendgrid_api_key.strip()
        if req.from_email:
            settings_dict["FROM_EMAIL"] = req.from_email.strip()
    elif provider == "smtp":
        settings_dict["SMTP_HOST"] = (req.smtp_host or "").strip()
        settings_dict["SMTP_PORT"] = (req.smtp_port or "587").strip()
        if req.smtp_user:
            settings_dict["SMTP_USER"] = req.smtp_user.strip()
        if req.smtp_pass and req.smtp_pass.strip():
            settings_dict["SMTP_PASS"] = req.smtp_pass.strip()
        if req.from_email:
            settings_dict["FROM_EMAIL"] = req.from_email.strip()
    elif provider == "none":
        settings_dict["EMAIL_PROVIDER"] = "none"

    res = save_email_settings(settings_dict)

    # Persist only non-sensitive email configuration to .env if writable and file exists
    try:
        env_file = Path(".env")
        if env_file.exists():
            from secret_manager import SENSITIVE_ENV_KEYS
            lines = env_file.read_text().splitlines(keepends=True)
            written = set()
            new_lines = []
            for l in lines:
                s = l.strip()
                if s and not s.startswith("#") and "=" in s:
                    k = s.split("=", 1)[0].strip()
                    if k in SENSITIVE_ENV_KEYS:
                        # Omit sensitive credentials from plaintext file
                        continue
                    if k in settings_dict and settings_dict[k]:
                        new_lines.append(f"{k}={settings_dict[k]}\n")
                        written.add(k)
                        continue
                new_lines.append(l)
            for k, v in settings_dict.items():
                if k not in written and v and k not in SENSITIVE_ENV_KEYS:
                    new_lines.append(f"{k}={v}\n")
            env_file.write_text("".join(new_lines))
    except Exception as e:
        print(f"[AUTH ERROR] Failed updating .env file: {e}")

    return res


@app.post("/api/auth/test-email")
def test_email_endpoint(req: TestEmailRequest):
    return test_email_connection(req.to_email)


# ============================================================================
# API ENDPOINTS: EXCHANGE API KEY MANAGEMENT
# ============================================================================
@app.get("/api/exchanges")
def get_exchanges():
    return {"exchanges": EXCHANGES_META}


@app.post("/api/exchange/test")
def test_exchange(req: TestExchangeRequest, authorization: Optional[str] = Header(None)):
    """Test connection and credential validity using CCXT."""
    user = get_current_user(authorization)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please create an account and log in to test exchange API keys."
        )

    if not hasattr(ccxt, req.exchange_id):
        raise HTTPException(status_code=400, detail=f"Unsupported exchange '{req.exchange_id}'")

    clean_key = req.api_key.strip().strip("'\"")
    clean_secret = req.api_secret.strip().strip("'\"")
    clean_passphrase = (req.api_passphrase or "").strip().strip("'\"")

    exchange_class = getattr(ccxt, req.exchange_id)

    def init_exchange_instance(use_sandbox: bool):
        opts = {
            "apiKey": clean_key,
            "secret": clean_secret,
            "enableRateLimit": True,
            "timeout": 15000,
            "options": {
                "adjustForTimeDifference": True,
                "recvWindow": 60000,
            },
        }
        if clean_passphrase:
            opts["password"] = clean_passphrase

        ex = exchange_class(opts)
        if use_sandbox and hasattr(ex, "set_sandbox_mode"):
            try:
                ex.set_sandbox_mode(True)
            except Exception:
                pass

        if ex.has.get("fetchTime") and hasattr(ex, "load_time_difference"):
            try:
                ex.load_time_difference()
            except Exception:
                pass
        return ex

    # 1. First attempt with user-requested sandbox setting
    actual_testnet = bool(req.testnet)
    exchange = init_exchange_instance(actual_testnet)

    try:
        balance = exchange.fetch_balance()
    except Exception as e:
        err_msg = str(e)

        # A. Clock skew / recvWindow retry
        if any(w in err_msg.lower() for w in ["recvwindow", "timestamp", "nonce", "outside of the recvwindow"]):
            try:
                if hasattr(exchange, "load_time_difference"):
                    exchange.load_time_difference()
                balance = exchange.fetch_balance()
                err_msg = ""
            except Exception as retry_e:
                err_msg = str(retry_e)

        # B. Sandbox vs Live Key Mismatch (e.g. Binance code -2014)
        if err_msg and any(w in err_msg.lower() for w in ["-2014", "api-key format invalid", "invalid api-key"]):
            if actual_testnet:
                try:
                    live_ex = init_exchange_instance(False)
                    balance = live_ex.fetch_balance()
                    actual_testnet = False
                    err_msg = ""
                except Exception as live_err:
                    err_msg = f"{err_msg} (Also tested Live mode: {live_err})"
            else:
                try:
                    testnet_ex = init_exchange_instance(True)
                    balance = testnet_ex.fetch_balance()
                    actual_testnet = True
                    err_msg = ""
                except Exception as testnet_err:
                    err_msg = f"{err_msg} (Also tested Testnet mode: {testnet_err})"

        if err_msg:
            if "-2014" in err_msg or "api-key format invalid" in err_msg.lower():
                return {
                    "success": False,
                    "message": (
                        "Binance Error (-2014: API-key format invalid). "
                        "Please verify: 1) You copied the complete API Key & Secret without extra spaces or quotes. "
                        "2) If your key was created on Binance.com (real exchange), uncheck 'Use Testnet / Sandbox Mode'. "
                        "3) Ensure the key has Spot & Margin trading enabled."
                    )
                }
            if isinstance(e, ccxt.AuthenticationError):
                return {"success": False, "message": f"Authentication Error: {err_msg}"}
            return {"success": False, "message": f"Connection check failed: {err_msg}"}

    free_balances = {k: v for k, v in balance.get("free", {}).items() if v and float(v) > 0}
    env_label = "Testnet / Sandbox" if actual_testnet else "Live / Real Trading"
    note = ""
    if actual_testnet != req.testnet:
        note = f" (Note: Automatically connected to {env_label} because your credentials match {env_label})"
    return {
        "success": True,
        "message": f"Successfully authenticated with {req.exchange_id} ({env_label})! Credentials are valid.{note}",
        "testnet": actual_testnet,
        "balances": free_balances or {"INFO": "Connected successfully (0 balance found)"}
    }


@app.post("/api/exchange/save")
def save_exchange_credentials(req: SaveExchangeKeyRequest, authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please create an account and log in to manage exchange API keys."
        )

    # Store in DB per user (encrypted at rest)
    AuthService.save_exchange_key(
        user["id"], req.exchange_id, req.api_key, req.api_secret, req.api_passphrase or "", req.testnet
    )

    # In-memory updates for non-sensitive routing metadata only
    os.environ["EXCHANGE_ID"] = req.exchange_id
    os.environ["TESTNET"] = "true" if req.testnet else "false"

    # Remove sensitive keys from os.environ to prevent runtime exposure
    for k in ["EXCHANGE_API_KEY", "EXCHANGE_API_SECRET", "EXCHANGE_API_PASSPHRASE"]:
        os.environ.pop(k, None)

    # Persist only non-sensitive configuration to .env (do NOT write API keys/secrets to .env)
    env_file = Path(".env")
    if env_file.exists():
        try:
            from secret_manager import SENSITIVE_ENV_KEYS
            lines = env_file.read_text().splitlines(keepends=True)
            new_lines = []
            written = set()
            for l in lines:
                s = l.strip()
                if s and not s.startswith("#") and "=" in s:
                    k = s.split("=", 1)[0].strip()
                    if k in SENSITIVE_ENV_KEYS:
                        # Strip plaintext credential from .env
                        continue
                    if k == "EXCHANGE_ID":
                        new_lines.append(f"EXCHANGE_ID={req.exchange_id}\n")
                        written.add(k)
                        continue
                    if k == "TESTNET":
                        new_lines.append(f"TESTNET={'true' if req.testnet else 'false'}\n")
                        written.add(k)
                        continue
                new_lines.append(l)

            if "EXCHANGE_ID" not in written:
                new_lines.append(f"EXCHANGE_ID={req.exchange_id}\n")
            if "TESTNET" not in written:
                new_lines.append(f"TESTNET={'true' if req.testnet else 'false'}\n")

            env_file.write_text("".join(new_lines))
        except Exception as e:
            logger.warning(f"Failed updating .env with exchange metadata: {e}")

    # Refresh runner exchange config
    runner.cfg = BotConfig()
    return {
        "success": True,
        "message": f"API keys for {req.exchange_id} saved securely to encrypted vault and applied to active configuration.",
        "exchange_id": req.exchange_id,
        "testnet": req.testnet,
    }


# ============================================================================
# API ENDPOINTS: CODE DOWNLOAD
# ============================================================================
@app.api_route("/api/download/bot", methods=["GET", "HEAD"])
def download_bot_zip():
    """Packages the bot into BOT.zip and serves it as a file download."""
    zip_path = create_bot_zip()
    return FileResponse(
        path=zip_path,
        filename="BOT.zip",
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=BOT.zip"}
    )


# ============================================================================
# API ENDPOINTS: TRADING BOT DASHBOARD & CONTROLS
# ============================================================================
@app.get("/api/status")
def get_status():
    cfg = runner.cfg
    bot = runner.bot
    risk_state = bot.risk.state if bot else None
    exchange = bot.exchange if bot else None

    kill_file_exists = os.path.exists(cfg.kill_switch_file)
    strat_meta = next((s for s in STRATEGIES if s["id"] == cfg.strategy.strategy_name), STRATEGIES[0])

    return {
        "running": runner.running,
        "mode": cfg.mode,
        "execution_mode": getattr(cfg, "execution_mode", "auto"),
        "strategy": cfg.strategy.strategy_name,
        "strategy_name": strat_meta["name"],
        "strategy_short_name": strat_meta.get("short_name", strat_meta["name"]),
        "exchange": cfg.exchange.exchange_id,
        "symbol": cfg.exchange.symbol,
        "timeframe": cfg.exchange.timeframe,
        "testnet": cfg.exchange.testnet,
        "has_api_key": bool(cfg.exchange.api_key),
        "last_step_time": runner.last_step_time,
        "last_error": runner.last_error,
        "kill_file_exists": kill_file_exists,
        "equity": round(risk_state.equity, 2) if risk_state else cfg.starting_equity,
        "peak_equity": round(risk_state.peak_equity, 2) if risk_state else cfg.starting_equity,
        "daily_start_equity": round(risk_state.daily_start_equity, 2) if risk_state else cfg.starting_equity,
        "drawdown_pct": round(((risk_state.peak_equity - risk_state.equity) / risk_state.peak_equity) * 100, 2) if risk_state and risk_state.peak_equity > 0 else 0.0,
        "daily_loss_pct": round(((risk_state.daily_start_equity - risk_state.equity) / risk_state.daily_start_equity) * 100, 2) if risk_state and risk_state.daily_start_equity > 0 else 0.0,
        "consecutive_losses": risk_state.consecutive_losses if risk_state else 0,
        "trading_halted_today": risk_state.trading_halted_today if risk_state else False,
        "kill_switch_triggered": risk_state.kill_switch_triggered if risk_state else False,
        "paper_cash": round(exchange.paper_cash, 2) if exchange else cfg.starting_equity,
        "paper_position": round(exchange.paper_position, 6) if exchange else 0.0,
        "active_position": bot.position if bot else None,
    }


@app.get("/api/bot/settings")
def get_bot_settings():
    cfg = runner.cfg
    strat_meta = next((s for s in STRATEGIES if s["id"] == cfg.strategy.strategy_name), STRATEGIES[0])
    return {
        "execution_mode": getattr(cfg, "execution_mode", "auto"),
        "strategy": cfg.strategy.strategy_name,
        "strategy_name": strat_meta["name"],
        "strategy_short_name": strat_meta.get("short_name", strat_meta["name"]),
        "available_strategies": STRATEGIES,
        "timeframe": cfg.exchange.timeframe,
        "available_timeframes": AVAILABLE_TIMEFRAMES,
        "symbol": cfg.exchange.symbol,
        "use_fib_filter": cfg.strategy.use_fib_filter,
        "fast_ema": cfg.strategy.fast_ema,
        "slow_ema": cfg.strategy.slow_ema,
        "rsi_period": cfg.strategy.rsi_period,
        "rsi_overbought": cfg.strategy.rsi_overbought,
        "adx_trend_threshold": cfg.strategy.adx_trend_threshold,
        "risk_per_trade_pct": cfg.risk.risk_per_trade_pct,
        "stop_loss_atr_mult": cfg.risk.stop_loss_atr_mult,
        "take_profit_atr_mult": cfg.risk.take_profit_atr_mult,
    }


@app.post("/api/bot/settings")
def save_bot_settings(req: BotSettingsRequest):
    exec_mode = (req.execution_mode or "auto").strip().lower()
    if exec_mode not in ["auto", "manual"]:
        raise HTTPException(status_code=400, detail="Invalid execution_mode. Must be 'auto' or 'manual'.")

    timeframe = (req.timeframe or "15m").strip().lower()
    if timeframe not in AVAILABLE_TIMEFRAMES:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe '{timeframe}'. Supported: {', '.join(AVAILABLE_TIMEFRAMES)}")

    valid_strat_ids = [s["id"] for s in STRATEGIES]
    strategy_id = (req.strategy or "ema_cross_fib").strip().lower()
    if strategy_id not in valid_strat_ids:
        raise HTTPException(status_code=400, detail=f"Invalid strategy '{strategy_id}'. Supported: {', '.join(valid_strat_ids)}")

    symbol = (req.symbol or "BTC/USDT").strip().upper()

    # Update OS env vars
    os.environ["EXECUTION_MODE"] = exec_mode
    os.environ["STRATEGY"] = strategy_id
    os.environ["TIMEFRAME"] = timeframe
    os.environ["SYMBOL"] = symbol

    # Persist to .env file
    try:
        env_file = Path(".env")
        if env_file.exists():
            lines = env_file.read_text().splitlines(keepends=True)
            updates = {
                "EXECUTION_MODE": exec_mode,
                "STRATEGY": strategy_id,
                "TIMEFRAME": timeframe,
                "SYMBOL": symbol,
            }
            written = set()
            new_lines = []
            for l in lines:
                s = l.strip()
                if s and not s.startswith("#") and "=" in s:
                    k = s.split("=", 1)[0].strip()
                    if k in updates:
                        new_lines.append(f"{k}={updates[k]}\n")
                        written.add(k)
                        continue
                new_lines.append(l)
            for k, v in updates.items():
                if k not in written:
                    new_lines.append(f"{k}={v}\n")
            env_file.write_text("".join(new_lines))
    except Exception as e:
        logger.warning(f"Failed updating .env file with new bot settings: {e}")

    # Re-apply to active runner
    was_running = runner.running
    if was_running:
        runner.stop()

    runner.cfg.execution_mode = exec_mode
    runner.cfg.strategy.strategy_name = strategy_id
    runner.cfg.exchange.timeframe = timeframe
    runner.cfg.exchange.symbol = symbol

    if req.use_fib_filter is not None:
        runner.cfg.strategy.use_fib_filter = req.use_fib_filter
    if req.fast_ema is not None and req.fast_ema > 0:
        runner.cfg.strategy.fast_ema = req.fast_ema
    if req.slow_ema is not None and req.slow_ema > 0:
        runner.cfg.strategy.slow_ema = req.slow_ema
    if req.rsi_period is not None and req.rsi_period > 0:
        runner.cfg.strategy.rsi_period = req.rsi_period
    if req.rsi_overbought is not None and req.rsi_overbought > 0:
        runner.cfg.strategy.rsi_overbought = req.rsi_overbought
    if req.adx_trend_threshold is not None and req.adx_trend_threshold >= 0:
        runner.cfg.strategy.adx_trend_threshold = req.adx_trend_threshold
    if req.risk_per_trade_pct is not None and req.risk_per_trade_pct > 0:
        runner.cfg.risk.risk_per_trade_pct = req.risk_per_trade_pct
    if req.stop_loss_atr_mult is not None and req.stop_loss_atr_mult > 0:
        runner.cfg.risk.stop_loss_atr_mult = req.stop_loss_atr_mult
    if req.take_profit_atr_mult is not None and req.take_profit_atr_mult > 0:
        runner.cfg.risk.take_profit_atr_mult = req.take_profit_atr_mult

    # Recreate bot instance with updated configuration
    runner.bot = TradingBot(runner.cfg)

    if was_running:
        runner.start()

    strat_meta = next((s for s in STRATEGIES if s["id"] == strategy_id), STRATEGIES[0])
    mode_label = "Manual Trading (Signals Only)" if exec_mode == "manual" else "Auto Trading (Algorithmic Execution)"

    return {
        "success": True,
        "message": f"Settings saved! Trading mode set to {mode_label}, strategy set to {strat_meta['name']}, and timeframe set to {timeframe}.",
        "execution_mode": exec_mode,
        "strategy": strategy_id,
        "strategy_name": strat_meta["name"],
        "timeframe": timeframe,
        "symbol": symbol,
    }


@app.post("/api/bot/trade/manual")
def execute_manual_trade(req: ManualTradeRequest):
    bot = runner.get_or_create_bot()
    act = req.action.strip().lower()

    if act == "buy":
        try:
            res = bot.manual_buy(custom_size=req.size or 0.0)
            return {
                "success": True,
                "message": f"Manual BUY order executed at ${res['price']:,.2f} (Size: {res['size']:.6f})",
                "trade": res,
            }
        except Exception as e:
            logger.exception("Manual BUY order failed")
            raise HTTPException(status_code=400, detail=str(e))
    elif act == "sell":
        try:
            res = bot.manual_sell()
            return {
                "success": True,
                "message": f"Manual SELL order executed at ${res['price']:,.2f} (PnL: ${res['pnl']:+,.2f})",
                "trade": res,
            }
        except Exception as e:
            logger.exception("Manual SELL order failed")
            raise HTTPException(status_code=400, detail=str(e))
    else:
        raise HTTPException(status_code=400, detail=f"Invalid manual trade action '{req.action}'. Use 'buy' or 'sell'.")


@app.post("/api/bot/start")
def start_bot():
    if runner.running:
        return {"success": False, "message": "Bot is already running"}
    started = runner.start()
    return {"success": started, "message": "Bot started in " + runner.cfg.mode + " mode (" + runner.cfg.execution_mode.upper() + ")"}


@app.post("/api/bot/stop")
def stop_bot():
    runner.stop()
    return {"success": True, "message": "Bot execution stopped"}


@app.post("/api/bot/kill")
def kill_bot():
    runner.trigger_kill()
    return {"success": True, "message": "Kill switch file created; bot halted immediately"}


@app.post("/api/bot/reset_day")
def reset_daily_metrics():
    if runner.bot:
        runner.bot.risk.reset_day()
        return {"success": True, "message": "Daily risk metrics reset"}
    return {"success": False, "message": "Bot is not initialized"}


@app.post("/api/bot/mode")
def set_trading_mode_endpoint(req: TradingModeRequest):
    """Switch trading mode between Testnet / Paper and Real Trading (Live)."""
    is_real = (req.trading_mode.lower() == "real")
    new_mode = "live" if is_real else "paper"
    new_testnet = not is_real

    # Update environment variables
    os.environ["MODE"] = new_mode
    os.environ["TESTNET"] = "false" if is_real else "true"

    # Persist to .env file if it exists
    try:
        env_file = Path(".env")
        if env_file.exists():
            lines = env_file.read_text().splitlines(keepends=True)
            updates = {
                "MODE": new_mode,
                "TESTNET": "false" if is_real else "true",
            }
            written = set()
            new_lines = []
            for l in lines:
                s = l.strip()
                if s and not s.startswith("#") and "=" in s:
                    k = s.split("=", 1)[0].strip()
                    if k in updates:
                        new_lines.append(f"{k}={updates[k]}\n")
                        written.add(k)
                        continue
                new_lines.append(l)
            for k, v in updates.items():
                if k not in written:
                    new_lines.append(f"{k}={v}\n")
            env_file.write_text("".join(new_lines))
    except Exception as e:
        print(f"[CONFIG ERROR] Failed updating .env file for trading mode: {e}")

    # Re-apply to active runner config
    was_running = runner.running
    if was_running:
        runner.stop()

    runner.cfg = BotConfig()
    runner.cfg.mode = new_mode
    runner.cfg.exchange.testnet = new_testnet

    if was_running:
        runner.start()

    mode_label = "Real Trading (LIVE)" if is_real else "Testnet / Paper Trading"
    return {
        "success": True,
        "message": f"Switched trading environment to {mode_label}.",
        "mode": runner.cfg.mode,
        "testnet": runner.cfg.exchange.testnet,
        "is_real": is_real,
    }


LAST_ANALYSIS_CACHE: Dict[str, Any] = {}
PUBLIC_EXCHANGE_CLIENTS: Dict[str, ExchangeClient] = {}


def get_public_exchange_client(exchange_id: str, symbol: str, timeframe: str) -> ExchangeClient:
    key = f"{exchange_id}_{symbol}_{timeframe}"
    if key not in PUBLIC_EXCHANGE_CLIENTS:
        ex_cfg = ExchangeConfig(exchange_id=exchange_id, symbol=symbol, timeframe=timeframe, testnet=False)
        PUBLIC_EXCHANGE_CLIENTS[key] = ExchangeClient(ex_cfg, mode="paper", fee_pct=0.001, slippage_pct=0.0005, starting_equity=1000.0)
    return PUBLIC_EXCHANGE_CLIENTS[key]


@app.get("/api/analysis")
def get_analysis(symbol: Optional[str] = None, timeframe: Optional[str] = None, exchange: Optional[str] = None):
    global LAST_ANALYSIS_CACHE
    req_symbol = symbol or runner.cfg.exchange.symbol or "BTC/USDT"
    req_timeframe = timeframe or runner.cfg.exchange.timeframe or "15m"
    req_exchange = exchange or runner.cfg.exchange.exchange_id or "binance"

    df = None
    used_exchange = req_exchange

    # 1. Try to fetch candles from requested exchange (live public OHLCV data)
    try:
        client = get_public_exchange_client(req_exchange, req_symbol, req_timeframe)
        df = client.fetch_ohlcv(limit=100)
    except Exception as e:
        logger.debug(f"Primary exchange {req_exchange} OHLCV unavailable ({e}), trying fallback...")

    # 2. Fallback to reliable major exchange if requested exchange fails or returns too few candles
    if df is None or len(df) < 30:
        for fb_id in ["binance", "coinbase", "kraken"]:
            if fb_id == req_exchange:
                continue
            try:
                client = get_public_exchange_client(fb_id, req_symbol, req_timeframe)
                fb_df = client.fetch_ohlcv(limit=100)
                if fb_df is not None and len(fb_df) >= 30:
                    df = fb_df
                    used_exchange = fb_id
                    break
            except Exception:
                continue

    # 3. If exchange calls fail, return last known cached analysis if available
    if (df is None or len(df) < 2) and LAST_ANALYSIS_CACHE:
        return LAST_ANALYSIS_CACHE

    if df is None or len(df) < 2:
        raise HTTPException(status_code=500, detail="Insufficient market data returned from exchange network")

    try:
        strat_cfg = runner.cfg.strategy
        df = add_indicators(df, strat_cfg)
        curr = df.iloc[-1]

        price = float(curr["close"])
        ema_fast_val = float(curr["ema_fast"]) if pd.notna(curr["ema_fast"]) else price
        ema_slow_val = float(curr["ema_slow"]) if pd.notna(curr["ema_slow"]) else price
        rsi_val = float(curr["rsi"]) if pd.notna(curr["rsi"]) else 50.0
        adx_val = float(curr["adx"]) if pd.notna(curr["adx"]) else 15.0
        atr_val = float(curr["atr"]) if pd.notna(curr["atr"]) else 0.0
        swing_h = float(curr["swing_high"]) if pd.notna(curr["swing_high"]) else None
        swing_l = float(curr["swing_low"]) if pd.notna(curr["swing_low"]) else None
        fib_382 = float(curr["fib_382"]) if pd.notna(curr["fib_382"]) else None
        fib_500 = float(curr["fib_500"]) if pd.notna(curr["fib_500"]) else None
        fib_618 = float(curr["fib_618"]) if pd.notna(curr["fib_618"]) else None

        near_fib = _near_fib_support(curr, strat_cfg)
        signal = generate_signal(df, strat_cfg, position_open=bool(runner.bot and runner.bot.position))

        recent_candles = []
        for _, row in df.tail(30).iterrows():
            recent_candles.append({
                "ts": row["ts"].strftime("%H:%M") if hasattr(row["ts"], "strftime") else str(row["ts"]),
                "open": round(float(row["open"]), 2),
                "high": round(float(row["high"]), 2),
                "low": round(float(row["low"]), 2),
                "close": round(float(row["close"]), 2),
                "volume": round(float(row["volume"]), 2),
                "ema_fast": round(float(row["ema_fast"]), 2) if pd.notna(row["ema_fast"]) else round(float(row["close"]), 2),
                "ema_slow": round(float(row["ema_slow"]), 2) if pd.notna(row["ema_slow"]) else round(float(row["close"]), 2),
            })

        result = {
            "symbol": req_symbol,
            "exchange": used_exchange,
            "timeframe": req_timeframe,
            "strategy": strat_cfg.strategy_name,
            "strategy_name": next((s["name"] for s in STRATEGIES if s["id"] == strat_cfg.strategy_name), strat_cfg.strategy_name),
            "price": round(price, 2),
            "signal": signal.value,
            "ema_fast": round(ema_fast_val, 2),
            "ema_slow": round(ema_slow_val, 2),
            "trend": "BULLISH" if ema_fast_val > ema_slow_val else "BEARISH",
            "rsi": round(rsi_val, 2),
            "rsi_status": "OVERBOUGHT" if rsi_val >= strat_cfg.rsi_overbought else ("OVERSOLD" if rsi_val <= 30 else "NEUTRAL"),
            "adx": round(adx_val, 2),
            "adx_status": "TRENDING" if adx_val >= strat_cfg.adx_trend_threshold else "CHOPPY / RANGING",
            "atr": round(atr_val, 2),
            "fib": {
                "swing_high": round(swing_h, 2) if swing_h else None,
                "swing_low": round(swing_l, 2) if swing_l else None,
                "fib_382": round(fib_382, 2) if fib_382 else None,
                "fib_500": round(fib_500, 2) if fib_500 else None,
                "fib_618": round(fib_618, 2) if fib_618 else None,
                "near_fib_support": near_fib,
            },
            "candles": recent_candles,
        }
        LAST_ANALYSIS_CACHE = result
        return result
    except Exception as e:
        logger.exception("Analysis calculation failed")
        if LAST_ANALYSIS_CACHE:
            return LAST_ANALYSIS_CACHE
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/backtest")
def run_backtest_endpoint(req: BacktestRequest):
    try:
        cfg = BotConfig()
        cfg.exchange.exchange_id = req.exchange
        cfg.exchange.symbol = req.symbol
        cfg.exchange.timeframe = req.timeframe
        cfg.starting_equity = req.starting_equity
        cfg.strategy.use_fib_filter = req.use_fib_filter
        cfg.risk.risk_per_trade_pct = req.risk_per_trade_pct

        df = fetch_historical(req.exchange, req.symbol, req.timeframe, req.days)
        if df is None or len(df) < 60:
            raise HTTPException(status_code=400, detail="Insufficient historical bars returned for backtest")

        results = run_backtest(df, cfg)
        return {
            "success": True,
            "params": req.dict(),
            "bars_analyzed": len(df),
            "results": results,
        }
    except Exception as e:
        logger.exception("Backtest execution failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/logs")
def get_logs(lines: int = 50):
    log_file = Path("/tmp/bot.log") if (os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME")) else Path("bot.log")
    if not log_file.exists() or log_file.stat().st_size == 0:
        if runner.running:
            return {"logs": [f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] [INFO] bot: Engine is running. Waiting for cycle step..."]}
        return {"logs": ["No logs recorded yet. Start the bot to generate activity."]}
    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            all_lines = [line.strip() for line in f.readlines() if line.strip()]
            if not all_lines:
                if runner.running:
                    return {"logs": [f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] [INFO] bot: Engine is running. Waiting for cycle step..."]}
                return {"logs": ["No logs recorded yet. Start the bot to generate activity."]}
            return {"logs": all_lines[-lines:]}
    except Exception as e:
        return {"logs": [f"Error reading log file: {e}"]}


# ============================================================================
# HTML PAGES & DASHBOARD
# ============================================================================
@app.get("/activate", response_class=HTMLResponse)
def activate_page(token: str = Query(...)):
    res = AuthService.activate_account(token)
    success = res["success"]
    status_icon = "fa-check-circle text-emerald-400" if success else "fa-times-circle text-rose-400"
    title = "Account Activated!" if success else "Activation Failed"
    session_token = res.get("session_token", "")

    auto_script = ""
    if success and session_token:
        auto_script = f"""
        <script>
          try {{
            localStorage.setItem('bot_auth_token', '{session_token}');
            document.cookie = "bot_auth_token=" + encodeURIComponent('{session_token}') + "; path=/; max-age=604800; SameSite=Lax; Secure";
          }} catch (e) {{}}

          let count = 3;
          const countdownEl = document.getElementById('countdown');
          const timer = setInterval(() => {{
            count--;
            if (countdownEl) countdownEl.innerText = count;
            if (count <= 0) {{
              clearInterval(timer);
              window.location.href = '/';
            }}
          }}, 1000);
        </script>
        """

    html = f"""<!DOCTYPE html>
    <html lang="en" class="dark">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>{title}</title>
      <script src="https://cdn.tailwindcss.com"></script>
      <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    </head>
    <body class="bg-slate-950 text-slate-100 flex items-center justify-center min-h-screen p-4 font-sans">
      <div class="max-w-md w-full bg-slate-900 border border-slate-800 rounded-2xl p-6 sm:p-8 text-center shadow-2xl space-y-4">
        <i class="fa-solid {status_icon} text-5xl"></i>
        <h1 class="text-2xl font-bold">{title}</h1>
        <p class="text-sm text-slate-300 leading-relaxed">{res['message']}</p>
        {f'<p class="text-xs text-emerald-400 font-medium"><i class="fa-solid fa-spinner fa-spin mr-1.5"></i> You are logged in! Entering trading dashboard in <span id="countdown">3</span>s...</p>' if success else ''}
        <div class="pt-3 space-y-2">
          <a href="/" class="inline-block w-full py-3 {'bg-emerald-600 hover:bg-emerald-500' if success else 'bg-slate-800 hover:bg-slate-700 border border-slate-700'} font-semibold text-white rounded-xl shadow transition">
            {'Enter Trading Dashboard' if success else 'Return to Home'}
          </a>
          {'' if success else '<a href="/?action=resend" class="inline-block w-full py-2.5 bg-amber-600 hover:bg-amber-500 font-semibold text-white rounded-xl shadow transition text-xs"><i class="fa-solid fa-paper-plane mr-1.5"></i> Request New Verification Link</a>'}
        </div>
      </div>
      {auto_script}
    </body>
    </html>"""
    return HTMLResponse(content=html, status_code=200)


@app.get("/magic-login", response_class=HTMLResponse)
def magic_login_page(token: str = Query(...)):
    res = AuthService.magic_login(token)
    success = res["success"]
    status_icon = "fa-check-circle text-emerald-400" if success else "fa-times-circle text-rose-400"
    title = "Signed In Successfully!" if success else "Sign-In Failed"
    session_token = res.get("session_token", "")

    auto_script = ""
    if success and session_token:
        auto_script = f"""
        <script>
          try {{
            localStorage.setItem('bot_auth_token', '{session_token}');
            document.cookie = "bot_auth_token=" + encodeURIComponent('{session_token}') + "; path=/; max-age=2592000; SameSite=Lax; Secure";
          }} catch (e) {{}}

          let count = 2;
          const countdownEl = document.getElementById('countdown');
          const timer = setInterval(() => {{
            count--;
            if (countdownEl) countdownEl.innerText = count;
            if (count <= 0) {{
              clearInterval(timer);
              window.location.href = '/';
            }}
          }}, 1000);
        </script>
        """

    html = f"""<!DOCTYPE html>
    <html lang="en" class="dark">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>{title}</title>
      <script src="https://cdn.tailwindcss.com"></script>
      <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    </head>
    <body class="bg-slate-950 text-slate-100 flex items-center justify-center min-h-screen p-4 font-sans">
      <div class="max-w-md w-full bg-slate-900 border border-slate-800 rounded-2xl p-6 sm:p-8 text-center shadow-2xl space-y-4">
        <i class="fa-solid {status_icon} text-5xl"></i>
        <h1 class="text-2xl font-bold">{title}</h1>
        <p class="text-sm text-slate-300 leading-relaxed">{res['message']}</p>
        {f'<p class="text-xs text-emerald-400 font-medium"><i class="fa-solid fa-spinner fa-spin mr-1.5"></i> Entering trading dashboard in <span id="countdown">2</span>s...</p>' if success else ''}
        <div class="pt-3 space-y-2">
          <a href="/" class="inline-block w-full py-3 {'bg-emerald-600 hover:bg-emerald-500' if success else 'bg-slate-800 hover:bg-slate-700 border border-slate-700'} font-semibold text-white rounded-xl shadow transition">
            {'Enter Trading Dashboard' if success else 'Return to Home'}
          </a>
        </div>
      </div>
      {auto_script}
    </body>
    </html>"""
    return HTMLResponse(content=html, status_code=200)


@app.get("/reset-password", response_class=HTMLResponse)
def reset_password_page(token: str = Query(...)):
    html = f"""<!DOCTYPE html>
    <html lang="en" class="dark">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Reset Your Password</title>
      <script src="https://cdn.tailwindcss.com"></script>
      <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    </head>
    <body class="bg-slate-950 text-slate-100 flex items-center justify-center min-h-screen p-4 font-sans">
      <div class="max-w-md w-full bg-slate-900 border border-slate-800 rounded-2xl p-5 sm:p-8 shadow-2xl space-y-5">
        <div class="text-center">
          <i class="fa-solid fa-key text-4xl text-amber-400 mb-2"></i>
          <h1 class="text-2xl font-bold">Set New Password</h1>
          <p class="text-xs text-slate-400 mt-1">Please enter your new account password below.</p>
        </div>
        <form onsubmit="handleReset(event)" class="space-y-4 text-xs">
          <input type="hidden" id="reset-token" value="{token}">
          <div>
            <label class="block text-slate-400 mb-1">New Password (min 6 characters)</label>
            <div class="relative">
              <input type="password" id="new-pw" required minlength="6" class="w-full bg-slate-800 border border-slate-700 rounded-lg pl-3 pr-10 py-2 text-white focus:outline-none focus:border-amber-500">
              <button type="button" onclick="togglePasswordVisibility('new-pw', 'new-pw-eye')" class="absolute inset-y-0 right-0 pr-3 flex items-center text-slate-400 hover:text-slate-200 transition" tabindex="-1">
                <i class="fa-solid fa-eye text-xs" id="new-pw-eye"></i>
              </button>
            </div>
          </div>
          <div>
            <label class="block text-slate-400 mb-1">Confirm New Password</label>
            <div class="relative">
              <input type="password" id="confirm-pw" required minlength="6" class="w-full bg-slate-800 border border-slate-700 rounded-lg pl-3 pr-10 py-2 text-white focus:outline-none focus:border-amber-500">
              <button type="button" onclick="togglePasswordVisibility('confirm-pw', 'confirm-pw-eye')" class="absolute inset-y-0 right-0 pr-3 flex items-center text-slate-400 hover:text-slate-200 transition" tabindex="-1">
                <i class="fa-solid fa-eye text-xs" id="confirm-pw-eye"></i>
              </button>
            </div>
          </div>
          <div id="reset-msg" class="hidden text-xs p-2 rounded"></div>
          <button type="submit" class="w-full py-2.5 bg-amber-600 hover:bg-amber-500 font-bold text-white rounded-lg shadow transition">
            Update Password
          </button>
        </form>
        <div class="text-center pt-2">
          <a href="/" class="text-xs text-slate-400 hover:text-white">Back to Dashboard</a>
        </div>
      </div>
      <script>
        async function handleReset(e) {{
          e.preventDefault();
          const pw = document.getElementById('new-pw').value;
          const cpw = document.getElementById('confirm-pw').value;
          const msg = document.getElementById('reset-msg');
          if (pw !== cpw) {{
            msg.className = 'text-xs p-2 rounded bg-rose-950 text-rose-300 block';
            msg.innerText = 'Passwords do not match';
            return;
          }}
          const res = await fetch('/api/auth/reset-password', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{ token: document.getElementById('reset-token').value, new_password: pw }})
          }});
          const data = await res.json();
          if (res.ok && data.success) {{
            msg.className = 'text-xs p-2 rounded bg-emerald-950 text-emerald-300 block';
            msg.innerText = data.message + ' Redirecting to login...';
            setTimeout(() => window.location.href = '/', 2000);
          }} else {{
            msg.className = 'text-xs p-2 rounded bg-rose-950 text-rose-300 block';
            msg.innerText = data.detail || data.message || 'Reset failed';
          }}
        }}

        function togglePasswordVisibility(inputId, iconId) {{
          const input = document.getElementById(inputId);
          const icon = document.getElementById(iconId);
          if (!input || !icon) return;
          if (input.type === 'password') {{
            input.type = 'text';
            icon.classList.remove('fa-eye');
            icon.classList.add('fa-eye-slash');
          }} else {{
            input.type = 'password';
            icon.classList.remove('fa-eye-slash');
            icon.classList.add('fa-eye');
          }}
        }}
      </script>
    </body>
    </html>"""
    return HTMLResponse(content=html, status_code=200)


# Main Dashboard HTML with integrated Auth & Exchange Key manager
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
  <title>Crypto Trading Bot - Dashboard & Exchange Keys</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            brand: { 50: '#f0fdf4', 500: '#22c55e', 600: '#16a34a' },
            darkbg: '#0b1120',
            cardbg: '#162032',
            bordercol: '#283548'
          }
        }
      }
    }
  </script>
  <style>
    html, body {
      overflow-x: hidden;
      max-width: 100vw;
      -webkit-tap-highlight-color: transparent;
    }
    .safe-area-bottom {
      padding-bottom: max(0.5rem, env(safe-area-inset-bottom));
    }
    .pb-safe-main {
      padding-bottom: max(5.5rem, calc(4.5rem + env(safe-area-inset-bottom)));
    }
    button, a {
      touch-action: manipulation;
    }
    ::-webkit-scrollbar {
      width: 5px;
      height: 5px;
    }
    ::-webkit-scrollbar-track {
      background: #0b1120;
    }
    ::-webkit-scrollbar-thumb {
      background: #334155;
      border-radius: 9999px;
    }
    #log-container {
      word-break: break-all;
      white-space: pre-wrap;
    }
    @keyframes slideDown {
      from { opacity: 0; transform: translateY(-6px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .animate-slideDown {
      animation: slideDown 0.2s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
  </style>
</head>
<body class="bg-darkbg text-slate-100 min-h-screen font-sans antialiased selection:bg-emerald-500/30 selection:text-emerald-200">

  <!-- Top Navigation -->
  <header class="border-b border-bordercol bg-cardbg/95 backdrop-blur-md sticky top-0 z-40">
    <div class="max-w-7xl mx-auto px-3 sm:px-4 py-2 sm:py-2.5">
      <div class="flex items-center justify-between gap-2">
        <!-- Brand & Title -->
        <div class="flex items-center space-x-2.5 sm:space-x-3 min-w-0">
          <div class="h-8 w-8 sm:h-9 sm:w-9 rounded-lg bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center text-emerald-400 font-bold shrink-0">
            <i class="fa-solid fa-chart-line text-sm sm:text-base"></i>
          </div>
          <div class="min-w-0">
            <h1 class="text-sm sm:text-base font-bold tracking-tight text-white flex items-center gap-1.5 sm:gap-2 truncate">
              <span class="truncate">Crypto Bot</span>
              <span id="badge-mode" class="text-[9px] sm:text-[10px] px-1.5 sm:px-2 py-0.5 rounded-full font-bold bg-blue-500/20 text-blue-400 border border-blue-500/30 shrink-0">PAPER</span>
            </h1>
            <p class="text-[10px] sm:text-[11px] text-slate-400 truncate hidden xs:block sm:block">EMA Cross + RSI + ADX + Fib Retracement + Risk Layer</p>
          </div>
        </div>

        <!-- Desktop Quick Action Controls (Hidden on mobile < md) -->
        <div class="hidden md:flex items-center space-x-2 text-xs">
          <!-- Bot Status Indicator -->
          <div class="flex items-center gap-2 bg-slate-900/80 px-2.5 py-1.5 rounded-lg border border-bordercol">
            <span class="inline-block w-2 h-2 rounded-full" id="status-dot"></span>
            <span id="status-text" class="font-semibold text-slate-300">IDLE</span>
          </div>

          <!-- Start / Stop Button -->
          <button id="btn-toggle" onclick="toggleBot()" class="px-3.5 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold rounded-lg shadow transition flex items-center gap-1.5">
            <i class="fa-solid fa-play" id="btn-toggle-icon"></i>
            <span id="btn-toggle-text">Start Bot</span>
          </button>

          <!-- Kill Switch -->
          <button onclick="triggerKillSwitch()" class="px-2.5 py-1.5 bg-red-600/20 hover:bg-red-600 text-red-400 hover:text-white border border-red-500/40 font-semibold rounded-lg transition flex items-center gap-1" title="Emergency Stop">
            <i class="fa-solid fa-skull-crossbones"></i>
          </button>

          <!-- Bot Settings Button -->
          <button type="button" onclick="openSettingsModal()" class="px-2.5 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white border border-bordercol font-semibold rounded-lg transition flex items-center gap-1.5" title="Configure Strategy, Timeframe, and Auto vs Manual Trading">
            <i class="fa-solid fa-gear text-amber-400"></i>
            <span>Settings</span>
          </button>

          <!-- Email Settings Button -->
          <button type="button" onclick="openEmailSettingsModal()" class="px-2.5 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white border border-bordercol font-semibold rounded-lg transition flex items-center gap-1.5" title="Configure Email & Verification Credentials">
            <span id="email-status-dot" class="w-2 h-2 rounded-full bg-slate-500 inline-block"></span>
            <i class="fa-solid fa-envelope text-sky-400"></i>
            <span>Email</span>
          </button>

          <!-- Auth User Menu Button -->
          <div id="auth-buttons" class="flex items-center gap-2">
            <button onclick="openAuthModal('login')" class="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-bordercol font-semibold rounded-lg transition">
              Login
            </button>
            <button onclick="openAuthModal('register')" class="px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold rounded-lg shadow transition">
              Register
            </button>
          </div>
          <div id="user-badge" class="hidden items-center gap-2 bg-slate-900 border border-bordercol px-2.5 py-1 rounded-lg">
            <i class="fa-solid fa-user-circle text-emerald-400"></i>
            <span id="user-name-display" class="font-semibold text-slate-200"></span>
            <button onclick="confirmDeleteAccount()" class="text-slate-500 hover:text-rose-400 text-xs ml-1" title="Delete My Account"><i class="fa-solid fa-trash-can"></i></button>
            <button onclick="logout()" class="text-slate-400 hover:text-rose-400 ml-1" title="Logout"><i class="fa-solid fa-arrow-right-from-bracket"></i></button>
          </div>
        </div>

        <!-- Mobile Header Action Controls (Visible only on < md) -->
        <div class="flex md:hidden items-center gap-1.5">
          <!-- Compact Status Badge -->
          <div class="flex items-center gap-1.5 bg-slate-900/90 px-2 py-1 rounded-lg border border-bordercol text-[11px]">
            <span class="inline-block w-2 h-2 rounded-full" id="status-dot-mobile"></span>
            <span id="status-text-mobile" class="font-semibold text-slate-300">IDLE</span>
          </div>

          <!-- Quick Mobile Start/Stop Toggle Button -->
          <button id="btn-toggle-mobile" onclick="toggleBot()" class="px-2.5 py-1.5 bg-emerald-600 active:bg-emerald-700 text-white text-xs font-semibold rounded-lg shadow transition flex items-center gap-1">
            <i class="fa-solid fa-play text-[10px]" id="btn-toggle-icon-mobile"></i>
            <span id="btn-toggle-text-mobile">Start</span>
          </button>

          <!-- Mobile Menu Hamburger Button -->
          <button id="btn-mobile-menu" onclick="toggleMobileMenu()" class="p-1.5 px-2.5 bg-slate-800 hover:bg-slate-700 active:bg-slate-900 text-slate-200 border border-bordercol rounded-lg transition text-sm focus:outline-none flex items-center justify-center" aria-label="Toggle navigation menu">
            <i class="fa-solid fa-bars" id="mobile-menu-icon"></i>
          </button>
        </div>
      </div>

      <!-- Mobile Dropdown Navigation Drawer -->
      <div id="mobile-menu-panel" class="hidden md:hidden pt-3 mt-2 border-t border-bordercol/80 space-y-2.5 text-xs animate-slideDown">
        <!-- Emergency Stop Banner Button -->
        <div>
          <button onclick="triggerKillSwitch()" class="w-full py-2.5 bg-red-600/20 active:bg-red-600 text-red-300 hover:text-white border border-red-500/40 font-semibold rounded-lg transition flex items-center justify-center gap-2">
            <i class="fa-solid fa-skull-crossbones"></i>
            <span>Emergency Kill Switch</span>
          </button>
        </div>

        <!-- Settings & Email Shortcut Buttons -->
        <div class="grid grid-cols-2 gap-2">
          <button type="button" onclick="openSettingsModal(); toggleMobileMenu();" class="py-2.5 px-3 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-bordercol font-semibold rounded-lg transition flex items-center justify-center gap-1.5">
            <i class="fa-solid fa-gear text-amber-400"></i>
            <span>Bot Settings</span>
          </button>
          <button type="button" onclick="openEmailSettingsModal(); toggleMobileMenu();" class="py-2.5 px-3 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-bordercol font-semibold rounded-lg transition flex items-center justify-center gap-1.5">
            <span id="email-status-dot-mobile" class="w-2 h-2 rounded-full bg-slate-500 inline-block"></span>
            <i class="fa-solid fa-envelope text-sky-400"></i>
            <span>Email & SMTP</span>
          </button>
        </div>

        <!-- Auth Controls (Mobile Drawer) -->
        <div id="auth-buttons-mobile" class="grid grid-cols-2 gap-2 pt-2 border-t border-bordercol/60">
          <button onclick="openAuthModal('login'); toggleMobileMenu();" class="py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-bordercol font-semibold rounded-lg transition text-center flex items-center justify-center gap-1.5">
            <i class="fa-solid fa-right-to-bracket text-xs"></i>
            <span>Login</span>
          </button>
          <button onclick="openAuthModal('register'); toggleMobileMenu();" class="py-2.5 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold rounded-lg shadow transition text-center flex items-center justify-center gap-1.5">
            <i class="fa-solid fa-user-plus text-xs"></i>
            <span>Register</span>
          </button>
        </div>

        <div id="user-badge-mobile" class="hidden flex-col gap-2 p-2.5 bg-slate-900 rounded-lg border border-bordercol">
          <div class="flex items-center justify-between">
            <div class="flex items-center gap-2">
              <i class="fa-solid fa-user-circle text-emerald-400 text-base"></i>
              <span id="user-name-display-mobile" class="font-semibold text-slate-200"></span>
            </div>
            <button onclick="logout(); toggleMobileMenu();" class="text-xs px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-rose-300 rounded border border-bordercol flex items-center gap-1">
              <i class="fa-solid fa-arrow-right-from-bracket"></i> Logout
            </button>
          </div>
          <div class="pt-1.5 border-t border-bordercol/50 flex justify-end">
            <button onclick="confirmDeleteAccount()" class="text-[11px] text-rose-400 hover:text-rose-300 underline flex items-center gap-1">
              <i class="fa-solid fa-trash-can"></i> Delete My Account
            </button>
          </div>
        </div>
      </div>
    </div>
  </header>

  <!-- Main Container -->
  <main class="max-w-7xl mx-auto px-3 sm:px-4 py-4 sm:py-5 space-y-4 sm:space-y-5 pb-safe-main md:pb-6">

    <!-- KPI Cards Grid -->
    <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5 sm:gap-3">
      <div class="bg-cardbg border border-bordercol rounded-xl p-2.5 sm:p-3">
        <div class="text-slate-400 text-[10px] sm:text-[11px] font-medium">Portfolio Equity</div>
        <div class="text-base sm:text-lg font-bold mt-0.5 text-white truncate" id="val-equity">$1,000.00</div>
        <div class="text-[9px] sm:text-[10px] text-slate-400 mt-0.5 truncate">Peak: <span id="val-peak">$1,000.00</span></div>
      </div>

      <div class="bg-cardbg border border-bordercol rounded-xl p-2.5 sm:p-3">
        <div class="text-slate-400 text-[10px] sm:text-[11px] font-medium" id="val-cash-label">Paper Cash</div>
        <div class="text-base sm:text-lg font-bold mt-0.5 text-emerald-400 truncate" id="val-cash">$1,000.00</div>
        <div class="text-[9px] sm:text-[10px] text-slate-400 mt-0.5 truncate">Position: <span id="val-pos">0.000000</span></div>
      </div>

      <div class="bg-cardbg border border-bordercol rounded-xl p-2.5 sm:p-3">
        <div class="text-slate-400 text-[10px] sm:text-[11px] font-medium">Max Drawdown</div>
        <div class="text-base sm:text-lg font-bold mt-0.5 text-slate-200 truncate" id="val-drawdown">0.0%</div>
        <div class="text-[9px] sm:text-[10px] text-slate-400 mt-0.5 truncate">Limit: 15.0% Kill</div>
      </div>

      <div class="bg-cardbg border border-bordercol rounded-xl p-2.5 sm:p-3">
        <div class="text-slate-400 text-[10px] sm:text-[11px] font-medium">Daily Loss</div>
        <div class="text-base sm:text-lg font-bold mt-0.5 text-slate-200 truncate" id="val-daily-loss">0.0%</div>
        <div class="text-[9px] sm:text-[10px] text-slate-400 mt-0.5 truncate">Limit: 3.0% Circuit</div>
      </div>

      <div class="bg-cardbg border border-bordercol rounded-xl p-2.5 sm:p-3">
        <div class="text-slate-400 text-[10px] sm:text-[11px] font-medium">Loss Streak</div>
        <div class="text-base sm:text-lg font-bold mt-0.5 text-slate-200 truncate" id="val-streak">0 / 4</div>
        <div class="text-[9px] sm:text-[10px] text-slate-400 mt-0.5 truncate">Cooloff at 4</div>
      </div>

      <div class="bg-cardbg border border-bordercol rounded-xl p-2.5 sm:p-3">
        <div class="text-slate-400 text-[10px] sm:text-[11px] font-medium">Live Signal</div>
        <div class="text-base sm:text-lg font-bold mt-0.5 text-yellow-400 truncate" id="val-signal">HOLD</div>
        <div class="text-[9px] sm:text-[10px] text-slate-400 mt-0.5 truncate" id="val-symbol-tag">BTC/USDT 15m</div>
      </div>
    </div>

    <!-- Active Position Banner -->
    <div id="position-banner" class="hidden bg-emerald-950/40 border border-emerald-500/40 rounded-xl p-3 sm:p-3.5 flex flex-wrap items-center justify-between gap-2">
      <div class="flex items-center gap-2.5 sm:gap-3 min-w-0">
        <div class="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse shrink-0"></div>
        <div class="min-w-0">
          <span class="font-bold text-emerald-400 text-xs">ACTIVE POSITION:</span>
          <span class="text-xs text-slate-300 ml-1.5 break-all" id="pos-details"></span>
        </div>
      </div>
      <span class="text-[10px] text-emerald-300 font-semibold px-2 py-0.5 bg-emerald-900/60 rounded shrink-0">IN MARKET</span>
    </div>

    <!-- ========================================================================= -->
    <!-- EXCHANGE API KEYS CONFIGURATION SECTION                                   -->
    <!-- ========================================================================= -->
    <div class="bg-cardbg border border-bordercol rounded-xl p-5 space-y-4">
      <div class="flex flex-col sm:flex-row sm:items-center justify-between border-b border-bordercol pb-3 gap-2">
        <div class="flex items-center gap-2">
          <i class="fa-solid fa-key text-amber-400"></i>
          <div>
            <h2 class="font-bold text-sm text-white">Exchange API Keys & Credential Management</h2>
            <p class="text-[11px] text-slate-400">Select an exchange first, then enter your API credentials with trade permissions.</p>
          </div>
        </div>
        <span class="self-start sm:self-auto text-[10px] px-2 py-1 bg-amber-500/10 text-amber-400 border border-amber-500/30 rounded font-medium shrink-0">
          <i class="fa-solid fa-shield-halved"></i> Trade-only keys (No withdrawal)
        </span>
      </div>

      <!-- LOCKED STATE: Shown when user is not logged in -->
      <div id="api-keys-locked" class="bg-slate-900/90 border border-bordercol/80 rounded-xl p-5 sm:p-8 text-center space-y-3">
        <div class="inline-flex p-3 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20 mb-1">
          <i class="fa-solid fa-lock text-2xl"></i>
        </div>
        <h3 class="text-base font-bold text-white">Exchange API Keys are Protected</h3>
        <p class="text-xs text-slate-400 max-w-md mx-auto">
          You must create an account and log in before you can access, configure, or test exchange API keys.
        </p>
        <div class="flex flex-col sm:flex-row items-center justify-center gap-2.5 pt-2">
          <button onclick="openAuthModal('login')" class="w-full sm:w-auto px-4 py-2.5 sm:py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-bordercol text-xs font-semibold rounded-lg transition flex items-center justify-center gap-1.5">
            <i class="fa-solid fa-right-to-bracket"></i> Log In
          </button>
          <button onclick="openAuthModal('register')" class="w-full sm:w-auto px-5 py-2.5 sm:py-2 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold rounded-lg shadow transition flex items-center justify-center gap-1.5">
            <i class="fa-solid fa-user-plus"></i> Create Account
          </button>
        </div>
      </div>

      <!-- UNLOCKED STATE: Shown only after account creation and login -->
      <div id="api-keys-unlocked" class="hidden space-y-4">
        <!-- Mode Switcher Card -->
        <div class="p-3 sm:p-3.5 bg-slate-900/90 rounded-xl border border-bordercol flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 shadow-sm">
          <div class="flex items-center gap-3">
            <div id="env-icon" class="w-9 h-9 rounded-lg bg-blue-500/20 text-blue-400 flex items-center justify-center font-bold text-base shrink-0">
              <i class="fa-solid fa-flask"></i>
            </div>
            <div>
              <div class="text-xs font-bold text-white flex items-center gap-2">
                Active Environment: <span id="env-mode-title" class="text-blue-400 font-bold">TESTNET / SIMULATION</span>
              </div>
              <p id="env-mode-desc" class="text-[11px] text-slate-400">Safe simulated execution. Zero real funds at risk.</p>
            </div>
          </div>
          <div class="flex flex-wrap sm:flex-nowrap items-center gap-2 w-full sm:w-auto">
            <button type="button" onclick="switchTradingMode('testnet')" id="env-btn-testnet" class="flex-1 sm:flex-none justify-center px-3 py-2 sm:py-1.5 bg-blue-600 text-white text-xs font-bold rounded-lg shadow transition flex items-center gap-1.5">
              <i class="fa-solid fa-flask"></i> Testnet Mode
            </button>
            <button type="button" onclick="switchTradingMode('real')" id="env-btn-real" class="flex-1 sm:flex-none justify-center px-3 py-2 sm:py-1.5 bg-slate-800 text-slate-300 hover:text-white border border-slate-700 text-xs font-bold rounded-lg transition flex items-center gap-1.5">
              <i class="fa-solid fa-bolt text-rose-400"></i> Real Trading (LIVE)
            </button>
            <button type="button" onclick="openSettingsModal()" class="flex-1 sm:flex-none justify-center px-3 py-2 sm:py-1.5 bg-indigo-600/30 hover:bg-indigo-600 text-indigo-300 hover:text-white border border-indigo-500/40 text-xs font-semibold rounded-lg transition flex items-center gap-1.5" title="Configure Strategy, Timeframe, and Auto vs Manual Trading">
              <i class="fa-solid fa-gear"></i> Settings
            </button>
          </div>
        </div>

        <form id="exchange-key-form" onsubmit="saveExchangeKey(event)" class="grid grid-cols-1 md:grid-cols-4 gap-3 sm:gap-4 text-xs">
          <!-- 1. Select Exchange -->
          <div>
            <label class="block text-slate-300 font-semibold mb-1">1. Choose Exchange</label>
            <select id="key-exchange-select" onchange="onExchangeChange()" class="w-full bg-slate-900 border border-bordercol rounded-lg px-3 py-2 text-white font-medium focus:border-emerald-500">
              <!-- Populated via JS -->
            </select>
            <div id="exchange-pass-hint" class="text-[10px] text-slate-400 mt-1"></div>
          </div>

          <!-- 2. API Key -->
          <div>
            <label class="block text-slate-300 font-semibold mb-1">2. API Key</label>
            <input type="text" id="key-api-key" required placeholder="Paste API Key" class="w-full bg-slate-900 border border-bordercol rounded-lg px-3 py-2 text-white focus:border-emerald-500">
          </div>

          <!-- 3. API Secret -->
          <div>
            <label class="block text-slate-300 font-semibold mb-1">3. API Secret</label>
            <div class="relative">
              <input type="password" id="key-api-secret" required placeholder="Paste API Secret" class="w-full bg-slate-900 border border-bordercol rounded-lg pl-3 pr-9 py-2 text-white focus:border-emerald-500">
              <button type="button" onclick="togglePasswordVisibility('key-api-secret', 'key-api-secret-eye')" class="absolute inset-y-0 right-0 pr-2.5 flex items-center text-slate-400 hover:text-slate-200 transition" title="Toggle visibility" tabindex="-1">
                <i class="fa-solid fa-eye text-xs" id="key-api-secret-eye"></i>
              </button>
            </div>
          </div>

          <!-- 4. Passphrase (Dynamic) -->
          <div id="passphrase-container">
            <label class="block text-slate-300 font-semibold mb-1" id="passphrase-label">4. Passphrase</label>
            <div class="relative">
              <input type="password" id="key-api-passphrase" placeholder="Passphrase (if needed)" class="w-full bg-slate-900 border border-bordercol rounded-lg pl-3 pr-9 py-2 text-white focus:border-emerald-500">
              <button type="button" onclick="togglePasswordVisibility('key-api-passphrase', 'key-api-passphrase-eye')" class="absolute inset-y-0 right-0 pr-2.5 flex items-center text-slate-400 hover:text-slate-200 transition" title="Toggle visibility" tabindex="-1">
                <i class="fa-solid fa-eye text-xs" id="key-api-passphrase-eye"></i>
              </button>
            </div>
          </div>

          <!-- Options & Action Buttons -->
          <div class="md:col-span-4 flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 pt-2 border-t border-bordercol/50">
            <div class="flex items-center gap-2">
              <label class="flex items-center gap-2 cursor-pointer text-xs">
                <input type="checkbox" id="key-testnet" class="rounded bg-slate-900 border-bordercol text-emerald-500">
                <span id="key-testnet-label" class="text-slate-300 font-medium">Use Testnet / Sandbox Mode (Check only if using Testnet keys)</span>
              </label>
            </div>

            <div class="flex flex-col sm:flex-row items-stretch sm:items-center gap-2 w-full sm:w-auto">
              <button type="button" onclick="testExchangeConnection()" id="btn-test-key" class="w-full sm:w-auto px-4 py-2.5 sm:py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-bordercol font-semibold rounded-lg transition flex items-center justify-center gap-1.5">
                <i class="fa-solid fa-plug"></i> Test Connection
              </button>
              <button type="submit" id="btn-save-key" class="w-full sm:w-auto px-5 py-2.5 sm:py-2 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold rounded-lg shadow transition flex items-center justify-center gap-1.5">
                <i class="fa-solid fa-floppy-disk"></i> Save & Apply Credentials
              </button>
            </div>
          </div>
        </form>

        <!-- Connection Test Result Banner -->
        <div id="key-test-result" class="hidden text-xs p-3 rounded-lg border"></div>

        <!-- Currently Saved Keys for User -->
        <div id="saved-keys-section" class="pt-2">
          <div class="text-xs font-semibold text-slate-400 mb-2">Active Exchange Credentials</div>
          <div id="saved-keys-list" class="space-y-1.5 text-xs">
            <div class="text-slate-500 italic">No custom keys saved yet. Bot currently defaults to Binance public data / paper mode.</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Live Analysis & Backtester 2-col layout -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-5">
      <!-- Live Market Analysis (2 cols) -->
      <div class="lg:col-span-2 bg-cardbg border border-bordercol rounded-xl p-5 space-y-4">
        <div class="flex items-center justify-between border-b border-bordercol pb-3 flex-wrap gap-2">
          <div class="flex items-center gap-2">
            <i class="fa-solid fa-radar text-emerald-400"></i>
            <div>
              <h2 class="font-bold text-sm text-white">Live Market Analysis & Technical Indicators</h2>
              <div class="text-[11px] text-slate-400 flex items-center gap-1.5 flex-wrap">
                <span id="active-strat-tag" class="text-emerald-400 font-semibold">Strategy: EMA Cross + Fib</span>
                <span class="text-slate-600">•</span>
                <span id="active-tf-tag" class="text-sky-300 font-semibold">Timeframe: 15m</span>
                <span class="text-slate-600">•</span>
                <span id="active-mode-tag" class="text-purple-300 font-semibold">Auto Trading</span>
              </div>
            </div>
          </div>
          <div class="flex items-center gap-2">
            <button onclick="refreshAnalysis()" class="text-xs text-slate-300 hover:text-white transition flex items-center gap-1 px-2.5 py-1.5 bg-slate-800 hover:bg-slate-700 rounded-lg border border-bordercol">
              <i class="fa-solid fa-rotate"></i> Refresh
            </button>
          </div>
        </div>

        <!-- Manual Trading Execution Panel (Visible when Execution Mode is Manual) -->
        <div id="manual-control-panel" class="hidden p-3 rounded-xl border border-amber-500/40 bg-amber-950/30 space-y-2">
          <div class="flex flex-wrap items-center justify-between gap-2">
            <div class="flex items-center gap-2">
              <span class="w-2.5 h-2.5 rounded-full bg-amber-400 animate-pulse"></span>
              <span class="text-xs font-bold text-amber-300">MANUAL TRADING MODE ACTIVE</span>
              <span class="text-[11px] text-slate-400 hidden sm:inline">— Bot monitors market & generates signals; place orders below</span>
            </div>
            <button type="button" onclick="openSettingsModal()" class="text-[11px] text-amber-300 hover:text-amber-100 underline font-medium">
              <i class="fa-solid fa-gear mr-1"></i>Change Strategy / Timeframe / Switch to Auto
            </button>
          </div>
          <div class="flex flex-col sm:flex-row items-stretch sm:items-center gap-2 sm:gap-3">
            <button type="button" id="btn-manual-buy" onclick="triggerManualTrade('buy')" class="w-full sm:flex-1 py-2.5 bg-emerald-600 hover:bg-emerald-500 active:bg-emerald-700 text-white font-bold text-xs rounded-lg shadow transition flex items-center justify-center gap-1.5 min-h-[42px]">
              <i class="fa-solid fa-arrow-trend-up"></i> Manual BUY (Market)
            </button>
            <button type="button" id="btn-manual-sell" onclick="triggerManualTrade('sell')" class="w-full sm:flex-1 py-2.5 bg-rose-600 hover:bg-rose-500 active:bg-rose-700 text-white font-bold text-xs rounded-lg shadow transition flex items-center justify-center gap-1.5 min-h-[42px]">
              <i class="fa-solid fa-arrow-trend-down"></i> Manual SELL (Market)
            </button>
          </div>
        </div>

        <div class="grid grid-cols-2 md:grid-cols-4 gap-2.5 sm:gap-3 text-xs">
          <div class="bg-slate-900/80 p-2.5 sm:p-3 rounded-lg border border-bordercol/60">
            <div class="text-slate-400">Market Price</div>
            <div class="text-sm sm:text-base font-bold text-white mt-0.5 truncate" id="an-price">Loading...</div>
            <div class="text-[10px] text-slate-400 mt-1 truncate" id="an-exchange">Exchange: binance</div>
          </div>

          <div class="bg-slate-900/80 p-2.5 sm:p-3 rounded-lg border border-bordercol/60">
            <div class="text-slate-400">EMA Trend (12 / 26)</div>
            <div class="text-sm sm:text-base font-bold mt-0.5 truncate" id="an-trend">--</div>
            <div class="text-[10px] text-slate-400 mt-1 truncate" id="an-emas">Fast: -- | Slow: --</div>
          </div>

          <div class="bg-slate-900/80 p-2.5 sm:p-3 rounded-lg border border-bordercol/60">
            <div class="text-slate-400">RSI (14)</div>
            <div class="text-sm sm:text-base font-bold mt-0.5 truncate" id="an-rsi">--</div>
            <div class="text-[10px] text-slate-400 mt-1 truncate" id="an-rsi-status">Neutral</div>
          </div>

          <div class="bg-slate-900/80 p-2.5 sm:p-3 rounded-lg border border-bordercol/60">
            <div class="text-slate-400">ADX Trend Strength</div>
            <div class="text-sm sm:text-base font-bold mt-0.5 truncate" id="an-adx">--</div>
            <div class="text-[10px] text-slate-400 mt-1 truncate" id="an-adx-status">Thresh: 20</div>
          </div>
        </div>

        <!-- Automatic Fibonacci Zone Breakdown -->
        <div class="bg-slate-900/80 p-3 sm:p-3.5 rounded-lg border border-bordercol/60">
          <div class="flex flex-wrap items-center justify-between text-xs mb-2 gap-1.5">
            <span class="font-semibold text-slate-300 flex items-center gap-1.5">
              <i class="fa-solid fa-layer-group text-blue-400"></i> Rolling Fibonacci Retracement Support Zone (50 bars)
            </span>
            <span id="an-fib-confluence" class="px-2 py-0.5 rounded text-[10px] font-bold bg-slate-800 text-slate-400 shrink-0">CHECKING</span>
          </div>
          <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2 text-center text-xs">
            <div class="bg-slate-800/60 p-2 rounded">
              <div class="text-[10px] text-slate-400">Swing High</div>
              <div class="font-semibold text-slate-200 mt-0.5 truncate" id="fib-high">--</div>
            </div>
            <div class="bg-slate-800/60 p-2 rounded">
              <div class="text-[10px] text-slate-400">38.2% Level</div>
              <div class="font-semibold text-blue-300 mt-0.5 truncate" id="fib-382">--</div>
            </div>
            <div class="bg-slate-800/60 p-2 rounded">
              <div class="text-[10px] text-slate-400">50.0% Level</div>
              <div class="font-semibold text-purple-300 mt-0.5 truncate" id="fib-500">--</div>
            </div>
            <div class="bg-slate-800/60 p-2 rounded">
              <div class="text-[10px] text-slate-400">61.8% Level</div>
              <div class="font-semibold text-amber-300 mt-0.5 truncate" id="fib-618">--</div>
            </div>
            <div class="bg-slate-800/60 p-2 rounded col-span-2 sm:col-span-1">
              <div class="text-[10px] text-slate-400">Swing Low</div>
              <div class="font-semibold text-slate-200 mt-0.5 truncate" id="fib-low">--</div>
            </div>
          </div>
        </div>

        <!-- Mini Candle Visualizer -->
        <div class="bg-slate-900/80 p-3.5 rounded-lg border border-bordercol/60">
          <div class="text-xs font-semibold text-slate-300 mb-2">Recent Candles & EMA Trajectory (15m)</div>
          <div id="candle-list" class="flex items-end gap-1.5 h-24 overflow-x-auto pt-4 pb-1"></div>
        </div>
      </div>

      <!-- Quick Backtester Panel (1 col) -->
      <div class="bg-cardbg border border-bordercol rounded-xl p-5 space-y-4">
        <div class="flex items-center justify-between border-b border-bordercol pb-3">
          <div class="flex items-center gap-2">
            <i class="fa-solid fa-vial text-purple-400"></i>
            <h2 class="font-bold text-sm text-white">Honest Backtester</h2>
          </div>
          <span class="text-[10px] text-slate-400">No lookahead bias</span>
        </div>

        <form id="backtest-form" onsubmit="executeBacktest(event)" class="space-y-3 text-xs">
          <div>
            <label class="block text-slate-400 mb-1">Exchange</label>
            <select id="bt-exchange" class="w-full bg-slate-900 border border-bordercol rounded-lg px-2.5 py-1.5 text-white">
              <option value="binance" selected>Binance</option>
              <option value="coinbase">Coinbase Advanced</option>
              <option value="kraken">Kraken</option>
              <option value="kucoin">KuCoin</option>
              <option value="bybit">Bybit</option>
              <option value="okx">OKX</option>
              <option value="bitget">Bitget</option>
              <option value="gateio">Gate.io</option>
              <option value="mexc">MEXC</option>
              <option value="bitfinex">Bitfinex</option>
              <option value="cryptocom">Crypto.com</option>
              <option value="htx">HTX (Huobi)</option>
              <option value="bingx">BingX</option>
              <option value="whitebit">WhiteBIT</option>
              <option value="gemini">Gemini</option>
              <option value="phemex">Phemex</option>
              <option value="poloniex">Poloniex</option>
              <option value="deribit">Deribit</option>
            </select>
          </div>

          <div class="grid grid-cols-2 gap-2">
            <div>
              <label class="block text-slate-400 mb-1">Symbol</label>
              <input type="text" id="bt-symbol" value="BTC/USDT" class="w-full bg-slate-900 border border-bordercol rounded-lg px-2.5 py-1.5 text-white">
            </div>
            <div>
              <label class="block text-slate-400 mb-1">Timeframe</label>
              <select id="bt-timeframe" class="w-full bg-slate-900 border border-bordercol rounded-lg px-2.5 py-1.5 text-white">
                <option value="5m">5m</option>
                <option value="15m" selected>15m</option>
                <option value="1h">1h</option>
                <option value="4h">4h</option>
                <option value="1d">1d</option>
              </select>
            </div>
          </div>

          <div class="grid grid-cols-2 gap-2">
            <div>
              <label class="block text-slate-400 mb-1">Lookback (Days)</label>
              <input type="number" id="bt-days" value="30" min="1" max="365" class="w-full bg-slate-900 border border-bordercol rounded-lg px-2.5 py-1.5 text-white">
            </div>
            <div>
              <label class="block text-slate-400 mb-1">Start Equity ($)</label>
              <input type="number" id="bt-equity" value="1000" min="10" class="w-full bg-slate-900 border border-bordercol rounded-lg px-2.5 py-1.5 text-white">
            </div>
          </div>

          <div class="flex items-center gap-2 pt-1">
            <input type="checkbox" id="bt-use-fib" checked class="rounded border-bordercol bg-slate-900 text-emerald-500">
            <label for="bt-use-fib" class="text-slate-300">Enable Fibonacci Zone Filter</label>
          </div>

          <button type="submit" id="btn-run-bt" class="w-full py-2.5 bg-purple-600 hover:bg-purple-500 active:bg-purple-700 text-white font-semibold rounded-lg shadow transition flex items-center justify-center gap-2 min-h-[44px]">
            <i class="fa-solid fa-play"></i> Run Historical Backtest
          </button>
        </form>

        <div id="bt-results-box" class="hidden bg-slate-900/90 rounded-lg p-3 border border-bordercol text-xs space-y-2">
          <div class="font-bold text-slate-200 border-b border-bordercol/60 pb-1 flex justify-between">
            <span>Backtest Results</span>
            <span id="bt-res-return" class="font-bold">--</span>
          </div>
          <div class="grid grid-cols-2 gap-2 text-[11px]">
            <div>Trades: <span id="bt-res-trades" class="font-semibold text-white">0</span></div>
            <div>Win Rate: <span id="bt-res-winrate" class="font-semibold text-white">0%</span></div>
            <div>Final Equity: <span id="bt-res-final" class="font-semibold text-white">$0</span></div>
            <div>Sharpe (approx): <span id="bt-res-sharpe" class="font-semibold text-white">0</span></div>
            <div>Avg Win: <span id="bt-res-avgwin" class="font-semibold text-emerald-400">$0</span></div>
            <div>Avg Loss: <span id="bt-res-avgloss" class="font-semibold text-red-400">$0</span></div>
            <div class="col-span-2">Max Drawdown: <span id="bt-res-drawdown" class="font-semibold text-amber-400">0%</span></div>
          </div>
        </div>
      </div>
    </div>

    <!-- Live Activity Logs -->
    <div class="bg-cardbg border border-bordercol rounded-xl p-5 space-y-3">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-2">
          <i class="fa-solid fa-terminal text-slate-400"></i>
          <h3 class="font-bold text-sm text-white">Live Bot Activity Logs</h3>
        </div>
        <button onclick="refreshLogs()" class="text-xs text-slate-400 hover:text-white transition flex items-center gap-1">
          <i class="fa-solid fa-rotate"></i> Refresh
        </button>
      </div>
      <div id="log-container" class="bg-slate-950 font-mono text-[11px] p-3 rounded-lg border border-bordercol/60 h-40 overflow-y-auto space-y-1 text-slate-300">
        <div class="text-slate-500">Loading recent logs...</div>
      </div>
    </div>
  </main>

  <!-- Sticky Mobile Bottom Quick-Action Bar -->
  <nav id="mobile-bottom-nav" class="md:hidden fixed bottom-0 left-0 right-0 z-40 bg-cardbg/95 backdrop-blur-md border-t border-bordercol py-1.5 px-2 flex items-center justify-around text-center shadow-2xl safe-area-bottom">
    <!-- 1. Bot Start/Stop Quick Action -->
    <button id="bottom-btn-toggle" onclick="toggleBot()" class="flex flex-col items-center justify-center py-1 px-1.5 text-slate-300 hover:text-white active:scale-95 transition flex-1">
      <i id="bottom-btn-toggle-icon" class="fa-solid fa-play text-emerald-400 text-base"></i>
      <span id="bottom-btn-toggle-text" class="text-[10px] font-semibold mt-0.5">Start</span>
    </button>

    <!-- 2. Emergency Kill Switch -->
    <button onclick="triggerKillSwitch()" class="flex flex-col items-center justify-center py-1 px-1.5 text-rose-400 hover:text-rose-300 active:scale-95 transition flex-1">
      <i class="fa-solid fa-skull-crossbones text-base"></i>
      <span class="text-[10px] font-semibold mt-0.5">Kill</span>
    </button>

    <!-- 3. Bot Settings -->
    <button onclick="openSettingsModal()" class="flex flex-col items-center justify-center py-1 px-1.5 text-slate-300 hover:text-white active:scale-95 transition flex-1">
      <i class="fa-solid fa-gear text-amber-400 text-base"></i>
      <span class="text-[10px] font-semibold mt-0.5">Settings</span>
    </button>

    <!-- 4. Email & Verification Settings -->
    <button onclick="openEmailSettingsModal()" class="flex flex-col items-center justify-center py-1 px-1.5 text-slate-300 hover:text-white active:scale-95 transition flex-1 relative">
      <span id="email-status-dot-bottom" class="w-2 h-2 rounded-full bg-slate-500 absolute top-1 right-1/4"></span>
      <i class="fa-solid fa-envelope text-sky-400 text-base"></i>
      <span class="text-[10px] font-semibold mt-0.5">Email</span>
    </button>

    <!-- 5. User Account -->
    <button id="bottom-btn-account" onclick="handleMobileAccountClick()" class="flex flex-col items-center justify-center py-1 px-1.5 text-slate-300 hover:text-white active:scale-95 transition flex-1">
      <i class="fa-solid fa-circle-user text-emerald-400 text-base"></i>
      <span id="bottom-account-text" class="text-[10px] font-semibold mt-0.5 truncate max-w-[60px]">Account</span>
    </button>
  </nav>

  <!-- ========================================================================= -->
  <!-- AUTH MODALS (LOGIN / REGISTER / FORGOT PASSWORD / ACTIVATION NOTICE)       -->
  <!-- ========================================================================= -->
  <div id="auth-modal" class="hidden fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-3 sm:p-4 overflow-y-auto">
    <div class="bg-slate-900 border border-slate-800 rounded-2xl max-w-md w-full p-4 sm:p-6 shadow-2xl space-y-4 relative my-auto max-h-[92vh] overflow-y-auto">
      <button onclick="closeAuthModal()" class="absolute top-4 right-4 text-slate-400 hover:text-white"><i class="fa-solid fa-xmark text-lg"></i></button>

      <!-- Modal Title -->
      <div class="text-center">
        <h3 id="modal-title" class="text-xl font-bold text-white">Account Login</h3>
        <p id="modal-subtitle" class="text-xs text-slate-400 mt-1">Sign in to manage exchange keys and view trades.</p>
      </div>

      <!-- LOGIN FORM -->
      <form id="form-login" onsubmit="handleLogin(event)" class="space-y-3 text-xs">
        <div>
          <label class="block text-slate-400 mb-1">Username or Email</label>
          <input type="text" id="login-user" required placeholder="Enter username or email" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-emerald-500">
        </div>
        <div>
          <div class="flex justify-between mb-1">
            <label class="text-slate-400">Password</label>
            <a href="javascript:void(0)" onclick="openAuthModal('forgot')" class="text-amber-400 hover:underline">Forgot password?</a>
          </div>
          <div class="relative">
            <input type="password" id="login-pass" required placeholder="Enter your password" class="w-full bg-slate-800 border border-slate-700 rounded-lg pl-3 pr-10 py-2 text-white focus:outline-none focus:border-emerald-500">
            <button type="button" onclick="togglePasswordVisibility('login-pass', 'login-pass-eye')" class="absolute inset-y-0 right-0 pr-3 flex items-center text-slate-400 hover:text-slate-200 focus:outline-none cursor-pointer" title="Toggle password visibility" tabindex="-1">
              <i class="fa-solid fa-eye text-xs" id="login-pass-eye"></i>
            </button>
          </div>
        </div>
        <button type="submit" class="w-full py-2.5 bg-emerald-600 hover:bg-emerald-500 font-bold text-white rounded-lg shadow transition">
          Sign In
        </button>
        <div class="relative flex py-2 items-center">
          <div class="flex-grow border-t border-slate-800"></div>
          <span class="flex-shrink mx-2 text-[10px] text-slate-500 uppercase">Or</span>
          <div class="flex-grow border-t border-slate-800"></div>
        </div>
        <button type="button" id="btn-magic-login" onclick="handleSendMagicLink()" class="w-full py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold rounded-lg border border-slate-700 shadow transition flex items-center justify-center gap-1.5 text-xs">
          <i class="fa-solid fa-wand-magic-sparkles text-amber-400"></i> Email 1-Click Instant Sign-In Link
        </button>
        <div class="text-center text-slate-400 pt-2 space-y-1">
          <div>Don't have an account? <a href="javascript:void(0)" onclick="openAuthModal('register')" class="text-emerald-400 font-semibold hover:underline">Register now</a></div>
          <div><a href="javascript:void(0)" onclick="openAuthModal('resend')" class="text-[11px] text-slate-400 hover:text-emerald-400 underline">Didn't receive verification email? Resend link</a></div>
        </div>
      </form>

      <!-- REGISTER FORM -->
      <form id="form-register" onsubmit="handleRegister(event)" class="hidden space-y-3 text-xs">
        <div>
          <label class="block text-slate-400 mb-1">Username</label>
          <input type="text" id="reg-user" required minlength="3" placeholder="TraderPro" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-emerald-500">
        </div>
        <div>
          <label class="block text-slate-400 mb-1">Email Address</label>
          <input type="email" id="reg-email" required placeholder="trader@example.com" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-emerald-500">
        </div>
        <div>
          <label class="block text-slate-400 mb-1">Password (min 6 chars)</label>
          <div class="relative">
            <input type="password" id="reg-pass" required minlength="6" placeholder="Create a strong password" class="w-full bg-slate-800 border border-slate-700 rounded-lg pl-3 pr-10 py-2 text-white focus:outline-none focus:border-emerald-500">
            <button type="button" onclick="togglePasswordVisibility('reg-pass', 'reg-pass-eye')" class="absolute inset-y-0 right-0 pr-3 flex items-center text-slate-400 hover:text-slate-200 focus:outline-none cursor-pointer" title="Toggle password visibility" tabindex="-1">
              <i class="fa-solid fa-eye text-xs" id="reg-pass-eye"></i>
            </button>
          </div>
        </div>
        <button type="submit" class="w-full py-2.5 bg-emerald-600 hover:bg-emerald-500 font-bold text-white rounded-lg shadow transition">
          Create Account
        </button>
        <div class="text-center text-slate-400 pt-2 space-y-1">
          <div>Already have an account? <a href="javascript:void(0)" onclick="openAuthModal('login')" class="text-emerald-400 font-semibold hover:underline">Log in</a></div>
          <div><a href="javascript:void(0)" onclick="openAuthModal('resend')" class="text-[11px] text-slate-400 hover:text-emerald-400 underline">Need another verification link? Resend here</a></div>
        </div>
      </form>

      <!-- FORGOT PASSWORD FORM -->
      <form id="form-forgot" onsubmit="handleForgot(event)" class="hidden space-y-3 text-xs">
        <div>
          <label class="block text-slate-400 mb-1">Registered Email Address</label>
          <input type="email" id="forgot-email" required placeholder="trader@example.com" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-amber-500">
        </div>
        <button type="submit" class="w-full py-2.5 bg-amber-600 hover:bg-amber-500 font-bold text-white rounded-lg shadow transition">
          Send Password Reset Email
        </button>
        <div class="text-center text-slate-400 pt-2">
          <a href="javascript:void(0)" onclick="openAuthModal('login')" class="text-slate-300 hover:underline">Back to Login</a>
        </div>
      </form>

      <!-- RESEND VERIFICATION FORM -->
      <form id="form-resend" onsubmit="handleResendForm(event)" class="hidden space-y-3 text-xs">
        <div>
          <label class="block text-slate-400 mb-1">Registered Email Address</label>
          <input type="email" id="resend-email" required placeholder="trader@example.com" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-emerald-500">
        </div>
        <button type="submit" id="btn-submit-resend" class="w-full py-2.5 bg-emerald-600 hover:bg-emerald-500 font-bold text-white rounded-lg shadow transition flex items-center justify-center gap-1.5">
          <i class="fa-solid fa-paper-plane"></i> Resend Verification Link
        </button>
        <div class="text-center text-slate-400 pt-2 flex items-center justify-between text-[11px]">
          <a href="javascript:void(0)" onclick="openAuthModal('login')" class="text-slate-300 hover:underline">Back to Login</a>
          <a href="javascript:void(0)" onclick="openAuthModal('register')" class="text-emerald-400 hover:underline">Create Account</a>
        </div>
      </form>

      <!-- ACTIVATION NOTICE BOX -->
      <div id="modal-notice" class="hidden p-3 rounded-lg border text-xs space-y-2"></div>
    </div>
  </div>

  <!-- EMAIL & SMTP SETTINGS MODAL -->
  <div id="email-settings-modal" class="hidden fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-3 sm:p-4 overflow-y-auto">
    <div class="bg-cardbg border border-bordercol rounded-2xl max-w-lg w-full p-4 sm:p-6 shadow-2xl space-y-4 my-auto max-h-[92vh] overflow-y-auto">
      <div class="flex items-center justify-between border-b border-bordercol pb-3">
        <div class="flex items-center gap-2">
          <div class="w-8 h-8 rounded-lg bg-sky-500/20 text-sky-400 flex items-center justify-center font-bold shrink-0">
            <i class="fa-solid fa-envelope"></i>
          </div>
          <div>
            <h3 class="text-base font-bold text-white">Email & SMTP Settings</h3>
            <p class="text-[11px] text-slate-400">Configure email delivery for account verifications & password resets</p>
          </div>
        </div>
        <button onclick="closeEmailSettingsModal()" class="text-slate-400 hover:text-white text-lg p-1 -mr-1 -mt-1"><i class="fa-solid fa-xmark"></i></button>
      </div>

      <!-- Current Status Badge -->
      <div id="smtp-current-status-banner" class="p-3 rounded-lg border border-bordercol bg-slate-900/80 text-xs flex flex-wrap items-center justify-between gap-2">
        <div class="flex items-center gap-2">
          <span id="smtp-status-indicator" class="w-2.5 h-2.5 rounded-full bg-amber-500 shrink-0"></span>
          <span id="smtp-status-text" class="font-medium text-slate-200">Checking email service...</span>
        </div>
        <span id="smtp-provider-tag" class="px-2 py-0.5 rounded text-[10px] font-mono uppercase bg-slate-800 text-slate-400 shrink-0">NONE</span>
      </div>

      <!-- Provider Selection Tabs -->
      <div>
        <label class="block text-slate-400 text-xs mb-1.5 font-medium">Select Email Provider</label>
        <div class="grid grid-cols-3 sm:grid-cols-5 gap-1.5 text-xs">
          <button type="button" onclick="selectEmailProvider('gmail')" id="tab-prov-gmail" class="py-2 px-1.5 rounded-lg border font-semibold text-center transition flex flex-col items-center gap-1 bg-emerald-600/20 border-emerald-500 text-emerald-300">
            <i class="fa-brands fa-google text-sm"></i>
            <span class="text-[11px]">Gmail SMTP</span>
          </button>
          <button type="button" onclick="selectEmailProvider('brevo')" id="tab-prov-brevo" class="py-2 px-1.5 rounded-lg border border-bordercol text-slate-400 hover:text-white font-semibold text-center transition flex flex-col items-center gap-1">
            <i class="fa-solid fa-paper-plane text-sm"></i>
            <span class="text-[11px]">Brevo API</span>
          </button>
          <button type="button" onclick="selectEmailProvider('resend')" id="tab-prov-resend" class="py-2 px-1.5 rounded-lg border border-bordercol text-slate-400 hover:text-white font-semibold text-center transition flex flex-col items-center gap-1">
            <i class="fa-solid fa-bolt text-sm"></i>
            <span class="text-[11px]">Resend API</span>
          </button>
          <button type="button" onclick="selectEmailProvider('sendgrid')" id="tab-prov-sendgrid" class="py-2 px-1.5 rounded-lg border border-bordercol text-slate-400 hover:text-white font-semibold text-center transition flex flex-col items-center gap-1">
            <i class="fa-solid fa-cloud text-sm"></i>
            <span class="text-[11px]">SendGrid</span>
          </button>
          <button type="button" onclick="selectEmailProvider('smtp')" id="tab-prov-smtp" class="py-2 px-1.5 rounded-lg border border-bordercol text-slate-400 hover:text-white font-semibold text-center transition flex flex-col items-center gap-1">
            <i class="fa-solid fa-server text-sm"></i>
            <span class="text-[11px]">Custom SMTP</span>
          </button>
        </div>
      </div>

      <!-- Form Inputs -->
      <form id="form-email-settings" onsubmit="handleSaveEmailSettings(event)" class="space-y-3 text-xs">
        <input type="hidden" id="email-provider-val" value="gmail">

        <!-- Gmail Guide Alert -->
        <div id="gmail-guide-box" class="p-3 bg-slate-900/90 rounded-lg border border-emerald-500/30 space-y-1 text-[11px] text-slate-300">
          <div class="font-bold text-emerald-400 flex items-center gap-1.5">
            <i class="fa-brands fa-google"></i> Gmail SMTP (Delivers to All Users - No Domain Required)
          </div>
          <p>1. Go to your <a href="https://myaccount.google.com/apppasswords" target="_blank" class="text-sky-400 underline font-semibold">Google Account Security &rarr; App Passwords</a>.</p>
          <p>2. Create a new App Password (name it <em>Trading Bot</em>) and copy the 16-letter code.</p>
          <p>3. Enter your Gmail address and the 16-letter App Password below.</p>
          <p class="text-slate-400 text-[10px] italic pt-0.5">Note: If deployed on Vercel, cloud serverless firewalls block SMTP ports; use Brevo API instead.</p>
        </div>

        <!-- Brevo Guide Alert -->
        <div id="brevo-guide-box" class="hidden p-3 bg-slate-900/90 rounded-lg border border-sky-500/30 space-y-1 text-[11px] text-slate-300">
          <div class="font-bold text-sky-400 flex items-center gap-1.5">
            <i class="fa-solid fa-paper-plane"></i> Brevo API (Recommended for Vercel - No Domain Required)
          </div>
          <p>1. Sign up for free at <a href="https://www.brevo.com" target="_blank" class="text-sky-400 underline font-semibold">brevo.com</a> (300 free emails/day to any user).</p>
          <p>2. Go to <strong>Senders & IP</strong> and add any sender email (e.g. your personal Gmail). Verify the confirmation code.</p>
          <p>3. Generate an API Key under <strong>SMTP & API &rarr; API Keys</strong> and paste it below.</p>
        </div>

        <!-- Resend Verified Domain Banner (shown when a verified domain exists) -->
        <div id="resend-verified-domain-box" class="hidden p-3 bg-emerald-950/40 rounded-lg border border-emerald-500/40 space-y-1 text-[11px] text-slate-200">
          <div class="font-bold text-emerald-400 flex items-center justify-between">
            <span class="flex items-center gap-1.5">
              <i class="fa-solid fa-circle-check text-emerald-400"></i> Domain Verified & Active
            </span>
            <span id="resend-domain-name-badge" class="font-mono text-[10px] bg-emerald-500/20 text-emerald-300 px-2 py-0.5 rounded border border-emerald-500/30"></span>
          </div>
          <p class="text-slate-300">
            Account verification emails are active and will deliver to <strong>all users</strong> worldwide from <code id="resend-active-sender-code" class="bg-slate-900 px-1.5 py-0.5 rounded text-emerald-300 font-mono"></code>.
          </p>
        </div>

        <!-- Resend Guide Alert -->
        <div id="resend-guide-box" class="hidden p-3 bg-slate-900/90 rounded-lg border border-amber-500/30 space-y-1.5 text-[11px] text-slate-300">
          <div class="font-bold text-amber-400 flex items-center gap-1.5">
            <i class="fa-solid fa-triangle-exclamation"></i> Important Resend Setup for All Users
          </div>
          <p class="leading-relaxed">
            Resend's default <code class="bg-slate-800 px-1 py-0.5 rounded text-amber-300">onboarding@resend.dev</code> sender is restricted to <strong>only deliver to the account owner</strong>.
          </p>
          <p class="leading-relaxed">
            To deliver verification emails to <strong>ALL users creating accounts</strong>, you must add & verify your custom domain at <a href="https://resend.com/domains" target="_blank" class="text-sky-400 underline font-semibold">resend.com/domains</a> and set your verified sender address below (e.g. <code class="bg-slate-800 px-1 py-0.5 rounded text-emerald-300">Crypto Bot &lt;noreply@yourdomain.com&gt;</code>).
          </p>
          <p class="text-slate-400 text-[10px] italic">Don't have a custom domain? Select <strong>Gmail SMTP</strong> or <strong>Brevo API</strong> above to send to all users without domain verification.</p>
        </div>

        <!-- SendGrid Guide Alert -->
        <div id="sendgrid-guide-box" class="hidden p-3 bg-slate-900/90 rounded-lg border border-sky-500/30 space-y-1 text-[11px] text-slate-300">
          <div class="font-bold text-sky-400 flex items-center gap-1.5">
            <i class="fa-solid fa-cloud"></i> SendGrid API (100 free emails/day)
          </div>
          <p>1. Sign up at <a href="https://sendgrid.com" target="_blank" class="text-sky-400 underline font-semibold">sendgrid.com</a> and complete Single Sender Verification.</p>
          <p>2. Create an API Key under <strong>Settings &rarr; API Keys</strong> with "Mail Send" permissions.</p>
          <p>3. Paste the API key and verified sender email below.</p>
        </div>

        <!-- Custom SMTP Host & Port (shown for custom SMTP) -->
        <div id="field-smtp-host-row" class="hidden grid grid-cols-3 gap-2">
          <div class="col-span-2">
            <label class="block text-slate-400 mb-1">SMTP Host</label>
            <input type="text" id="smtp-input-host" placeholder="smtp.mailgun.org / smtp.postmarkapp.com" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white">
          </div>
          <div>
            <label class="block text-slate-400 mb-1">Port</label>
            <input type="text" id="smtp-input-port" value="587" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white">
          </div>
        </div>

        <!-- Brevo API Key (shown for Brevo) -->
        <div id="field-brevo-row" class="hidden space-y-2">
          <div>
            <label class="block text-slate-400 mb-1">Brevo API Key (v3) (<a href="https://app.brevo.com/settings/keys/api" target="_blank" class="text-sky-400 hover:underline">Get free key from brevo.com</a>)</label>
            <input type="password" id="brevo-input-key" placeholder="xkeysib-..." class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white font-mono">
          </div>
        </div>

        <!-- Resend API Key (shown for Resend) -->
        <div id="field-resend-row" class="hidden space-y-2">
          <div>
            <label class="block text-slate-400 mb-1">Resend API Key (<a href="https://resend.com/api-keys" target="_blank" class="text-sky-400 hover:underline">Get key from resend.com</a>)</label>
            <input type="password" id="resend-input-key" placeholder="re_123456789..." class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white font-mono">
          </div>
        </div>

        <!-- SendGrid API Key (shown for SendGrid) -->
        <div id="field-sendgrid-row" class="hidden space-y-2">
          <div>
            <label class="block text-slate-400 mb-1">SendGrid API Key (<a href="https://app.sendgrid.com/settings/api_keys" target="_blank" class="text-sky-400 hover:underline">Get key from sendgrid.com</a>)</label>
            <input type="password" id="sendgrid-input-key" placeholder="SG.xxxxxxxx..." class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white font-mono">
          </div>
        </div>

        <!-- SMTP User & Password (shown for Gmail & Custom SMTP) -->
        <div id="field-smtp-creds-row" class="space-y-3">
          <div>
            <label id="lbl-smtp-user" class="block text-slate-400 mb-1">Gmail Address / SMTP User</label>
            <input type="email" id="smtp-input-user" placeholder="trader@gmail.com" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white">
          </div>
          <div>
            <label id="lbl-smtp-pass" class="block text-slate-400 mb-1">16-Character App Password</label>
            <div class="relative">
              <input type="password" id="smtp-input-pass" placeholder="abcd efgh ijkl mnop" class="w-full bg-slate-800 border border-slate-700 rounded-lg pl-3 pr-10 py-2 text-white">
              <button type="button" onclick="togglePasswordVisibility('smtp-input-pass', 'smtp-pass-eye')" class="absolute inset-y-0 right-0 pr-3 flex items-center text-slate-400 hover:text-slate-200 transition" tabindex="-1">
                <i class="fa-solid fa-eye text-xs" id="smtp-pass-eye"></i>
              </button>
            </div>
          </div>
        </div>

        <div>
          <label class="block text-slate-400 mb-1">Sender "From" Email Address</label>
          <input type="text" id="smtp-input-from" placeholder="Crypto Trading Bot <noreply@yourdomain.com>" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white">
        </div>

        <button type="submit" id="btn-save-email-settings" class="w-full py-2.5 bg-emerald-600 hover:bg-emerald-500 font-bold text-white rounded-lg shadow transition flex items-center justify-center gap-1.5">
          <i class="fa-solid fa-floppy-disk"></i> Save Email Settings
        </button>
      </form>

      <!-- TEST EMAIL DISPATCH SECTION -->
      <div class="border-t border-bordercol pt-3 space-y-2">
        <h4 class="text-xs font-bold text-white flex items-center gap-1.5">
          <i class="fa-solid fa-paper-plane text-sky-400"></i> Send Test Verification Email
        </h4>
        <div class="flex flex-col sm:flex-row gap-2 text-xs">
          <input type="email" id="test-email-dest" placeholder="Enter recipient email (e.g. your email)" class="flex-1 bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white">
          <button type="button" id="btn-run-test-email" onclick="runTestEmail()" class="w-full sm:w-auto px-4 py-2.5 sm:py-2 bg-sky-600 hover:bg-sky-500 text-white font-semibold rounded-lg shadow transition flex items-center justify-center gap-1.5">
            <i class="fa-solid fa-paper-plane"></i> Send Test
          </button>
        </div>
        <div id="test-email-result" class="hidden p-3 rounded-lg border text-xs"></div>
      </div>
    </div>
  </div>

  <!-- ========================================================================= -->
  <!-- TRADING BOT SETTINGS MODAL                                                -->
  <!-- ========================================================================= -->
  <div id="settings-modal" class="hidden fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-3 sm:p-4 overflow-y-auto">
    <div class="bg-cardbg border border-bordercol rounded-2xl max-w-xl w-full p-4 sm:p-6 shadow-2xl space-y-4 my-auto max-h-[92vh] overflow-y-auto">
      <div class="flex items-center justify-between border-b border-bordercol pb-3">
        <div class="flex items-center gap-2.5">
          <div class="w-9 h-9 rounded-lg bg-amber-500/20 text-amber-400 flex items-center justify-center font-bold shrink-0">
            <i class="fa-solid fa-gear text-base"></i>
          </div>
          <div>
            <h3 class="text-base font-bold text-white">Bot & Trading Settings</h3>
            <p class="text-[11px] text-slate-400">Configure manual vs auto trading, strategy choice, timeframe, and risk</p>
          </div>
        </div>
        <button onclick="closeSettingsModal()" class="text-slate-400 hover:text-white text-lg"><i class="fa-solid fa-xmark"></i></button>
      </div>

      <form id="form-settings" onsubmit="handleSaveSettings(event)" class="space-y-4 text-xs">
        <!-- 1. Trading Execution Mode (Auto vs Manual) -->
        <div>
          <label class="block text-slate-300 font-bold mb-2">1. Trading Execution Mode</label>
          <div class="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
            <label id="opt-mode-auto-card" class="p-3 rounded-xl border border-purple-500/60 bg-purple-950/30 cursor-pointer transition flex items-start gap-2.5">
              <input type="radio" name="settings-exec-mode" id="radio-mode-auto" value="auto" checked onchange="onExecutionModeOptionChange()" class="mt-0.5 text-purple-500 focus:ring-0">
              <div>
                <div class="font-bold text-white flex items-center gap-1.5">
                  <i class="fa-solid fa-robot text-purple-400"></i> Auto Trading
                </div>
                <p class="text-[11px] text-slate-400 mt-0.5 leading-tight">Bot automatically enters and exits trades when strategy signals fire.</p>
              </div>
            </label>

            <label id="opt-mode-manual-card" class="p-3 rounded-xl border border-bordercol bg-slate-900/80 cursor-pointer transition flex items-start gap-2.5 hover:border-slate-600">
              <input type="radio" name="settings-exec-mode" id="radio-mode-manual" value="manual" onchange="onExecutionModeOptionChange()" class="mt-0.5 text-amber-500 focus:ring-0">
              <div>
                <div class="font-bold text-white flex items-center gap-1.5">
                  <i class="fa-solid fa-hand-pointer text-amber-400"></i> Manual Trading
                </div>
                <p class="text-[11px] text-slate-400 mt-0.5 leading-tight">Bot monitors market and generates signals, but YOU place trades manually.</p>
              </div>
            </label>
          </div>
        </div>

        <!-- 2. Strategy Selection -->
        <div>
          <label class="block text-slate-300 font-bold mb-1">2. Select Trading Strategy</label>
          <select id="settings-strategy" onchange="onStrategyOptionChange()" class="w-full bg-slate-900 border border-bordercol rounded-lg px-3 py-2 text-white font-medium focus:border-emerald-500">
            <!-- Populated dynamically -->
          </select>
          <div id="settings-strat-desc" class="p-2.5 bg-slate-900/80 border border-bordercol/60 rounded-lg text-[11px] text-slate-300 mt-1.5 leading-relaxed">
            <!-- Populated dynamically -->
          </div>
        </div>

        <!-- 3. Timeframe & Symbol -->
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div>
            <label class="block text-slate-300 font-bold mb-1">
              3. Timeframe <span class="text-emerald-400 font-semibold text-[11px]">(5m included)</span>
            </label>
            <select id="settings-timeframe" class="w-full bg-slate-900 border border-bordercol rounded-lg px-3 py-2 text-white font-medium focus:border-emerald-500">
              <option value="1m">1m (1 minute)</option>
              <option value="3m">3m (3 minutes)</option>
              <option value="5m" class="font-bold text-emerald-400">5m (5 minutes) ★</option>
              <option value="15m" selected>15m (15 minutes)</option>
              <option value="30m">30m (30 minutes)</option>
              <option value="1h">1h (1 hour)</option>
              <option value="2h">2h (2 hours)</option>
              <option value="4h">4h (4 hours)</option>
              <option value="1d">1d (1 day)</option>
            </select>
          </div>

          <div>
            <label class="block text-slate-300 font-bold mb-1">Trading Pair / Symbol</label>
            <input type="text" id="settings-symbol" value="BTC/USDT" placeholder="BTC/USDT" class="w-full bg-slate-900 border border-bordercol rounded-lg px-3 py-2 text-white font-medium focus:border-emerald-500">
          </div>
        </div>

        <!-- 4. Advanced Technical & Risk Settings (collapsible) -->
        <div class="border border-bordercol rounded-xl overflow-hidden">
          <button type="button" onclick="toggleAdvancedSettings()" class="w-full p-3 bg-slate-900 flex items-center justify-between text-left text-slate-300 hover:text-white font-semibold">
            <span class="flex items-center gap-1.5"><i class="fa-solid fa-sliders text-slate-400"></i> Advanced Indicators & Risk Parameters</span>
            <i class="fa-solid fa-chevron-down text-xs transition-transform" id="adv-chevron"></i>
          </button>
          <div id="adv-settings-body" class="hidden p-3.5 space-y-3 bg-slate-950 border-t border-bordercol">
            <div class="grid grid-cols-2 gap-3">
              <div>
                <label class="block text-slate-400 mb-1">Fast EMA Period</label>
                <input type="number" id="settings-fast-ema" value="12" min="2" max="100" class="w-full bg-slate-900 border border-bordercol rounded-lg px-2.5 py-1.5 text-white">
              </div>
              <div>
                <label class="block text-slate-400 mb-1">Slow EMA Period</label>
                <input type="number" id="settings-slow-ema" value="26" min="5" max="200" class="w-full bg-slate-900 border border-bordercol rounded-lg px-2.5 py-1.5 text-white">
              </div>
            </div>
            <div class="grid grid-cols-2 gap-3">
              <div>
                <label class="block text-slate-400 mb-1">RSI Overbought Max</label>
                <input type="number" id="settings-rsi-ob" value="65" min="50" max="95" class="w-full bg-slate-900 border border-bordercol rounded-lg px-2.5 py-1.5 text-white">
              </div>
              <div>
                <label class="block text-slate-400 mb-1">ADX Trend Threshold</label>
                <input type="number" id="settings-adx-thresh" value="20" min="5" max="50" class="w-full bg-slate-900 border border-bordercol rounded-lg px-2.5 py-1.5 text-white">
              </div>
            </div>
            <div class="grid grid-cols-3 gap-2">
              <div>
                <label class="block text-slate-400 mb-1">Risk Per Trade %</label>
                <input type="number" id="settings-risk-pct" value="1.0" step="0.1" min="0.1" max="10.0" class="w-full bg-slate-900 border border-bordercol rounded-lg px-2 py-1.5 text-white">
              </div>
              <div>
                <label class="block text-slate-400 mb-1">Stop Loss (ATR)</label>
                <input type="number" id="settings-stop-atr" value="2.0" step="0.1" min="0.5" max="10.0" class="w-full bg-slate-900 border border-bordercol rounded-lg px-2 py-1.5 text-white">
              </div>
              <div>
                <label class="block text-slate-400 mb-1">Take Profit (ATR)</label>
                <input type="number" id="settings-target-atr" value="4.0" step="0.1" min="1.0" max="20.0" class="w-full bg-slate-900 border border-bordercol rounded-lg px-2 py-1.5 text-white">
              </div>
            </div>
            <div class="flex items-center gap-2 pt-1">
              <input type="checkbox" id="settings-use-fib" checked class="rounded border-bordercol bg-slate-900 text-emerald-500">
              <label for="settings-use-fib" class="text-slate-300">Enable Fibonacci Pullback Confluence Filter</label>
            </div>
          </div>
        </div>

        <!-- Result Message Box -->
        <div id="settings-result-msg" class="hidden p-3 rounded-lg border text-xs"></div>

        <!-- Action Buttons -->
        <div class="flex flex-col-reverse sm:flex-row items-stretch sm:items-center justify-end gap-2 pt-2 border-t border-bordercol">
          <button type="button" onclick="closeSettingsModal()" class="w-full sm:w-auto px-4 py-2.5 sm:py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 font-semibold rounded-lg transition text-center">
            Cancel
          </button>
          <button type="submit" id="btn-save-settings" class="w-full sm:w-auto px-5 py-2.5 sm:py-2 bg-emerald-600 hover:bg-emerald-500 text-white font-bold rounded-lg shadow transition flex items-center justify-center gap-1.5">
            <i class="fa-solid fa-floppy-disk"></i> Save & Apply Settings
          </button>
        </div>
      </form>
    </div>
  </div>

  <!-- Javascript Logic -->
  <script>
    let botRunning = false;
    let currentUser = null;
    let exchangesList = [];

    // --- Exchange Meta & Selection ---
    async function loadExchanges() {
      try {
        const res = await fetch('/api/exchanges');
        const data = await res.json();
        exchangesList = data.exchanges;

        const sel = document.getElementById('key-exchange-select');
        const btSel = document.getElementById('bt-exchange');
        if (sel) sel.innerHTML = '';
        if (btSel) btSel.innerHTML = '';

        exchangesList.forEach(ex => {
          if (sel) {
            const opt = document.createElement('option');
            opt.value = ex.id;
            opt.innerText = ex.name + (ex.needs_passphrase ? ' (Passphrase required)' : '');
            sel.appendChild(opt);
          }
          if (btSel) {
            const btOpt = document.createElement('option');
            btOpt.value = ex.id;
            btOpt.innerText = ex.name;
            btSel.appendChild(btOpt);
          }
        });
        onExchangeChange();
      } catch (e) {
        console.error('Failed to load exchanges:', e);
      }
    }

    function onExchangeChange() {
      const exId = document.getElementById('key-exchange-select').value;
      const ex = exchangesList.find(e => e.id === exId);
      const hint = document.getElementById('exchange-pass-hint');
      const passContainer = document.getElementById('passphrase-container');
      const passInput = document.getElementById('key-api-passphrase');

      if (ex && ex.needs_passphrase) {
        hint.innerText = `Note: ${ex.name} requires an API Passphrase in addition to Key & Secret.`;
        hint.className = 'text-[10px] text-amber-400 mt-1';
        passInput.required = true;
        passInput.placeholder = 'API Passphrase (Required)';
      } else {
        hint.innerText = `${ex ? ex.name : exId} does not require an API Passphrase.`;
        hint.className = 'text-[10px] text-slate-400 mt-1';
        passInput.required = false;
        passInput.placeholder = 'Passphrase (Optional)';
      }

      const testnetCheckbox = document.getElementById('key-testnet');
      const testnetLabel = document.getElementById('key-testnet-label');
      if (ex && !ex.sandbox) {
        if (testnetCheckbox) testnetCheckbox.checked = false;
        if (testnetLabel) {
          testnetLabel.innerHTML = `Use Testnet / Sandbox Mode <span class="text-amber-400 font-semibold text-[10px]">(${ex.name} operates in Real Trading / Live Mode; no spot sandbox)</span>`;
        }
      } else {
        if (testnetLabel) {
          testnetLabel.innerText = 'Use Testnet / Sandbox Mode (Recommended first)';
        }
      }
    }

    async function testExchangeConnection() {
      const token = localStorage.getItem('bot_auth_token');
      if (!token) {
        alert('Authentication required: please sign up or log in first to test exchange keys.');
        openAuthModal('login');
        return;
      }

      const btn = document.getElementById('btn-test-key');
      btn.disabled = true;
      btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Testing...';

      const payload = {
        exchange_id: document.getElementById('key-exchange-select').value,
        api_key: document.getElementById('key-api-key').value,
        api_secret: document.getElementById('key-api-secret').value,
        api_passphrase: document.getElementById('key-api-passphrase').value,
        testnet: document.getElementById('key-testnet').checked
      };

      const resBox = document.getElementById('key-test-result');
      resBox.classList.remove('hidden');

      try {
        const res = await fetch('/api/exchange/test', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + token
          },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.success) {
          if (typeof data.testnet === 'boolean') {
            const chk = document.getElementById('key-testnet');
            if (chk) chk.checked = data.testnet;
          }
          resBox.className = 'text-xs p-3 rounded-lg border bg-emerald-950/60 border-emerald-500/40 text-emerald-200';
          resBox.innerHTML = `<strong><i class="fa-solid fa-check"></i> ${data.message}</strong><br><span class="text-[11px] text-emerald-300/80">Balances: ${JSON.stringify(data.balances)}</span>`;
        } else {
          resBox.className = 'text-xs p-3 rounded-lg border bg-rose-950/60 border-rose-500/40 text-rose-200';
          resBox.innerHTML = `<strong><i class="fa-solid fa-xmark"></i> ${data.message}</strong>`;
        }
      } catch (e) {
        resBox.className = 'text-xs p-3 rounded-lg border bg-rose-950/60 border-rose-500/40 text-rose-200';
        resBox.innerText = 'Test connection request error: ' + e;
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-plug"></i> Test Connection';
      }
    }

    async function saveExchangeKey(e) {
      e.preventDefault();
      const token = localStorage.getItem('bot_auth_token');
      if (!token) {
        alert('Authentication required: please create an account and log in first to configure exchange credentials.');
        openAuthModal('login');
        return;
      }

      const btn = document.getElementById('btn-save-key');
      btn.disabled = true;
      btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Saving...';

      const payload = {
        exchange_id: document.getElementById('key-exchange-select').value,
        api_key: document.getElementById('key-api-key').value,
        api_secret: document.getElementById('key-api-secret').value,
        api_passphrase: document.getElementById('key-api-passphrase').value,
        testnet: document.getElementById('key-testnet').checked
      };

      const headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + token
      };

      try {
        const res = await fetch('/api/exchange/save', {
          method: 'POST',
          headers: headers,
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        const resBox = document.getElementById('key-test-result');
        resBox.classList.remove('hidden');
        if (data.success) {
          resBox.className = 'text-xs p-3 rounded-lg border bg-emerald-950/60 border-emerald-500/40 text-emerald-200';
          resBox.innerHTML = `<strong><i class="fa-solid fa-check"></i> ${data.message}</strong>`;
          await checkAuth();
          await fetchStatus();
        } else {
          resBox.className = 'text-xs p-3 rounded-lg border bg-rose-950/60 border-rose-500/40 text-rose-200';
          resBox.innerText = data.detail || data.message || 'Save failed';
        }
      } catch (e) {
        alert('Save key error: ' + e);
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-floppy-disk"></i> Save & Apply Credentials';
      }
    }

    // --- Authentication Modals & API ---
    function openAuthModal(mode) {
      document.getElementById('auth-modal').classList.remove('hidden');
      document.getElementById('modal-notice').classList.add('hidden');
      document.getElementById('form-login').classList.add('hidden');
      document.getElementById('form-register').classList.add('hidden');
      document.getElementById('form-forgot').classList.add('hidden');
      const fResend = document.getElementById('form-resend');
      if (fResend) fResend.classList.add('hidden');

      if (mode === 'login') {
        document.getElementById('modal-title').innerText = 'Account Login';
        document.getElementById('modal-subtitle').innerText = 'Sign in to access your bot and exchange credentials.';
        document.getElementById('form-login').classList.remove('hidden');
      } else if (mode === 'register') {
        document.getElementById('modal-title').innerText = 'Create Account';
        document.getElementById('modal-subtitle').innerText = 'Register for instant, persistent access across all devices.';
        document.getElementById('form-register').classList.remove('hidden');
        const lUser = document.getElementById('login-user') ? document.getElementById('login-user').value.trim() : '';
        if (lUser) {
          if (lUser.includes('@')) {
            const regEmail = document.getElementById('reg-email');
            if (regEmail && !regEmail.value) regEmail.value = lUser;
          } else {
            const regUser = document.getElementById('reg-user');
            if (regUser && !regUser.value) regUser.value = lUser;
          }
        }
      } else if (mode === 'forgot') {
        document.getElementById('modal-title').innerText = 'Reset Password';
        document.getElementById('modal-subtitle').innerText = 'Enter your email to receive a password reset message.';
        document.getElementById('form-forgot').classList.remove('hidden');
      } else if (mode === 'resend') {
        document.getElementById('modal-title').innerText = 'Resend Verification Link';
        document.getElementById('modal-subtitle').innerText = 'Enter your registered email address to receive a new verification link.';
        if (fResend) fResend.classList.remove('hidden');
      }
    }

    function switchToRegister(prefillUser) {
      openAuthModal('register');
      if (prefillUser) {
        if (prefillUser.includes('@')) {
          const regEmail = document.getElementById('reg-email');
          if (regEmail) regEmail.value = prefillUser;
        } else {
          const regUser = document.getElementById('reg-user');
          if (regUser) regUser.value = prefillUser;
        }
      }
    }

    function closeAuthModal() {
      document.getElementById('auth-modal').classList.add('hidden');
    }

    function togglePasswordVisibility(inputId, iconId) {
      const input = document.getElementById(inputId);
      const icon = document.getElementById(iconId);
      if (!input || !icon) return;
      if (input.type === 'password') {
        input.type = 'text';
        icon.classList.remove('fa-eye');
        icon.classList.add('fa-eye-slash');
      } else {
        input.type = 'password';
        icon.classList.remove('fa-eye-slash');
        icon.classList.add('fa-eye');
      }
    }

    async function resendVerificationLink(email, btnId) {
      const btn = btnId ? document.getElementById(btnId) : null;
      const oldHtml = btn ? btn.innerHTML : '';
      if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin text-xs"></i> Resending...';
      }

      try {
        const res = await fetch('/api/auth/resend-activation', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email: email })
        });
        const data = await res.json();
        const notice = document.getElementById('modal-notice');
        if (notice) {
          notice.classList.remove('hidden');
          if (data.success) {
            if (data.email_sent) {
              notice.className = 'p-3.5 rounded-lg border bg-emerald-950/80 border-emerald-500/40 text-emerald-200 text-xs space-y-1.5';
              notice.innerHTML = `
                <div class="flex items-center gap-2 font-bold text-emerald-400">
                  <i class="fa-solid fa-circle-check text-base"></i>
                  <span>Verification Link Resent!</span>
                </div>
                <p class="text-slate-200 leading-relaxed">${data.message}</p>
              `;
            } else {
              notice.className = 'p-3.5 rounded-lg border bg-amber-950/80 border-amber-500/40 text-amber-200 text-xs space-y-2.5';
              notice.innerHTML = `
                <div class="flex items-center gap-2 font-bold text-amber-400">
                  <i class="fa-solid fa-triangle-exclamation text-base"></i>
                  <span>Email Not Delivered</span>
                </div>
                <p class="text-slate-200 leading-relaxed">${data.message}</p>
                ${data.dev_activation_link ? `
                  <div class="p-2.5 bg-slate-900/90 rounded border border-amber-500/30 space-y-1">
                    <span class="text-[11px] text-amber-300 font-semibold block">Local / Direct Activation:</span>
                    <a href="${data.dev_activation_link}" target="_blank" class="inline-block px-3 py-1.5 bg-amber-600 hover:bg-amber-500 text-white font-bold text-xs rounded transition shadow">
                      Activate Account Now
                    </a>
                  </div>
                ` : ''}
              `;
            }
          } else {
            notice.className = 'p-3 rounded-lg border bg-amber-950/80 border-amber-500/40 text-amber-200 text-xs';
            notice.innerText = data.message || 'Unable to resend verification link.';
          }
        }
      } catch (err) {
        alert('Error resending verification link: ' + err);
      } finally {
        if (btn) {
          btn.disabled = false;
          btn.innerHTML = oldHtml;
        }
      }
    }

    async function handleResendForm(e) {
      e.preventDefault();
      const email = document.getElementById('resend-email').value;
      await resendVerificationLink(email, 'btn-submit-resend');
    }

    async function handleLogin(e) {
      e.preventDefault();
      const user = document.getElementById('login-user').value.trim();
      const pass = document.getElementById('login-pass').value;
      const notice = document.getElementById('modal-notice');
      let clientToken = localStorage.getItem('bot_auth_token') || '';
      if (!clientToken) {
        try {
          const saved = JSON.parse(localStorage.getItem('bot_saved_account') || '{}');
          if (saved && saved.token) clientToken = saved.token;
        } catch (e) {}
      }

      try {
        const res = await fetch('/api/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username_or_email: user, password: pass, client_token: clientToken })
        });
        const data = await res.json();
        if (res.ok && data.success) {
          localStorage.setItem('bot_auth_token', data.token);
          try {
            document.cookie = "bot_auth_token=" + encodeURIComponent(data.token) + "; path=/; max-age=2592000; SameSite=Lax; Secure";
          } catch (ce) {}
          localStorage.setItem('bot_saved_account', JSON.stringify({
            username: data.user ? data.user.username : user,
            email: data.user ? data.user.email : '',
            token: data.token
          }));
          closeAuthModal();
          await checkAuth();
        } else {
          notice.classList.remove('hidden');
          notice.className = 'p-3 rounded-lg border bg-rose-950/70 border-rose-500/40 text-rose-200 text-xs';
          if (data.needs_activation) {
            notice.innerHTML = `
              <div class="flex items-center gap-2 font-bold text-amber-400">
                <i class="fa-solid fa-envelope-open-text text-base"></i>
                <span>Account Verification Required</span>
              </div>
              <p class="text-slate-300 leading-relaxed mt-1">
                Your account is pending verification. Please open the email sent to <strong>${data.email || user}</strong> and click the link to activate your account.
              </p>
              <div class="pt-2 flex flex-col gap-2">
                <button type="button" id="btn-resend-login" onclick="resendVerificationLink('${data.email || user}', 'btn-resend-login')" class="w-full py-2 bg-amber-600 hover:bg-amber-500 text-white font-semibold rounded-lg shadow transition flex items-center justify-center gap-1.5">
                  <i class="fa-solid fa-paper-plane text-xs"></i> Resend Verification Link
                </button>
              </div>
            `;
          } else if (data.not_found) {
            notice.innerHTML = `
              <div class="flex items-center gap-2 font-bold text-amber-400">
                <i class="fa-solid fa-triangle-exclamation text-base"></i>
                <span>Account Not Found</span>
              </div>
              <p class="text-slate-300 leading-relaxed mt-1">
                Unable to locate credentials for <strong>${user}</strong>. If you haven't registered on this deployment yet, create your account in seconds:
              </p>
              <div class="pt-2 flex flex-col gap-2">
                <button type="button" onclick="switchToRegister('${user.replace(/'/g, "\\'")}')" class="w-full py-2.5 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold rounded-lg shadow transition flex items-center justify-center gap-1.5 text-xs">
                  <i class="fa-solid fa-user-plus text-xs"></i> Register "${user}" Now (Instant Access)
                </button>
                <div class="flex items-center gap-2 my-0.5">
                  <div class="h-px flex-1 bg-slate-700"></div>
                  <span class="text-[10px] text-slate-400">OR SIGN IN VIA EMAIL</span>
                  <div class="h-px flex-1 bg-slate-700"></div>
                </div>
                <button type="button" id="btn-magic-notfound" onclick="handleSendMagicLink('${data.email || (user.includes('@') ? user : '')}', 'btn-magic-notfound')" class="w-full py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 font-semibold rounded-lg shadow transition flex items-center justify-center gap-1.5 text-xs">
                  <i class="fa-solid fa-wand-magic-sparkles text-amber-400 text-xs"></i> Email 1-Click Instant Sign-In Link
                </button>
              </div>
            `;
          } else {
            notice.innerText = data.detail || data.message || 'Login failed';
          }
        }
      } catch (err) {
        alert('Login error: ' + err);
      }
    }

    async function handleSendMagicLink(presetEmail, btnId) {
      let email = presetEmail;
      if (!email || !email.includes('@')) {
        const userInput = document.getElementById('login-user') ? document.getElementById('login-user').value.trim() : '';
        if (userInput && userInput.includes('@')) {
          email = userInput;
        } else {
          email = prompt('Enter your registered email address to receive an Instant Sign-In Link:');
        }
      }
      if (!email || !email.includes('@')) {
        return;
      }

      const btn = btnId ? document.getElementById(btnId) : document.getElementById('btn-magic-login');
      const oldHtml = btn ? btn.innerHTML : '';
      if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin text-xs"></i> Sending Link...';
      }

      const notice = document.getElementById('modal-notice');
      try {
        const res = await fetch('/api/auth/magic-link', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email: email })
        });
        const data = await res.json();
        if (notice) {
          notice.classList.remove('hidden');
          if (data.success) {
            notice.className = 'p-3.5 rounded-lg border bg-emerald-950/80 border-emerald-500/40 text-emerald-200 text-xs space-y-2';
            notice.innerHTML = `
              <div class="flex items-center gap-2 font-bold text-emerald-400">
                <i class="fa-solid fa-envelope-circle-check text-base"></i>
                <span>1-Click Sign-In Link Sent!</span>
              </div>
              <p class="text-slate-200 leading-relaxed">
                We sent an instant sign-in link to <strong>${email}</strong>. Open your email and click the button to log in immediately on this device.
              </p>
            `;
          } else {
            notice.className = 'p-3 rounded-lg border bg-rose-950/70 border-rose-500/40 text-rose-200 text-xs';
            notice.innerText = data.detail || data.message || 'Failed to send sign-in link.';
          }
        }
      } catch (e) {
        alert('Sign-in link error: ' + e);
      } finally {
        if (btn) {
          btn.disabled = false;
          btn.innerHTML = oldHtml;
        }
      }
    }

    async function handleRegister(e) {
      e.preventDefault();
      const user = document.getElementById('reg-user').value;
      const email = document.getElementById('reg-email').value;
      const pass = document.getElementById('reg-pass').value;
      const notice = document.getElementById('modal-notice');

      try {
        const res = await fetch('/api/auth/register', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username: user, email: email, password: pass })
        });
        const data = await res.json();
        notice.classList.remove('hidden');
        if (res.ok && data.success) {
          if (data.token) {
            localStorage.setItem('bot_auth_token', data.token);
            try {
              document.cookie = "bot_auth_token=" + encodeURIComponent(data.token) + "; path=/; max-age=2592000; SameSite=Lax; Secure";
            } catch (ce) {}
            localStorage.setItem('bot_saved_account', JSON.stringify({
              username: data.user ? data.user.username : user,
              email: data.user ? data.user.email : email,
              token: data.token
            }));
            closeAuthModal();
            await checkAuth();
            alert('Welcome, ' + (data.user ? data.user.username : user) + '! Your account is active and you are now signed in.');
            return;
          }
          if (data.email_sent) {
            notice.className = 'p-4 rounded-xl border bg-emerald-950/80 border-emerald-500/40 text-emerald-200 text-xs space-y-3';
            notice.innerHTML = `
              <div class="flex items-center gap-2.5 text-emerald-400 font-bold text-sm">
                <i class="fa-solid fa-envelope-circle-check text-xl"></i>
                <span>Verification Email Sent!</span>
              </div>
              <p class="text-slate-200 leading-relaxed">
                We have sent an account verification email to <strong>${email}</strong>.
              </p>
              <div class="p-3 bg-slate-900/90 rounded-lg border border-bordercol/80 space-y-1 text-[11px] text-slate-300">
                <div class="font-semibold text-white flex items-center gap-1.5">
                  <i class="fa-solid fa-circle-info text-emerald-400"></i> Next Step:
                </div>
                <p>Please check your email inbox (and spam/junk folder) and click the verification link inside the email to activate your account.</p>
                <p class="text-slate-400 text-[10px] italic pt-1">Account verification must be accepted directly from the email message.</p>
              </div>
              <div class="flex flex-col sm:flex-row gap-2 pt-1">
                <button type="button" id="btn-resend-reg" onclick="resendVerificationLink('${email}', 'btn-resend-reg')" class="flex-1 py-2 bg-emerald-700/90 hover:bg-emerald-600 text-white font-semibold rounded-lg shadow transition flex items-center justify-center gap-1.5">
                  <i class="fa-solid fa-paper-plane text-xs"></i> Resend Verification Link
                </button>
                <button type="button" onclick="openAuthModal('login')" class="flex-1 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold rounded-lg border border-slate-700 transition flex items-center justify-center gap-1.5">
                  <i class="fa-solid fa-right-to-bracket text-xs"></i> Proceed to Login
                </button>
              </div>
            `;
          } else {
            notice.className = 'p-4 rounded-xl border bg-amber-950/80 border-amber-500/40 text-amber-200 text-xs space-y-3';
            notice.innerHTML = `
              <div class="flex items-center gap-2.5 text-amber-400 font-bold text-sm">
                <i class="fa-solid fa-triangle-exclamation text-xl"></i>
                <span>Account Created (Verification Email Delivery Issue)</span>
              </div>
              <p class="text-slate-200 leading-relaxed">${data.message}</p>
              ${data.dev_activation_link ? `
                <div class="p-3 bg-slate-900/90 rounded-lg border border-amber-500/40 space-y-1.5 text-xs text-amber-200">
                  <div class="font-bold text-amber-400 flex items-center gap-1.5">
                    <i class="fa-solid fa-bolt"></i> Local / Dev Direct Activation:
                  </div>
                  <p class="text-[11px] text-slate-300">Because live email delivery was not completed, you can activate this account directly:</p>
                  <div class="pt-1">
                    <a href="${data.dev_activation_link}" target="_blank" class="inline-block px-3.5 py-1.5 bg-amber-600 hover:bg-amber-500 text-white font-bold rounded-lg shadow transition text-xs">
                      Activate Account Directly
                    </a>
                  </div>
                </div>
              ` : ''}
              <div class="flex flex-col sm:flex-row gap-2 pt-1">
                <button type="button" id="btn-resend-reg-failed" onclick="resendVerificationLink('${email}', 'btn-resend-reg-failed')" class="flex-1 py-2 bg-emerald-700 hover:bg-emerald-600 text-white font-semibold rounded-lg shadow transition flex items-center justify-center gap-1.5 text-xs">
                  <i class="fa-solid fa-paper-plane text-xs"></i> Resend Verification Link
                </button>
                <button type="button" onclick="openAuthModal('login')" class="flex-1 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold rounded-lg border border-slate-700 transition flex items-center justify-center gap-1.5 text-xs">
                  <i class="fa-solid fa-right-to-bracket text-xs"></i> Proceed to Login
                </button>
              </div>
            `;
          }
          document.getElementById('form-register').reset();
        } else {
          notice.className = 'p-3 rounded-lg border bg-rose-950/70 border-rose-500/40 text-rose-200 text-xs';
          notice.innerText = data.detail || data.message || 'Registration failed';
        }
      } catch (err) {
        alert('Registration error: ' + err);
      }
    }

    async function handleForgot(e) {
      e.preventDefault();
      const email = document.getElementById('forgot-email').value;
      const notice = document.getElementById('modal-notice');

      try {
        const res = await fetch('/api/auth/forgot-password', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email: email })
        });
        const data = await res.json();
        notice.classList.remove('hidden');
        notice.className = 'p-3.5 rounded-lg border bg-amber-950/80 border-amber-500/40 text-amber-200 text-xs space-y-2';
        notice.innerHTML = `
          <div class="flex items-center gap-2 font-bold text-amber-400">
            <i class="fa-solid fa-envelope"></i>
            <span>Password Reset Email Sent</span>
          </div>
          <p class="text-slate-300 leading-relaxed">
            If an account exists for <strong>${email}</strong>, password reset instructions have been dispatched. Please check your inbox (and spam folder) to set a new password.
          </p>
        `;
      } catch (err) {
        alert('Reset error: ' + err);
      }
    }

    function toggleMobileMenu() {
      const panel = document.getElementById('mobile-menu-panel');
      const icon = document.getElementById('mobile-menu-icon');
      if (!panel) return;
      const isHidden = panel.classList.toggle('hidden');
      if (icon) {
        icon.className = isHidden ? 'fa-solid fa-bars' : 'fa-solid fa-xmark';
      }
    }

    function handleMobileAccountClick() {
      if (currentUser) {
        const panel = document.getElementById('mobile-menu-panel');
        if (panel && panel.classList.contains('hidden')) {
          toggleMobileMenu();
        }
      } else {
        openAuthModal('login');
      }
    }

    async function confirmDeleteAccount() {
      if (!confirm('⚠️ Permanently delete your registered account, active sessions, and saved exchange keys?\\\\n\\\\nThis cannot be undone. Click OK to delete.')) return;
      const token = localStorage.getItem('bot_auth_token');
      if (!token) return;
      try {
        const res = await fetch('/api/auth/account', {
          method: 'DELETE',
          headers: { 'Authorization': 'Bearer ' + token }
        });
        const data = await res.json();
        if (data.success) {
          alert('Account deleted successfully.');
          logout();
        } else {
          alert('Failed to delete account: ' + (data.detail || data.message));
        }
      } catch (err) {
        alert('Error deleting account: ' + err);
      }
    }

    async function checkAuth() {
      let token = localStorage.getItem('bot_auth_token');
      if (!token) {
        const m = document.cookie.match(/(?:^|;[ \t]*)bot_auth_token=([^;]+)/);
        if (m) {
          try {
            token = decodeURIComponent(m[1]);
            localStorage.setItem('bot_auth_token', token);
          } catch (e) {}
        }
      }
      const authButtons = document.getElementById('auth-buttons');
      const userBadge = document.getElementById('user-badge');
      const authButtonsMob = document.getElementById('auth-buttons-mobile');
      const userBadgeMob = document.getElementById('user-badge-mobile');
      const lockedDiv = document.getElementById('api-keys-locked');
      const unlockedDiv = document.getElementById('api-keys-unlocked');
      const bottomAccountText = document.getElementById('bottom-account-text');

      if (!token) {
        if (authButtons) authButtons.classList.remove('hidden');
        if (userBadge) userBadge.classList.add('hidden');
        if (authButtonsMob) authButtonsMob.classList.remove('hidden');
        if (userBadgeMob) userBadgeMob.classList.add('hidden');
        if (lockedDiv) lockedDiv.classList.remove('hidden');
        if (unlockedDiv) unlockedDiv.classList.add('hidden');
        if (bottomAccountText) bottomAccountText.innerText = 'Account';
        return;
      }

      try {
        const res = await fetch('/api/auth/me', {
          headers: { 'Authorization': 'Bearer ' + token }
        });
        const data = await res.json();
        if (data.authenticated) {
          currentUser = data.user;
          try {
            localStorage.setItem('bot_saved_account', JSON.stringify({
              username: currentUser.username,
              email: currentUser.email,
              token: token
            }));
          } catch (e) {}
          if (authButtons) authButtons.classList.add('hidden');
          if (userBadge) userBadge.classList.remove('hidden');
          if (authButtonsMob) authButtonsMob.classList.add('hidden');
          if (userBadgeMob) userBadgeMob.classList.remove('hidden');

          if (document.getElementById('user-name-display')) {
            document.getElementById('user-name-display').innerText = currentUser.username;
          }
          if (document.getElementById('user-name-display-mobile')) {
            document.getElementById('user-name-display-mobile').innerText = currentUser.username;
          }
          if (bottomAccountText) {
            bottomAccountText.innerText = currentUser.username.length > 7 ? currentUser.username.slice(0, 6) + '…' : currentUser.username;
          }

          if (lockedDiv) lockedDiv.classList.add('hidden');
          if (unlockedDiv) unlockedDiv.classList.remove('hidden');

          // Render saved exchange keys
          const list = document.getElementById('configured-keys-list');
          list.innerHTML = '';
          if (data.saved_exchanges && data.saved_exchanges.length > 0) {
            data.saved_exchanges.forEach(k => {
              const div = document.createElement('div');
              div.className = 'flex items-center justify-between bg-slate-900/90 border border-slate-800 rounded-lg p-2 text-xs';
              div.innerHTML = `
                <div class="flex items-center gap-2 min-w-0">
                  <span class="font-bold text-white uppercase">${k.exchange_id}</span>
                  <span class="text-[10px] px-1.5 py-0.5 rounded ${k.testnet ? 'bg-blue-500/20 text-blue-300' : 'bg-emerald-500/20 text-emerald-300'} shrink-0">${k.testnet ? 'TESTNET' : 'LIVE'}</span>
                  <span class="font-mono text-slate-400 text-[11px] truncate">${k.api_key_masked}</span>
                </div>
                <span class="text-[10px] text-slate-500 shrink-0">Configured</span>
              `;
              list.appendChild(div);
            });
          }
        } else {
          localStorage.removeItem('bot_auth_token');
          document.cookie = "bot_auth_token=; path=/; max-age=0; SameSite=Lax; Secure";
          currentUser = null;
          if (authButtons) authButtons.classList.remove('hidden');
          if (userBadge) userBadge.classList.add('hidden');
          if (authButtonsMob) authButtonsMob.classList.remove('hidden');
          if (userBadgeMob) userBadgeMob.classList.add('hidden');
          if (lockedDiv) lockedDiv.classList.remove('hidden');
          if (unlockedDiv) unlockedDiv.classList.add('hidden');
          if (bottomAccountText) bottomAccountText.innerText = 'Account';
        }
      } catch (e) {
        console.error('Check auth error:', e);
        if (lockedDiv) lockedDiv.classList.remove('hidden');
        if (unlockedDiv) unlockedDiv.classList.add('hidden');
      }
    }

    function logout() {
      localStorage.removeItem('bot_auth_token');
      document.cookie = "bot_auth_token=; path=/; max-age=0; SameSite=Lax; Secure";
      currentUser = null;
      checkAuth();
    }

    // --- Trading Environment Switcher (Testnet vs Real Trading) ---
    async function switchTradingMode(targetMode) {
      const isReal = (targetMode === 'real');
      if (isReal) {
        const confirmed = confirm('⚠️ SWITCH TO REAL TRADING (LIVE)?\\n\\nWhen started, the bot will place REAL orders with REAL FUNDS on your configured exchange.\\n\\nPlease make sure you have verified your exchange credentials and risk settings.\\n\\nClick OK to switch to Real Trading.');
        if (!confirmed) return;
      }

      try {
        const res = await fetch('/api/bot/mode', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ trading_mode: targetMode })
        });
        const data = await res.json();
        if (res.ok && data.success) {
          updateTradingModeUI(data.is_real, data.mode, data.testnet);
          await fetchStatus();
          await refreshLogs();
        } else {
          alert('Failed to switch trading mode: ' + (data.message || 'Unknown error'));
        }
      } catch (err) {
        alert('Error switching trading mode: ' + err);
      }
    }

    function updateTradingModeUI(isReal, mode, testnet) {
      const badgeMode = document.getElementById('badge-mode');
      const envBtnTestnet = document.getElementById('env-btn-testnet');
      const envBtnReal = document.getElementById('env-btn-real');
      const envTitle = document.getElementById('env-mode-title');
      const envDesc = document.getElementById('env-mode-desc');
      const envIcon = document.getElementById('env-icon');
      const cashLabel = document.getElementById('val-cash-label');

      if (isReal) {
        if (badgeMode) {
          badgeMode.innerText = 'REAL TRADING (LIVE)';
          badgeMode.className = 'text-[10px] px-2 py-0.5 rounded-full font-bold bg-rose-500/20 text-rose-400 border border-rose-500/40 animate-pulse';
        }
        if (envBtnReal) envBtnReal.className = 'px-3 py-1.5 bg-rose-600 text-white text-xs font-bold rounded-lg shadow transition flex items-center gap-1.5';
        if (envBtnTestnet) envBtnTestnet.className = 'px-3 py-1.5 bg-slate-800 text-slate-300 hover:text-white border border-slate-700 text-xs font-bold rounded-lg transition flex items-center gap-1.5';
        if (envTitle) {
          envTitle.innerText = 'REAL TRADING (LIVE)';
          envTitle.className = 'text-rose-400 font-bold';
        }
        if (envDesc) envDesc.innerText = 'Live execution with real capital on your exchange account.';
        if (envIcon) {
          envIcon.className = 'w-9 h-9 rounded-lg bg-rose-500/20 text-rose-400 flex items-center justify-center font-bold text-base';
          envIcon.innerHTML = '<i class="fa-solid fa-bolt"></i>';
        }
        if (cashLabel) cashLabel.innerText = 'Exchange Balance';
      } else {
        if (badgeMode) {
          badgeMode.innerText = 'TESTNET';
          badgeMode.className = 'text-[10px] px-2 py-0.5 rounded-full font-bold bg-blue-500/20 text-blue-400 border border-blue-500/30';
        }
        if (envBtnTestnet) envBtnTestnet.className = 'px-3 py-1.5 bg-blue-600 text-white text-xs font-bold rounded-lg shadow transition flex items-center gap-1.5';
        if (envBtnReal) envBtnReal.className = 'px-3 py-1.5 bg-slate-800 text-slate-300 hover:text-white border border-slate-700 text-xs font-bold rounded-lg transition flex items-center gap-1.5';
        if (envTitle) {
          envTitle.innerText = 'TESTNET / SIMULATION';
          envTitle.className = 'text-blue-400 font-bold';
        }
        if (envDesc) envDesc.innerText = 'Safe simulated execution. Zero real funds at risk.';
        if (envIcon) {
          envIcon.className = 'w-9 h-9 rounded-lg bg-blue-500/20 text-blue-400 flex items-center justify-center font-bold text-base';
          envIcon.innerHTML = '<i class="fa-solid fa-flask"></i>';
        }
        if (cashLabel) cashLabel.innerText = 'Paper Cash';
      }
    }

    // --- Bot Settings & Manual Trading Handlers ---
    let currentSettings = null;
    let availableStrategies = [];

    async function openSettingsModal() {
      const modal = document.getElementById('settings-modal');
      if (modal) modal.classList.remove('hidden');
      await loadBotSettings();
    }

    function closeSettingsModal() {
      const modal = document.getElementById('settings-modal');
      if (modal) modal.classList.add('hidden');
      const msg = document.getElementById('settings-result-msg');
      if (msg) msg.classList.add('hidden');
    }

    function toggleAdvancedSettings() {
      const body = document.getElementById('adv-settings-body');
      const chev = document.getElementById('adv-chevron');
      if (body) {
        if (body.classList.contains('hidden')) {
          body.classList.remove('hidden');
          if (chev) chev.style.transform = 'rotate(180deg)';
        } else {
          body.classList.add('hidden');
          if (chev) chev.style.transform = 'rotate(0deg)';
        }
      }
    }

    function onExecutionModeOptionChange() {
      const autoRadio = document.getElementById('radio-mode-auto');
      const isAuto = autoRadio ? autoRadio.checked : true;
      const autoCard = document.getElementById('opt-mode-auto-card');
      const manualCard = document.getElementById('opt-mode-manual-card');

      if (isAuto) {
        if (autoCard) autoCard.className = 'p-3 rounded-xl border border-purple-500/60 bg-purple-950/30 cursor-pointer transition flex items-start gap-2.5';
        if (manualCard) manualCard.className = 'p-3 rounded-xl border border-bordercol bg-slate-900/80 cursor-pointer transition flex items-start gap-2.5 hover:border-slate-600';
      } else {
        if (autoCard) autoCard.className = 'p-3 rounded-xl border border-bordercol bg-slate-900/80 cursor-pointer transition flex items-start gap-2.5 hover:border-slate-600';
        if (manualCard) manualCard.className = 'p-3 rounded-xl border border-amber-500/60 bg-amber-950/30 cursor-pointer transition flex items-start gap-2.5';
      }
    }

    function onStrategyOptionChange() {
      const sel = document.getElementById('settings-strategy');
      if (!sel) return;
      const stratId = sel.value;
      const strat = availableStrategies.find(s => s.id === stratId);
      const descEl = document.getElementById('settings-strat-desc');
      if (descEl && strat) {
        descEl.innerHTML = `<span class="font-bold text-emerald-400">${strat.name}</span>: ${strat.description}`;
      }
    }

    async function loadBotSettings() {
      try {
        const res = await fetch('/api/bot/settings');
        const data = await res.json();
        currentSettings = data;
        availableStrategies = data.available_strategies || [];

        // 1. Populate strategy select
        const stratSel = document.getElementById('settings-strategy');
        if (stratSel) {
          stratSel.innerHTML = '';
          availableStrategies.forEach(s => {
            const opt = document.createElement('option');
            opt.value = s.id;
            opt.innerText = s.name + (s.recommended ? ' ★ (Recommended)' : '');
            if (s.id === data.strategy) opt.selected = true;
            stratSel.appendChild(opt);
          });
          onStrategyOptionChange();
        }

        // 2. Set execution mode
        const radioManual = document.getElementById('radio-mode-manual');
        const radioAuto = document.getElementById('radio-mode-auto');
        if (data.execution_mode === 'manual') {
          if (radioManual) radioManual.checked = true;
        } else {
          if (radioAuto) radioAuto.checked = true;
        }
        onExecutionModeOptionChange();

        // 3. Set timeframe (with 5m included and verified)
        const tfSel = document.getElementById('settings-timeframe');
        if (tfSel) {
          tfSel.value = data.timeframe || '15m';
        }

        // 4. Set symbol
        const symInput = document.getElementById('settings-symbol');
        if (symInput) symInput.value = data.symbol || 'BTC/USDT';

        // 5. Advanced technical parameters
        if (document.getElementById('settings-fast-ema')) document.getElementById('settings-fast-ema').value = data.fast_ema;
        if (document.getElementById('settings-slow-ema')) document.getElementById('settings-slow-ema').value = data.slow_ema;
        if (document.getElementById('settings-rsi-ob')) document.getElementById('settings-rsi-ob').value = data.rsi_overbought;
        if (document.getElementById('settings-adx-thresh')) document.getElementById('settings-adx-thresh').value = data.adx_trend_threshold;
        if (document.getElementById('settings-risk-pct')) document.getElementById('settings-risk-pct').value = (data.risk_per_trade_pct * 100).toFixed(1);
        if (document.getElementById('settings-stop-atr')) document.getElementById('settings-stop-atr').value = data.stop_loss_atr_mult;
        if (document.getElementById('settings-target-atr')) document.getElementById('settings-target-atr').value = data.take_profit_atr_mult;
        if (document.getElementById('settings-use-fib')) document.getElementById('settings-use-fib').checked = data.use_fib_filter;
      } catch (err) {
        console.error('Failed to load bot settings:', err);
      }
    }

    async function handleSaveSettings(e) {
      e.preventDefault();
      const btn = document.getElementById('btn-save-settings');
      const oldHtml = btn ? btn.innerHTML : '';
      if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Saving...';
      }

      const radioManual = document.getElementById('radio-mode-manual');
      const isManual = radioManual ? radioManual.checked : false;
      const payload = {
        execution_mode: isManual ? 'manual' : 'auto',
        strategy: document.getElementById('settings-strategy').value,
        timeframe: document.getElementById('settings-timeframe').value,
        symbol: document.getElementById('settings-symbol').value.trim().toUpperCase(),
        use_fib_filter: document.getElementById('settings-use-fib').checked,
        fast_ema: parseInt(document.getElementById('settings-fast-ema').value) || 12,
        slow_ema: parseInt(document.getElementById('settings-slow-ema').value) || 26,
        rsi_overbought: parseFloat(document.getElementById('settings-rsi-ob').value) || 65.0,
        adx_trend_threshold: parseFloat(document.getElementById('settings-adx-thresh').value) || 20.0,
        risk_per_trade_pct: (parseFloat(document.getElementById('settings-risk-pct').value) || 1.0) / 100.0,
        stop_loss_atr_mult: parseFloat(document.getElementById('settings-stop-atr').value) || 2.0,
        take_profit_atr_mult: parseFloat(document.getElementById('settings-target-atr').value) || 4.0,
      };

      const msgBox = document.getElementById('settings-result-msg');
      if (msgBox) msgBox.classList.remove('hidden');

      try {
        const res = await fetch('/api/bot/settings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (res.ok && data.success) {
          if (msgBox) {
            msgBox.className = 'p-3 rounded-lg border bg-emerald-950/80 border-emerald-500/40 text-emerald-200 text-xs';
            msgBox.innerHTML = `<strong><i class="fa-solid fa-check"></i> ${data.message}</strong>`;
          }
          await fetchStatus();
          await refreshAnalysis();
          await refreshLogs();
          setTimeout(() => closeSettingsModal(), 1800);
        } else {
          if (msgBox) {
            msgBox.className = 'p-3 rounded-lg border bg-rose-950/80 border-rose-500/40 text-rose-200 text-xs';
            msgBox.innerText = data.detail || data.message || 'Failed to save settings';
          }
        }
      } catch (err) {
        if (msgBox) {
          msgBox.className = 'p-3 rounded-lg border bg-rose-950/80 border-rose-500/40 text-rose-200 text-xs';
          msgBox.innerText = 'Network error: ' + err;
        }
      } finally {
        if (btn) {
          btn.disabled = false;
          btn.innerHTML = oldHtml;
        }
      }
    }

    async function quickSetExecutionMode(targetMode) {
      try {
        const res = await fetch('/api/bot/settings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ execution_mode: targetMode })
        });
        const data = await res.json();
        if (res.ok && data.success) {
          await fetchStatus();
          await refreshLogs();
        } else {
          alert('Failed to switch execution mode: ' + (data.detail || data.message));
        }
      } catch (err) {
        alert('Error changing execution mode: ' + err);
      }
    }

    async function triggerManualTrade(action) {
      const actName = action.toUpperCase();
      if (!confirm(`Confirm manual ${actName} order execution at current market price?`)) return;

      const btn = document.getElementById('btn-manual-' + action.toLowerCase());
      const oldHtml = btn ? btn.innerHTML : '';
      if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Placing...';
      }

      try {
        const res = await fetch('/api/bot/trade/manual', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: action.toLowerCase() })
        });
        const data = await res.json();
        if (res.ok && data.success) {
          alert(data.message);
          await fetchStatus();
          await refreshLogs();
        } else {
          alert(`Manual ${actName} failed: ` + (data.detail || data.message));
        }
      } catch (err) {
        alert(`Error executing manual ${actName}: ` + err);
      } finally {
        if (btn) {
          btn.disabled = false;
          btn.innerHTML = oldHtml;
        }
      }
    }

    // --- Status, Analysis, Backtest & Logs ---
    async function fetchStatus() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        botRunning = data.running;

        const isReal = (data.mode === 'live' && !data.testnet);
        updateTradingModeUI(isReal, data.mode, data.testnet);
        document.getElementById('val-symbol-tag').innerText = `${data.symbol} ${data.timeframe}`;

        // Update Execution Mode & Strategy UI on Dashboard
        const isManual = (data.execution_mode === 'manual');
        const activeStratTag = document.getElementById('active-strat-tag');
        const activeTfTag = document.getElementById('active-tf-tag');
        const activeModeTag = document.getElementById('active-mode-tag');
        const manualPanel = document.getElementById('manual-control-panel');

        if (activeStratTag) activeStratTag.innerText = `Strategy: ${data.strategy_short_name || data.strategy}`;
        if (activeTfTag) activeTfTag.innerText = `Timeframe: ${data.timeframe}`;
        if (activeModeTag) {
          activeModeTag.innerText = isManual ? 'Manual Trading' : 'Auto Trading';
          activeModeTag.className = isManual ? 'text-amber-400 font-semibold' : 'text-purple-300 font-semibold';
        }
        if (manualPanel) {
          manualPanel.classList.toggle('hidden', !isManual);
        }

        const dot = document.getElementById('status-dot');
        const text = document.getElementById('status-text');
        const dotMob = document.getElementById('status-dot-mobile');
        const textMob = document.getElementById('status-text-mobile');

        const btnToggle = document.getElementById('btn-toggle');
        const btnToggleText = document.getElementById('btn-toggle-text');
        const btnToggleIcon = document.getElementById('btn-toggle-icon');

        const btnToggleMob = document.getElementById('btn-toggle-mobile');
        const btnToggleTextMob = document.getElementById('btn-toggle-text-mobile');
        const btnToggleIconMob = document.getElementById('btn-toggle-icon-mobile');

        const bottomBtnText = document.getElementById('bottom-btn-toggle-text');
        const bottomBtnIcon = document.getElementById('bottom-btn-toggle-icon');

        if (data.running) {
          if (dot) dot.className = 'inline-block w-2 h-2 rounded-full bg-emerald-400 animate-pulse';
          if (text) {
            text.innerText = 'RUNNING (' + data.mode.toUpperCase() + ')';
            text.className = 'font-semibold text-emerald-400';
          }
          if (dotMob) dotMob.className = 'inline-block w-2 h-2 rounded-full bg-emerald-400 animate-pulse';
          if (textMob) {
            textMob.innerText = 'RUNNING';
            textMob.className = 'font-semibold text-emerald-400';
          }

          if (btnToggleText) btnToggleText.innerText = 'Stop Bot';
          if (btnToggleIcon) btnToggleIcon.className = 'fa-solid fa-stop';
          if (btnToggle) btnToggle.className = 'px-3.5 py-1.5 bg-amber-600 hover:bg-amber-500 text-white font-semibold rounded-lg shadow transition flex items-center gap-1.5';

          if (btnToggleTextMob) btnToggleTextMob.innerText = 'Stop';
          if (btnToggleIconMob) btnToggleIconMob.className = 'fa-solid fa-stop text-[10px]';
          if (btnToggleMob) btnToggleMob.className = 'px-2.5 py-1.5 bg-amber-600 active:bg-amber-700 text-white text-xs font-semibold rounded-lg shadow transition flex items-center gap-1';

          if (bottomBtnText) { bottomBtnText.innerText = 'Stop'; bottomBtnText.className = 'text-[10px] font-semibold mt-0.5 text-amber-400'; }
          if (bottomBtnIcon) { bottomBtnIcon.className = 'fa-solid fa-stop text-amber-400 text-base'; }
        } else {
          if (dot) dot.className = 'inline-block w-2 h-2 rounded-full bg-slate-500';
          if (text) {
            text.innerText = data.kill_file_exists ? 'HALTED (KILL SWITCH)' : 'IDLE';
            text.className = data.kill_file_exists ? 'font-semibold text-red-400' : 'font-semibold text-slate-400';
          }
          if (dotMob) dotMob.className = 'inline-block w-2 h-2 rounded-full ' + (data.kill_file_exists ? 'bg-red-500' : 'bg-slate-500');
          if (textMob) {
            textMob.innerText = data.kill_file_exists ? 'HALTED' : 'IDLE';
            textMob.className = data.kill_file_exists ? 'font-semibold text-red-400' : 'font-semibold text-slate-400';
          }

          if (btnToggleText) btnToggleText.innerText = 'Start Bot';
          if (btnToggleIcon) btnToggleIcon.className = 'fa-solid fa-play';
          if (btnToggle) btnToggle.className = 'px-3.5 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold rounded-lg shadow transition flex items-center gap-1.5';

          if (btnToggleTextMob) btnToggleTextMob.innerText = 'Start';
          if (btnToggleIconMob) btnToggleIconMob.className = 'fa-solid fa-play text-[10px]';
          if (btnToggleMob) btnToggleMob.className = 'px-2.5 py-1.5 bg-emerald-600 active:bg-emerald-700 text-white text-xs font-semibold rounded-lg shadow transition flex items-center gap-1';

          if (bottomBtnText) { bottomBtnText.innerText = 'Start'; bottomBtnText.className = 'text-[10px] font-semibold mt-0.5 text-slate-300'; }
          if (bottomBtnIcon) { bottomBtnIcon.className = 'fa-solid fa-play text-emerald-400 text-base'; }
        }

        document.getElementById('val-equity').innerText = '$' + data.equity.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
        document.getElementById('val-peak').innerText = '$' + data.peak_equity.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
        document.getElementById('val-cash').innerText = '$' + data.paper_cash.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
        document.getElementById('val-pos').innerText = data.paper_position.toFixed(6);
        document.getElementById('val-drawdown').innerText = data.drawdown_pct.toFixed(1) + '%';
        document.getElementById('val-daily-loss').innerText = data.daily_loss_pct.toFixed(1) + '%';
        document.getElementById('val-streak').innerText = `${data.consecutive_losses} / 4`;

        const posBanner = document.getElementById('position-banner');
        if (data.active_position) {
          posBanner.classList.remove('hidden');
          document.getElementById('pos-details').innerText = 
            `Entry: $${data.active_position.entry.toFixed(2)} | Stop: $${data.active_position.stop.toFixed(2)} | Target: $${data.active_position.target.toFixed(2)} | Size: ${data.active_position.size.toFixed(6)}`;
        } else {
          posBanner.classList.add('hidden');
        }
      } catch (e) {
        console.error('Error fetching status:', e);
      }
    }

    async function toggleBot() {
      const endpoint = botRunning ? '/api/bot/stop' : '/api/bot/start';
      try {
        const res = await fetch(endpoint, { method: 'POST' });
        await fetchStatus();
        await refreshLogs();
      } catch (e) {
        alert('Action failed: ' + e);
      }
    }

    async function triggerKillSwitch() {
      if (!confirm('Are you sure you want to trigger the emergency KILL SWITCH? This halts all execution immediately.')) return;
      try {
        await fetch('/api/bot/kill', { method: 'POST' });
        await fetchStatus();
        await refreshLogs();
      } catch (e) {
        alert('Kill switch failed: ' + e);
      }
    }

    async function refreshAnalysis() {
      try {
        const res = await fetch('/api/analysis');
        if (!res.ok) {
          console.warn('Analysis endpoint returned HTTP status:', res.status);
          return;
        }
        const data = await res.json();
        if (!data || data.price === undefined || data.price === null) {
          console.warn('Invalid analysis payload:', data);
          return;
        }

        const priceEl = document.getElementById('an-price');
        if (priceEl) priceEl.innerText = '$' + Number(data.price).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
        
        const exEl = document.getElementById('an-exchange');
        if (exEl) exEl.innerText = `Exchange: ${data.exchange} (${data.symbol})`;

        const trendEl = document.getElementById('an-trend');
        if (trendEl) {
          trendEl.innerText = data.trend;
          trendEl.className = data.trend === 'BULLISH' ? 'text-base font-bold mt-0.5 text-emerald-400' : 'text-base font-bold mt-0.5 text-rose-400';
        }
        
        const emasEl = document.getElementById('an-emas');
        if (emasEl) emasEl.innerText = `Fast: $${data.ema_fast} | Slow: $${data.ema_slow}`;

        const rsiEl = document.getElementById('an-rsi');
        if (rsiEl) rsiEl.innerText = Number(data.rsi).toFixed(1);
        const rsiStatusEl = document.getElementById('an-rsi-status');
        if (rsiStatusEl) {
          rsiStatusEl.innerText = data.rsi_status;
          rsiStatusEl.className = data.rsi_status === 'OVERBOUGHT' ? 'text-[10px] mt-1 text-rose-400' : (data.rsi_status === 'OVERSOLD' ? 'text-[10px] mt-1 text-emerald-400' : 'text-[10px] mt-1 text-slate-400');
        }

        const adxEl = document.getElementById('an-adx');
        if (adxEl) adxEl.innerText = Number(data.adx).toFixed(1);
        const adxStatusEl = document.getElementById('an-adx-status');
        if (adxStatusEl) {
          adxStatusEl.innerText = data.adx_status;
          adxStatusEl.className = Number(data.adx) >= 20 ? 'text-[10px] mt-1 text-emerald-400 font-semibold' : 'text-[10px] mt-1 text-slate-400';
        }

        const sigEl = document.getElementById('val-signal');
        if (sigEl) {
          sigEl.innerText = data.signal;
          sigEl.className = data.signal === 'BUY' ? 'text-lg font-bold mt-0.5 text-emerald-400 animate-pulse' : (data.signal === 'SELL' ? 'text-lg font-bold mt-0.5 text-rose-400' : 'text-lg font-bold mt-0.5 text-yellow-400');
        }

        if (data.fib) {
          const fHigh = document.getElementById('fib-high');
          if (fHigh) fHigh.innerText = data.fib.swing_high ? '$' + Number(data.fib.swing_high).toLocaleString() : '--';
          const f382 = document.getElementById('fib-382');
          if (f382) f382.innerText = data.fib.fib_382 ? '$' + Number(data.fib.fib_382).toLocaleString() : '--';
          const f500 = document.getElementById('fib-500');
          if (f500) f500.innerText = data.fib.fib_500 ? '$' + Number(data.fib.fib_500).toLocaleString() : '--';
          const f618 = document.getElementById('fib-618');
          if (f618) f618.innerText = data.fib.fib_618 ? '$' + Number(data.fib.fib_618).toLocaleString() : '--';
          const fLow = document.getElementById('fib-low');
          if (fLow) fLow.innerText = data.fib.swing_low ? '$' + Number(data.fib.swing_low).toLocaleString() : '--';

          const fibBadge = document.getElementById('an-fib-confluence');
          if (fibBadge) {
            if (data.fib.near_fib_support) {
              fibBadge.innerText = 'AT PULLBACK SUPPORT';
              fibBadge.className = 'px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30';
            } else {
              fibBadge.innerText = 'OUTSIDE SUPPORT ZONE';
              fibBadge.className = 'px-2 py-0.5 rounded text-[10px] font-bold bg-slate-800 text-slate-400';
            }
          }
        }

        const candleList = document.getElementById('candle-list');
        if (candleList) {
          candleList.innerHTML = '';
          if (data.candles && data.candles.length) {
            const prices = data.candles.map(c => Number(c.close));
            const minP = Math.min(...prices);
            const maxP = Math.max(...prices);
            const range = maxP - minP || 1;

            data.candles.forEach(c => {
              const heightPct = Math.max(15, Math.min(100, ((Number(c.close) - minP) / range) * 80 + 15));
              const isUp = Number(c.close) >= Number(c.open);
              const bar = document.createElement('div');
              bar.className = 'flex flex-col items-center flex-1 group relative';
              bar.innerHTML = `
                <div class="w-full ${isUp ? 'bg-emerald-500/80' : 'bg-rose-500/80'} rounded-t" style="height: ${heightPct}px" title="${c.ts} Close: $${c.close}"></div>
                <span class="text-[7px] text-slate-500 mt-0.5 opacity-60">${c.ts}</span>
              `;
              candleList.appendChild(bar);
            });
          }
        }
      } catch (e) {
        console.error('Error fetching analysis:', e);
      }
    }

    async function executeBacktest(e) {
      e.preventDefault();
      const btn = document.getElementById('btn-run-bt');
      btn.disabled = true;
      btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Running Backtest...';

      const payload = {
        exchange: document.getElementById('bt-exchange').value,
        symbol: document.getElementById('bt-symbol').value,
        timeframe: document.getElementById('bt-timeframe').value,
        days: parseInt(document.getElementById('bt-days').value),
        starting_equity: parseFloat(document.getElementById('bt-equity').value),
        use_fib_filter: document.getElementById('bt-use-fib').checked,
        risk_per_trade_pct: 0.01
      };

      try {
        const res = await fetch('/api/backtest', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!data.success) throw new Error(data.detail || 'Backtest failed');

        const r = data.results;
        document.getElementById('bt-results-box').classList.remove('hidden');
        const retEl = document.getElementById('bt-res-return');
        retEl.innerText = (r.total_return_pct >= 0 ? '+' : '') + r.total_return_pct + '%';
        retEl.className = r.total_return_pct >= 0 ? 'font-bold text-emerald-400' : 'font-bold text-rose-400';

        document.getElementById('bt-res-trades').innerText = r.num_trades;
        document.getElementById('bt-res-winrate').innerText = r.win_rate_pct + '%';
        document.getElementById('bt-res-final').innerText = '$' + r.final_equity.toLocaleString();
        document.getElementById('bt-res-sharpe').innerText = r.sharpe_approx;
        document.getElementById('bt-res-avgwin').innerText = '$' + r.avg_win;
        document.getElementById('bt-res-avgloss').innerText = '$' + r.avg_loss;
        document.getElementById('bt-res-drawdown').innerText = r.max_drawdown_pct + '%';
      } catch (err) {
        alert('Backtest error: ' + err.message);
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-play"></i> Run Historical Backtest';
      }
    }

    async function refreshLogs() {
      try {
        const res = await fetch('/api/logs?lines=40');
        const data = await res.json();
        const container = document.getElementById('log-container');
        container.innerHTML = '';
        if (data.logs && data.logs.length) {
          data.logs.forEach(l => {
            const div = document.createElement('div');
            if (l.includes('[CRITICAL]')) div.className = 'text-rose-400 font-bold';
            else if (l.includes('[WARNING]')) div.className = 'text-amber-400';
            else if (l.includes('BUY') || l.includes('Closed position')) div.className = 'text-emerald-400';
            else div.className = 'text-slate-300';
            div.innerText = l;
            container.appendChild(div);
          });
          container.scrollTop = container.scrollHeight;
        } else {
          container.innerHTML = '<div class="text-slate-500">No logs recorded yet.</div>';
        }
      } catch (e) {
        console.error('Error refreshing logs:', e);
      }
    }

    // --- Email & SMTP Settings Management ---
    let currentEmailProvider = 'gmail';
    let cachedEmailSettings = null;

    async function checkEmailStatusBadge() {
      try {
        const res = await fetch('/api/auth/email-settings');
        const data = await res.json();
        let cls = 'w-2 h-2 rounded-full bg-rose-500 inline-block animate-pulse';
        if (data.is_configured) {
          cls = data.can_send_to_all_users
            ? 'w-2 h-2 rounded-full bg-emerald-400 inline-block'
            : 'w-2 h-2 rounded-full bg-amber-400 inline-block animate-pulse';
        }
        ['email-status-dot', 'email-status-dot-mobile', 'email-status-dot-bottom'].forEach(id => {
          const el = document.getElementById(id);
          if (el) el.className = cls;
        });
      } catch (e) {
        console.error('Error checking email status:', e);
      }
    }

    async function openEmailSettingsModal() {
      document.getElementById('email-settings-modal').classList.remove('hidden');
      document.getElementById('test-email-result').classList.add('hidden');

      try {
        const res = await fetch('/api/auth/email-settings');
        const data = await res.json();
        cachedEmailSettings = data;

        // Update banner
        const indicator = document.getElementById('smtp-status-indicator');
        const text = document.getElementById('smtp-status-text');
        const tag = document.getElementById('smtp-provider-tag');

        if (data.is_configured) {
          if (data.can_send_to_all_users) {
            indicator.className = 'w-2.5 h-2.5 rounded-full bg-emerald-400';
            const domainInfo = data.verified_domain ? ` &bull; Domain: ${data.verified_domain}` : '';
            text.innerHTML = `<span class="font-bold text-emerald-400">Ready for All Users</span> <span class="text-slate-300">(${data.provider.toUpperCase()}${domainInfo})</span>`;
            tag.innerText = data.provider.toUpperCase();
            tag.className = 'px-2 py-0.5 rounded text-[10px] font-mono uppercase bg-emerald-950/80 text-emerald-300 border border-emerald-500/40';
          } else {
            indicator.className = 'w-2.5 h-2.5 rounded-full bg-amber-400 animate-pulse';
            text.innerHTML = `<span class="font-bold text-amber-400">Sandbox Testing Only</span> <span class="text-slate-300">(${data.provider.toUpperCase()} - Custom domain required for all users)</span>`;
            tag.innerText = 'SANDBOX ONLY';
            tag.className = 'px-2 py-0.5 rounded text-[10px] font-mono uppercase bg-amber-950/80 text-amber-300 border border-amber-500/40';
          }
        } else {
          indicator.className = 'w-2.5 h-2.5 rounded-full bg-rose-500 animate-pulse';
          text.innerText = 'No email service configured (verification emails cannot be delivered)';
          tag.innerText = 'NOT CONFIGURED';
          tag.className = 'px-2 py-0.5 rounded text-[10px] font-mono uppercase bg-rose-950/80 text-rose-300 border border-rose-500/40';
        }

        // Configure verified domain badge details
        const resendDomainBadge = document.getElementById('resend-domain-name-badge');
        const resendSenderCode = document.getElementById('resend-active-sender-code');
        if (data.verified_domain) {
          if (resendDomainBadge) resendDomainBadge.innerText = data.verified_domain;
          if (resendSenderCode) resendSenderCode.innerText = data.from_email || `Crypto Trading Bot <noreply@${data.verified_domain}>`;
        }

        // Populate fields
        selectEmailProvider(data.provider !== 'none' ? data.provider : 'gmail');
        document.getElementById('smtp-input-host').value = data.smtp_host || '';
        document.getElementById('smtp-input-port').value = data.smtp_port || '587';
        document.getElementById('smtp-input-user').value = data.smtp_user || '';
        document.getElementById('smtp-input-from').value = data.from_email || '';
        if (data.resend_key_masked) {
          document.getElementById('resend-input-key').placeholder = `Saved: ${data.resend_key_masked} (leave blank to keep)`;
        }
        if (data.brevo_key_masked) {
          document.getElementById('brevo-input-key').placeholder = `Saved: ${data.brevo_key_masked} (leave blank to keep)`;
        }
        if (data.sendgrid_key_masked) {
          document.getElementById('sendgrid-input-key').placeholder = `Saved: ${data.sendgrid_key_masked} (leave blank to keep)`;
        }
        if (data.smtp_pass_masked) {
          document.getElementById('smtp-input-pass').placeholder = `Saved: ${data.smtp_pass_masked} (leave blank to keep)`;
        }
      } catch (err) {
        console.error('Failed to load email settings:', err);
      }
    }

    function closeEmailSettingsModal() {
      document.getElementById('email-settings-modal').classList.add('hidden');
    }

    function selectEmailProvider(p) {
      currentEmailProvider = p;
      document.getElementById('email-provider-val').value = p;

      const tabs = ['gmail', 'brevo', 'resend', 'sendgrid', 'smtp'];
      const activeClass = 'py-2 px-1.5 rounded-lg border font-semibold text-center transition flex flex-col items-center gap-1 bg-emerald-600/20 border-emerald-500 text-emerald-300';
      const inactiveClass = 'py-2 px-1.5 rounded-lg border border-bordercol text-slate-400 hover:text-white font-semibold text-center transition flex flex-col items-center gap-1';

      tabs.forEach(t => {
        const tabEl = document.getElementById(`tab-prov-${t}`);
        if (tabEl) tabEl.className = (p === t) ? activeClass : inactiveClass;
      });

      const gmailBox = document.getElementById('gmail-guide-box');
      const brevoBox = document.getElementById('brevo-guide-box');
      const resendBox = document.getElementById('resend-guide-box');
      const resendVerifiedBox = document.getElementById('resend-verified-domain-box');
      const sendgridBox = document.getElementById('sendgrid-guide-box');

      const hostRow = document.getElementById('field-smtp-host-row');
      const brevoRow = document.getElementById('field-brevo-row');
      const resendRow = document.getElementById('field-resend-row');
      const sendgridRow = document.getElementById('field-sendgrid-row');
      const credsRow = document.getElementById('field-smtp-creds-row');
      const lblUser = document.getElementById('lbl-smtp-user');
      const lblPass = document.getElementById('lbl-smtp-pass');

      if (gmailBox) gmailBox.classList.toggle('hidden', p !== 'gmail');
      if (brevoBox) brevoBox.classList.toggle('hidden', p !== 'brevo');
      if (sendgridBox) sendgridBox.classList.toggle('hidden', p !== 'sendgrid');

      if (p === 'resend') {
        const hasVerified = cachedEmailSettings && cachedEmailSettings.verified_domain;
        if (resendVerifiedBox) resendVerifiedBox.classList.toggle('hidden', !hasVerified);
        if (resendBox) resendBox.classList.toggle('hidden', Boolean(hasVerified));
      } else {
        if (resendVerifiedBox) resendVerifiedBox.classList.add('hidden');
        if (resendBox) resendBox.classList.add('hidden');
      }

      if (hostRow) hostRow.classList.toggle('hidden', p !== 'smtp');
      if (brevoRow) brevoRow.classList.toggle('hidden', p !== 'brevo');
      if (resendRow) resendRow.classList.toggle('hidden', p !== 'resend');
      if (sendgridRow) sendgridRow.classList.toggle('hidden', p !== 'sendgrid');
      if (credsRow) credsRow.classList.toggle('hidden', p !== 'gmail' && p !== 'smtp');

      if (p === 'gmail') {
        if (lblUser) lblUser.innerText = 'Gmail Address';
        if (lblPass) lblPass.innerText = '16-Character Google App Password';
      } else if (p === 'smtp') {
        if (lblUser) lblUser.innerText = 'SMTP Username';
        if (lblPass) lblPass.innerText = 'SMTP Password';
      }
    }

    async function handleSaveEmailSettings(e) {
      e.preventDefault();
      const btn = document.getElementById('btn-save-email-settings');
      const oldHtml = btn.innerHTML;
      btn.disabled = true;
      btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Saving...';

      const provider = currentEmailProvider;
      const payload = {
        provider: provider,
        smtp_host: document.getElementById('smtp-input-host').value,
        smtp_port: document.getElementById('smtp-input-port').value,
        smtp_user: document.getElementById('smtp-input-user').value,
        smtp_pass: document.getElementById('smtp-input-pass').value,
        from_email: document.getElementById('smtp-input-from').value,
        resend_api_key: document.getElementById('resend-input-key').value,
        brevo_api_key: document.getElementById('brevo-input-key') ? document.getElementById('brevo-input-key').value : '',
        sendgrid_api_key: document.getElementById('sendgrid-input-key') ? document.getElementById('sendgrid-input-key').value : '',
      };

      try {
        const res = await fetch('/api/auth/email-settings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (res.ok && data.success) {
          alert('Email settings saved successfully!');
          await openEmailSettingsModal();
          await checkEmailStatusBadge();
        } else {
          alert('Failed to save email settings: ' + (data.detail || data.message));
        }
      } catch (err) {
        alert('Error saving email settings: ' + err);
      } finally {
        btn.disabled = false;
        btn.innerHTML = oldHtml;
      }
    }

    async function runTestEmail() {
      const dest = document.getElementById('test-email-dest').value.trim();
      const resultBox = document.getElementById('test-email-result');
      if (!dest) {
        alert('Please enter a recipient email address to send a test to.');
        return;
      }

      const btn = document.getElementById('btn-run-test-email');
      const oldHtml = btn.innerHTML;
      btn.disabled = true;
      btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Sending...';

      resultBox.classList.remove('hidden');
      resultBox.className = 'p-3 rounded-lg border bg-slate-900 border-bordercol text-slate-300 text-xs flex items-center gap-2';
      resultBox.innerHTML = '<i class="fa-solid fa-spinner fa-spin text-sky-400"></i> Dispatching test email...';

      try {
        const res = await fetch('/api/auth/test-email', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ to_email: dest })
        });
        const data = await res.json();
        if (res.ok && data.success) {
          resultBox.className = 'p-3.5 rounded-lg border bg-emerald-950/80 border-emerald-500/40 text-emerald-200 text-xs space-y-1.5';
          resultBox.innerHTML = `
            <div class="font-bold text-emerald-400 flex items-center gap-1.5 text-sm">
              <i class="fa-solid fa-circle-check"></i> Test Email Sent Successfully!
            </div>
            <p>${data.message}</p>
            <p class="text-[11px] text-slate-300">A test verification message was dispatched to <strong>${dest}</strong>. Please check the inbox and spam folder.</p>
            ${!data.can_send_to_all_users ? `
              <div class="p-2.5 bg-amber-950/60 border border-amber-500/40 rounded text-amber-200 text-[11px] mt-1 space-y-1">
                <div class="font-bold text-amber-300"><i class="fa-solid fa-triangle-exclamation"></i> Sandbox Notice:</div>
                <p>This test succeeded because ${dest} matches your verified testing address. However, Resend is using <code>onboarding@resend.dev</code>, which cannot send to other users. To send verification emails to all registering users, please verify your custom domain at <a href="https://resend.com/domains" target="_blank" class="text-sky-300 underline font-semibold">resend.com/domains</a>, or switch to Brevo API / Gmail SMTP.</p>
              </div>
            ` : ''}
          `;
          await checkEmailStatusBadge();
        } else {
          const isSandboxIssue = (data.message && (data.message.includes('403') || data.message.includes('Sandbox') || data.message.includes('testing emails to your own email address')));
          resultBox.className = 'p-3.5 rounded-lg border bg-rose-950/80 border-rose-500/40 text-rose-200 text-xs space-y-2';
          resultBox.innerHTML = `
            <div class="font-bold text-rose-400 flex items-center gap-1.5 text-sm">
              <i class="fa-solid fa-circle-xmark"></i> Test Email Failed
            </div>
            <p class="font-mono text-[11px] leading-relaxed text-slate-200">${data.message || data.detail}</p>
            ${isSandboxIssue ? `
              <div class="p-2.5 bg-slate-900/90 border border-amber-500/40 rounded text-amber-200 text-[11px] space-y-1">
                <div class="font-bold text-amber-300"><i class="fa-solid fa-lightbulb"></i> How to fix this:</div>
                <p><strong>Option 1 (Custom Domain):</strong> Add and verify your domain at <a href="https://resend.com/domains" target="_blank" class="text-sky-300 underline font-semibold">resend.com/domains</a>, then set your Sender "From" Email to <code>noreply@yourdomain.com</code>.</p>
                <p><strong>Option 2 (No Domain Needed):</strong> Switch to <strong>Brevo API</strong> (300 free emails/day to any user) or <strong>Gmail SMTP</strong> above.</p>
              </div>
            ` : ''}
          `;
        }
      } catch (err) {
        resultBox.className = 'p-3 rounded-lg border bg-rose-950/80 border-rose-500/40 text-rose-200 text-xs';
        resultBox.innerText = 'Network error: ' + err;
      } finally {
        btn.disabled = false;
        btn.innerHTML = oldHtml;
      }
    }

    // Startup
    loadExchanges();
    checkAuth();
    loadBotSettings();
    fetchStatus();
    refreshAnalysis();
    refreshLogs();
    checkEmailStatusBadge();

    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('action') === 'resend') {
      openAuthModal('resend');
    } else if (urlParams.get('action') === 'login') {
      openAuthModal('login');
    }

    setInterval(fetchStatus, 3000);
    setInterval(refreshAnalysis, 15000);
    setInterval(refreshLogs, 5000);
  </script>
</body>
</html>
"""

@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
def get_dashboard():
    return HTMLResponse(content=DASHBOARD_HTML, status_code=200)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"\n========================================================")
    print(f" Crypto Trading Bot Dashboard running on http://localhost:{port}")
    print(f"========================================================\n")
    uvicorn.run("web_server:app", host=host, port=port, reload=False)
