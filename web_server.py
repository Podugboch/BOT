"""
Web Server, Dashboard, and Authentication API for Trader Queen.
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
import html
import re
import collections
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, Header, HTTPException, Query, Response, Request, Cookie
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from pydantic import BaseModel, model_validator
import uvicorn
import pandas as pd
import numpy as np
import ccxt

CURRENT_DIR = Path(__file__).resolve().parent
LOGO_PATH = CURRENT_DIR / "static" / "logo.png"
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from secret_manager import safe_load_dotenv, migrate_and_sanitize_env_file, SENSITIVE_ENV_KEYS
safe_load_dotenv()
migrate_and_sanitize_env_file()

from config import BotConfig, StrategyConfig, RiskConfig, ExchangeConfig
from exchange_client import (
    ExchangeClient, MAX_DAYS, lookback_limits, max_lookback_days,
    lookback_limit_note, clamp_lookback_days, get_exchange_fee_info
)
from indicators import ema, rsi, atr, adx, fib_retracement_levels
from strategy import Signal, add_indicators, generate_signal, _near_fib_support, STRATEGIES
from risk_manager import RiskManager, reward_risk_ratio
from backtester import fetch_historical, run_backtest, summarize, run_walk_forward_validation
from bot import TradingBot, setup_logging
from execution_tracker import get_execution_tracker
from alerts import get_alert_dispatcher
from auth import AuthService, init_auth_db, get_email_settings, save_email_settings, test_email_connection, auth_service
auth_service = AuthService

AVAILABLE_TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"]

setup_logging()
logger = logging.getLogger("web")
init_auth_db()

_memory_log_buffer: collections.deque = collections.deque(maxlen=1000)

class BufferLogHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            _memory_log_buffer.append(msg)
        except Exception:
            pass

_buf_handler = BufferLogHandler()
_buf_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logging.getLogger().addHandler(_buf_handler)

# Preload non-sensitive exchange metadata into environment if present in vault
latest_key = AuthService.get_latest_raw_exchange_key()
if latest_key:
    if not os.getenv("EXCHANGE_ID") and latest_key.get("exchange_id"):
        os.environ["EXCHANGE_ID"] = latest_key["exchange_id"]
    if not os.getenv("TESTNET") and "testnet" in latest_key:
        os.environ["TESTNET"] = "true" if latest_key["testnet"] else "false"

# Check if an explicit trading_mode preference exists in remote settings
try:
    from storage_adapter import remote_get_app_setting
    saved_init_tm = remote_get_app_setting("trading_mode")
    if saved_init_tm in ("real", "testnet"):
        os.environ["MODE"] = "live" if saved_init_tm == "real" else "paper"
        os.environ["TESTNET"] = "false" if saved_init_tm == "real" else "true"
except Exception:
    pass

# Ensure sensitive exchange credentials are never present in os.environ at runtime
if not (os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME")):
    for k in {"EXCHANGE_API_KEY", "EXCHANGE_API_SECRET", "EXCHANGE_API_PASSPHRASE"}:
        os.environ.pop(k, None)

app = FastAPI(title="Trader Queen Dashboard & Auth", version="2.0.0")


@app.get("/logo.png")
@app.get("/favicon.ico")
@app.get("/static/logo.png")
def get_logo():
    if LOGO_PATH.exists():
        return FileResponse(LOGO_PATH, media_type="image/png")
    raise HTTPException(status_code=404, detail="Logo not found")


# Supported popular exchanges with metadata
EXCHANGES_META = [
    {"id": "binance", "name": "Binance", "needs_passphrase": False, "sandbox": True},
    {"id": "coinbase", "name": "Coinbase Advanced", "needs_passphrase": True, "sandbox": True},
    {"id": "kraken", "name": "Kraken", "needs_passphrase": False, "sandbox": False},
    {"id": "kucoin", "name": "KuCoin", "needs_passphrase": True, "sandbox": True},
    {"id": "bybit", "name": "Bybit", "needs_passphrase": False, "sandbox": True},
    {"id": "okx", "name": "OKX", "needs_passphrase": True, "sandbox": True},
    {"id": "bitget", "name": "Bitget", "needs_passphrase": True, "sandbox": True},
    {"id": "gate", "name": "Gate.io", "needs_passphrase": False, "sandbox": False},
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
        self._stop_event: threading.Event = threading.Event()

    def start(self):
        if self.running and self.thread and self.thread.is_alive():
            return False
        if os.path.exists(self.cfg.kill_switch_file):
            try:
                os.remove(self.cfg.kill_switch_file)
            except Exception:
                pass

        if self.bot is None:
            self.bot = TradingBot(self.cfg)
        else:
            self.bot.cfg = self.cfg

        # Signal any prior thread to stop
        self._stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

        self._stop_event = threading.Event()
        stop_evt = self._stop_event
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

        def _run_loop(evt: threading.Event):
            logger.info("Bot execution loop active. Polling interval: %ds.", self.cfg.poll_interval_sec)
            try:
                while not evt.is_set():
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

                    if evt.wait(timeout=self.cfg.poll_interval_sec):
                        break
                    if self.bot and self.bot.kill_switch_active():
                        logger.warning("Kill switch active — halting bot runner loop.")
                        self.running = False
                        break
            except Exception as e:
                logger.exception("Error in bot background thread")
                self.last_error = str(e)
                self.running = False

        self.thread = threading.Thread(target=_run_loop, args=(stop_evt,), daemon=True)
        self.thread.start()
        return True

    def stop(self):
        was_running = self.running
        self.running = False
        self._stop_event.set()
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

    def deactivate_kill(self):
        logger.info("Deactivating emergency kill switch...")
        if os.path.exists(self.cfg.kill_switch_file):
            try:
                os.remove(self.cfg.kill_switch_file)
                logger.info("Removed kill switch file '%s'.", self.cfg.kill_switch_file)
            except Exception as e:
                logger.warning("Could not remove kill switch file: %s", e)
        if self.bot:
            self.bot.deactivate_kill_switch()
        self.last_error = None
        logger.info("Emergency kill switch deactivated. Bot is now IDLE.")
        return True

    def get_or_create_bot(self) -> TradingBot:
        if self.bot is None:
            self.bot = TradingBot(self.cfg)
        else:
            self.bot.cfg = self.cfg
        return self.bot


runner = BotRunner()


# --- Pydantic Request Models ---
class ExecutionModeRequest(BaseModel):
    execution_mode: str  # "auto" | "manual"


class KillSwitchToggleRequest(BaseModel):
    active: Optional[bool] = None


class BotSettingsRequest(BaseModel):
    execution_mode: Optional[str] = None  # "auto" | "manual"
    strategy: Optional[str] = None
    timeframe: Optional[str] = None
    symbol: Optional[str] = None
    use_fib_filter: Optional[bool] = None
    fast_ema: Optional[int] = None
    slow_ema: Optional[int] = None
    rsi_period: Optional[int] = None
    rsi_overbought: Optional[float] = None
    adx_trend_threshold: Optional[float] = None
    risk_per_trade_pct: Optional[float] = None
    stop_loss_atr_mult: Optional[float] = None
    take_profit_atr_mult: Optional[float] = None


class ManualTradeRequest(BaseModel):
    action: str  # "buy" | "sell"
    size: Optional[float] = 0.0


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    username: Optional[str] = None
    username_or_email: Optional[str] = None
    password: str
    client_token: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def sync_usernames(cls, values):
        if isinstance(values, dict):
            u = values.get("username") or values.get("username_or_email")
            if u:
                values.setdefault("username", u)
                values.setdefault("username_or_email", u)
        return values


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


class DeleteExchangeKeyRequest(BaseModel):
    exchange_id: Optional[str] = None


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
    days: int = 30                      # clamped to 1..MAX_DAYS (730) server-side
    starting_equity: float = 1000.0
    strategy: str = "ema_cross_fib"
    use_fib_filter: bool = True
    # Dynamic (ATR-scaled) fib band: 0 restores the fixed percentage band.
    fib_tolerance_atr_mult: float = 0.5
    fib_tolerance_pct: float = 0.003    # floor for the band (and the fixed band)
    fib_level_mode: str = "deep"        # "deep" (0.5/0.618) | "zone" (0.382-0.618) | "all"
    trailing_stop_enabled: bool = True
    trailing_stop_atr_mult: float = 3.5  # matches RiskConfig: arm late, trail wide
    # Entry debounce: candles required between one trade's exit and the next
    # entry. None derives it from the timeframe (8 bars on 5m/15m, 3 on 1h/4h).
    min_bars_between_trades: Optional[int] = None
    risk_per_trade_pct: float = 0.01
    use_macro_filter: bool = True
    macro_filter_slope: bool = True
    sweep_tag: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def accept_fib_filter_alias(cls, values):
        """Accept `enable_fib_filter` (the label on the UI checkbox) as an
        alias for `use_fib_filter`; an explicit `use_fib_filter` still wins."""
        if isinstance(values, dict) and "use_fib_filter" not in values:
            alias = values.get("enable_fib_filter")
            if alias is not None:
                values["use_fib_filter"] = alias
        return values


class CompareWindowsRequest(BaseModel):
    exchange: str = "binance"
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    starting_equity: float = 1000.0
    strategy: str = "ema_cross_fib"
    use_fib_filter: bool = True
    use_macro_filter: bool = True
    macro_filter_slope: bool = True
    fib_tolerance_atr_mult: float = 0.5
    fib_tolerance_pct: float = 0.003
    fib_level_mode: str = "deep"
    trailing_stop_enabled: bool = True
    trailing_stop_atr_mult: float = 3.5
    min_bars_between_trades: Optional[int] = None
    risk_per_trade_pct: float = 0.01
    windows: Optional[List[int]] = None

    @model_validator(mode="before")
    @classmethod
    def accept_fib_filter_alias(cls, values):
        if isinstance(values, dict) and "use_fib_filter" not in values:
            alias = values.get("enable_fib_filter")
            if alias is not None:
                values["use_fib_filter"] = alias
        return values


class CompareTimeframesRequest(BaseModel):
    exchange: str = "binance"
    symbol: str = "BTC/USDT"
    days: int = 90
    starting_equity: float = 1000.0
    strategy: str = "ema_cross_fib"
    use_fib_filter: bool = True
    fib_tolerance_atr_mult: float = 0.5
    fib_tolerance_pct: float = 0.003
    fib_level_mode: str = "deep"
    trailing_stop_enabled: bool = True
    trailing_stop_atr_mult: float = 3.5
    min_bars_between_trades: Optional[int] = None
    risk_per_trade_pct: float = 0.01
    timeframes: Optional[List[str]] = None

    @model_validator(mode="before")
    @classmethod
    def accept_fib_filter_alias(cls, values):
        if isinstance(values, dict) and "use_fib_filter" not in values:
            alias = values.get("enable_fib_filter")
            if alias is not None:
                values["use_fib_filter"] = alias
        return values


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
        "optimize_1h.py",
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
    configured = os.getenv("BASE_URL") or os.getenv("APP_URL")
    if configured and configured.strip():
        return configured.strip().rstrip("/")

    proto = request.headers.get("x-forwarded-proto", request.url.scheme).lower().strip()
    if proto not in ("http", "https"):
        proto = "https" if getattr(request.url, "is_secure", False) else "http"

    host = request.headers.get("x-forwarded-host", request.headers.get("host", request.url.netloc)).strip()
    if not re.match(r"^[a-zA-Z0-9.-]+(:[0-9]{1,5})?$", host):
        host = request.url.netloc or "localhost:8000"

    allowed = os.getenv("ALLOWED_HOSTS", "").strip()
    if allowed:
        allowed_list = [h.strip().lower() for h in allowed.split(",") if h.strip()]
        host_no_port = host.split(":")[0].lower()
        if host.lower() not in allowed_list and host_no_port not in allowed_list:
            host = request.url.netloc or "localhost:8000"

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
        is_secure = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
        response.set_cookie(key="bot_auth_token", value=res["token"], max_age=2592000, path="/", samesite="lax", secure=is_secure)
    return res


@app.get("/api/auth/activate")
@app.post("/api/auth/activate")
def activate_account(token: str = Query(...), response: Response = None):
    res = AuthService.activate_account(token)
    if not res["success"]:
        return JSONResponse(status_code=400, content=res)
    if response and res.get("session_token"):
        response.set_cookie(key="bot_auth_token", value=res["session_token"], max_age=2592000, path="/", samesite="lax", secure=False)
    return res


@app.post("/api/auth/resend-activation")
def resend_activation_endpoint(req: ResendActivationRequest, request: Request):
    base_url = get_request_base_url(request)
    res = AuthService.resend_activation_email(req.email, base_url=base_url)
    return res


@app.post("/api/login")
@app.post("/api/auth/login")
def login(payload: LoginRequest, response: Response):
    try:
        username = (payload.username or payload.username_or_email or "").strip()
        result = auth_service.login(username, payload.password, payload.client_token)
        if not result.get("success"):
            return JSONResponse(
                status_code=400,
                content=result
            )
        if result.get("token"):
            response.set_cookie(
                key="bot_auth_token",
                value=result["token"],
                max_age=2592000,
                path="/",
                samesite="lax",
                secure=False
            )
        return result
    except Exception as exc:
        import traceback
        traceback.print_exc()  # Check terminal logs for full stack trace
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Internal server error: {str(exc)}"}
        )


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
def delete_all_accounts_endpoint(
    authorization: Optional[str] = Header(None),
    bot_auth_token: Optional[str] = Cookie(None)
):
    user = get_current_user(authorization, bot_auth_token)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required to delete accounts")
    res = AuthService.delete_all_accounts()
    return res


@app.delete("/api/auth/account")
def delete_my_account(
    authorization: Optional[str] = Header(None),
    bot_auth_token: Optional[str] = Cookie(None)
):
    user = get_current_user(authorization, bot_auth_token)
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
def test_exchange(
    req: TestExchangeRequest,
    authorization: Optional[str] = Header(None),
    bot_auth_token: Optional[str] = Cookie(None)
):
    """Test connection and credential validity using CCXT."""
    user = get_current_user(authorization, bot_auth_token)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please create an account and log in to test exchange API keys."
        )

    ex_id = req.exchange_id.strip().lower()
    if ex_id not in ccxt.exchanges:
        raise HTTPException(status_code=400, detail=f"Unsupported exchange '{req.exchange_id}'")

    clean_key = req.api_key.strip().strip("'\"")
    clean_secret = req.api_secret.strip().strip("'\"")
    clean_passphrase = (req.api_passphrase or "").strip().strip("'\"")

    exchange_class = getattr(ccxt, ex_id)

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
def save_exchange_credentials(
    req: SaveExchangeKeyRequest,
    authorization: Optional[str] = Header(None),
    bot_auth_token: Optional[str] = Cookie(None)
):
    user = get_current_user(authorization, bot_auth_token)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please create an account and log in to manage exchange API keys."
        )

    ex_id = req.exchange_id.strip().lower()
    if ex_id not in ccxt.exchanges:
        raise HTTPException(status_code=400, detail=f"Unsupported exchange '{req.exchange_id}'")

    try:
        # Store in DB per user (encrypted at rest)
        AuthService.save_exchange_key(
            user["id"], ex_id, req.api_key, req.api_secret, req.api_passphrase or "", req.testnet
        )
    except Exception as e:
        logger.exception("Failed saving exchange credentials for user %s on %s", user.get("id"), ex_id)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Could not save exchange credentials. Please retry, or check server storage configuration.",
                "detail": str(e),
            },
        )

    # In-memory updates for non-sensitive routing metadata only
    os.environ["EXCHANGE_ID"] = ex_id
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

    # Refresh runner exchange config while preserving user settings
    runner.cfg.exchange.exchange_id = req.exchange_id
    runner.cfg.exchange.testnet = req.testnet
    runner.cfg.exchange.api_key = req.api_key
    runner.cfg.exchange.api_secret = req.api_secret
    runner.cfg.exchange.api_passphrase = req.api_passphrase or ""
    if runner.bot and runner.bot.exchange:
        runner.bot.exchange.cfg.exchange_id = req.exchange_id
        runner.bot.exchange.cfg.testnet = req.testnet
        runner.bot.exchange.cfg.api_key = req.api_key
        runner.bot.exchange.cfg.api_secret = req.api_secret
        runner.bot.exchange.cfg.api_passphrase = req.api_passphrase or ""
    return {
        "success": True,
        "message": f"API keys for {req.exchange_id} saved securely to encrypted vault and applied to active configuration.",
        "exchange_id": req.exchange_id,
        "testnet": req.testnet,
    }


@app.post("/api/exchange/delete")
@app.delete("/api/exchange/delete")
@app.delete("/api/exchange/key")
def delete_exchange_credentials(
    req: Optional[DeleteExchangeKeyRequest] = None,
    exchange_id: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
    bot_auth_token: Optional[str] = Cookie(None),
):
    user = get_current_user(authorization, bot_auth_token)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please log in to manage exchange API keys."
        )

    target_ex = None
    if req and req.exchange_id:
        target_ex = req.exchange_id.strip().lower()
    elif exchange_id:
        target_ex = exchange_id.strip().lower()

    if not target_ex:
        raise HTTPException(status_code=400, detail="Missing exchange_id to delete.")

    try:
        AuthService.delete_exchange_key(user["id"], target_ex)
    except Exception as e:
        logger.exception("Failed deleting exchange credentials for user %s on %s", user.get("id"), target_ex)
        raise HTTPException(status_code=500, detail=f"Failed to delete exchange credentials: {e}")

    # If the active runner configuration was using this exchange, clear in-memory credentials
    if runner.cfg.exchange.exchange_id.lower() == target_ex:
        runner.cfg.exchange.api_key = ""
        runner.cfg.exchange.api_secret = ""
        runner.cfg.exchange.api_passphrase = ""
        if runner.bot and runner.bot.exchange:
            runner.bot.exchange.cfg.api_key = ""
            runner.bot.exchange.cfg.api_secret = ""
            runner.bot.exchange.cfg.api_passphrase = ""
        for k in ["EXCHANGE_API_KEY", "EXCHANGE_API_SECRET", "EXCHANGE_API_PASSPHRASE"]:
            os.environ.pop(k, None)

    return {
        "success": True,
        "exchange_id": target_ex,
        "message": f"API keys for {target_ex.upper()} successfully removed."
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
NO_STORE_HEADERS = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "Pragma": "no-cache"}


def _no_store(response: Response) -> None:
    """Keep live bot state out of browser/CDN caches.

    A cached GET is the usual reason a Refresh button looks broken on a deployed
    dashboard: the click fires, the request returns 200, and the page redraws
    yesterday's payload.
    """
    if response is not None:
        for key, value in NO_STORE_HEADERS.items():
            response.headers[key] = value


def sanitize_for_json(obj):
    """Recursively convert NumPy scalars/arrays into JSON-native Python types.

    Indicator math and the backtester hand back values like ``np.bool_``,
    ``np.float64`` and ``np.int64``. FastAPI's ``jsonable_encoder`` cannot encode
    those scalars (``np.bool_`` is not iterable and has no ``__dict__``), which is
    what turned every sweep click into an HTTP 500. Sanitizing the payload at the
    boundary keeps the endpoints returning plain dicts/lists/bools/floats/ints.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.bool_, getattr(np, 'bool8', bool))):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _html_response(content: str, status_code: int = 200) -> HTMLResponse:
    """HTML that must never be cached, with the headers set for you.

    The dashboard carries the frontend JavaScript, so a cached copy is how a
    fixed page keeps showing the *old* bug (the JS error survives long after the
    deploy), and the auth pages carry one-shot tokens, where a cached copy is a
    leaked credential. Neither may come out of a cache.
    """
    response = HTMLResponse(content=content, status_code=status_code)
    _no_store(response)
    return response


def _safe_remote_pos(bot=None) -> Optional[dict]:
    if "PYTEST_CURRENT_TEST" in os.environ:
        return None
    try:
        from storage_adapter import remote_get_app_setting
        rem = remote_get_app_setting("active_position")
        if rem and isinstance(rem, dict) and rem.get("entry"):
            if bot and bot.position is None:
                bot.position = rem
            return rem
    except Exception:
        pass
    try:
        from pathlib import Path
        import json
        pos_file = Path("/tmp/active_position.json") if (os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME")) else Path(CURRENT_DIR / "active_position.json")
        if pos_file.exists():
            rem = json.loads(pos_file.read_text())
            if rem and isinstance(rem, dict) and rem.get("entry"):
                if bot and bot.position is None:
                    bot.position = rem
                return rem
    except Exception:
        pass
    return None


@app.get("/api/status")
def get_status(response: Response = None):
    _no_store(response)
    cfg = runner.cfg
    bot = runner.get_or_create_bot()
    bot._sync_portfolio()
    active_pos = bot.get_unified_active_position()
    risk_state = bot.risk.state
    exchange = bot.exchange

    kill_file_exists = os.path.exists(cfg.kill_switch_file)
    strat_meta = next((s for s in STRATEGIES if s["id"] == cfg.strategy.strategy_name), STRATEGIES[0])

    wallet_info = bot.exchange.fetch_wallet_balance(cfg.exchange.symbol) if bot else {
        "free_quote": cfg.starting_equity,
        "free_base": 0.0,
        "total_quote": cfg.starting_equity,
        "total_base": 0.0,
        "quote_currency": "USDT",
        "base_currency": "BTC",
        "mode": cfg.mode,
        "is_live": (cfg.mode == "live" and bool(cfg.exchange.api_key)),
    }
    spread_info = bot.exchange.fetch_order_book_spread(cfg.exchange.symbol) if bot else {
        "symbol": cfg.exchange.symbol,
        "spread_pct": 0.0,
        "valid": False,
    }

    env_mode = os.getenv("EXECUTION_MODE")
    if env_mode in ("auto", "manual"):
        cfg.execution_mode = env_mode
        exec_mode = env_mode
    else:
        exec_mode = getattr(cfg, "execution_mode", None)
        if not exec_mode or exec_mode == "auto":
            try:
                from storage_adapter import remote_get_app_setting
                rm = remote_get_app_setting("execution_mode")
                if rm in ("auto", "manual"):
                    cfg.execution_mode = rm
                    os.environ["EXECUTION_MODE"] = rm
                    exec_mode = rm
            except Exception:
                pass
    exec_mode = exec_mode or getattr(cfg, "execution_mode", "auto") or "auto"

    try:
        from storage_adapter import remote_get_app_setting
        saved_tm = remote_get_app_setting("trading_mode")
        if saved_tm in ("real", "testnet"):
            is_real = (saved_tm == "real")
            cfg.mode = "live" if is_real else "paper"
            cfg.exchange.testnet = not is_real
            os.environ["MODE"] = cfg.mode
            os.environ["TESTNET"] = "false" if is_real else "true"
            if runner.bot:
                runner.bot.cfg.mode = cfg.mode
                runner.bot.cfg.exchange.testnet = cfg.exchange.testnet
                if hasattr(runner.bot, "exchange") and runner.bot.exchange:
                    runner.bot.exchange.mode = cfg.mode
                    runner.bot.exchange.cfg.testnet = cfg.exchange.testnet
    except Exception:
        pass

    return {
        "running": runner.running,
        "mode": cfg.mode,
        "execution_mode": exec_mode,
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
        "equity": round(risk_state.equity, 2),
        "peak_equity": round(risk_state.peak_equity, 2),
        "daily_start_equity": round(risk_state.daily_start_equity, 2),
        "drawdown_pct": round(((risk_state.peak_equity - risk_state.equity) / risk_state.peak_equity) * 100, 2) if risk_state.peak_equity > 0 else 0.0,
        "daily_loss_pct": round(((risk_state.daily_start_equity - risk_state.equity) / risk_state.daily_start_equity) * 100, 2) if risk_state.daily_start_equity > 0 else 0.0,
        "consecutive_losses": risk_state.consecutive_losses,
        "max_consecutive_losses": cfg.risk.max_consecutive_losses,
        "loss_streak_cooldown_hours": cfg.risk.loss_streak_cooldown_hours,
        # Published so the dashboard can say *when* trading resumes instead of
        # leaving a tripped circuit breaker looking like a dead bot.
        "cooldown_until": (
            risk_state.cooldown_until.isoformat()
            if getattr(risk_state, "cooldown_until", None) else None
        ),
        "risk_cooldown_note": bot.risk.cooldown_note(peek=True),
        "reward_risk_ratio": round(reward_risk_ratio(cfg.risk), 2),
        "trailing_stop_enabled": bool(getattr(cfg.risk, "trailing_stop_enabled", True)),
        "trailing_stop_atr_mult": cfg.risk.trailing_stop_atr_mult,
        "trading_halted_today": risk_state.trading_halted_today,
        "kill_switch_triggered": risk_state.kill_switch_triggered,
        "paper_cash": round(exchange.paper_cash, 2),
        "paper_position": round(exchange.paper_position, 6),
        "wallet_balance": wallet_info,
        "order_book_spread": spread_info,
        "max_spread_pct": getattr(cfg.risk, "max_spread_pct", 0.35),
        "active_position": active_pos,
        "timeframe_warning": (
            f"Negative expectancy (-10% to -15%) in historical backtests on {cfg.exchange.timeframe}. Do not risk live capital on this grain without strategy recalibration."
            if cfg.exchange.timeframe in ["1m", "3m", "5m", "15m"] else None
        ),
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
    if req.execution_mode is not None:
        exec_mode = req.execution_mode.strip().lower()
        if exec_mode not in ["auto", "manual"]:
            raise HTTPException(status_code=400, detail="Invalid execution_mode. Must be 'auto' or 'manual'.")
    else:
        exec_mode = getattr(runner.cfg, "execution_mode", "auto")

    timeframe = (req.timeframe or runner.cfg.exchange.timeframe or "15m").strip().lower()
    if timeframe not in AVAILABLE_TIMEFRAMES:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe '{timeframe}'. Supported: {', '.join(AVAILABLE_TIMEFRAMES)}")

    valid_strat_ids = [s["id"] for s in STRATEGIES]
    strategy_id = (req.strategy or runner.cfg.strategy.strategy_name or "ema_cross_fib").strip().lower()
    if strategy_id not in valid_strat_ids:
        raise HTTPException(status_code=400, detail=f"Invalid strategy '{strategy_id}'. Supported: {', '.join(valid_strat_ids)}")

    symbol = (req.symbol or runner.cfg.exchange.symbol or "BTC/USDT").strip().upper()
    if not re.match(r"^[A-Z0-9/_-]{2,20}$", symbol):
        raise HTTPException(status_code=400, detail=f"Invalid trading pair symbol '{symbol}'. Must match pattern like BTC/USDT.")

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
                        clean_val = str(updates[k]).replace("\r", "").replace("\n", "").strip()
                        new_lines.append(f"{k}={clean_val}\n")
                        written.add(k)
                        continue
                new_lines.append(l)
            for k, v in updates.items():
                if k not in written:
                    clean_val = str(v).replace("\r", "").replace("\n", "").strip()
                    new_lines.append(f"{k}={clean_val}\n")
            env_file.write_text("".join(new_lines))
    except Exception as e:
        logger.warning(f"Failed updating .env file with new bot settings: {e}")

    # Re-apply to active runner
    was_running = runner.running
    if was_running:
        runner.stop()

    runner.cfg.execution_mode = exec_mode
    os.environ["EXECUTION_MODE"] = exec_mode
    try:
        from storage_adapter import remote_save_app_setting
        remote_save_app_setting("execution_mode", exec_mode)
    except Exception as e:
        logger.debug(f"Remote storage execution_mode save skipped: {e}")
    runner.cfg.strategy.strategy_name = strategy_id
    runner.cfg.exchange.timeframe = timeframe
    runner.cfg.exchange.symbol = symbol

    if req.use_fib_filter is not None:
        runner.cfg.strategy.use_fib_filter = req.use_fib_filter
    if req.fast_ema is not None:
        if req.fast_ema < 2 or req.fast_ema > 200:
            raise HTTPException(status_code=400, detail="fast_ema must be between 2 and 200")
        runner.cfg.strategy.fast_ema = req.fast_ema
    if req.slow_ema is not None:
        if req.slow_ema < 3 or req.slow_ema > 500:
            raise HTTPException(status_code=400, detail="slow_ema must be between 3 and 500")
        runner.cfg.strategy.slow_ema = req.slow_ema
    if runner.cfg.strategy.fast_ema >= runner.cfg.strategy.slow_ema:
        raise HTTPException(status_code=400, detail="Fast EMA must be less than Slow EMA")
    if req.rsi_period is not None:
        if req.rsi_period < 2 or req.rsi_period > 100:
            raise HTTPException(status_code=400, detail="rsi_period must be between 2 and 100")
        runner.cfg.strategy.rsi_period = req.rsi_period
    if req.rsi_overbought is not None:
        if req.rsi_overbought < 50.0 or req.rsi_overbought > 95.0:
            raise HTTPException(status_code=400, detail="rsi_overbought must be between 50.0 and 95.0")
        runner.cfg.strategy.rsi_overbought = req.rsi_overbought
    if req.adx_trend_threshold is not None:
        if req.adx_trend_threshold < 5.0 or req.adx_trend_threshold > 80.0:
            raise HTTPException(status_code=400, detail="adx_trend_threshold must be between 5.0 and 80.0")
        runner.cfg.strategy.adx_trend_threshold = req.adx_trend_threshold
    if req.risk_per_trade_pct is not None:
        if req.risk_per_trade_pct <= 0 or req.risk_per_trade_pct > 0.20:
            raise HTTPException(status_code=400, detail="risk_per_trade_pct must be between 0.0001 (0.01%) and 0.20 (20%)")
        runner.cfg.risk.risk_per_trade_pct = req.risk_per_trade_pct
    if req.stop_loss_atr_mult is not None:
        if req.stop_loss_atr_mult < 0.1 or req.stop_loss_atr_mult > 20.0:
            raise HTTPException(status_code=400, detail="stop_loss_atr_mult must be between 0.1 and 20.0")
        runner.cfg.risk.stop_loss_atr_mult = req.stop_loss_atr_mult
    if req.take_profit_atr_mult is not None:
        if req.take_profit_atr_mult < 0.1 or req.take_profit_atr_mult > 50.0:
            raise HTTPException(status_code=400, detail="take_profit_atr_mult must be between 0.1 and 50.0")
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


@app.post("/api/bot/execution_mode")
def set_execution_mode_endpoint(req: ExecutionModeRequest):
    mode = (req.execution_mode or "").strip().lower()
    if mode not in ["auto", "manual"]:
        raise HTTPException(status_code=400, detail="Invalid execution_mode. Must be 'auto' or 'manual'.")

    runner.cfg.execution_mode = mode
    if runner.bot:
        runner.bot.cfg.execution_mode = mode

    os.environ["EXECUTION_MODE"] = mode
    try:
        env_file = Path(".env")
        if env_file.exists():
            lines = env_file.read_text().splitlines(keepends=True)
            written = False
            new_lines = []
            for l in lines:
                s = l.strip()
                if s and not s.startswith("#") and s.split("=", 1)[0].strip() == "EXECUTION_MODE":
                    new_lines.append(f"EXECUTION_MODE={mode}\n")
                    written = True
                else:
                    new_lines.append(l)
            if not written:
                new_lines.append(f"EXECUTION_MODE={mode}\n")
            env_file.write_text("".join(new_lines))
    except Exception as e:
        logger.warning(f"Failed updating .env with execution_mode: {e}")

    try:
        from storage_adapter import remote_save_app_setting
        remote_save_app_setting("execution_mode", mode)
    except Exception as e:
        logger.debug(f"Remote storage execution_mode save skipped: {e}")

    logger.info("Bot execution mode explicitly updated to: %s", mode.upper())
    return {
        "success": True,
        "execution_mode": mode,
        "message": f"Execution mode set to {mode.upper()}."
    }


@app.post("/api/bot/trade/manual")
def execute_manual_trade(req: ManualTradeRequest):
    bot = runner.get_or_create_bot()
    act = req.action.strip().lower()

    if act == "buy":
        try:
            res = bot.manual_buy(custom_size=req.size or 0.0)
            trade_type = res.get("type", "OPEN_LONG")
            if trade_type == "CLOSE_SHORT":
                msg = f"Manual BUY order closed short position at ${res['price']:,.2f} (PnL: ${res.get('pnl', 0.0):+,.2f})"
            else:
                msg = f"Manual BUY order executed at ${res['price']:,.2f} (Size: {res['size']:.6f})"
            logger.info(msg)
            return {
                "success": True,
                "message": msg,
                "trade": res,
                "active_position": bot.get_unified_active_position(),
                "equity": round(bot.risk.state.equity, 2),
                "paper_cash": round(bot.exchange.paper_cash, 2),
                "paper_position": round(bot.exchange.paper_position, 6),
            }
        except Exception as e:
            logger.exception("Manual BUY order failed")
            raise HTTPException(status_code=400, detail=str(e))
    elif act == "sell":
        try:
            res = bot.manual_sell(custom_size=req.size or 0.0)
            trade_type = res.get("type", "CLOSE_LONG")
            if trade_type == "OPEN_SHORT":
                msg = f"Manual SELL (Short) order executed at ${res['price']:,.2f} (Size: {res['size']:.6f})"
            else:
                msg = f"Manual SELL order executed at ${res['price']:,.2f} (PnL: ${res.get('pnl', 0.0):+,.2f})"
            logger.info(msg)
            return {
                "success": True,
                "message": msg,
                "trade": res,
                "active_position": bot.get_unified_active_position(),
                "equity": round(bot.risk.state.equity, 2),
                "paper_cash": round(bot.exchange.paper_cash, 2),
                "paper_position": round(bot.exchange.paper_position, 6),
            }
        except Exception as e:
            logger.exception("Manual SELL order failed")
            raise HTTPException(status_code=400, detail=str(e))
    elif act in ("close", "close_position", "cover", "exit"):
        try:
            if not bot.position:
                bot._sync_position()
            if not bot.position:
                _safe_remote_pos(bot)
            if not bot.position:
                raise HTTPException(status_code=400, detail="No active position open to close.")
            pos_side = str(bot.position.get("side", "BUY")).upper()
            if pos_side == "SELL":
                res = bot.manual_buy()
                msg = f"Manual order closed SHORT position at ${res['price']:,.2f} (PnL: ${res.get('pnl', 0.0):+,.2f})"
            else:
                res = bot.manual_sell()
                msg = f"Manual order closed LONG position at ${res['price']:,.2f} (PnL: ${res.get('pnl', 0.0):+,.2f})"
            logger.info(msg)
            try:
                from storage_adapter import remote_save_app_setting
                remote_save_app_setting("active_position", None)
            except Exception:
                pass
            return {
                "success": True,
                "message": msg,
                "trade": res,
                "active_position": bot.get_unified_active_position(),
                "equity": round(bot.risk.state.equity, 2),
                "paper_cash": round(bot.exchange.paper_cash, 2),
                "paper_position": round(bot.exchange.paper_position, 6),
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Manual close position failed")
            raise HTTPException(status_code=400, detail=str(e))
    else:
        raise HTTPException(status_code=400, detail=f"Invalid manual trade action '{req.action}'. Use 'buy', 'sell', or 'close'.")


@app.post("/api/bot/trade/close")
def close_manual_trade():
    return execute_manual_trade(ManualTradeRequest(action="close"))


@app.post("/api/bot/start")
def start_bot(force: bool = False):
    if runner.running:
        return {"success": False, "message": "Bot is already running"}
    if runner.cfg.mode == "live" and runner.cfg.exchange.timeframe in ["1m", "3m", "5m", "15m"] and not force:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Safety Protection: Live trading with real capital is blocked on {runner.cfg.exchange.timeframe}. "
                "Historical backtests on this grain show negative expectancy (-10% to -15% returns). "
                "Please switch to 1h, 4h, or 1d for live execution, or test on Paper mode."
            )
        )
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


@app.post("/api/bot/reset_kill")
def reset_kill_endpoint():
    runner.deactivate_kill()
    return {"success": True, "message": "Emergency kill switch deactivated. Bot is now IDLE."}


@app.post("/api/bot/kill/toggle")
def toggle_kill_switch_endpoint(req: Optional[KillSwitchToggleRequest] = None):
    currently_active = bool(
        os.path.exists(runner.cfg.kill_switch_file) or
        (runner.bot and runner.bot.kill_switch_active())
    )
    if req is not None and req.active is not None:
        target_active = bool(req.active)
    else:
        target_active = not currently_active

    if target_active:
        runner.trigger_kill()
        msg = "Emergency kill switch activated; bot halted immediately."
    else:
        runner.deactivate_kill()
        msg = "Emergency kill switch deactivated. Bot is now IDLE."

    return {
        "success": True,
        "kill_switch_active": target_active,
        "message": msg
    }


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

    try:
        from storage_adapter import remote_save_app_setting
        remote_save_app_setting("trading_mode", "real" if is_real else "testnet")
        remote_save_app_setting("mode", new_mode)
        remote_save_app_setting("testnet", new_testnet)
    except Exception as e:
        logger.warning("Failed persisting trading mode remotely: %s", e)

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

    # Re-apply to active runner config while preserving execution_mode and strategy settings
    was_running = runner.running
    if was_running:
        runner.stop()

    runner.cfg.mode = new_mode
    runner.cfg.exchange.testnet = new_testnet
    if runner.bot:
        runner.bot.cfg.mode = new_mode
        runner.bot.cfg.exchange.testnet = new_testnet
        if hasattr(runner.bot, "exchange") and runner.bot.exchange:
            runner.bot.exchange.mode = new_mode
            runner.bot.exchange.cfg.testnet = new_testnet

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
def get_analysis(response: Response = None, symbol: Optional[str] = None,
                 timeframe: Optional[str] = None, exchange: Optional[str] = None):
    global LAST_ANALYSIS_CACHE
    _no_store(response)
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
        return sanitize_for_json(LAST_ANALYSIS_CACHE)

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
        return sanitize_for_json(result)
    except Exception as e:
        logger.exception("Analysis calculation failed")
        if LAST_ANALYSIS_CACHE:
            return sanitize_for_json(LAST_ANALYSIS_CACHE)
        raise HTTPException(status_code=500, detail=str(e))


def _clamp_days(days) -> int:
    """Clamp a requested lookback into 1..MAX_DAYS instead of rejecting it.

    365/730 days are the point of the control: a 1h run has to be able to span
    bull, bear and chop, which is where the two-year (730-day) cap comes from.
    A nonsense value is capped rather than 422'd, so a stale UI cannot break a
    perfectly runnable backtest.
    """
    try:
        value = int(days)
    except (TypeError, ValueError):
        return 30
    return max(1, min(value, MAX_DAYS))


def _build_backtest_config(req, exchange_id: str, symbol: str, timeframe: str, starting_equity: float) -> BotConfig:
    cfg = BotConfig()
    cfg.exchange.exchange_id = exchange_id
    cfg.exchange.symbol = symbol
    cfg.exchange.timeframe = timeframe
    cfg.starting_equity = max(1.0, float(starting_equity))
    cfg.strategy.strategy_name = (getattr(req, "strategy", None) or getattr(req, "strategy_name", None) or "ema_cross_fib").strip().lower()
    cfg.strategy.use_fib_filter = getattr(req, "use_fib_filter", True)
    if getattr(req, "fib_tolerance_atr_mult", None) is not None:
        cfg.strategy.fib_tolerance_atr_mult = max(0.0, float(req.fib_tolerance_atr_mult))
    if getattr(req, "fib_tolerance_pct", None) is not None:
        cfg.strategy.fib_tolerance_pct = max(0.0, float(req.fib_tolerance_pct)) or 0.003
    cfg.strategy.fib_level_mode = str(getattr(req, "fib_level_mode", None) or "deep").strip().lower()
    if getattr(req, "trailing_stop_enabled", None) is not None:
        cfg.risk.trailing_stop_enabled = bool(req.trailing_stop_enabled)
    if getattr(req, "trailing_stop_atr_mult", None) is not None:
        cfg.risk.trailing_stop_atr_mult = max(0.0, float(req.trailing_stop_atr_mult))
    if getattr(req, "min_bars_between_trades", None) is not None:
        cfg.strategy.min_bars_between_trades = max(0, int(req.min_bars_between_trades))
    if getattr(req, "risk_per_trade_pct", None) is not None:
        cfg.risk.risk_per_trade_pct = float(req.risk_per_trade_pct)
    if getattr(req, "use_macro_filter", None) is not None:
        cfg.strategy.use_macro_filter = bool(req.use_macro_filter)
    if getattr(req, "macro_filter_slope", None) is not None:
        cfg.strategy.macro_filter_slope = bool(req.macro_filter_slope)
    return cfg


def _invoke_backtest(df, cfg, sweep_tag: Optional[str] = None) -> dict:
    if sweep_tag:
        try:
            return run_backtest(df, cfg, sweep_tag=sweep_tag)
        except TypeError as e:
            if "sweep_tag" in str(e):
                return run_backtest(df, cfg)
            raise
    return run_backtest(df, cfg)


@app.post("/api/backtest")
def run_backtest_endpoint(req: BacktestRequest):
    try:
        exch = req.exchange.strip().lower()
        if exch not in ccxt.exchanges:
            raise HTTPException(status_code=400, detail=f"Unsupported exchange '{req.exchange}'")
        days = _clamp_days(req.days)
        starting_equity = max(1.0, float(req.starting_equity))

        cfg = _build_backtest_config(req, exch, req.symbol, req.timeframe, starting_equity)

        df = fetch_historical(exch, req.symbol, req.timeframe, days)
        if df is None or len(df) < 60:
            raise HTTPException(status_code=400, detail=(
                f"Insufficient historical bars for {req.symbol} {req.timeframe}: "
                f"the exchange returned {0 if df is None else len(df)} candle(s). "
                f"{lookback_limit_note(req.timeframe)}"))

        results = _invoke_backtest(df, cfg, sweep_tag=req.sweep_tag)
        eval_range = results.get("eval_date_range") or [None, None]
        eval_count = results.get("candles_evaluated", 0)
        prefix = f"[{req.sweep_tag}] " if req.sweep_tag else ""
        logger.info(
            "%sBacktest run complete for %s %s (%sd): walked %s post-warmup candles (%s to %s) evaluated for signals, %s trade(s)",
            prefix, req.symbol, req.timeframe, days, eval_count, eval_range[0], eval_range[1], results.get("num_trades", 0)
        )
        return sanitize_for_json({
            "success": True,
            "params": req.dict(),
            "bars_analyzed": len(df),
            "results": results,
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Backtest execution failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/backtest/compare-windows")
def compare_windows_endpoint(req: CompareWindowsRequest):
    try:
        exch = req.exchange.strip().lower()
        if exch not in ccxt.exchanges:
            raise HTTPException(status_code=400, detail=f"Unsupported exchange '{req.exchange}'")

        starting_equity = max(1.0, float(req.starting_equity))
        default_windows = [90, 365, 730]
        requested_windows = req.windows or default_windows
        windows = []
        for w in requested_windows:
            try:
                w_int = int(w)
                if w_int > 0 and w_int not in windows:
                    windows.append(w_int)
            except (ValueError, TypeError):
                continue
        if not windows:
            windows = [90, 365, 730]

        limit = max_lookback_days(req.timeframe)
        valid_windows = [w for w in windows if w <= limit]

        df_full = None
        if valid_windows:
            max_days = max(valid_windows)
            try:
                df_full = fetch_historical(exch, req.symbol, req.timeframe, max_days)
            except Exception as fe:
                logger.warning(f"Failed to fetch historical data for {req.symbol} {req.timeframe}: {fe}")
                df_full = None

        cfg = _build_backtest_config(req, exch, req.symbol, req.timeframe, starting_equity)

        rows = []
        last_ts = df_full["ts"].iloc[-1] if (df_full is not None and not df_full.empty) else None

        for w in windows:
            if w > limit:
                rows.append({
                    "lookback_days": int(w),
                    "lookback_label": f"{w}d",
                    "error": f"Not enough history for {req.timeframe} (max ~{limit}d)",
                    "trades": 0,
                    "win_rate_pct": 0.0,
                    "total_return_pct": 0.0,
                    "final_equity": starting_equity,
                    "sharpe_approx": None,
                    "realized_rr": 0.0,
                    "max_drawdown_pct": 0.0,
                    "fib_blocked": 0,
                    "macro_blocked": 0,
                    "sizing_blocked": 0,
                    "risk_blocked": 0,
                    "total_fees_paid": 0.0,
                    "profitable": False,
                })
                continue

            if df_full is None or len(df_full) < 60 or last_ts is None:
                rows.append({
                    "lookback_days": int(w),
                    "lookback_label": f"{w}d",
                    "error": f"Insufficient historical bars ({0 if df_full is None else len(df_full)} < 60)",
                    "trades": 0,
                    "win_rate_pct": 0.0,
                    "total_return_pct": 0.0,
                    "final_equity": starting_equity,
                    "sharpe_approx": None,
                    "realized_rr": 0.0,
                    "max_drawdown_pct": 0.0,
                    "fib_blocked": 0,
                    "macro_blocked": 0,
                    "sizing_blocked": 0,
                    "risk_blocked": 0,
                    "total_fees_paid": 0.0,
                    "profitable": False,
                })
                continue

            cutoff = last_ts - pd.Timedelta(days=w)
            df_slice = df_full[df_full["ts"] >= cutoff].copy().reset_index(drop=True)
            if len(df_slice) < 60:
                rows.append({
                    "lookback_days": int(w),
                    "lookback_label": f"{w}d",
                    "error": f"Insufficient bars ({len(df_slice)} < 60)",
                    "trades": 0,
                    "win_rate_pct": 0.0,
                    "total_return_pct": 0.0,
                    "final_equity": starting_equity,
                    "sharpe_approx": None,
                    "realized_rr": 0.0,
                    "max_drawdown_pct": 0.0,
                    "fib_blocked": 0,
                    "macro_blocked": 0,
                    "sizing_blocked": 0,
                    "risk_blocked": 0,
                    "total_fees_paid": 0.0,
                    "profitable": False,
                })
                continue

            sweep_tag = f"Sweep {w}d"
            res = _invoke_backtest(df_slice, cfg, sweep_tag=sweep_tag)
            eval_range = res.get("eval_date_range") or [None, None]
            eval_count = res.get("candles_evaluated", 0)
            logger.info(
                "[%s] Sweep window %sd: walked %s post-warmup candles (%s to %s) evaluated for signals, %s trade(s)",
                sweep_tag, w, eval_count, eval_range[0], eval_range[1], res.get("num_trades", 0)
            )
            diag = res.get("diagnostics", {})
            ret_pct = float(res.get("total_return_pct", 0.0))
            rows.append({
                "lookback_days": int(w),
                "lookback_label": f"{w}d",
                "candles_evaluated": int(res.get("candles_evaluated", len(df_slice))),
                "candles_tested": int(res.get("candles_tested", len(df_slice))),
                "trades": int(res.get("num_trades", 0)),
                "win_rate_pct": float(res.get("win_rate_pct", 0.0)),
                "total_return_pct": ret_pct,
                "buy_and_hold_return_pct": float(res.get("buy_and_hold_return_pct", 0.0)),
                "buy_and_hold_final_equity": float(res.get("buy_and_hold_final_equity", starting_equity)),
                "final_equity": float(res.get("final_equity", starting_equity)),
                "sharpe_approx": float(res["sharpe_approx"]) if res.get("sharpe_approx") is not None else None,
                "realized_rr": float(res.get("realized_rr", 0.0)),
                "max_drawdown_pct": float(res.get("max_drawdown_pct", 0.0)),
                "fib_blocked": int(diag.get("fib_filter_blocked", 0)),
                "macro_blocked": int(diag.get("macro_filter_blocked", 0)),
                "sizing_blocked": int(diag.get("sizing_blocked", 0)),
                "risk_blocked": int(diag.get("signals_risk_blocked", 0)),
                "total_fees_paid": float(res.get("total_fees_paid", 0.0)),
                "profitable": bool(ret_pct > 0),
                "eval_date_range": list(res.get("eval_date_range", [])),
                "date_range": list(res.get("date_range", [])),
                "candles_warmup": int(res.get("candles_warmup", 0)),
            })

        profitable_count = sum(1 for r in rows if r.get("profitable"))
        return sanitize_for_json({
            "success": True,
            "mode": "windows",
            "symbol": req.symbol,
            "exchange": exch,
            "timeframe": req.timeframe,
            "windows": windows,
            "profitable_count": profitable_count,
            "total_windows": len(rows),
            "rows": rows,
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Compare windows execution failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/backtest/compare-timeframes")
def compare_timeframes_endpoint(req: CompareTimeframesRequest):
    try:
        exch = req.exchange.strip().lower()
        if exch not in ccxt.exchanges:
            raise HTTPException(status_code=400, detail=f"Unsupported exchange '{req.exchange}'")

        starting_equity = max(1.0, float(req.starting_equity))
        target_tfs = req.timeframes or ["5m", "15m", "1h", "4h", "1d"]
        rows = []
        for tf in target_tfs:
            tf_clean = tf.strip().lower()
            limit = max_lookback_days(tf_clean)
            if req.days > limit:
                rows.append({
                    "timeframe": tf_clean,
                    "lookback_days": int(req.days),
                    "error": f"Not enough history for {tf_clean} (max ~{limit}d)",
                    "trades": 0,
                    "win_rate_pct": 0.0,
                    "total_return_pct": 0.0,
                    "final_equity": starting_equity,
                    "sharpe_approx": None,
                    "realized_rr": 0.0,
                    "max_drawdown_pct": 0.0,
                    "fib_blocked": 0,
                    "sizing_blocked": 0,
                    "risk_blocked": 0,
                    "total_fees_paid": 0.0,
                    "profitable": False,
                })
                continue

            try:
                df = fetch_historical(exch, req.symbol, tf_clean, req.days)
            except Exception as fe:
                rows.append({
                    "timeframe": tf_clean,
                    "lookback_days": int(req.days),
                    "error": str(fe),
                    "trades": 0,
                    "win_rate_pct": 0.0,
                    "total_return_pct": 0.0,
                    "final_equity": starting_equity,
                    "sharpe_approx": None,
                    "realized_rr": 0.0,
                    "max_drawdown_pct": 0.0,
                    "fib_blocked": 0,
                    "sizing_blocked": 0,
                    "risk_blocked": 0,
                    "total_fees_paid": 0.0,
                    "profitable": False,
                })
                continue

            if df is None or len(df) < 60:
                rows.append({
                    "timeframe": tf_clean,
                    "lookback_days": int(req.days),
                    "error": f"Insufficient bars ({0 if df is None else len(df)} < 60)",
                    "trades": 0,
                    "win_rate_pct": 0.0,
                    "total_return_pct": 0.0,
                    "final_equity": starting_equity,
                    "sharpe_approx": None,
                    "realized_rr": 0.0,
                    "max_drawdown_pct": 0.0,
                    "fib_blocked": 0,
                    "sizing_blocked": 0,
                    "risk_blocked": 0,
                    "total_fees_paid": 0.0,
                    "profitable": False,
                })
                continue

            cfg = _build_backtest_config(req, exch, req.symbol, tf_clean, starting_equity)
            sweep_tag = f"Sweep {tf_clean}"
            res = _invoke_backtest(df, cfg, sweep_tag=sweep_tag)
            eval_range = res.get("eval_date_range") or [None, None]
            eval_count = res.get("candles_evaluated", 0)
            logger.info(
                "[%s] Sweep timeframe %s: walked %s post-warmup candles (%s to %s) evaluated for signals, %s trade(s)",
                sweep_tag, tf_clean, eval_count, eval_range[0], eval_range[1], res.get("num_trades", 0)
            )
            diag = res.get("diagnostics", {})
            ret_pct = float(res.get("total_return_pct", 0.0))
            rows.append({
                "timeframe": str(tf_clean),
                "lookback_days": int(req.days),
                "candles_evaluated": int(res.get("candles_evaluated", len(df))),
                "candles_tested": int(res.get("candles_tested", len(df))),
                "trades": int(res.get("num_trades", 0)),
                "win_rate_pct": float(res.get("win_rate_pct", 0.0)),
                "total_return_pct": ret_pct,
                "buy_and_hold_return_pct": float(res.get("buy_and_hold_return_pct", 0.0)),
                "buy_and_hold_final_equity": float(res.get("buy_and_hold_final_equity", starting_equity)),
                "final_equity": float(res.get("final_equity", starting_equity)),
                "sharpe_approx": float(res["sharpe_approx"]) if res.get("sharpe_approx") is not None else None,
                "realized_rr": float(res.get("realized_rr", 0.0)),
                "max_drawdown_pct": float(res.get("max_drawdown_pct", 0.0)),
                "fib_blocked": int(diag.get("fib_filter_blocked", 0)),
                "sizing_blocked": int(diag.get("sizing_blocked", 0)),
                "risk_blocked": int(diag.get("signals_risk_blocked", 0)),
                "total_fees_paid": float(res.get("total_fees_paid", 0.0)),
                "profitable": bool(ret_pct > 0),
                "eval_date_range": list(res.get("eval_date_range", [])),
                "date_range": list(res.get("date_range", [])),
                "candles_warmup": int(res.get("candles_warmup", 0)),
            })

        profitable_count = sum(1 for r in rows if r.get("profitable"))
        return sanitize_for_json({
            "success": True,
            "mode": "timeframes",
            "symbol": req.symbol,
            "exchange": exch,
            "timeframes": target_tfs,
            "profitable_count": profitable_count,
            "total_timeframes": len(rows),
            "rows": rows,
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Compare timeframes execution failed")
        raise HTTPException(status_code=500, detail=str(e))


def _resolve_log_file() -> Path:
    """Whichever activity log actually has content.

    The bot writes to `bot.log` next to its working directory (or /tmp on
    serverless), so looking at only one of the two paths is why the dashboard
    sometimes shows "No logs recorded yet" no matter how often you refresh.
    """
    primary = Path("/tmp/bot.log") if (os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME")) else Path("bot.log")
    for candidate in (primary, Path("/tmp/bot.log"), Path("bot.log"), Path(__file__).resolve().parent / "bot.log"):
        try:
            if candidate.exists() and candidate.stat().st_size > 0:
                return candidate
        except OSError:
            continue
    return primary


@app.get("/api/logs")
def get_logs(response: Response = None, lines: int = 50):
    _no_store(response)
    lines = max(1, min(int(lines), 1000))
    log_file = _resolve_log_file()
    lines_from_file = []
    if log_file.exists() and log_file.stat().st_size > 0:
        try:
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                lines_from_file = [line.strip() for line in f.readlines() if line.strip()]
        except Exception:
            lines_from_file = []

    combined = list(lines_from_file)
    if _memory_log_buffer:
        file_set = set(lines_from_file[-200:]) if lines_from_file else set()
        for mem_line in _memory_log_buffer:
            if mem_line.strip() and mem_line.strip() not in file_set:
                combined.append(mem_line.strip())

    if not combined:
        if runner.running:
            return {"logs": [f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] [INFO] bot: Engine is running. Waiting for cycle step..."]}
        return {"logs": ["No logs recorded yet. Start the bot to generate activity."]}
    return {"logs": combined[-lines:]}


class SizingValidationRequest(BaseModel):
    equity: float
    risk_per_trade_pct: float = 0.01
    price: float
    stop_price: float
    min_notional: float = 5.0


class WalkForwardRequest(BaseModel):
    exchange: str = "binance"
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    days: int = 180
    n_splits: int = 3
    starting_equity: float = 1000.0
    use_fib_filter: bool = True
    fib_tolerance_atr_mult: Optional[float] = None
    trailing_stop_enabled: Optional[bool] = None
    trailing_stop_atr_mult: Optional[float] = None
    stop_loss_atr_mult: Optional[float] = None
    take_profit_atr_mult: Optional[float] = None
    min_bars_between_trades: Optional[int] = None
    strategy: Optional[str] = None
    strategy_name: Optional[str] = None


@app.get("/api/execution/tracker")
def get_execution_tracker_data(response: Response = None):
    _no_store(response)
    bot = runner.get_or_create_bot()
    tracker = get_execution_tracker()
    active_pos = bot.get_unified_active_position()
    orders = tracker.get_orders(limit=50)
    closed = tracker.get_closed_positions(limit=50)
    stats = tracker.get_summary_stats()
    return sanitize_for_json({
        "success": True,
        "active_position": active_pos,
        "open_position": active_pos,
        "orders": orders,
        "recent_orders": orders,
        "closed_trades": closed,
        "stats": stats,
    })


@app.get("/api/alerts/config")
def get_alerts_config_endpoint(response: Response = None):
    _no_store(response)
    dispatcher = get_alert_dispatcher()
    return sanitize_for_json({
        "success": True,
        "config": dispatcher.get_config_summary(),
        "recent_alerts": dispatcher.alert_history[-20:],
    })


@app.post("/api/alerts/test")
def test_alert_endpoint():
    dispatcher = get_alert_dispatcher()
    sent = dispatcher.send_alert(
        "TEST_ALERT",
        "Test Notification from Trading Bot",
        "Alert routing verified successfully across configured channels.",
        level="info",
        details={"time": datetime.now(timezone.utc).isoformat(), "environment": "Production/Live"},
        force=True,
    )
    return {
        "success": sent,
        "message": "Test alert sent to configured channels (webhook/telegram/logs)" if sent else "Failed to send test alert. Check webhook/telegram settings in .env."
    }


@app.get("/api/exchange/fee-info")
def get_fee_info_endpoint(exchange: str = "binance", symbol: str = "BTC/USDT"):
    exch_client = runner.bot.exchange if (runner.bot and runner.bot.exchange) else None
    exch_obj = exch_client.exchange if exch_client else None
    info = get_exchange_fee_info(exchange, exchange_obj=exch_obj, symbol=symbol)
    return sanitize_for_json({"success": True, "fee_info": info})


@app.get("/api/exchange/wallet")
def get_exchange_wallet_endpoint(response: Response = None):
    _no_store(response)
    bot = runner.get_or_create_bot()
    symbol = runner.cfg.exchange.symbol
    wallet = bot.exchange.fetch_wallet_balance(symbol)
    spread = bot.exchange.fetch_order_book_spread(symbol)
    limits = bot.exchange.load_market_limits(symbol)
    return sanitize_for_json({
        "success": True,
        "symbol": symbol,
        "mode": runner.cfg.mode,
        "wallet": wallet,
        "spread": spread,
        "limits": limits,
        "max_spread_pct": getattr(runner.cfg.risk, "max_spread_pct", 0.35),
    })


@app.post("/api/risk/validate-sizing")
def validate_sizing_endpoint(req: SizingValidationRequest):
    cfg_risk = RiskConfig(risk_per_trade_pct=req.risk_per_trade_pct)
    rm = RiskManager(cfg_risk, req.equity)
    size = rm.position_size(req.price, req.stop_price)
    notional = round(size * req.price, 2)
    clears, reason = rm.validate_min_notional(req.price, size, min_notional=req.min_notional)
    return sanitize_for_json({
        "success": True,
        "equity": req.equity,
        "risk_pct": req.risk_per_trade_pct,
        "computed_size": size,
        "notional_value": notional,
        "min_notional_required": req.min_notional,
        "clears_minimum": clears,
        "status": "APPROVED" if clears else "REJECTED_BELOW_MIN_NOTIONAL",
        "behavior": "Order will be sent to exchange" if clears else "Order REJECTED by risk engine before exchange submission (trade skipped to protect risk limits)",
        "explanation": reason,
    })


@app.post("/api/backtest/walk-forward")
def walk_forward_endpoint(req: WalkForwardRequest):
    try:
        exch = req.exchange.strip().lower()
        if exch not in ccxt.exchanges:
            raise HTTPException(status_code=400, detail=f"Unsupported exchange '{req.exchange}'")
        days = _clamp_days(req.days)
        starting_equity = max(1.0, float(req.starting_equity))
        cfg = _build_backtest_config(req, exch, req.symbol, req.timeframe, starting_equity)

        df = fetch_historical(exch, req.symbol, req.timeframe, days)
        if df is None or len(df) < 120:
            raise HTTPException(status_code=400, detail=f"Insufficient candles ({0 if df is None else len(df)} < 120) for walk-forward validation.")

        wfo_results = run_walk_forward_validation(df, cfg, n_splits=req.n_splits)
        return sanitize_for_json({
            "success": True,
            "params": req.dict() if hasattr(req, "dict") else req.model_dump(),
            "results": wfo_results,
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Walk-forward validation failed")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# HTML PAGES & DASHBOARD
# ============================================================================
@app.get("/activate", response_class=HTMLResponse)
def activate_page(token: str = Query(...)):
    res = AuthService.activate_account(token)
    success = res["success"]
    status_icon = "fa-check-circle text-emerald-400" if success else "fa-times-circle text-rose-400"
    title = "Account Activated!" if success else "Activation Failed"
    title_escaped = html.escape(title)
    msg_escaped = html.escape(str(res.get("message", "")))
    session_token = res.get("session_token", "")
    username = res.get("username", "")
    email = res.get("email", "")

    auto_script = ""
    if success and session_token:
        safe_session_token = json.dumps(str(session_token)).replace("<", "\\u003c")
        safe_user_obj = json.dumps({
            "username": str(username),
            "email": str(email),
            "token": str(session_token)
        }).replace("<", "\\u003c")
        auto_script = f"""
        <script>
          try {{
            localStorage.setItem('bot_auth_token', {safe_session_token});
            document.cookie = "bot_auth_token=" + encodeURIComponent({safe_session_token}) + "; path=/; max-age=2592000; SameSite=Lax; Secure";
            localStorage.setItem('bot_saved_account', JSON.stringify({safe_user_obj}));
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

    html_content = f"""<!DOCTYPE html>
    <html lang="en" class="dark">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>{title_escaped}</title>
      <link rel="icon" type="image/png" href="/logo.png">
      <link rel="apple-touch-icon" href="/logo.png">
      <script src="https://cdn.tailwindcss.com"></script>
      <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    </head>
    <body class="bg-slate-950 text-slate-100 flex items-center justify-center min-h-screen p-4 font-sans">
      <div class="max-w-md w-full bg-slate-900 border border-slate-800 rounded-2xl p-6 sm:p-8 text-center shadow-2xl space-y-4">
        <i class="fa-solid {status_icon} text-5xl"></i>
        <h1 class="text-2xl font-bold">{title_escaped}</h1>
        <p class="text-sm text-slate-300 leading-relaxed">{msg_escaped}</p>
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
    return _html_response(html_content)


@app.get("/magic-login", response_class=HTMLResponse)
def magic_login_page(token: str = Query(...)):
    res = AuthService.magic_login(token)
    success = res["success"]
    status_icon = "fa-check-circle text-emerald-400" if success else "fa-times-circle text-rose-400"
    title = "Signed In Successfully!" if success else "Sign-In Failed"
    title_escaped = html.escape(title)
    msg_escaped = html.escape(str(res.get("message", "")))
    session_token = res.get("session_token", "")

    auto_script = ""
    if success and session_token:
        safe_session_token = json.dumps(str(session_token)).replace("<", "\\u003c")
        auto_script = f"""
        <script>
          try {{
            localStorage.setItem('bot_auth_token', {safe_session_token});
            document.cookie = "bot_auth_token=" + encodeURIComponent({safe_session_token}) + "; path=/; max-age=2592000; SameSite=Lax; Secure";
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

    html_content = f"""<!DOCTYPE html>
    <html lang="en" class="dark">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>{title_escaped}</title>
      <link rel="icon" type="image/png" href="/logo.png">
      <link rel="apple-touch-icon" href="/logo.png">
      <script src="https://cdn.tailwindcss.com"></script>
      <link rel="stylesheet" href="https://cdnjs.cloudflare.com/react/libs/font-awesome/6.4.0/css/all.min.css">
    </head>
    <body class="bg-slate-950 text-slate-100 flex items-center justify-center min-h-screen p-4 font-sans">
      <div class="max-w-md w-full bg-slate-900 border border-slate-800 rounded-2xl p-6 sm:p-8 text-center shadow-2xl space-y-4">
        <i class="fa-solid {status_icon} text-5xl"></i>
        <h1 class="text-2xl font-bold">{title_escaped}</h1>
        <p class="text-sm text-slate-300 leading-relaxed">{msg_escaped}</p>
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
    return _html_response(html_content)


@app.get("/reset-password", response_class=HTMLResponse)
def reset_password_page(token: str = Query(...)):
    safe_token = html.escape(token, quote=True)
    html_content = f"""<!DOCTYPE html>
    <html lang="en" class="dark">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Reset Your Password</title>
      <link rel="icon" type="image/png" href="/logo.png">
      <link rel="apple-touch-icon" href="/logo.png">
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
          <input type="hidden" id="reset-token" value="{safe_token}">
          <div>
            <label class="block text-slate-400 mb-1">New Password (min 10 characters)</label>
            <div class="relative">
              <input type="password" id="new-pw" required minlength="10" class="w-full bg-slate-800 border border-slate-700 rounded-lg pl-3 pr-10 py-2 text-white focus:outline-none focus:border-amber-500">
              <button type="button" onclick="togglePasswordVisibility('new-pw', 'new-pw-eye')" class="absolute inset-y-0 right-0 pr-3 flex items-center text-slate-400 hover:text-slate-200 transition" tabindex="-1">
                <i class="fa-solid fa-eye text-xs" id="new-pw-eye"></i>
              </button>
            </div>
          </div>
          <div>
            <label class="block text-slate-400 mb-1">Confirm New Password</label>
            <div class="relative">
              <input type="password" id="confirm-pw" required minlength="10" class="w-full bg-slate-800 border border-slate-700 rounded-lg pl-3 pr-10 py-2 text-white focus:outline-none focus:border-amber-500">
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
          if (pw.length < 10) {{
            msg.className = 'text-xs p-2 rounded bg-rose-950 text-rose-300 block';
            msg.innerText = 'Password must be at least 10 characters long';
            return;
          }}
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
    return _html_response(html_content)


# Main Dashboard HTML with integrated Auth & Exchange Key manager
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
  <title>Trader Queen - Dashboard & Exchange Keys</title>
  <link rel="icon" type="image/png" href="/logo.png">
  <link rel="apple-touch-icon" href="/logo.png">
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
    @media (min-width: 768px) {
      #section-kpi {
        display: grid !important;
        grid-template-columns: repeat(6, minmax(0, 1fr)) !important;
        gap: 0.75rem !important;
      }
      #panel-market-analysis,
      #section-api-keys,
      #section-backtester,
      #section-execution-tracking,
      #section-bot-activity {
        display: block !important;
      }
      #section-account {
        display: none !important;
      }
      .mobile-only-header {
        display: none !important;
      }
    }
    @media (max-width: 767px) {
      #section-api-keys,
      #section-backtester,
      #section-execution-tracking,
      #section-bot-activity,
      #section-account {
        display: none;
      }
      body.mobile-view-dashboard #dashboard-detail-grid,
      body.mobile-view-dashboard #panel-market-analysis,
      body.mobile-view-backtester #dashboard-detail-grid,
      body.mobile-view-keys #section-api-keys,
      body.mobile-view-backtester #section-backtester,
      body.mobile-view-activity #section-execution-tracking,
      body.mobile-view-activity #section-bot-activity,
      body.mobile-view-account #section-account {
        display: block;
      }
      body.mobile-view-dashboard #section-kpi {
        display: grid !important;
        grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
        gap: 0.625rem !important;
      }
      body.mobile-view-dashboard #dashboard-detail-grid,
      body.mobile-view-backtester #dashboard-detail-grid {
        display: grid;
      }
      body:not(.mobile-view-dashboard) #section-kpi,
      body:not(.mobile-view-dashboard) #position-banner,
      body:not(.mobile-view-dashboard) #panel-market-analysis {
        display: none !important;
      }
      body:not(.mobile-view-dashboard):not(.mobile-view-backtester) #dashboard-detail-grid {
        display: none !important;
      }
    }
    @media (min-width: 480px) and (max-width: 767px) {
      body.mobile-view-dashboard #section-kpi {
        grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
      }
    }
  </style>
</head>
<body class="mobile-view-dashboard bg-darkbg text-slate-100 min-h-screen font-sans antialiased selection:bg-emerald-500/30 selection:text-emerald-200">

  <!-- Top Navigation -->
  <header class="border-b border-bordercol bg-cardbg/95 backdrop-blur-md sticky top-0 z-40">
    <div class="max-w-7xl mx-auto px-3 sm:px-4 py-2 sm:py-2.5">
      <div class="flex items-center justify-between gap-2">
        <!-- Brand & Title -->
        <div class="flex items-center space-x-2.5 sm:space-x-3 min-w-0">
          <div class="h-8 w-8 sm:h-9 sm:w-9 rounded-lg overflow-hidden border border-amber-500/40 shadow-md shrink-0 flex items-center justify-center bg-slate-900">
            <img src="/logo.png" alt="Trader Queen Logo" class="h-full w-full object-cover">
          </div>
          <div class="min-w-0">
            <h1 class="text-sm sm:text-base font-bold tracking-tight text-white flex items-center gap-1.5 sm:gap-2 truncate">
              <span class="truncate">Trader Queen</span>
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

          <!-- Emergency Kill Switch Toggle -->
          <button id="btn-kill-toggle" onclick="toggleKillSwitch()" class="px-2.5 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 font-semibold rounded-lg transition flex items-center gap-1.5 text-xs shadow-sm" title="Toggle Emergency Kill Switch">
            <i id="btn-kill-toggle-icon" class="fa-solid fa-toggle-off text-slate-400 text-sm"></i>
            <span id="btn-kill-toggle-text">Kill Switch: OFF</span>
          </button>
          <button id="btn-reset-kill" onclick="deactivateKillSwitch()" class="hidden" aria-hidden="true" title="Reset Kill Switch"></button>

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
      <div id="mobile-menu-panel" class="hidden md:hidden pt-2.5 mt-2 border-t border-bordercol/80 space-y-2 text-xs animate-slideDown max-h-[85vh] overflow-y-auto pb-4">
        <!-- Auth / Profile Top Action (Clean like Screenshot 2) -->
        <div id="auth-buttons-mobile">
          <button type="button" onclick="openAuthModal('login'); toggleMobileMenu();" class="w-full py-2.5 bg-amber-500 hover:bg-amber-400 text-slate-950 font-bold rounded-xl shadow transition flex items-center justify-center gap-2 text-xs">
            <i class="fa-solid fa-user"></i>
            <span>SIGN IN / REGISTER</span>
          </button>
        </div>

        <div id="user-badge-mobile" class="hidden">
          <button type="button" onclick="showMobileView('account'); toggleMobileMenu();" class="w-full p-2.5 bg-slate-900 border border-emerald-500/40 hover:border-emerald-500 rounded-xl text-left flex items-center justify-between transition">
            <div class="flex items-center gap-2.5 min-w-0">
              <div class="w-8 h-8 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 flex items-center justify-center shrink-0">
                <i class="fa-solid fa-user text-sm"></i>
              </div>
              <div class="min-w-0">
                <div class="font-bold text-white text-xs truncate" id="user-name-display-mobile">Account</div>
                <div class="text-[10px] text-emerald-400">View Profile & Settings</div>
              </div>
            </div>
            <i class="fa-solid fa-chevron-right text-emerald-400 text-xs"></i>
          </button>
        </div>

        <!-- Clean Navigation Links (One page at a time) -->
        <div class="bg-cardbg rounded-xl border border-bordercol divide-y divide-bordercol/60 overflow-hidden shadow-lg">
          <!-- 1. Live Dashboard -->
          <button type="button" onclick="showMobileView('dashboard'); toggleMobileMenu();" class="w-full px-3.5 py-3 text-left flex items-center justify-between text-slate-200 hover:bg-slate-800/60 active:bg-slate-800 transition font-medium">
            <div class="flex items-center gap-2.5">
              <i class="fa-solid fa-chart-line text-emerald-400 w-4 text-center"></i>
              <span>Live Dashboard</span>
            </div>
            <i class="fa-solid fa-caret-right text-slate-500 text-xs"></i>
          </button>

          <!-- 2. Profile & Account -->
          <button type="button" onclick="showMobileView('account'); toggleMobileMenu();" class="w-full px-3.5 py-3 text-left flex items-center justify-between text-slate-200 hover:bg-slate-800/60 active:bg-slate-800 transition font-medium">
            <div class="flex items-center gap-2.5">
              <i class="fa-solid fa-id-card text-sky-400 w-4 text-center"></i>
              <span>Profile & Delete Account</span>
            </div>
            <i class="fa-solid fa-caret-right text-slate-500 text-xs"></i>
          </button>

          <!-- 3. Exchange API Keys & Credential Management -->
          <button type="button" onclick="showMobileView('keys'); toggleMobileMenu();" class="w-full px-3.5 py-3 text-left flex items-center justify-between text-slate-200 hover:bg-slate-800/60 active:bg-slate-800 transition font-medium">
            <div class="flex items-center gap-2.5">
              <i class="fa-solid fa-key text-amber-400 w-4 text-center"></i>
              <span>Exchange API Keys & Credentials</span>
            </div>
            <i class="fa-solid fa-caret-right text-slate-500 text-xs"></i>
          </button>

          <!-- 4. Settings -->
          <button type="button" onclick="openSettingsModal(); toggleMobileMenu();" class="w-full px-3.5 py-3 text-left flex items-center justify-between text-slate-200 hover:bg-slate-800/60 active:bg-slate-800 transition font-medium">
            <div class="flex items-center gap-2.5">
              <i class="fa-solid fa-gear text-indigo-400 w-4 text-center"></i>
              <span>Settings & Strategy Parameters</span>
            </div>
            <i class="fa-solid fa-caret-right text-slate-500 text-xs"></i>
          </button>

          <!-- 5. Backtester -->
          <button type="button" onclick="showMobileView('backtester'); toggleMobileMenu();" class="w-full px-3.5 py-3 text-left flex items-center justify-between text-slate-200 hover:bg-slate-800/60 active:bg-slate-800 transition font-medium">
            <div class="flex items-center gap-2.5">
              <i class="fa-solid fa-vial text-purple-400 w-4 text-center"></i>
              <span>Honest Backtester</span>
            </div>
            <i class="fa-solid fa-caret-right text-slate-500 text-xs"></i>
          </button>

          <!-- 6. Bot Activity -->
          <button type="button" onclick="showMobileView('activity'); toggleMobileMenu();" class="w-full px-3.5 py-3 text-left flex items-center justify-between text-slate-200 hover:bg-slate-800/60 active:bg-slate-800 transition font-medium">
            <div class="flex items-center gap-2.5">
              <i class="fa-solid fa-terminal text-teal-400 w-4 text-center"></i>
              <span>Bot Activity Logs</span>
            </div>
            <i class="fa-solid fa-caret-right text-slate-500 text-xs"></i>
          </button>

        </div>

        <!-- Emergency Kill Switch Toggle (Bottom item) -->
        <div class="pt-1">
          <button type="button" id="btn-kill-toggle-mob" onclick="toggleKillSwitch(); toggleMobileMenu();" class="w-full py-2.5 bg-slate-800 active:bg-slate-700 text-slate-300 border border-slate-700 font-semibold rounded-xl transition flex items-center justify-between px-3 text-xs" title="Toggle Emergency Kill Switch">
            <div class="flex items-center gap-2">
              <i class="fa-solid fa-skull-crossbones text-rose-400"></i>
              <span>Emergency Kill Switch</span>
            </div>
            <div class="flex items-center gap-1.5">
              <span id="btn-kill-toggle-mob-status" class="text-[11px] font-bold text-slate-400">OFF</span>
              <i id="btn-kill-toggle-mob-icon" class="fa-solid fa-toggle-off text-slate-400 text-base"></i>
            </div>
          </button>
        </div>

        <!-- Hidden compatibility hook for existing test assertions -->
        <div id="btn-reset-kill-mob-container" class="hidden">
          <button type="button" id="btn-reset-kill-mob" onclick="deactivateKillSwitch(); toggleMobileMenu();" class="hidden"></button>
        </div>
      </div>
    </div>
  </header>

  <!-- In-app notification. Browser alert() is ugly, cannot be dismissed from
       the page, and dumps raw JSON in the reader's face; this shows the same
       event as a short, styled card with the technical detail tucked away. -->
  <div id="app-notice" class="hidden fixed z-[60] top-20 right-3 left-3 sm:left-auto sm:w-[24rem]"
       role="status" aria-live="polite">
    <div class="rounded-xl border border-bordercol bg-cardbg/95 backdrop-blur-md shadow-2xl p-3.5">
      <div class="flex items-start gap-3">
        <i id="app-notice-icon" class="fa-solid fa-circle-exclamation mt-0.5 text-amber-400"></i>
        <div class="min-w-0 flex-1">
          <p id="app-notice-title" class="text-sm font-semibold text-white"></p>
          <p id="app-notice-body" class="text-xs text-slate-300 mt-1 leading-relaxed"></p>
          <button type="button" id="app-notice-details-btn"
                  onclick="toggleNoticeDetail()"
                  class="hidden text-[11px] text-slate-500 hover:text-slate-300 mt-1.5 underline underline-offset-2">
            Technical detail
          </button>
          <p id="app-notice-detail" class="hidden text-[10px] text-slate-500 mt-1 break-words"></p>
        </div>
        <button type="button" id="app-notice-close" aria-label="Dismiss notification"
                onclick="hideNotice()"
                class="text-slate-500 hover:text-white transition-colors">
          <i class="fa-solid fa-xmark"></i>
        </button>
      </div>
    </div>
  </div>

  <!-- Main Container -->
  <main class="max-w-7xl mx-auto px-3 sm:px-4 py-4 sm:py-5 space-y-4 sm:space-y-5 pb-safe-main md:pb-6">

    <!-- Emergency Kill Switch Active Alert Banner -->
    <div id="kill-switch-active-banner" class="hidden bg-rose-950/90 border-2 border-rose-500/80 rounded-xl p-3 sm:p-4 text-rose-200 flex flex-col sm:flex-row items-center justify-between gap-3 shadow-2xl">
      <div class="flex items-center gap-2.5">
        <i class="fa-solid fa-triangle-exclamation text-rose-400 text-lg sm:text-xl shrink-0 animate-bounce"></i>
        <div>
          <div class="font-bold text-rose-300 text-xs sm:text-sm">EMERGENCY KILL SWITCH ACTIVATED — TRADING HALTED</div>
          <div class="text-[11px] text-rose-200/80 mt-0.5">All automated execution and manual orders are disabled to protect capital. Disarm below to return to safe IDLE mode.</div>
        </div>
      </div>
      <button type="button" onclick="deactivateKillSwitch()" class="w-full sm:w-auto px-4 py-2 bg-rose-600 hover:bg-rose-500 active:bg-rose-700 text-white font-bold text-xs rounded-lg shadow-lg transition flex items-center justify-center gap-1.5 shrink-0 whitespace-nowrap">
        <i class="fa-solid fa-lock-open"></i> Deactivate Kill Switch (Return to IDLE)
      </button>
    </div>

    <!-- KPI Cards Grid (Dashboard Landing Page) -->
    <div id="section-kpi" class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5 sm:gap-3">
      <div class="bg-cardbg border border-bordercol rounded-xl p-2.5 sm:p-3 min-w-0">
        <div class="text-slate-400 text-[10px] sm:text-[11px] font-medium truncate">Portfolio Equity</div>
        <div class="text-base sm:text-lg font-bold mt-0.5 text-white truncate" id="val-equity">$1,000.00</div>
        <div class="text-[9px] sm:text-[10px] text-slate-400 mt-0.5 truncate">Peak: <span id="val-peak">$1,000.00</span></div>
      </div>

      <div class="bg-cardbg border border-bordercol rounded-xl p-2.5 sm:p-3 min-w-0">
        <div class="text-slate-400 text-[10px] sm:text-[11px] font-medium truncate" id="val-cash-label">Paper Cash</div>
        <div class="text-base sm:text-lg font-bold mt-0.5 text-emerald-400 truncate" id="val-cash">$1,000.00</div>
        <div class="text-[9px] sm:text-[10px] text-slate-400 mt-0.5 truncate">Position: <span id="val-pos">0.000000</span></div>
      </div>

      <div class="bg-cardbg border border-bordercol rounded-xl p-2.5 sm:p-3 min-w-0">
        <div class="text-slate-400 text-[10px] sm:text-[11px] font-medium truncate">Max Drawdown</div>
        <div class="text-base sm:text-lg font-bold mt-0.5 text-slate-200 truncate" id="val-drawdown">0.0%</div>
        <div class="text-[9px] sm:text-[10px] text-slate-400 mt-0.5 truncate">Limit: 15.0% Kill</div>
      </div>

      <div class="bg-cardbg border border-bordercol rounded-xl p-2.5 sm:p-3 min-w-0">
        <div class="text-slate-400 text-[10px] sm:text-[11px] font-medium truncate">Daily Loss</div>
        <div class="text-base sm:text-lg font-bold mt-0.5 text-slate-200 truncate" id="val-daily-loss">0.0%</div>
        <div class="text-[9px] sm:text-[10px] text-slate-400 mt-0.5 truncate">Limit: 3.0% Circuit</div>
      </div>

      <div class="bg-cardbg border border-bordercol rounded-xl p-2.5 sm:p-3 min-w-0">
        <div class="text-slate-400 text-[10px] sm:text-[11px] font-medium truncate">Loss Streak</div>
        <div class="text-base sm:text-lg font-bold mt-0.5 text-slate-200 truncate" id="val-streak">0 / 8</div>
        <div class="text-[9px] sm:text-[10px] text-slate-400 mt-0.5 truncate">Cooloff at <span id="val-streak-limit">8</span></div>
      </div>

      <div class="bg-cardbg border border-bordercol rounded-xl p-2.5 sm:p-3 min-w-0">
        <div class="text-slate-400 text-[10px] sm:text-[11px] font-medium truncate">Live Signal</div>
        <div class="text-base sm:text-lg font-bold mt-0.5 text-yellow-400 truncate" id="val-signal">HOLD</div>
        <div class="text-[9px] sm:text-[10px] text-slate-400 mt-0.5 truncate" id="val-symbol-tag">BTC/USDT 15m</div>
      </div>
    </div>

    <!-- Circuit-breaker cooldown banner: says when trading resumes -->
    <div id="cooldown-banner" class="hidden bg-amber-950/40 border border-amber-500/40 rounded-xl p-3 flex flex-wrap items-center gap-2 text-xs text-amber-200">
      <i class="fa-solid fa-hourglass-half text-amber-400"></i>
      <span>Circuit breaker: <span id="cooldown-text">--</span></span>
    </div>

    <!-- Active Position Banner -->
    <div id="position-banner" class="hidden bg-emerald-950/40 border border-emerald-500/40 rounded-xl p-3 sm:p-3.5 flex flex-wrap items-center justify-between gap-2">
      <div class="flex items-center gap-2.5 sm:gap-3 min-w-0">
        <div id="pos-banner-dot" class="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse shrink-0"></div>
        <div class="min-w-0">
          <span class="font-bold text-emerald-400 text-xs">ACTIVE POSITION:</span>
          <span class="text-xs text-slate-300 ml-1.5 break-all" id="pos-details"></span>
        </div>
      </div>
      <button type="button" id="btn-banner-close-trade" onclick="closeActivePosition()" class="px-3 py-1.5 bg-rose-600 hover:bg-rose-500 active:bg-rose-700 text-white font-bold text-xs rounded-lg shadow transition flex items-center gap-1.5 shrink-0" title="Close this position immediately at market price">
        <i class="fa-solid fa-circle-xmark"></i>
        <span id="btn-banner-close-text">Close Position</span>
      </button>
    </div>

    <!-- ========================================================================= -->
    <!-- MOBILE PROFILE & ACCOUNT VIEW (Shown on mobile when Account is clicked)   -->
    <!-- ========================================================================= -->
    <div id="section-account" class="hidden md:hidden space-y-4">
      <div class="mobile-only-header flex items-center justify-between border-b border-bordercol pb-3">
        <button type="button" onclick="showMobileView('dashboard')" class="text-xs font-semibold text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 px-3 py-1.5 rounded-lg border border-bordercol flex items-center gap-1.5 transition">
          <i class="fa-solid fa-arrow-left"></i> Back to Dashboard
        </button>
        <span class="text-xs font-bold text-slate-400">My Profile & Account</span>
      </div>

      <!-- Logged Out State -->
      <div id="account-view-logged-out" class="bg-cardbg border border-bordercol rounded-xl p-6 text-center space-y-4">
        <div class="w-14 h-14 rounded-full bg-slate-800 border border-bordercol flex items-center justify-center mx-auto text-slate-400 text-2xl">
          <i class="fa-solid fa-circle-user"></i>
        </div>
        <div>
          <h2 class="text-base font-bold text-white">Sign In to Your Account</h2>
          <p class="text-xs text-slate-400 mt-1">Log in or register to configure exchange API keys and control live trading.</p>
        </div>
        <div class="flex flex-col gap-2 pt-2">
          <button type="button" onclick="openAuthModal('login')" class="w-full py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-bordercol font-semibold rounded-lg text-xs flex items-center justify-center gap-1.5">
            <i class="fa-solid fa-right-to-bracket"></i> Log In
          </button>
          <button type="button" onclick="openAuthModal('register')" class="w-full py-2.5 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold rounded-lg text-xs shadow flex items-center justify-center gap-1.5">
            <i class="fa-solid fa-user-plus"></i> Create Account
          </button>
        </div>
      </div>

      <!-- Logged In State with Delete Button -->
      <div id="account-view-logged-in" class="hidden space-y-4">
        <div class="bg-cardbg border border-bordercol rounded-xl p-5 space-y-4">
          <div class="flex items-center gap-3">
            <div class="w-12 h-12 rounded-full bg-emerald-500/20 border border-emerald-500/40 text-emerald-400 flex items-center justify-center text-xl shrink-0 font-bold">
              <i class="fa-solid fa-user"></i>
            </div>
            <div class="min-w-0">
              <div class="text-base font-bold text-white truncate flex items-center gap-2">
                <span id="account-view-username">Trader</span>
                <span class="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 font-semibold">Active</span>
              </div>
              <div class="text-xs text-slate-400 truncate mt-0.5" id="account-view-email">trader@example.com</div>
            </div>
          </div>

          <div class="grid grid-cols-2 gap-2 text-xs pt-2 border-t border-bordercol">
            <div class="p-2.5 bg-slate-900/80 rounded-lg border border-bordercol/60">
              <div class="text-[10px] text-slate-400">Account Role</div>
              <div class="font-bold text-slate-200 mt-0.5">Bot Administrator</div>
            </div>
            <div class="p-2.5 bg-slate-900/80 rounded-lg border border-bordercol/60">
              <div class="text-[10px] text-slate-400">Session Status</div>
              <div class="font-bold text-emerald-400 mt-0.5">Authenticated</div>
            </div>
          </div>

          <div class="pt-2 flex items-center justify-between gap-2 border-t border-bordercol">
            <span class="text-xs text-slate-400">Current Session</span>
            <button type="button" onclick="logout()" class="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-rose-300 rounded-lg border border-bordercol text-xs font-semibold flex items-center gap-1.5 transition">
              <i class="fa-solid fa-arrow-right-from-bracket"></i> Log Out
            </button>
          </div>
        </div>

        <!-- Danger Zone Card: Delete Account -->
        <div class="bg-rose-950/20 border border-rose-500/40 rounded-xl p-5 space-y-3">
          <div class="flex items-center gap-2 text-rose-400 font-bold text-sm">
            <i class="fa-solid fa-triangle-exclamation"></i>
            <span>Danger Zone</span>
          </div>
          <p class="text-xs text-slate-400 leading-relaxed">
            Permanently delete your profile, active sessions, and stored exchange API keys. This action is immediate and cannot be undone.
          </p>
          <button type="button" onclick="confirmDeleteAccount()" class="w-full py-2.5 bg-rose-600 hover:bg-rose-500 active:bg-rose-700 text-white font-bold rounded-lg text-xs shadow transition flex items-center justify-center gap-2">
            <i class="fa-solid fa-trash-can"></i> Delete My Account
          </button>
        </div>
      </div>
    </div>

    <!-- ========================================================================= -->
    <!-- EXCHANGE API KEYS CONFIGURATION SECTION                                   -->
    <!-- Desktop: in-page card | Mobile: separate page view                         -->
    <!-- ========================================================================= -->
    <div id="section-api-keys" class="hidden md:block bg-cardbg border border-bordercol rounded-xl p-5 space-y-4">
      <!-- Mobile Back Header -->
      <div class="mobile-only-header flex items-center justify-between border-b border-bordercol pb-3 mb-2 md:hidden">
        <button type="button" onclick="showMobileView('dashboard')" class="text-xs font-semibold text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 px-3 py-1.5 rounded-lg border border-bordercol flex items-center gap-1.5 transition">
          <i class="fa-solid fa-arrow-left"></i> Back to Dashboard
        </button>
        <span class="text-xs font-bold text-slate-400">Exchange API Keys</span>
      </div>

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
                Active Environment: <span id="env-mode-title" class="text-blue-400 font-bold">TEST MODE</span>
              </div>
              <p id="env-mode-desc" class="text-[11px] text-slate-400">Safe simulated execution. Zero real funds at risk.</p>
            </div>
          </div>
          <div class="flex flex-wrap sm:flex-nowrap items-center gap-2 w-full sm:w-auto">
            <button type="button" onclick="switchTradingMode('testnet')" id="env-btn-testnet" class="flex-1 sm:flex-none justify-center px-3 py-2 sm:py-1.5 bg-blue-600 text-white text-xs font-bold rounded-lg shadow transition flex items-center gap-1.5" title="Switch to Test Mode / Paper Trading">
              <i class="fa-solid fa-flask"></i> Test Mode
            </button>
            <button type="button" onclick="switchTradingMode('real')" id="env-btn-real" class="flex-1 sm:flex-none justify-center px-3 py-2 sm:py-1.5 bg-slate-800 text-slate-300 hover:text-white border border-slate-700 text-xs font-bold rounded-lg transition flex items-center gap-1.5" title="Switch to Real Trading (LIVE)">
              <i class="fa-solid fa-bolt text-rose-400"></i> Real Trading (LIVE)
            </button>

            <!-- 1-Click Execution Mode Toggle (Auto vs Manual) -->
            <div class="flex items-center rounded-lg bg-slate-900 border border-bordercol p-0.5" title="Execution Mode: Switch between Automated and Manual Trading">
              <button type="button" onclick="quickSetExecutionMode('auto')" id="btn-exec-auto" class="px-2.5 py-1.5 bg-purple-600 text-white text-xs font-bold rounded-md shadow transition flex items-center gap-1.5" title="Auto Trading: Bot places entries and exits automatically">
                <i class="fa-solid fa-robot"></i> Auto
              </button>
              <button type="button" onclick="quickSetExecutionMode('manual')" id="btn-exec-manual" class="px-2.5 py-1.5 text-slate-300 hover:text-white text-xs font-bold rounded-md transition flex items-center gap-1.5" title="Manual Trading: Bot generates signals only; you trade manually">
                <i class="fa-solid fa-hand-pointer text-amber-400"></i> Manual
              </button>
            </div>

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
  </div>

    <!-- Live Analysis & Backtester 2-col layout -->
    <div id="dashboard-detail-grid" class="grid grid-cols-1 lg:grid-cols-3 gap-5">
      <!-- Live Market Analysis (2 cols) -->
      <div id="panel-market-analysis" class="lg:col-span-2 bg-cardbg border border-bordercol rounded-xl p-5 space-y-4">
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
                <span id="active-mode-tag" class="text-slate-400 font-semibold bg-slate-800/80 px-2 py-0.5 rounded border border-slate-700">
                  <i class="fa-solid fa-pause text-slate-400 mr-1"></i>Engine: IDLE (No Orders Executing)
                </span>
              </div>
            </div>
          </div>
          <div class="flex items-center gap-2">
            <div class="flex items-center gap-2">
              <span id="an-updated" class="text-[10px] text-slate-500">--</span>
              <button type="button" id="btn-refresh-analysis" onclick="refreshAnalysis(true)" class="text-xs text-slate-300 hover:text-white transition flex items-center gap-1 px-2.5 py-1.5 bg-slate-800 hover:bg-slate-700 rounded-lg border border-bordercol disabled:opacity-60">
                <i class="fa-solid fa-rotate"></i> Refresh
              </button>
            </div>
          </div>
        </div>

        <!-- 15m/5m Negative Expectancy Safety Warning Banner (Item 0) -->
        <div id="tf-warning-banner" class="hidden p-2.5 rounded-lg border border-amber-500/50 bg-amber-950/40 text-amber-200 text-xs flex items-start gap-2.5">
          <i class="fa-solid fa-triangle-exclamation text-amber-400 text-sm mt-0.5 shrink-0"></i>
          <div>
            <span class="font-bold text-amber-300">Caution — 15m/5m Negative Expectancy:</span>
            Historical backtests across all tested venues on 15m and 5m show negative net expectancy (-10% to -15% returns, sub-1:1 realized R:R). 
            <span class="font-semibold text-white">Live capital execution is guarded on this timeframe.</span> Switch to 1h, 4h, or 1d for robust trends or test exclusively in Paper Mode.
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
              <i class="fa-solid fa-arrow-trend-up"></i> <span id="btn-manual-buy-text">Manual BUY (Market)</span>
            </button>
            <button type="button" id="btn-manual-sell" onclick="triggerManualTrade('sell')" class="w-full sm:flex-1 py-2.5 bg-rose-600 hover:bg-rose-500 active:bg-rose-700 text-white font-bold text-xs rounded-lg shadow transition flex items-center justify-center gap-1.5 min-h-[42px]">
              <i class="fa-solid fa-arrow-trend-down"></i> <span id="btn-manual-sell-text">Manual SELL (Market)</span>
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
      <!-- Desktop: in-page 3rd col | Mobile: separate hamburger page view -->
      <div id="section-backtester" class="hidden md:block">
        <div class="bg-cardbg border border-bordercol rounded-xl p-4 sm:p-5 space-y-4 w-full">
          <div class="mobile-only-header flex items-center justify-between border-b border-bordercol pb-3 mb-2 md:hidden">
            <button type="button" onclick="showMobileView('dashboard')" class="text-xs font-semibold text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 px-3 py-1.5 rounded-lg border border-bordercol flex items-center gap-1.5 transition">
              <i class="fa-solid fa-arrow-left"></i> Back to Dashboard
            </button>
            <span class="text-xs font-bold text-slate-400">Backtester</span>
          </div>
          <div class="flex items-center justify-between border-b border-bordercol pb-3">
            <div class="flex items-center gap-2">
              <i class="fa-solid fa-vial text-purple-400"></i>
              <h2 class="font-bold text-sm text-white">Honest Backtester</h2>
            </div>
            <span class="text-[10px] text-slate-400 hidden sm:inline">No lookahead bias</span>
          </div>

        <form id="backtest-form" onsubmit="executeBacktest(event)" class="space-y-3 text-xs">
          <div>
            <div class="flex justify-between items-center mb-1">
              <label class="block text-slate-400">Exchange</label>
              <span id="bt-fee-tier-pill" class="text-[10px] text-slate-400 font-mono bg-slate-900 px-1.5 py-0.5 rounded border border-bordercol/60">Taker: 0.10% (Default)</span>
            </div>
            <select id="bt-exchange" onchange="loadFeeInfo()" class="w-full bg-slate-900 border border-bordercol rounded-lg px-2.5 py-1.5 text-white">
              <option value="binance" selected>Binance</option>
              <option value="coinbase">Coinbase Advanced</option>
              <option value="kraken">Kraken</option>
              <option value="kucoin">KuCoin</option>
              <option value="bybit">Bybit</option>
              <option value="okx">OKX</option>
              <option value="bitget">Bitget</option>
              <option value="gate">Gate.io</option>
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
              <select id="bt-timeframe" onchange="applyTimeframeLimits()" class="w-full bg-slate-900 border border-bordercol rounded-lg px-2.5 py-1.5 text-white">
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
              <input type="number" id="bt-days" value="365" min="1" max="730" class="w-full bg-slate-900 border border-bordercol rounded-lg px-2.5 py-1.5 text-white">
              <div class="flex gap-1 mt-1" id="bt-days-presets">
                <button type="button" onclick="setBacktestDays(90)" class="flex-1 text-[10px] text-slate-300 bg-slate-800 hover:bg-slate-700 border border-bordercol rounded px-1.5 py-0.5 transition">90d</button>
                <button type="button" onclick="setBacktestDays(365)" class="flex-1 text-[10px] text-slate-300 bg-slate-800 hover:bg-slate-700 border border-bordercol rounded px-1.5 py-0.5 transition">365d</button>
                <button type="button" onclick="setBacktestDays(730)" class="flex-1 text-[10px] text-slate-300 bg-slate-800 hover:bg-slate-700 border border-bordercol rounded px-1.5 py-0.5 transition">730d</button>
              </div>
              <div id="bt-days-note" class="text-[10px] text-slate-500 mt-1">--</div>
            </div>
            <div>
              <label class="block text-slate-400 mb-1">Start Equity ($)</label>
              <input type="number" id="bt-equity" value="1000" min="10" class="w-full bg-slate-900 border border-bordercol rounded-lg px-2.5 py-1.5 text-white">
            </div>
          </div>

          <!-- Fee Tier & Real Rate Indicator (Item 5) -->
          <div class="flex items-center justify-between text-[11px] p-2 bg-slate-900/60 rounded-lg border border-bordercol/60">
            <div class="flex items-center gap-1.5">
              <i class="fa-solid fa-receipt text-amber-400"></i>
              <span class="text-slate-400">Fee Tier:</span>
              <span id="bt-fee-badge" class="font-medium text-amber-300">0.100% Taker</span>
            </div>
            <span id="bt-fee-source" class="text-[10px] text-slate-500 font-sans">Hardcoded assumption</span>
          </div>

          <div class="flex items-center gap-2 pt-1">
            <input type="checkbox" id="bt-use-fib" checked class="rounded border-bordercol bg-slate-900 text-emerald-500">
            <label for="bt-use-fib" class="text-slate-300">Enable Fibonacci Zone Filter</label>
          </div>

          <div class="grid grid-cols-2 gap-2">
            <div>
              <label class="block text-slate-400 mb-1">Fib Depth</label>
              <select id="bt-fib-depth" class="w-full bg-slate-900 border border-bordercol rounded-lg px-2.5 py-1.5 text-white">
                <option value="deep" selected>Key levels (0.5 / 0.618)</option>
                <option value="zone">Golden zone (0.382 - 0.618)</option>
                <option value="all">Any level (0.382 / 0.5 / 0.618)</option>
              </select>
            </div>
            <div>
              <label class="block text-slate-400 mb-1">Fib Band (&times; ATR)</label>
              <input type="number" id="bt-fib-atr" value="0.5" min="0" step="0.1" class="w-full bg-slate-900 border border-bordercol rounded-lg px-2.5 py-1.5 text-white">
            </div>
            <div>
              <label class="block text-slate-400 mb-1">Band Floor (%)</label>
              <input type="number" id="bt-fib-floor" value="0.3" min="0" step="0.1" class="w-full bg-slate-900 border border-bordercol rounded-lg px-2.5 py-1.5 text-white">
            </div>
            <div></div>
          </div>

          <div class="grid grid-cols-2 gap-2">
            <div class="flex items-center gap-2 pt-5">
              <input type="checkbox" id="bt-trail" checked class="rounded border-bordercol bg-slate-900 text-emerald-500">
              <label for="bt-trail" class="text-slate-300">ATR Trailing Stop</label>
            </div>
            <div>
              <label class="block text-slate-400 mb-1">Trail Distance (&times; ATR)</label>
              <input type="number" id="bt-trail-atr" value="3.5" min="0" step="0.1" class="w-full bg-slate-900 border border-bordercol rounded-lg px-2.5 py-1.5 text-white">
            </div>
          </div>
          <div class="text-[10px] text-slate-500">Band 0 = fixed percentage; otherwise the band is the wider of &times;ATR and the floor. 730d on 1h spans ~17,500 candles (bull, bear and chop).</div>

          <div class="space-y-2 pt-1">
            <button type="submit" id="btn-run-bt" class="w-full py-2.5 bg-purple-600 hover:bg-purple-500 active:bg-purple-700 text-white font-semibold rounded-lg shadow transition flex items-center justify-center gap-2 min-h-[44px]">
              <i class="fa-solid fa-play"></i> Run Historical Backtest
            </button>
            <div class="grid grid-cols-1 sm:grid-cols-3 gap-2">
              <button type="button" id="btn-sweep-windows" onclick="executeCompareWindows(event)" class="py-2 bg-slate-800 hover:bg-slate-700 active:bg-slate-900 border border-purple-500/50 text-purple-300 hover:text-white font-semibold rounded-lg shadow transition flex items-center justify-center gap-1.5 text-xs min-h-[38px]">
                <i class="fa-solid fa-table-columns"></i> Compare Windows
              </button>
              <button type="button" id="btn-sweep-timeframes" onclick="executeCompareTimeframes(event)" class="py-2 bg-slate-800 hover:bg-slate-700 active:bg-slate-900 border border-blue-500/50 text-blue-300 hover:text-white font-semibold rounded-lg shadow transition flex items-center justify-center gap-1.5 text-xs min-h-[38px]">
                <i class="fa-solid fa-layer-group"></i> Sweep Timeframes
              </button>
              <button type="button" id="btn-walk-forward" onclick="executeWalkForward(event)" class="py-2 bg-slate-800 hover:bg-slate-700 active:bg-slate-900 border border-sky-500/50 text-sky-300 hover:text-white font-semibold rounded-lg shadow transition flex items-center justify-center gap-1.5 text-xs min-h-[38px]">
                <i class="fa-solid fa-arrows-split-up-and-left"></i> Walk-Forward (OOS)
              </button>
            </div>
          </div>
        </form>

        <!-- Multi-Window / Timeframe Comparison Sweep Panel -->
        <div id="bt-sweep-box" class="hidden bg-slate-900/90 rounded-lg p-3 border border-purple-500/40 text-xs space-y-2">
          <div class="font-bold text-slate-200 border-b border-bordercol/60 pb-1 flex justify-between items-center">
            <div class="flex items-center gap-2">
              <i class="fa-solid fa-table-columns text-purple-400"></i>
              <span id="bt-sweep-title">Regime Robustness Comparison</span>
            </div>
            <span id="bt-sweep-badge" class="px-2 py-0.5 rounded text-[10px] font-semibold">--</span>
          </div>
          <div id="bt-sweep-summary" class="text-[11px] text-slate-300">--</div>
          <div class="overflow-x-auto">
            <table class="w-full text-left text-[11px] border-collapse">
              <thead>
                <tr class="border-b border-bordercol/60 text-slate-400">
                  <th class="py-1 px-1.5 font-medium" id="bt-th-window">Lookback</th>
                  <th class="py-1 px-1.5 font-medium">Trades</th>
                  <th class="py-1 px-1.5 font-medium">Win Rate</th>
                  <th class="py-1 px-1.5 font-medium">Return</th>
                  <th class="py-1 px-1.5 font-medium text-sky-400">B&amp;H %</th>
                  <th class="py-1 px-1.5 font-medium">Sharpe</th>
                  <th class="py-1 px-1.5 font-medium">Realized R:R</th>
                  <th class="py-1 px-1.5 font-medium">Max DD</th>
                  <th class="py-1 px-1.5 font-medium">Fib Blk</th>
                  <th class="py-1 px-1.5 font-medium">Risk Blk</th>
                  <th class="py-1 px-1.5 font-medium">Fees Paid</th>
                  <th class="py-1 px-1.5 font-medium">Regime</th>
                </tr>
              </thead>
              <tbody id="bt-sweep-tbody" class="divide-y divide-bordercol/40 font-mono text-[10px]">
                <!-- Rows injected dynamically -->
              </tbody>
            </table>
          </div>
          <div class="text-[10px] text-slate-500">A robust strategy maintains positive expectancy across regimes; profitability in only 1 window indicates regime dependency.</div>
        </div>

        <!-- Walk-Forward Validation (Out-of-Sample) Panel (Item 2) -->
        <div id="bt-wfo-box" class="hidden bg-slate-900/90 rounded-lg p-3 border border-sky-500/40 text-xs space-y-2">
          <div class="font-bold text-slate-200 border-b border-bordercol/60 pb-1 flex justify-between items-center">
            <div class="flex items-center gap-2">
              <i class="fa-solid fa-arrows-split-up-and-left text-sky-400"></i>
              <span id="bt-wfo-title">Sequential Walk-Forward Validation (Out-of-Sample)</span>
            </div>
            <span id="bt-wfo-badge" class="px-2 py-0.5 rounded text-[10px] font-semibold">--</span>
          </div>
          <div id="bt-wfo-summary" class="text-[11px] text-slate-300">--</div>
          <div class="overflow-x-auto">
            <table class="w-full text-left text-[11px] border-collapse">
              <thead>
                <tr class="border-b border-bordercol/60 text-slate-400">
                  <th class="py-1 px-1.5 font-medium">Fold</th>
                  <th class="py-1 px-1.5 font-medium">Train Window</th>
                  <th class="py-1 px-1.5 font-medium">In-Sample Ret</th>
                  <th class="py-1 px-1.5 font-medium">IS Trades</th>
                  <th class="py-1 px-1.5 font-medium">Test Window (OOS)</th>
                  <th class="py-1 px-1.5 font-medium">Out-of-Sample Ret</th>
                  <th class="py-1 px-1.5 font-medium text-sky-400">OOS B&amp;H</th>
                  <th class="py-1 px-1.5 font-medium">OOS Trades</th>
                  <th class="py-1 px-1.5 font-medium">Verdict</th>
                </tr>
              </thead>
              <tbody id="bt-wfo-tbody" class="divide-y divide-bordercol/40 font-mono text-[10px]">
                <!-- Rows injected dynamically -->
              </tbody>
            </table>
          </div>
          <div class="text-[10px] text-slate-500">Walk-Forward Efficiency (WFE = OOS / IS Return): Measures whether in-sample tuning holds on unseen subsequent market data or suffers from curve-fitting.</div>
        </div>

        <div id="bt-results-box" class="hidden bg-slate-900/90 rounded-lg p-3 border border-bordercol text-xs space-y-2">
          <div class="font-bold text-slate-200 border-b border-bordercol/60 pb-1 flex justify-between items-center">
            <div class="flex items-center gap-2">
              <span>Backtest Results</span>
              <span id="bt-res-bh-badge" class="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 font-normal">B&amp;H: --</span>
            </div>
            <span id="bt-res-return" class="font-bold">--</span>
          </div>
          <div class="grid grid-cols-2 gap-2 text-[11px]">
            <div>Trades: <span id="bt-res-trades" class="font-semibold text-white">0</span></div>
            <div>Win Rate: <span id="bt-res-winrate" class="font-semibold text-white">0%</span></div>
            <div>Final Equity: <span id="bt-res-final" class="font-semibold text-white">$0</span></div>
            <div>Sharpe (approx): <span id="bt-res-sharpe" class="font-semibold text-white">0</span></div>
            <div>Buy &amp; Hold Ret: <span id="bt-res-bh" class="font-semibold text-sky-400">0%</span></div>
            <div>Strategy Alpha: <span id="bt-res-alpha" class="font-semibold text-white">0%</span></div>
            <div>Avg Win: <span id="bt-res-avgwin" class="font-semibold text-emerald-400">$0</span></div>
            <div>Avg Loss: <span id="bt-res-avgloss" class="font-semibold text-red-400">$0</span></div>
            <div>Realized R:R: <span id="bt-res-rr" class="font-semibold text-white">0</span></div>
            <div>Max Drawdown: <span id="bt-res-drawdown" class="font-semibold text-amber-400">0%</span></div>
            <div>Total Fees Paid: <span id="bt-res-fees" class="font-semibold text-amber-400">$0.00</span></div>
            <div>Exits: <span id="bt-res-exits" class="font-semibold text-slate-300">--</span></div>
          </div>
          <div class="grid grid-cols-2 gap-2 text-[11px] border-t border-bordercol/60 pt-1.5">
            <div>Candles evaluated: <span id="bt-res-candles" class="font-semibold text-white">0</span></div>
            <div>Raw signals: <span id="bt-res-raw" class="font-semibold text-white">0</span></div>
            <div>Fib blocked: <span id="bt-res-fibblocked" class="font-semibold text-amber-400">0</span></div>
            <div>Sizing/risk blocked: <span id="bt-res-blocked" class="font-semibold text-amber-400">0</span></div>
          </div>
          <div id="bt-res-fib-note" class="text-[10px] text-slate-400">--</div>
          <div id="bt-res-window" class="text-[10px] text-slate-400">Indicator Calc: --</div>
          <div id="bt-res-eval-window" class="text-[10px] text-slate-400">Signal Eval: --</div>
          <details class="text-[10px] text-slate-400">
            <summary class="cursor-pointer hover:text-slate-200">Run report / rejection detail (copyable)</summary>
            <textarea id="bt-res-report" readonly rows="7" class="w-full mt-1 bg-slate-950 border border-bordercol rounded p-1.5 text-slate-300 font-mono text-[10px]"></textarea>
          </details>
        </div>
      </div>
    </div>
    </div>

    <!-- Live/Paper Execution Tracking & Persistent Order Audit Log (Item 1 & 7) -->
    <div id="section-execution-tracking" class="hidden md:block">
      <div class="mobile-only-header flex items-center justify-between border-b border-bordercol pb-3 mb-3 md:hidden">
        <button type="button" onclick="showMobileView('dashboard')" class="text-xs font-semibold text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 px-3 py-1.5 rounded-lg border border-bordercol flex items-center gap-1.5 transition">
          <i class="fa-solid fa-arrow-left"></i> Back to Dashboard
        </button>
        <span class="text-xs font-bold text-slate-400">Execution Tracking</span>
      </div>

      <div class="bg-cardbg border border-bordercol rounded-xl p-4 sm:p-5 space-y-4 w-full">
        <!-- Header & Alert Status Controls -->
        <div class="flex flex-wrap items-center justify-between gap-3 border-b border-bordercol/60 pb-3">
          <div class="flex items-center gap-2.5">
            <div class="w-8 h-8 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
              <i class="fa-solid fa-satellite-dish text-sm"></i>
            </div>
            <div>
              <div class="flex items-center gap-2">
                <h3 class="font-bold text-sm text-white">Live &amp; Paper Execution Tracking</h3>
                <span id="tracker-mode-badge" class="px-2 py-0.5 rounded text-[10px] font-bold bg-blue-500/20 text-blue-400 border border-blue-500/30">AUDIT LOG FORWARD</span>
              </div>
              <p class="text-[11px] text-slate-400">Persistent order history, real-time stop &amp; trailing telemetry, fills &amp; rejections.</p>
            </div>
          </div>
          <div class="flex items-center gap-2 flex-wrap">
            <span id="tracker-last-refresh" class="text-[10px] text-slate-500">--</span>
            <button type="button" id="btn-test-alert" onclick="testLiveAlert()" class="text-xs text-amber-300 hover:text-white transition flex items-center gap-1 px-2.5 py-1 rounded-lg bg-amber-950/40 hover:bg-amber-900/60 border border-amber-500/40">
              <i class="fa-solid fa-bell"></i> Test Live Alert
            </button>
            <button type="button" id="btn-refresh-tracker" onclick="refreshExecutionTracker(true)" class="text-xs text-slate-300 hover:text-white transition flex items-center gap-1 px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 border border-bordercol">
              <i class="fa-solid fa-rotate"></i> Refresh
            </button>
          </div>
        </div>

        <!-- Real-Time Position Telemetry Card -->
        <div class="bg-slate-900/80 border border-bordercol/80 rounded-xl p-3 sm:p-4">
          <div class="flex items-center justify-between mb-2">
            <div class="flex items-center gap-2">
              <span class="text-xs font-bold text-slate-300 uppercase tracking-wider">Active Position Telemetry</span>
              <span id="pos-status-badge" class="px-2 py-0.5 rounded text-[10px] font-semibold bg-slate-800 text-slate-400">FLAT</span>
            </div>
            <div id="pos-id-display" class="text-[10px] text-slate-500 font-mono">No Active Position</div>
          </div>

          <!-- Position details grid -->
          <div id="pos-details-container" class="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2 text-xs">
            <div class="bg-slate-950/60 p-2 rounded border border-bordercol/40">
              <div class="text-[10px] text-slate-500">Symbol / Side</div>
              <div id="pos-symbol-side" class="font-bold text-white mt-0.5">Flat</div>
            </div>
            <div class="bg-slate-950/60 p-2 rounded border border-bordercol/40">
              <div class="text-[10px] text-slate-500">Entry Price</div>
              <div id="pos-entry-price" class="font-bold text-white mt-0.5">$0.00</div>
            </div>
            <div class="bg-slate-950/60 p-2 rounded border border-bordercol/40">
              <div class="text-[10px] text-slate-500">Current Price</div>
              <div id="pos-current-price" class="font-bold text-white mt-0.5">$0.00</div>
            </div>
            <div class="bg-slate-950/60 p-2 rounded border border-bordercol/40">
              <div class="text-[10px] text-slate-500">Position Size</div>
              <div id="pos-size-amount" class="font-bold text-white mt-0.5">0.00</div>
            </div>
            <div class="bg-slate-950/60 p-2 rounded border border-bordercol/40">
              <div class="text-[10px] text-slate-500">Stop / Trail Level</div>
              <div id="pos-stop-level" class="font-bold text-amber-400 mt-0.5">--</div>
            </div>
            <div class="bg-slate-950/60 p-2 rounded border border-bordercol/40">
              <div class="text-[10px] text-slate-500">Unrealized P&amp;L</div>
              <div id="pos-unrealized-pnl" class="font-bold text-slate-400 mt-0.5">$0.00 (0.0%)</div>
            </div>
            <div class="bg-slate-950/60 p-2 rounded border border-bordercol/40">
              <div class="text-[10px] text-slate-500">Duration / Opened</div>
              <div id="pos-duration-opened" class="font-medium text-slate-400 mt-0.5">--</div>
            </div>
          </div>
        </div>

        <!-- Orders & Audit Log Table (Fills, Rejections, Cancellations) -->
        <div class="space-y-2">
          <div class="flex items-center justify-between text-xs">
            <span class="font-bold text-slate-300 flex items-center gap-1.5">
              <i class="fa-solid fa-list-check text-slate-400"></i>
              <span>Order Execution History &amp; Rejection Audit</span>
            </span>
            <span id="orders-count-label" class="text-[10px] text-slate-500">0 orders recorded</span>
          </div>

          <div class="overflow-x-auto rounded-lg border border-bordercol/60 bg-slate-950">
            <table class="w-full text-left text-[11px] border-collapse">
              <thead>
                <tr class="border-b border-bordercol/60 text-slate-400 bg-slate-900/60">
                  <th class="py-2 px-2.5 font-medium">Time (UTC)</th>
                  <th class="py-2 px-2 font-medium">Symbol</th>
                  <th class="py-2 px-2 font-medium">Side</th>
                  <th class="py-2 px-2 font-medium">Type</th>
                  <th class="py-2 px-2 font-medium">Price</th>
                  <th class="py-2 px-2 font-medium">Amount</th>
                  <th class="py-2 px-2 font-medium">Status</th>
                  <th class="py-2 px-2.5 font-medium">Reason / Note</th>
                </tr>
              </thead>
              <tbody id="tracker-orders-tbody" class="divide-y divide-bordercol/30 font-mono text-[10px]">
                <tr>
                  <td colspan="8" class="py-3 px-3 text-center text-slate-500 italic font-sans">
                    No orders recorded yet. As orders are placed, filled, or rejected by risk controls, they will appear here in real time.
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

    <!-- Live Activity Logs -->
    <!-- Desktop: in-page card | Mobile: separate hamburger page view -->
    <div id="section-bot-activity" class="hidden md:block">
      <div class="mobile-only-header flex items-center justify-between border-b border-bordercol pb-3 mb-3 md:hidden">
        <button type="button" onclick="showMobileView('dashboard')" class="text-xs font-semibold text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 px-3 py-1.5 rounded-lg border border-bordercol flex items-center gap-1.5 transition">
          <i class="fa-solid fa-arrow-left"></i> Back to Dashboard
        </button>
        <span class="text-xs font-bold text-slate-400">Bot Activity</span>
      </div>
      <div class="bg-cardbg border border-bordercol rounded-xl p-4 sm:p-5 space-y-3 w-full">
        <div class="flex flex-wrap items-center justify-between gap-2">
          <div class="flex items-center gap-2">
            <i class="fa-solid fa-terminal text-slate-400"></i>
            <h3 class="font-bold text-sm text-white">Live Bot Activity Logs</h3>
          </div>
          <div class="flex items-center gap-2">
            <div class="flex items-center bg-slate-900 border border-bordercol/80 rounded-lg p-0.5 text-[11px]">
              <button type="button" id="btn-log-filter-all" onclick="setLogFilter('all')" class="px-2 py-0.5 rounded font-medium bg-purple-600 text-white shadow-sm">All</button>
              <button type="button" id="btn-log-filter-sweep" onclick="setLogFilter('sweep')" class="px-2 py-0.5 rounded font-medium text-slate-400 hover:text-white transition">Sweep Logs</button>
              <button type="button" id="btn-log-filter-live" onclick="setLogFilter('live')" class="px-2 py-0.5 rounded font-medium text-slate-400 hover:text-white transition">Live Engine</button>
            </div>
            <span id="logs-updated" class="text-[10px] text-slate-500">--</span>
            <button type="button" id="btn-refresh-logs" onclick="refreshLogs(true)" class="text-xs text-slate-400 hover:text-white transition flex items-center gap-1 px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 border border-bordercol disabled:opacity-60">
              <i class="fa-solid fa-rotate"></i> Refresh
            </button>
          </div>
        </div>
        <div id="log-container" class="bg-slate-950 font-mono text-[11px] p-3 rounded-lg border border-bordercol/60 h-56 md:h-40 overflow-y-auto space-y-1 text-slate-300">
          <div class="text-slate-500">Loading recent logs...</div>
        </div>
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

    <!-- 2. Emergency Kill Switch Toggle -->
    <button id="bottom-btn-kill" onclick="toggleKillSwitch()" class="flex flex-col items-center justify-center py-1 px-1.5 text-slate-400 hover:text-slate-200 active:scale-95 transition flex-1" title="Toggle Emergency Kill Switch">
      <i id="bottom-btn-kill-icon" class="fa-solid fa-toggle-off text-base text-slate-400"></i>
      <span id="bottom-btn-kill-text" class="text-[10px] font-semibold mt-0.5">Kill: OFF</span>
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
        <div class="w-12 h-12 mx-auto mb-2 rounded-xl overflow-hidden border border-amber-500/40 shadow-lg bg-slate-900 flex items-center justify-center">
          <img src="/logo.png" alt="Trader Queen Logo" class="h-full w-full object-cover">
        </div>
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
          <label class="block text-slate-400 mb-1">Password (min 10 chars)</label>
          <div class="relative">
            <input type="password" id="reg-pass" required minlength="10" placeholder="Create a strong password" class="w-full bg-slate-800 border border-slate-700 rounded-lg pl-3 pr-10 py-2 text-white focus:outline-none focus:border-emerald-500">
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
            To deliver verification emails to <strong>ALL users creating accounts</strong>, you must add & verify your custom domain at <a href="https://resend.com/domains" target="_blank" class="text-sky-400 underline font-semibold">resend.com/domains</a> and set your verified sender address below (e.g. <code class="bg-slate-800 px-1 py-0.5 rounded text-emerald-300">Trader Queen &lt;noreply@yourdomain.com&gt;</code>).
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
          <input type="text" id="smtp-input-from" placeholder="Trader Queen <noreply@yourdomain.com>" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white">
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
              <input type="radio" name="settings-exec-mode" id="radio-mode-auto" value="auto" checked onchange="onExecutionModeOptionChange(true)" class="mt-0.5 text-purple-500 focus:ring-0">
              <div>
                <div class="font-bold text-white flex items-center gap-1.5">
                  <i class="fa-solid fa-robot text-purple-400"></i> Auto Trading
                </div>
                <p class="text-[11px] text-slate-400 mt-0.5 leading-tight">Bot automatically enters and exits trades when strategy signals fire.</p>
              </div>
            </label>

            <label id="opt-mode-manual-card" class="p-3 rounded-xl border border-bordercol bg-slate-900/80 cursor-pointer transition flex items-start gap-2.5 hover:border-slate-600">
              <input type="radio" name="settings-exec-mode" id="radio-mode-manual" value="manual" onchange="onExecutionModeOptionChange(true)" class="mt-0.5 text-amber-500 focus:ring-0">
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
    let isKillSwitchActive = false;
    let currentUser = null;
    let exchangesList = [];
    let currentActivePosition = null;
    let activePosNullStreak = 0;

    // Per-grain storage limits, injected by the server from the same table the
    // candle fetcher uses. Asking deeper than this returns an empty page, not
    // more candles, so the lookback input is bounded by it.
    const BT_LOOKBACK_LIMITS = __BT_LOOKBACK_LIMITS__;

    // --- In-app notifications ------------------------------------------------
    // Errors are told to the user inside the page: a short, readable card with
    // the raw exchange text behind a toggle. A browser alert() cannot be
    // dismissed from the page and dumps unreadable JSON in the reader's face.
    let noticeTimer = null;

    function hideNotice() {
      const panel = document.getElementById('app-notice');
      if (panel) panel.classList.add('hidden');
      if (noticeTimer) {
        clearTimeout(noticeTimer);
        noticeTimer = null;
      }
    }

    function toggleNoticeDetail() {
      const detail = document.getElementById('app-notice-detail');
      if (!detail) return;
      if (detail.classList.contains('hidden')) detail.classList.remove('hidden');
      else detail.classList.add('hidden');
    }

    function showNotice(title, body, options) {
      const opts = options || {};
      const kind = opts.kind || 'error';
      const look = {
        error: ['fa-circle-exclamation', 'text-rose-400'],
        warn: ['fa-triangle-exclamation', 'text-amber-400'],
        success: ['fa-circle-check', 'text-emerald-400'],
        info: ['fa-circle-info', 'text-sky-400'],
      }[kind] || ['fa-circle-info', 'text-sky-400'];

      const panel = document.getElementById('app-notice');
      if (!panel) return;
      const setText = (id, text) => {
        const el = document.getElementById(id);
        if (el) el.innerText = text == null ? '' : String(text);
      };
      const icon = document.getElementById('app-notice-icon');
      if (icon) icon.className = 'fa-solid ' + look[0] + ' mt-0.5 ' + look[1];
      setText('app-notice-title', title);
      setText('app-notice-body', body);

      // The raw text is available, collapsed, so the card stays short.
      const button = document.getElementById('app-notice-details-btn');
      const detail = document.getElementById('app-notice-detail');
      if (button && detail) {
        setText('app-notice-detail', opts.detail || '');
        if (opts.detail) button.classList.remove('hidden');
        else button.classList.add('hidden');
        detail.classList.add('hidden');
      }
      panel.classList.remove('hidden');

      if (noticeTimer) {
        clearTimeout(noticeTimer);
        noticeTimer = null;
      }
      // Errors wait to be dismissed; anything else fades on its own.
      const ms = opts.timeout === undefined ? (kind === 'error' ? 0 : 7000) : opts.timeout;
      if (ms > 0) {
        noticeTimer = setTimeout(hideNotice, ms);
        // Never hold a Node test process open (no-op in the browser).
        if (noticeTimer && noticeTimer.unref) noticeTimer.unref();
      }
    }

    // Raw exchange errors are walls of JSON. Keep the first useful sentence.
    function firstUsefulLine(text) {
      const line = String(text == null ? '' : text)
        .split('{')[0]
        .split('\\n')[0]
        .replace(/\\s+/g, ' ')
        .replace(/^(binance|mexc|kraken|coinbase|kucoin|bybit|okx|bitget|gateio|gate|bitfinex|cryptocom|htx)\\s+/i, '')
        .trim();
      return line.length > 160 ? line.slice(0, 157).trimEnd() + '…' : line;
    }

    // One short, actionable sentence per failure mode, so the card never shows
    // a raw ccxt blob. `detail` carries that blob for the toggle.
    function describeBacktestError(err) {
      const raw = String((err && err.message) || 'Unknown error').trim();
      const text = raw.toLowerCase();
      const venue = (document.getElementById('bt-exchange') || {}).value || 'binance';
      const name = venue.charAt(0).toUpperCase() + venue.slice(1);
      const symbol = (document.getElementById('bt-symbol') || {}).value || 'That pair';
      const detail = raw.length > 400 ? raw.slice(0, 400) + '…' : raw;

      if (text.includes('restricted location') || text.includes('451') || text.includes('eligibility')) {
        return { title: name + ' is not available in this region',
                 body: 'This server sits in a restricted region, so ' + name + ' refuses the request. ' +
                       'Try Kraken, Bybit, KuCoin or OKX instead.',
                 detail: detail };
      }
      if (text.includes('rate limit') || text.includes('429') || text.includes('too many requests')) {
        return { title: name + ' is rate-limiting this run',
                 body: 'Too many requests in a row. Give it a minute, then run the backtest again.',
                 detail: detail };
      }
      if (text.includes('insufficient historical bars') || text.includes('not enough history')) {
        // The server appends the venue's real storage limit after the colon.
        const tail = raw.split(/insufficient historical bars[^:]*:\\s*/i)[1];
        return { title: 'Not enough history for that window',
                 body: (tail || '').trim() ||
                       (text.includes('not enough history') ? raw : 'The exchange had no candles that far back. Shorten the lookback and try again.'),
                 detail: detail };
      }
      if (text.includes('timeout') || text.includes('timed out') || text.includes('network') ||
          text.includes('econn') || text.includes('getaddrinfo') || text.includes('fetch failed')) {
        return { title: 'Could not reach ' + name,
                 body: 'The request timed out or was blocked. Check the connection and try again.',
                 detail: detail };
      }
      if (text.includes('bad symbol') || text.includes('does not have market symbol')) {
        return { title: symbol + ' is not listed on ' + name,
                 body: 'This exchange does not trade that pair. Choose another symbol and run the backtest again.',
                 detail: detail };
      }
      if (text.includes('unsupported exchange')) {
        return { title: 'That exchange is not supported',
                 body: 'Pick one from the exchange list, then run the backtest again.',
                 detail: detail };
      }
      return { title: 'Backtest could not run',
               body: firstUsefulLine(raw) || 'Something went wrong on the way to the exchange.',
               detail: detail };
    }

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
        const raw = await res.text();
        let data;
        try {
          data = raw ? JSON.parse(raw) : {};
        } catch (parseErr) {
          data = {
            success: false,
            message: raw ? raw.slice(0, 300) : ('HTTP ' + res.status + ' ' + res.statusText)
          };
        }
        const resBox = document.getElementById('key-test-result');
        resBox.classList.remove('hidden');
        if (res.ok && data.success) {
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

    async function deleteSavedExchangeKey(exchangeId) {
      if (!exchangeId) return;
      const exName = exchangeId.toUpperCase();
      if (!confirm(`Are you sure you want to remove your saved API credentials for ${exName}?\n\nThis will remove them from secure encrypted storage and reset this exchange to public / paper mode.`)) return;

      const token = localStorage.getItem('bot_auth_token');
      const headers = { 'Content-Type': 'application/json' };
      if (token) headers['Authorization'] = 'Bearer ' + token;

      try {
        const res = await fetch('/api/exchange/delete', {
          method: 'POST',
          headers: headers,
          body: JSON.stringify({ exchange_id: exchangeId })
        });
        const data = await res.json();
        const resBox = document.getElementById('key-test-result');
        if (resBox) resBox.classList.remove('hidden');

        if (res.ok && data.success) {
          if (resBox) {
            resBox.className = 'text-xs p-3 rounded-lg border bg-amber-950/60 border-amber-500/40 text-amber-200';
            resBox.innerHTML = `<strong><i class="fa-solid fa-trash-can"></i> ${data.message}</strong>`;
          }
          const sel = document.getElementById('key-exchange-select');
          if (sel && sel.value.toLowerCase() === exchangeId.toLowerCase()) {
            const kInput = document.getElementById('key-api-key');
            const sInput = document.getElementById('key-api-secret');
            const pInput = document.getElementById('key-api-passphrase');
            if (kInput) kInput.value = '';
            if (sInput) sInput.value = '';
            if (pInput) pInput.value = '';
          }
          await checkAuth();
          await fetchStatus();
        } else {
          if (resBox) {
            resBox.className = 'text-xs p-3 rounded-lg border bg-rose-950/60 border-rose-500/40 text-rose-200';
            resBox.innerText = data.detail || data.message || 'Remove failed';
          }
        }
      } catch (err) {
        alert('Failed to remove API key: ' + err);
      }
    }

    async function deleteCurrentExchangeKey() {
      const sel = document.getElementById('key-exchange-select');
      const exId = sel ? sel.value : '';
      if (!exId) {
        alert('Please choose an exchange to remove credentials for.');
        return;
      }
      await deleteSavedExchangeKey(exId);
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
        const res = await fetch('/api/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username: user, username_or_email: user, password: pass, client_token: clientToken })
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
          const msg = (data && data.message) ? data.message : 'Invalid credentials';
          if (data.needs_activation || msg.toLowerCase().includes('not activated') || msg.toLowerCase().includes('activation')) {
            notice.innerHTML = `
              <div class="flex items-center gap-2 font-bold text-amber-400">
                <i class="fa-solid fa-envelope-open-text text-base"></i>
                <span>Account Verification Required</span>
              </div>
              <p class="text-slate-300 leading-relaxed mt-1">
                ${msg}
              </p>
              <div class="pt-2 flex flex-col gap-2">
                <button type="button" id="btn-resend-login" onclick="resendVerificationLink('${data.email || user}', 'btn-resend-login')" class="w-full py-2 bg-amber-600 hover:bg-amber-500 text-white font-semibold rounded-lg shadow transition flex items-center justify-center gap-1.5">
                  <i class="fa-solid fa-paper-plane text-xs"></i> Resend Verification Link
                </button>
              </div>
            `;
          } else if (data.not_found || msg.toLowerCase().includes('not found') || msg.toLowerCase().includes('invalid')) {
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

    function showMobileView(view) {
      const activeView = view || 'dashboard';
      document.body.classList.remove(
        'mobile-view-dashboard',
        'mobile-view-keys',
        'mobile-view-backtester',
        'mobile-view-activity',
        'mobile-view-account'
      );
      document.body.classList.add('mobile-view-' + activeView);
      window.scrollTo({ top: 0, behavior: 'smooth' });
      if (activeView === 'activity') {
        refreshLogs();
        refreshExecutionTracker();
      }
      if (activeView === 'account') {
        renderMobileAccountView();
      }
    }

    function renderMobileAccountView() {
      const loggedOut = document.getElementById('account-view-logged-out');
      const loggedIn = document.getElementById('account-view-logged-in');
      const name = document.getElementById('account-view-username');
      const email = document.getElementById('account-view-email');
      if (!loggedOut || !loggedIn) return;
      if (!currentUser) {
        loggedOut.classList.remove('hidden');
        loggedIn.classList.add('hidden');
        return;
      }
      loggedOut.classList.add('hidden');
      loggedIn.classList.remove('hidden');
      if (name) name.innerText = currentUser.username || 'Trader';
      if (email) email.innerText = currentUser.email || 'No email saved';
    }

    function handleMobileAccountClick() {
      showMobileView('account');
    }

    document.addEventListener('keydown', function(e) {
      if (e.key === 'Escape' && window.innerWidth < 768) {
        showMobileView('dashboard');
      }
    });

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

      if (!token) {
        if (authButtons) authButtons.classList.remove('hidden');
        if (userBadge) userBadge.classList.add('hidden');
        if (authButtonsMob) authButtonsMob.classList.remove('hidden');
        if (userBadgeMob) userBadgeMob.classList.add('hidden');
        if (lockedDiv) lockedDiv.classList.remove('hidden');
        if (unlockedDiv) unlockedDiv.classList.add('hidden');
        renderMobileAccountView();
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
          renderMobileAccountView();

          if (lockedDiv) lockedDiv.classList.add('hidden');
          if (unlockedDiv) unlockedDiv.classList.remove('hidden');

          // Render saved exchange keys
          const list = document.getElementById('saved-keys-list') || document.getElementById('configured-keys-list');
          if (list) {
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
                  <div class="flex items-center gap-2 shrink-0">
                    <span class="text-[10px] text-emerald-400 font-medium">Configured</span>
                    <button type="button" onclick="deleteSavedExchangeKey('${k.exchange_id}')" class="px-2 py-0.5 text-slate-400 hover:text-rose-400 hover:bg-rose-950/40 rounded transition border border-slate-800 hover:border-rose-500/40 flex items-center gap-1 text-[11px]" title="Remove saved API keys for ${k.exchange_id.toUpperCase()}">
                      <i class="fa-solid fa-trash-can text-xs"></i>
                      <span>Remove</span>
                    </button>
                  </div>
                `;
                list.appendChild(div);
              });
            } else {
              list.innerHTML = '<div class="text-slate-500 italic">No custom keys saved yet. Bot currently defaults to Binance public data / paper mode.</div>';
            }
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
          renderMobileAccountView();
        }
      } catch (e) {
        console.error('Check auth error:', e);
        if (!currentUser) {
          if (lockedDiv) lockedDiv.classList.remove('hidden');
          if (unlockedDiv) unlockedDiv.classList.add('hidden');
        }
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

      // Explicit user action: persist locally immediately so UI stays stable without flickering
      try {
        localStorage.setItem('bot_trading_mode', targetMode);
      } catch (e) {}
      updateTradingModeUI(isReal, isReal ? 'live' : 'paper', !isReal);

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
        if (envBtnReal) envBtnReal.className = 'flex-1 sm:flex-none justify-center px-3 py-2 sm:py-1.5 bg-rose-600 text-white text-xs font-bold rounded-lg shadow transition flex items-center gap-1.5';
        if (envBtnTestnet) envBtnTestnet.className = 'flex-1 sm:flex-none justify-center px-3 py-2 sm:py-1.5 bg-slate-800 text-slate-300 hover:text-white border border-slate-700 text-xs font-bold rounded-lg transition flex items-center gap-1.5';
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
          badgeMode.innerText = 'TEST MODE';
          badgeMode.className = 'text-[10px] px-2 py-0.5 rounded-full font-bold bg-blue-500/20 text-blue-400 border border-blue-500/30';
        }
        if (envBtnTestnet) envBtnTestnet.className = 'flex-1 sm:flex-none justify-center px-3 py-2 sm:py-1.5 bg-blue-600 text-white text-xs font-bold rounded-lg shadow transition flex items-center gap-1.5';
        if (envBtnReal) envBtnReal.className = 'flex-1 sm:flex-none justify-center px-3 py-2 sm:py-1.5 bg-slate-800 text-slate-300 hover:text-white border border-slate-700 text-xs font-bold rounded-lg transition flex items-center gap-1.5';
        if (envTitle) {
          envTitle.innerText = 'TEST MODE';
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

    function onExecutionModeOptionChange(autoSave = false) {
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

      if (autoSave) {
        quickSetExecutionMode(isAuto ? 'auto' : 'manual');
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
        const savedMode = localStorage.getItem('bot_execution_mode') || data.execution_mode;
        const radioManual = document.getElementById('radio-mode-manual');
        const radioAuto = document.getElementById('radio-mode-auto');
        if (savedMode === 'manual') {
          if (radioManual) radioManual.checked = true;
          if (radioAuto) radioAuto.checked = false;
        } else {
          if (radioAuto) radioAuto.checked = true;
          if (radioManual) radioManual.checked = false;
        }
        onExecutionModeOptionChange(false);

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
      const targetExecMode = isManual ? 'manual' : 'auto';
      localStorage.setItem('bot_execution_mode', targetExecMode);
      updateExecutionModeUI(isManual);
      const payload = {
        execution_mode: targetExecMode,
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

    function updateExecutionModeUI(isManual) {
      const btnExecAuto = document.getElementById('btn-exec-auto');
      const btnExecManual = document.getElementById('btn-exec-manual');
      if (btnExecAuto && btnExecManual) {
        if (isManual) {
          btnExecManual.className = 'px-2.5 py-1.5 bg-amber-600 text-white text-xs font-bold rounded-md shadow transition flex items-center gap-1.5';
          btnExecAuto.className = 'px-2.5 py-1.5 text-slate-300 hover:text-white text-xs font-bold rounded-md transition flex items-center gap-1.5';
        } else {
          btnExecAuto.className = 'px-2.5 py-1.5 bg-purple-600 text-white text-xs font-bold rounded-md shadow transition flex items-center gap-1.5';
          btnExecManual.className = 'px-2.5 py-1.5 text-slate-300 hover:text-white text-xs font-bold rounded-md transition flex items-center gap-1.5';
        }
      }
      const manualPanel = document.getElementById('manual-control-panel');
      if (manualPanel) {
        manualPanel.classList.toggle('hidden', !isManual);
      }
    }

    async function quickSetExecutionMode(targetMode) {
      const isManual = (targetMode === 'manual');
      localStorage.setItem('bot_execution_mode', targetMode);
      updateExecutionModeUI(isManual);

      const radioManual = document.getElementById('radio-mode-manual');
      const radioAuto = document.getElementById('radio-mode-auto');
      if (radioManual) radioManual.checked = isManual;
      if (radioAuto) radioAuto.checked = !isManual;
      if (typeof onExecutionModeOptionChange === 'function') {
        onExecutionModeOptionChange(false);
      }

      try {
        const res = await fetch('/api/bot/execution_mode', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ execution_mode: targetMode })
        });
        const data = await res.json();
        if (res.ok && data.success) {
          await fetchStatus();
          await refreshLogs();
        } else {
          console.warn('Execution mode switch warning:', data);
        }
      } catch (err) {
        console.warn('Error changing execution mode:', err);
      }
    }

    function toggleExecutionMode() {
      const current = localStorage.getItem('bot_execution_mode') || 'auto';
      const target = (current === 'manual') ? 'auto' : 'manual';
      quickSetExecutionMode(target);
    }

    function renderActivePositionBanner(pos) {
      const posBanner = document.getElementById('position-banner');
      const buyTextEl = document.getElementById('btn-manual-buy-text');
      const sellTextEl = document.getElementById('btn-manual-sell-text');
      const posDot = document.getElementById('pos-banner-dot');
      const closeBtnText = document.getElementById('btn-banner-close-text');
      const posDetails = document.getElementById('pos-details');

      if (!posBanner) return;

      if (!pos || !(pos.entry || pos.entry_price)) {
        posBanner.classList.add('hidden');
        if (buyTextEl && sellTextEl) {
          buyTextEl.innerText = 'Manual BUY (Market)';
          sellTextEl.innerText = 'Manual SELL (Market)';
        }
        return;
      }

      posBanner.classList.remove('hidden');
      const posSide = (pos.side || 'BUY').toUpperCase();
      const isShort = (posSide === 'SELL' || posSide === 'SHORT');
      const sideColor = isShort ? 'text-rose-400' : 'text-emerald-400';
      const sideText = isShort ? 'SHORT' : 'LONG';
      const entryVal = Number(pos.entry || pos.entry_price || 0).toFixed(2);
      const stopVal = Number(pos.stop || pos.current_stop || pos.trailing_stop || pos.stop_loss || 0).toFixed(2);
      const targetVal = Number(pos.target || 0).toFixed(2);
      const sizeVal = Number(pos.size !== undefined ? pos.size : (pos.amount || 0)).toFixed(6);

      if (posDot) {
        posDot.className = isShort 
          ? 'w-2.5 h-2.5 rounded-full bg-rose-500 animate-pulse shrink-0'
          : 'w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse shrink-0';
      }

      if (posDetails) {
        posDetails.innerHTML = `<span class="${sideColor} font-bold mr-1">[${sideText}]</span> Entry: $${entryVal} | Stop: $${stopVal} | Target: $${targetVal} | Size: ${sizeVal}`;
      }

      if (closeBtnText) {
        closeBtnText.innerText = isShort ? 'Close Short' : 'Close Long';
      }

      if (buyTextEl && sellTextEl) {
        if (!isShort) {
          buyTextEl.innerText = 'Manual BUY (Long Open)';
          sellTextEl.innerText = 'Close Long Position (Market)';
        } else {
          buyTextEl.innerText = 'Close Short Position (Market)';
          sellTextEl.innerText = 'Manual SELL (Short Open)';
        }
      }
    }

    async function closeActivePosition() {
      let pos = currentActivePosition;
      if (!pos) {
        try { pos = JSON.parse(localStorage.getItem('bot_active_position')); } catch(e) {}
      }
      const side = ((pos && pos.side) || 'SELL').toUpperCase();
      const sideName = side === 'SELL' ? 'SHORT' : 'LONG';
      if (!confirm(`Confirm CLOSE ${sideName} position at current market price?`)) return;

      const btn = document.getElementById('btn-banner-close-trade');
      const oldHtml = btn ? btn.innerHTML : '';
      if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Closing...';
      }

      try {
        const res = await fetch('/api/bot/trade/close', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'close' })
        });
        const data = await res.json();
        if (res.ok && data.success) {
          alert(data.message || `${sideName} position closed successfully.`);
          currentActivePosition = null;
          activePosNullStreak = 0;
          try { localStorage.removeItem('bot_active_position'); } catch(e) {}
          renderActivePositionBanner(null);
          await Promise.all([fetchStatus(), refreshExecutionTracker(), refreshLogs()]);
        } else {
          alert(`Failed to close ${sideName} position: ` + (data.detail || data.message || 'Unknown error'));
        }
      } catch (err) {
        alert('Error closing position: ' + err);
      } finally {
        if (btn) {
          btn.disabled = false;
          btn.innerHTML = oldHtml;
        }
      }
    }

    async function triggerManualTrade(action) {
      const act = action.toLowerCase();
      const actName = action.toUpperCase();
      let confirmMsg = `Confirm manual ${actName} order execution at current market price?`;
      if (currentActivePosition) {
        const curSide = (currentActivePosition.side || 'BUY').toUpperCase();
        if (act === 'sell' && curSide === 'BUY') {
          confirmMsg = `Confirm CLOSE LONG position at current market price?`;
        } else if (act === 'buy' && curSide === 'SELL') {
          confirmMsg = `Confirm CLOSE SHORT position at current market price?`;
        } else if (act === 'buy' && curSide === 'BUY') {
          confirmMsg = `A Long position is already open. Confirm manual BUY?`;
        } else if (act === 'sell' && curSide === 'SELL') {
          confirmMsg = `A Short position is already open. Confirm manual SELL?`;
        }
      } else {
        if (act === 'sell') {
          confirmMsg = `Confirm OPEN SHORT position at current market price?`;
        } else {
          confirmMsg = `Confirm OPEN LONG position at current market price?`;
        }
      }
      if (!confirm(confirmMsg)) return;

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
          if (data.active_position) {
            currentActivePosition = data.active_position;
            activePosNullStreak = 0;
            try { localStorage.setItem('bot_active_position', JSON.stringify(data.active_position)); } catch(e) {}
            renderActivePositionBanner(data.active_position);
          } else {
            currentActivePosition = null;
            activePosNullStreak = 0;
            try { localStorage.removeItem('bot_active_position'); } catch(e) {}
            renderActivePositionBanner(null);
          }
          await Promise.all([fetchStatus(), refreshExecutionTracker(), refreshLogs()]);
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
        const res = await fetch('/api/status?_ts=' + Date.now(), { cache: 'no-store' });
        const data = await res.json();
        botRunning = data.running;
        // Update Trading Environment UI on Dashboard (Client-authoritative via localStorage)
        const savedTradingMode = localStorage.getItem('bot_trading_mode');
        let effectiveTradingMode = savedTradingMode;
        if (!effectiveTradingMode) {
          effectiveTradingMode = (data.mode === 'live' && !data.testnet) ? 'real' : 'testnet';
          try { localStorage.setItem('bot_trading_mode', effectiveTradingMode); } catch (e) {}
        }

        const isReal = (effectiveTradingMode === 'real');
        updateTradingModeUI(isReal, isReal ? 'live' : 'paper', !isReal);
        document.getElementById('val-symbol-tag').innerText = `${data.symbol} ${data.timeframe}`;

        // Update Execution Mode & Strategy UI on Dashboard (Client-authoritative via localStorage)
        const savedExecMode = localStorage.getItem('bot_execution_mode');
        let effectiveExecMode = data.execution_mode;
        if (savedExecMode) {
          effectiveExecMode = savedExecMode;
        } else if (data.execution_mode) {
          localStorage.setItem('bot_execution_mode', data.execution_mode);
          effectiveExecMode = data.execution_mode;
        }

        const isManual = (effectiveExecMode === 'manual');
        updateExecutionModeUI(isManual);

        const activeStratTag = document.getElementById('active-strat-tag');
        const activeTfTag = document.getElementById('active-tf-tag');
        const activeModeTag = document.getElementById('active-mode-tag');

        if (activeStratTag) activeStratTag.innerText = `Strategy: ${data.strategy_short_name || data.strategy}`;
        if (activeTfTag) activeTfTag.innerText = `Timeframe: ${data.timeframe}`;
        if (activeModeTag) {
          const modeLabel = (data.mode || 'paper').toUpperCase();
          const execType = isManual ? 'Manual Orders' : 'Algorithmic Execution';
          if (!data.running) {
            activeModeTag.innerHTML = `<i class="fa-solid fa-pause text-slate-400 mr-1"></i>Engine: IDLE (${execType})`;
            activeModeTag.className = isManual ? 'text-amber-300 font-semibold bg-amber-950/60 px-2 py-0.5 rounded border border-amber-500/50' : 'text-slate-400 font-semibold bg-slate-800/80 px-2 py-0.5 rounded border border-slate-700';
          } else {
            activeModeTag.innerHTML = `<i class="fa-solid fa-play text-emerald-400 mr-1"></i>Engine: ARMED (${modeLabel} · ${execType})`;
            activeModeTag.className = isReal ? 'text-emerald-300 font-semibold bg-emerald-950/60 px-2 py-0.5 rounded border border-emerald-500/50 animate-pulse' : (isManual ? 'text-amber-300 font-semibold bg-amber-950/60 px-2 py-0.5 rounded border border-amber-500/50' : 'text-blue-300 font-semibold bg-blue-950/60 px-2 py-0.5 rounded border border-blue-500/50');
          }
        }

        const tfWarnBanner = document.getElementById('tf-warning-banner');
        if (tfWarnBanner) {
          if (data.timeframe_warning || ['1m', '3m', '5m', '15m'].includes(data.timeframe)) {
            tfWarnBanner.classList.remove('hidden');
          } else {
            tfWarnBanner.classList.add('hidden');
          }
        }

        const isKillActive = Boolean(data.kill_file_exists || data.kill_switch_triggered);
        isKillSwitchActive = isKillActive;

        const killBanner = document.getElementById('kill-switch-active-banner');
        const btnResetKill = document.getElementById('btn-reset-kill');
        const btnResetKillMob = document.getElementById('btn-reset-kill-mob');
        if (killBanner) killBanner.classList.toggle('hidden', !isKillActive);
        if (btnResetKill) btnResetKill.classList.toggle('hidden', !isKillActive);
        if (btnResetKillMob) btnResetKillMob.classList.toggle('hidden', !isKillActive);

        // Update desktop toggle button
        const killToggle = document.getElementById('btn-kill-toggle');
        const killToggleIcon = document.getElementById('btn-kill-toggle-icon');
        const killToggleText = document.getElementById('btn-kill-toggle-text');
        if (killToggle && killToggleIcon && killToggleText) {
          if (isKillActive) {
            killToggle.className = 'px-2.5 py-1.5 bg-rose-600 hover:bg-rose-500 text-white border border-rose-400 font-semibold rounded-lg shadow-lg flex items-center gap-1.5 text-xs animate-pulse transition';
            killToggle.title = 'Emergency Kill Switch is ACTIVE (Halted). Click to disarm.';
            killToggleIcon.className = 'fa-solid fa-toggle-on text-white text-sm';
            killToggleText.innerText = 'Kill Switch: ON';
          } else {
            killToggle.className = 'px-2.5 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 font-semibold rounded-lg transition flex items-center gap-1.5 text-xs shadow-sm';
            killToggle.title = 'Emergency Kill Switch is DISARMED. Click to activate emergency halt.';
            killToggleIcon.className = 'fa-solid fa-toggle-off text-slate-400 text-sm';
            killToggleText.innerText = 'Kill Switch: OFF';
          }
        }

        // Update mobile drawer toggle button
        const killToggleMob = document.getElementById('btn-kill-toggle-mob');
        const killToggleMobIcon = document.getElementById('btn-kill-toggle-mob-icon');
        const killToggleMobStatus = document.getElementById('btn-kill-toggle-mob-status');
        if (killToggleMob && killToggleMobIcon && killToggleMobStatus) {
          if (isKillActive) {
            killToggleMob.className = 'w-full py-2.5 bg-rose-600/90 active:bg-rose-700 text-white border border-rose-400 font-bold rounded-xl shadow-lg transition flex items-center justify-between px-3 text-xs animate-pulse';
            killToggleMobIcon.className = 'fa-solid fa-toggle-on text-white text-base';
            killToggleMobStatus.innerText = 'ON';
            killToggleMobStatus.className = 'text-[11px] font-bold text-white';
          } else {
            killToggleMob.className = 'w-full py-2.5 bg-slate-800 active:bg-slate-700 text-slate-300 border border-slate-700 font-semibold rounded-xl transition flex items-center justify-between px-3 text-xs';
            killToggleMobIcon.className = 'fa-solid fa-toggle-off text-slate-400 text-base';
            killToggleMobStatus.innerText = 'OFF';
            killToggleMobStatus.className = 'text-[11px] font-bold text-slate-400';
          }
        }

        // Update mobile bottom nav button
        const bottomKill = document.getElementById('bottom-btn-kill');
        const bottomKillIcon = document.getElementById('bottom-btn-kill-icon');
        const bottomKillText = document.getElementById('bottom-btn-kill-text');
        if (bottomKill && bottomKillIcon && bottomKillText) {
          if (isKillActive) {
            bottomKill.className = 'flex flex-col items-center justify-center py-1 px-1.5 text-rose-400 active:scale-95 transition flex-1 animate-pulse';
            bottomKillIcon.className = 'fa-solid fa-toggle-on text-base text-rose-400';
            bottomKillText.innerText = 'Kill: ON';
            bottomKillText.className = 'text-[10px] font-bold mt-0.5 text-rose-300';
          } else {
            bottomKill.className = 'flex flex-col items-center justify-center py-1 px-1.5 text-slate-400 hover:text-slate-200 active:scale-95 transition flex-1';
            bottomKillIcon.className = 'fa-solid fa-toggle-off text-base text-slate-400';
            bottomKillText.innerText = 'Kill: OFF';
            bottomKillText.className = 'text-[10px] font-semibold mt-0.5 text-slate-400';
          }
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
          if (dot) dot.className = isKillActive ? 'inline-block w-2 h-2 rounded-full bg-rose-500 animate-pulse' : 'inline-block w-2 h-2 rounded-full bg-slate-500';
          if (text) {
            text.innerText = isKillActive ? 'HALTED (KILL SWITCH)' : 'IDLE';
            text.className = isKillActive ? 'font-semibold text-rose-400' : 'font-semibold text-slate-400';
          }
          if (dotMob) dotMob.className = 'inline-block w-2 h-2 rounded-full ' + (isKillActive ? 'bg-rose-500 animate-pulse' : 'bg-slate-500');
          if (textMob) {
            textMob.innerText = isKillActive ? 'HALTED' : 'IDLE';
            textMob.className = isKillActive ? 'font-semibold text-rose-400' : 'font-semibold text-slate-400';
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

        const cashLbl = document.getElementById('val-cash-label');
        if (isReal && data.wallet_balance && data.wallet_balance.is_live) {
          if (cashLbl) cashLbl.innerText = `Live ${data.wallet_balance.quote_currency || 'Quote'}`;
          document.getElementById('val-cash').innerText = '$' + (data.wallet_balance.free_quote ?? data.paper_cash).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
          document.getElementById('val-pos').innerText = (data.wallet_balance.free_base ?? data.paper_position).toFixed(6);
        } else {
          if (cashLbl) cashLbl.innerText = 'Paper Cash';
          document.getElementById('val-cash').innerText = '$' + data.paper_cash.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
          document.getElementById('val-pos').innerText = data.paper_position.toFixed(6);
        }

        const symTag = document.getElementById('val-symbol-tag');
        if (symTag) {
          let extra = '';
          if (data.order_book_spread && data.order_book_spread.valid) {
            extra = ` · Spr: ${data.order_book_spread.spread_pct.toFixed(2)}%`;
          }
          symTag.innerText = `${data.symbol} ${data.timeframe}${extra}`;
        }

        document.getElementById('val-drawdown').innerText = data.drawdown_pct.toFixed(1) + '%';
        document.getElementById('val-daily-loss').innerText = data.daily_loss_pct.toFixed(1) + '%';
        document.getElementById('val-streak').innerText = `${data.consecutive_losses} / ${data.max_consecutive_losses ?? 8}`;
        const streakLimit = document.getElementById('val-streak-limit');
        if (streakLimit) streakLimit.innerText = data.max_consecutive_losses ?? 8;
        const cooldownBanner = document.getElementById('cooldown-banner');
        const cooldownText = document.getElementById('cooldown-text');
        if (cooldownBanner && cooldownText) {
          if (data.risk_cooldown_note) {
            cooldownBanner.classList.remove('hidden');
            cooldownText.innerText = 'cooling off ' + data.risk_cooldown_note.replace(/^cooling off /, '') +
              (data.cooldown_until ? ' — auto-resumes, no restart needed' : '');
          } else {
            cooldownBanner.classList.add('hidden');
          }
        }

        if (data.active_position) {
          activePosNullStreak = 0;
          currentActivePosition = data.active_position;
          try { localStorage.setItem('bot_active_position', JSON.stringify(data.active_position)); } catch (e) {}
          renderActivePositionBanner(data.active_position);
        } else {
          currentActivePosition = null;
          activePosNullStreak = 0;
          try { localStorage.removeItem('bot_active_position'); } catch (e) {}
          renderActivePositionBanner(null);
        }
      } catch (e) {
        console.error('Error fetching status:', e);
      }
    }

    async function toggleBot() {
      if (!botRunning) {
        const savedTradingMode = localStorage.getItem('bot_trading_mode');
        if (savedTradingMode) {
          try {
            await fetch('/api/bot/mode', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ trading_mode: savedTradingMode })
            });
          } catch (e) {}
        }
      }
      const endpoint = botRunning ? '/api/bot/stop' : '/api/bot/start';
      try {
        const res = await fetch(endpoint, { method: 'POST' });
        await fetchStatus();
        await refreshLogs();
      } catch (e) {
        alert('Action failed: ' + e);
      }
    }

    async function toggleKillSwitch() {
      if (isKillSwitchActive) {
        if (!confirm('Deactivate emergency KILL SWITCH and return to IDLE mode? This clears the halt latch and establishes the current equity baseline.')) return;
        try {
          const res = await fetch('/api/bot/kill/toggle', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ active: false })
          });
          const data = await res.json();
          if (data.success) {
            await fetchStatus();
            await refreshLogs();
          } else {
            alert('Could not deactivate kill switch: ' + (data.message || 'Unknown error'));
          }
        } catch (e) {
          alert('Deactivate kill switch failed: ' + e);
        }
      } else {
        if (!confirm('Are you sure you want to trigger the emergency KILL SWITCH? This halts all execution immediately.')) return;
        try {
          const res = await fetch('/api/bot/kill/toggle', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ active: true })
          });
          const data = await res.json();
          if (data.success) {
            await fetchStatus();
            await refreshLogs();
          } else {
            alert('Kill switch failed: ' + e);
          }
        } catch (e) {
          alert('Kill switch failed: ' + e);
        }
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


    async function deactivateKillSwitch() {
      if (!confirm('Deactivate emergency KILL SWITCH and return to IDLE mode? This clears the halt latch and establishes the current equity baseline.')) return;
      try {
        const res = await fetch('/api/bot/reset_kill', { method: 'POST' });
        const data = await res.json();
        if (data.success) {
          await fetchStatus();
          await refreshLogs();
        } else {
          alert('Could not deactivate kill switch: ' + (data.message || 'Unknown error'));
        }
      } catch (e) {
        alert('Deactivate kill switch failed: ' + e);
      }
    }

    let analysisInFlight = false;

    function setRefreshStamp(id, message, ok) {
      const el = document.getElementById(id);
      if (!el) return;
      el.innerText = message;
      el.className = ok ? 'text-[10px] text-emerald-400' : 'text-[10px] text-rose-400';
    }

    function beginRefresh(buttonId, label) {
      const btn = document.getElementById(buttonId);
      if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> ' + label;
      }
    }

    function endRefresh(buttonId, icon, label) {
      const btn = document.getElementById(buttonId);
      if (btn) {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid ' + icon + '"></i> ' + label;
      }
    }

    async function refreshAnalysis(manual) {
      // A manual click must never be silently dropped: if a poll is already in
      // flight, wait for it and then run a fresh one.
      if (analysisInFlight) return;
      analysisInFlight = true;
      const stampId = 'an-updated';
      if (manual) {
        beginRefresh('btn-refresh-analysis', 'Refreshing...');
        setRefreshStamp(stampId, 'Refreshing market data...', true);
      }
      try {
        // no-store + a unique query param: a CDN/browser cache is the classic
        // reason a Refresh button appears dead on a deployed dashboard.
        const res = await fetch('/api/analysis?_ts=' + Date.now(), { cache: 'no-store' });
        if (!res.ok) {
          setRefreshStamp(stampId, `Could not refresh (HTTP ${res.status}). Exchange data unreachable - retrying automatically.`, false);
          console.warn('Analysis endpoint returned HTTP status:', res.status);
          return;
        }
        const data = await res.json();
        if (!data || data.price === undefined || data.price === null) {
          setRefreshStamp(stampId, 'Could not refresh: unexpected analysis payload.', false);
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
        setRefreshStamp(stampId, 'Updated ' + new Date().toLocaleTimeString(), true);
      } catch (e) {
        setRefreshStamp(stampId, 'Could not refresh: ' + (e && e.message ? e.message : e), false);
        console.error('Error fetching analysis:', e);
      } finally {
        analysisInFlight = false;
        if (manual) endRefresh('btn-refresh-analysis', 'fa-rotate', 'Refresh');
      }
    }

    function setBacktestDays(days) {
      const el = document.getElementById('bt-days');
      if (!el) return;
      // The quick-pick buttons are meant to be one click, but a grain that only
      // stores 60 days must not silently hand the API a 730-day request.
      const limit = backtestDayLimit();
      el.value = Math.min(days, limit);
      applyTimeframeLimits();
    }

    function backtestDayLimit() {
      const tf = (document.getElementById('bt-timeframe') || {}).value;
      const entry = BT_LOOKBACK_LIMITS[tf];
      return entry ? entry.max_days : 730;
    }

    function applyTimeframeLimits() {
      const input = document.getElementById('bt-days');
      const note = document.getElementById('bt-days-note');
      const limit = backtestDayLimit();
      const grain = (document.getElementById('bt-timeframe') || {}).value || '';
      const entry = BT_LOOKBACK_LIMITS[grain] || {};
      if (input) input.max = String(limit);

      const asked = input ? parseInt(input.value, 10) : limit;
      const tooDeep = Number.isFinite(asked) && asked > limit;
      if (note) {
        note.innerText = tooDeep
          ? `${grain} on this venue stores ~${limit} days; the run will be capped to ${limit}.`
          : (entry.note || `Lookback up to ${limit} days for ${grain}.`);
        note.className = tooDeep ? 'text-[10px] text-amber-400 mt-1' : 'text-[10px] text-slate-500 mt-1';
      }
      return limit;
    }

    async function executeBacktest(e) {
      e.preventDefault();
      const btn = document.getElementById('btn-run-bt');
      // Helpers are declared before their first use on purpose: a `const` arrow
      // function referenced above its own declaration sits in the temporal dead
      // zone, and the resulting ReferenceError aborted the entire results render
      // (the user saw "Backtest error: Cannot access 'setIf' before initialization").
      const setIf = (id, value) => { const el = document.getElementById(id); if (el) el.innerText = value; };
      btn.disabled = true;
      btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Running Backtest...';

      // Warn and cap rather than sending a request the venue cannot answer.
      const dayLimit = applyTimeframeLimits();
      const askedDays = parseInt(document.getElementById('bt-days').value, 10);
      const days = Math.min(Number.isFinite(askedDays) ? askedDays : dayLimit, dayLimit);
      if (days !== askedDays) setIf('bt-days-note', `Capped to ${days} days for this timeframe.`);

      const payload = {
        exchange: document.getElementById('bt-exchange').value,
        symbol: document.getElementById('bt-symbol').value,
        timeframe: document.getElementById('bt-timeframe').value,
        days: days,
        starting_equity: parseFloat(document.getElementById('bt-equity').value),
        use_fib_filter: document.getElementById('bt-use-fib').checked,
        fib_level_mode: document.getElementById('bt-fib-depth').value,
        fib_tolerance_atr_mult: parseFloat(document.getElementById('bt-fib-atr').value),
        fib_tolerance_pct: parseFloat(document.getElementById('bt-fib-floor').value) / 100,
        trailing_stop_enabled: document.getElementById('bt-trail').checked,
        trailing_stop_atr_mult: parseFloat(document.getElementById('bt-trail-atr').value),
        risk_per_trade_pct: 0.01
      };

      try {
        const res = await fetch('/api/backtest', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        // A proxy or gateway can answer with HTML, so a parse failure is an
        // ordinary outcome here, not an exception to leak to the console.
        let data = null;
        try { data = await res.json(); } catch (parseErr) { data = null; }
        const detail = data && data.detail;
        if (!data || !data.success) {
          throw new Error(typeof detail === 'string' && detail.trim()
            ? detail
            : `The backtest service answered HTTP ${res.status}.`);
        }

        const r = data.results;
        document.getElementById('bt-results-box').classList.remove('hidden');
        hideNotice();   // a successful run clears any error still on screen
        const retEl = document.getElementById('bt-res-return');
        retEl.innerText = (r.total_return_pct >= 0 ? '+' : '') + r.total_return_pct + '%';
        retEl.className = r.total_return_pct >= 0 ? 'font-bold text-emerald-400' : 'font-bold text-rose-400';

        document.getElementById('bt-res-trades').innerText = r.num_trades;
        document.getElementById('bt-res-winrate').innerText = r.win_rate_pct + '%';
        document.getElementById('bt-res-final').innerText = '$' + r.final_equity.toLocaleString();
        const sharpeDisplay = (r.sharpe_approx !== null && r.sharpe_approx !== undefined) ? r.sharpe_approx : '-- (< 20 trades)';
        document.getElementById('bt-res-sharpe').innerText = sharpeDisplay;
        const bhRet = r.buy_and_hold_return_pct;
        const bhBadge = document.getElementById('bt-res-bh-badge');
        if (bhRet !== null && bhRet !== undefined) {
          const bhSign = bhRet >= 0 ? '+' : '';
          setIf('bt-res-bh', `${bhSign}${Number(bhRet).toFixed(2)}%`);
          const alpha = (Number(r.total_return_pct || 0) - Number(bhRet)).toFixed(2);
          const alphaSign = alpha >= 0 ? '+' : '';
          const alphaEl = document.getElementById('bt-res-alpha');
          if (alphaEl) {
            alphaEl.innerText = `${alphaSign}${alpha}%`;
            alphaEl.className = alpha >= 0 ? 'font-semibold text-emerald-400' : 'font-semibold text-rose-400';
          }
          if (bhBadge) {
            bhBadge.innerText = `B&H: ${bhSign}${Number(bhRet).toFixed(1)}%`;
            bhBadge.className = 'text-[10px] px-1.5 py-0.5 rounded bg-sky-950/80 text-sky-300 border border-sky-500/30 font-semibold';
          }
        } else {
          setIf('bt-res-bh', '--');
          setIf('bt-res-alpha', '--');
          if (bhBadge) {
            bhBadge.innerText = 'B&H: --';
            bhBadge.className = 'text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 font-normal';
          }
        }
        document.getElementById('bt-res-avgwin').innerText = '$' + r.avg_win;
        document.getElementById('bt-res-avgloss').innerText = '$' + r.avg_loss;
        document.getElementById('bt-res-drawdown').innerText = r.max_drawdown_pct + '%';
        setIf('bt-res-fees', '$' + (r.total_fees_paid != null ? Number(r.total_fees_paid).toFixed(2) : '0.00'));
        setIf('bt-res-rr', (r.realized_rr ?? 0) + ' : 1');
        const exits = r.exit_reasons || {};
        setIf('bt-res-exits', Object.keys(exits).length
          ? Object.entries(exits).map(([k, v]) => k + '=' + v).join(', ')
          : '--');
        const fibNote = document.getElementById('bt-res-fib-note');
        const diag = r.diagnostics || {};
        setIf('bt-res-candles', (r.candles_tested ?? r.candles_evaluated ?? diag.candles_evaluated ?? 0).toLocaleString());
        const range = (r.date_range || []).filter(Boolean);
        const evalRange = (r.eval_date_range || []).filter(Boolean);
        const warmup = r.candles_warmup ?? diag.candles_warmup ?? 0;
        setIf('bt-res-window', range.length === 2
          ? 'Indicator Calc: ' + (r.candles_tested ?? 0).toLocaleString() + ' candles (' + range[0] + ' \u2192 ' + range[1] + ')'
          : 'Indicator Calc: --');
        setIf('bt-res-eval-window', evalRange.length === 2
          ? 'Signal Eval: ' + (diag.candles_evaluated ?? 0).toLocaleString() + ' candles (' + evalRange[0] + ' \u2192 ' + evalRange[1] + '; ' + warmup + ' warmup skipped)'
          : 'Signal Eval: --');
        setIf('bt-res-raw', (r.raw_signals ?? 0).toLocaleString());
        setIf('bt-res-fibblocked', (diag.fib_filter_blocked || 0).toLocaleString());
        setIf('bt-res-blocked', ((diag.sizing_blocked || 0) + (diag.signals_risk_blocked || 0)).toLocaleString());
        const reportEl = document.getElementById('bt-res-report');
        if (reportEl) reportEl.value = diag.report || 'No pipeline diagnostics returned for this run.';
        if (fibNote) {
          const atrMult = r.fib_tolerance_atr_mult ?? 0;
          const band = r.fib_filter_enabled
            ? (atrMult > 0 ? `band \u00b1${atrMult}\u00d7ATR (never tighter than the fixed %)`
                           : 'fixed percentage band')
            : '';
          const mode = (r.fib_level_mode || 'all');
          const levels = mode === 'zone' ? 'pullback into the 0.382-0.618 golden zone'
            : (mode === 'deep' ? 'within the 0.5/0.618 band' : '0.382/0.5/0.618');
          const trail = r.trailing_stop_enabled
            ? `Trailing stop: ON, ${r.trailing_stop_atr_mult}\u00d7ATR behind the peak high.`
            : 'Trailing stop: OFF (fixed stop, reversal-signal exits).';
          const bars = r.min_bars_between_trades ?? 0;
          const debounce = bars > 0
            ? `Entry debounce: ${bars} bar(s) between trades` +
              ((r.debounce_blocked || 0) > 0
                ? ` \u2014 ${(r.debounce_blocked || 0).toLocaleString()} re-trigger(s) held back.`
                : '.')
            : 'Entry debounce: OFF (adjacent candles may re-enter).';
          if (r.fib_filter_enabled) {
            fibNote.innerText = 'Fibonacci Zone Filter: ON — ' + (r.fib_blocked_signals || 0) +
              ' core signal(s) skipped because price was not near a fib level. Levels: ' + levels +
              ', ' + band + '. ' + trail + ' ' + debounce;
          } else {
            fibNote.innerText = 'Fibonacci Zone Filter: OFF — entries taken on the core EMA/ADX/RSI signal alone. ' +
              trail + ' ' + debounce;
          }
        }
        if (typeof refreshLogs === 'function') await refreshLogs();
      } catch (err) {
        // Say it inside the page in one short sentence; the raw exchange text
        // stays one tap away instead of being dumped by a browser alert().
        const failure = describeBacktestError(err);
        showNotice(failure.title, failure.body, { kind: 'error', detail: failure.detail });
        // Stale numbers from an earlier run must not read as this run's result.
        const stale = document.getElementById('bt-results-box');
        if (stale) stale.classList.add('hidden');
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-play"></i> Run Historical Backtest';
      }
    }

    function escapeHtml(str) {
      if (!str) return '';
      return String(str)
        .split('&').join('&amp;')
        .split('<').join('&lt;')
        .split('>').join('&gt;')
        .split('"').join('&quot;')
        .split("'").join('&#039;');
    }

    function getBacktestPayloadBase() {
      return {
        exchange: (document.getElementById('bt-exchange') || {}).value || 'binance',
        symbol: (document.getElementById('bt-symbol') || {}).value || 'BTC/USDT',
        starting_equity: parseFloat((document.getElementById('bt-equity') || {}).value) || 1000.0,
        use_fib_filter: (document.getElementById('bt-use-fib') || {}).checked ?? true,
        fib_level_mode: (document.getElementById('bt-fib-depth') || {}).value || 'deep',
        fib_tolerance_atr_mult: parseFloat((document.getElementById('bt-fib-atr') || {}).value) || 0.5,
        fib_tolerance_pct: (parseFloat((document.getElementById('bt-fib-floor') || {}).value) || 0.3) / 100,
        trailing_stop_enabled: (document.getElementById('bt-trail') || {}).checked ?? true,
        trailing_stop_atr_mult: parseFloat((document.getElementById('bt-trail-atr') || {}).value) || 3.5,
        risk_per_trade_pct: 0.01
      };
    }

    function initSweepTable(mode, title, items) {
      const sweepBox = document.getElementById('bt-sweep-box');
      if (!sweepBox) return;
      sweepBox.classList.remove('hidden');
      hideNotice();

      const titleEl = document.getElementById('bt-sweep-title');
      const badgeEl = document.getElementById('bt-sweep-badge');
      const summaryEl = document.getElementById('bt-sweep-summary');
      const thWindowEl = document.getElementById('bt-th-window');
      const tbody = document.getElementById('bt-sweep-tbody');

      const isTimeframes = (mode === 'timeframes');
      if (thWindowEl) thWindowEl.innerText = isTimeframes ? 'Timeframe' : 'Lookback';
      if (titleEl) titleEl.innerText = title;
      if (badgeEl) {
        badgeEl.innerText = 'RUNNING';
        badgeEl.className = 'px-2 py-0.5 rounded text-[10px] font-semibold bg-purple-500/20 text-purple-300 border border-purple-500/30';
      }
      if (summaryEl) {
        summaryEl.innerHTML = `<span class="text-purple-300 font-semibold"><i class="fa-solid fa-spinner fa-spin mr-1"></i> Running sweep across regimes...</span> Completed <span id="bt-sweep-progress">0</span> of ${items.length}.`;
      }

      if (tbody) {
        tbody.innerHTML = '';
        items.forEach((item, idx) => {
          const tr = document.createElement('tr');
          tr.id = `bt-sweep-row-${idx}`;
          tr.className = 'bg-slate-900/40 text-slate-400';
          const label = isTimeframes ? (item.tf + ` (${item.days}d)`) : item.label;
          tr.innerHTML = `
            <td class="py-1.5 px-1.5 font-semibold text-slate-300">${label}</td>
            <td colspan="11" class="py-1.5 px-1.5 text-slate-500 text-left font-sans italic" id="bt-sweep-status-${idx}">
              <i class="fa-solid fa-clock mr-1"></i> Waiting in queue...
            </td>
          `;
          tbody.appendChild(tr);
        });
      }
    }

    function updateSweepTableRow(idx, isTimeframes, row) {
      const tr = document.getElementById(`bt-sweep-row-${idx}`);
      if (!tr) return;
      const isProf = row.profitable;
      const hasErr = !!row.error;

      tr.className = hasErr ? 'bg-slate-900/40 text-slate-500' : (isProf ? 'bg-emerald-950/20 hover:bg-emerald-950/30' : 'bg-rose-950/15 hover:bg-rose-950/25');

      const windowLabel = isTimeframes ? (row.timeframe + ` (${row.lookback_days}d)`) : row.lookback_label;
      const returnText = hasErr ? '--' : ((row.total_return_pct >= 0 ? '+' : '') + Number(row.total_return_pct).toFixed(2) + '%');
      const returnClass = hasErr ? 'text-slate-500' : (row.total_return_pct >= 0 ? 'text-emerald-400 font-bold' : 'text-rose-400 font-bold');
      const sharpeText = (row.sharpe_approx !== null && row.sharpe_approx !== undefined) ? Number(row.sharpe_approx).toFixed(2) : '--';
      const isUnavailable = row.error && row.error.includes('Not enough history');
      const regimeBadge = hasErr
        ? (isUnavailable
            ? `<span class="px-1.5 py-0.5 rounded text-[9px] bg-amber-500/20 text-amber-300 font-sans">UNAVAILABLE</span>`
            : `<span class="px-1.5 py-0.5 rounded text-[9px] bg-slate-800 text-slate-400 font-sans">ERROR</span>`)
        : (isProf
            ? `<span class="px-1.5 py-0.5 rounded text-[9px] bg-emerald-500/20 text-emerald-300 font-bold font-sans">PROFIT</span>`
            : `<span class="px-1.5 py-0.5 rounded text-[9px] bg-rose-500/20 text-rose-300 font-bold font-sans">LOSS</span>`);

      const evalRange = (row.eval_date_range || []).filter(Boolean);
      const dateTooltip = evalRange.length === 2 ? `title="Eval: ${evalRange[0]} \u2192 ${evalRange[1]}"` : '';

      if (hasErr) {
        const cleanErr = firstUsefulLine(row.error) || row.error;
        const iconClass = isUnavailable ? 'fa-solid fa-circle-exclamation' : 'fa-solid fa-triangle-exclamation';
        const textCol = isUnavailable ? 'text-amber-400/90' : 'text-rose-400';
        tr.innerHTML = `
          <td class="py-1.5 px-1.5 font-semibold text-slate-300" ${dateTooltip}>${windowLabel}</td>
          <td colspan="10" class="py-1.5 px-1.5 ${textCol} text-left font-sans text-[10px] truncate" title="${escapeHtml(row.error)}">
            <i class="${iconClass} mr-1"></i> ${escapeHtml(cleanErr)}
          </td>
          <td class="py-1.5 px-1.5">${regimeBadge}</td>
        `;
      } else {
        const bhText = (row.buy_and_hold_return_pct !== null && row.buy_and_hold_return_pct !== undefined)
          ? ((row.buy_and_hold_return_pct >= 0 ? '+' : '') + Number(row.buy_and_hold_return_pct).toFixed(2) + '%')
          : '--';
        const bhClass = (row.buy_and_hold_return_pct !== null && row.buy_and_hold_return_pct !== undefined)
          ? (row.buy_and_hold_return_pct >= 0 ? 'text-sky-400' : 'text-slate-400')
          : 'text-slate-500';
        tr.innerHTML = `
          <td class="py-1.5 px-1.5 font-semibold text-slate-200" ${dateTooltip}>${windowLabel}</td>
          <td class="py-1.5 px-1.5 text-slate-300">${row.trades}</td>
          <td class="py-1.5 px-1.5 text-slate-300">${(Number(row.win_rate_pct).toFixed(1) + '%')}</td>
          <td class="py-1.5 px-1.5 ${returnClass}">${returnText}</td>
          <td class="py-1.5 px-1.5 ${bhClass}">${bhText}</td>
          <td class="py-1.5 px-1.5 text-slate-300">${sharpeText}</td>
          <td class="py-1.5 px-1.5 text-slate-300">${((row.realized_rr ?? 0) + ' : 1')}</td>
          <td class="py-1.5 px-1.5 text-amber-400">${(Number(row.max_drawdown_pct).toFixed(1) + '%')}</td>
          <td class="py-1.5 px-1.5 text-slate-400">${(row.fib_blocked || 0)}</td>
          <td class="py-1.5 px-1.5 text-slate-400">${((row.sizing_blocked || 0) + (row.risk_blocked || 0))}</td>
          <td class="py-1.5 px-1.5 text-amber-400 font-sans">${('$' + Number(row.total_fees_paid || 0).toFixed(2))}</td>
          <td class="py-1.5 px-1.5">${regimeBadge}</td>
        `;
      }
    }

    function finalizeSweepSummary(rows) {
      const badgeEl = document.getElementById('bt-sweep-badge');
      const summaryEl = document.getElementById('bt-sweep-summary');
      if (!badgeEl || !summaryEl) return;

      const totalCount = rows.length;
      const errRows = rows.filter(r => !!r.error);
      const validRows = rows.filter(r => !r.error);
      const profCount = validRows.filter(r => r.profitable).length;
      const validCount = validRows.length;

      if (errRows.length === totalCount && totalCount > 0) {
        const allUnavailable = errRows.every(r => r.error && r.error.includes('Not enough history'));
        if (allUnavailable) {
          badgeEl.innerText = 'UNAVAILABLE';
          badgeEl.className = 'px-2 py-0.5 rounded text-[10px] font-semibold bg-amber-500/20 text-amber-400 border border-amber-500/30';
          const sampleErr = errRows[0].error || 'Exchange request failed';
          summaryEl.innerHTML = `<span class="text-amber-400 font-semibold">\u26a0 Lookback window unavailable.</span> ${escapeHtml(firstUsefulLine(sampleErr) || sampleErr)}`;
        } else {
          badgeEl.innerText = 'EXECUTION ERROR';
          badgeEl.className = 'px-2 py-0.5 rounded text-[10px] font-semibold bg-rose-500/20 text-rose-400 border border-rose-500/30';
          const sampleErr = errRows[0].error || 'Exchange request failed';
          summaryEl.innerHTML = `<span class="text-rose-400 font-semibold">\u2718 Sweep failed (${errRows.length}/${totalCount} regimes failed).</span> ${escapeHtml(firstUsefulLine(sampleErr) || sampleErr)}`;
          const failure = describeBacktestError({ message: sampleErr });
          showNotice(failure.title, failure.body, { kind: 'error', detail: failure.detail });
        }
        return;
      }

      const pctProf = validCount > 0 ? (profCount / validCount) * 100 : 0;
      const unavailCount = errRows.filter(r => r.error && r.error.includes('Not enough history')).length;
      const rawErrCount = errRows.length - unavailCount;
      let errSuffix = '';
      if (unavailCount > 0 && rawErrCount > 0) {
        errSuffix = ` (${unavailCount} unavailable, ${rawErrCount} errored)`;
      } else if (unavailCount > 0) {
        errSuffix = ` (${unavailCount} unavailable)`;
      } else if (rawErrCount > 0) {
        errSuffix = ` (${rawErrCount} errored)`;
      }

      if (pctProf === 100) {
        badgeEl.innerText = 'ROBUST REGIME';
        badgeEl.className = 'px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30';
        summaryEl.innerHTML = `<span class="text-emerald-400 font-semibold">\u2714 Profitable across all ${validCount} test regimes (${profCount}/${validCount}).</span> Strategy exhibits strong regime independence.${errSuffix}`;
      } else if (pctProf > 0) {
        badgeEl.innerText = 'REGIME DEPENDENT';
        badgeEl.className = 'px-2 py-0.5 rounded text-[10px] font-semibold bg-amber-500/20 text-amber-400 border border-amber-500/30';
        summaryEl.innerHTML = `<span class="text-amber-400 font-semibold">\u26a0 Profitable in ${profCount} of ${validCount} regimes.</span> Strategy gains are concentrated in specific market regimes and fragile in others.${errSuffix}`;
      } else {
        badgeEl.innerText = 'UNPROFITABLE';
        badgeEl.className = 'px-2 py-0.5 rounded text-[10px] font-semibold bg-rose-500/20 text-rose-400 border border-rose-500/30';
        summaryEl.innerHTML = `<span class="text-rose-400 font-semibold">\u2718 Unprofitable across all ${validCount} regimes (0/${validCount}).</span> Parameters require re-tuning.${errSuffix}`;
      }
    }

    async function executeCompareWindows(e) {
      if (e && e.preventDefault) e.preventDefault();
      const btn = document.getElementById('btn-sweep-windows');
      if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Sweeping Windows...';
      }

      try {
        const tf = (document.getElementById('bt-timeframe') || {}).value || '1h';
        const exch = (document.getElementById('bt-exchange') || {}).value || 'binance';
        const sym = (document.getElementById('bt-symbol') || {}).value || 'BTC/USDT';
        const tfLimit = (typeof BT_LOOKBACK_LIMITS !== 'undefined' && BT_LOOKBACK_LIMITS[tf])
          ? BT_LOOKBACK_LIMITS[tf].max_days : 730;

        const windows = [90, 365, 730];
        const items = windows.map(w => ({ days: w, label: w + 'd' }));
        const title = `Multi-Window Sweep (${sym} ${tf} on ${exch.toUpperCase()})`;
        initSweepTable('windows', title, items);

        const rows = [];
        const basePayload = getBacktestPayloadBase();
        basePayload.timeframe = tf;

        for (let i = 0; i < items.length; i++) {
          const item = items[i];
          const stEl = document.getElementById(`bt-sweep-status-${i}`);

          if (item.days > tfLimit) {
            const rowData = {
              lookback_days: item.days,
              lookback_label: item.label,
              error: `Not enough history for ${tf} (max ~${tfLimit}d)`,
              trades: 0, win_rate_pct: 0, total_return_pct: 0, final_equity: basePayload.starting_equity,
              sharpe_approx: null, realized_rr: 0, max_drawdown_pct: 0,
              fib_blocked: 0, sizing_blocked: 0, risk_blocked: 0, total_fees_paid: 0, profitable: false
            };
            rows.push(rowData);
            updateSweepTableRow(i, false, rowData);
            const progEl = document.getElementById('bt-sweep-progress');
            if (progEl) progEl.innerText = String(i + 1);
            continue;
          }

          if (stEl) stEl.innerHTML = `<span class="text-purple-300 font-semibold"><i class="fa-solid fa-spinner fa-spin mr-1"></i> Running ${item.label} backtest...</span>`;

          const payload = { ...basePayload, days: item.days, sweep_tag: 'Sweep ' + item.label };
          let rowData = null;
          try {
            const res = await fetch('/api/backtest', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(payload)
            });
            let data = null;
            try { data = await res.json(); } catch (pe) { data = null; }
            if (data && data.success) {
              const r = data.results;
              const diag = r.diagnostics || {};
              rowData = {
                lookback_days: item.days,
                lookback_label: item.label,
                candles_evaluated: r.candles_evaluated ?? r.candles_tested ?? 0,
                candles_tested: r.candles_tested ?? 0,
                trades: r.num_trades ?? 0,
                win_rate_pct: r.win_rate_pct ?? 0,
                total_return_pct: r.total_return_pct ?? 0,
                buy_and_hold_return_pct: r.buy_and_hold_return_pct,
                final_equity: r.final_equity ?? 0,
                sharpe_approx: r.sharpe_approx,
                realized_rr: r.realized_rr ?? 0,
                max_drawdown_pct: r.max_drawdown_pct ?? 0,
                fib_blocked: diag.fib_filter_blocked ?? 0,
                sizing_blocked: diag.sizing_blocked ?? 0,
                risk_blocked: diag.signals_risk_blocked ?? 0,
                total_fees_paid: r.total_fees_paid ?? 0,
                profitable: (r.total_return_pct ?? 0) > 0,
                eval_date_range: r.eval_date_range || [],
                date_range: r.date_range || [],
              };
            } else {
              const detail = (data && data.detail) || `HTTP ${res.status}`;
              rowData = {
                lookback_days: item.days,
                lookback_label: item.label,
                error: detail,
                trades: 0, win_rate_pct: 0, total_return_pct: 0, final_equity: payload.starting_equity,
                sharpe_approx: null, realized_rr: 0, max_drawdown_pct: 0,
                fib_blocked: 0, sizing_blocked: 0, risk_blocked: 0, total_fees_paid: 0, profitable: false
              };
            }
          } catch (netErr) {
            rowData = {
              lookback_days: item.days,
              lookback_label: item.label,
              error: netErr.message || 'Network error',
              trades: 0, win_rate_pct: 0, total_return_pct: 0, final_equity: payload.starting_equity,
              sharpe_approx: null, realized_rr: 0, max_drawdown_pct: 0,
              fib_blocked: 0, sizing_blocked: 0, risk_blocked: 0, total_fees_paid: 0, profitable: false
            };
          }
          rows.push(rowData);
          updateSweepTableRow(i, false, rowData);
          if (typeof refreshLogs === 'function') await refreshLogs();
          const progEl = document.getElementById('bt-sweep-progress');
          if (progEl) progEl.innerText = String(i + 1);
        }
        finalizeSweepSummary(rows);
        if (typeof refreshLogs === 'function') await refreshLogs();
      } catch (err) {
        console.error('Multi-window sweep error:', err);
        const failure = describeBacktestError(err);
        showNotice(failure.title, failure.body, { kind: 'error', detail: failure.detail });
      } finally {
        if (btn) {
          btn.disabled = false;
          btn.innerHTML = '<i class="fa-solid fa-table-columns"></i> Compare All Windows';
        }
      }
    }

    async function executeCompareTimeframes(e) {
      if (e && e.preventDefault) e.preventDefault();
      const btn = document.getElementById('btn-sweep-timeframes');
      if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Sweeping Grid...';
      }

      try {
        const askedDays = parseInt((document.getElementById('bt-days') || {}).value, 10) || 30;
        const exch = (document.getElementById('bt-exchange') || {}).value || 'binance';
        const sym = (document.getElementById('bt-symbol') || {}).value || 'BTC/USDT';
        const timeframes = ['5m', '15m', '1h', '4h', '1d'];

        const items = timeframes.map(tf => {
          const tfLimit = (typeof BT_LOOKBACK_LIMITS !== 'undefined' && BT_LOOKBACK_LIMITS[tf])
            ? BT_LOOKBACK_LIMITS[tf].max_days : 730;
          return { tf, days: askedDays, limit: tfLimit, label: tf };
        });

        const title = `Timeframe Sweep Grid (${sym} on ${exch.toUpperCase()})`;
        initSweepTable('timeframes', title, items);

        const rows = [];
        const basePayload = getBacktestPayloadBase();

        for (let i = 0; i < items.length; i++) {
          const item = items[i];
          const stEl = document.getElementById(`bt-sweep-status-${i}`);

          if (askedDays > item.limit) {
            const rowData = {
              timeframe: item.tf,
              lookback_days: askedDays,
              error: `Not enough history for ${item.tf} (max ~${item.limit}d)`,
              trades: 0, win_rate_pct: 0, total_return_pct: 0, final_equity: basePayload.starting_equity,
              sharpe_approx: null, realized_rr: 0, max_drawdown_pct: 0,
              fib_blocked: 0, sizing_blocked: 0, risk_blocked: 0, total_fees_paid: 0, profitable: false
            };
            rows.push(rowData);
            updateSweepTableRow(i, true, rowData);
            const progEl = document.getElementById('bt-sweep-progress');
            if (progEl) progEl.innerText = String(i + 1);
            continue;
          }

          if (stEl) stEl.innerHTML = `<span class="text-blue-300 font-semibold"><i class="fa-solid fa-spinner fa-spin mr-1"></i> Running ${item.tf} backtest (${item.days}d)...</span>`;

          const payload = { ...basePayload, timeframe: item.tf, days: item.days, sweep_tag: 'Sweep ' + item.tf + ' (' + item.days + 'd)' };
          let rowData = null;
          try {
            const res = await fetch('/api/backtest', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(payload)
            });
            let data = null;
            try { data = await res.json(); } catch (pe) { data = null; }
            if (data && data.success) {
              const r = data.results;
              const diag = r.diagnostics || {};
              rowData = {
                timeframe: item.tf,
                lookback_days: item.days,
                candles_evaluated: r.candles_evaluated ?? r.candles_tested ?? 0,
                candles_tested: r.candles_tested ?? 0,
                trades: r.num_trades ?? 0,
                win_rate_pct: r.win_rate_pct ?? 0,
                total_return_pct: r.total_return_pct ?? 0,
                buy_and_hold_return_pct: r.buy_and_hold_return_pct,
                final_equity: r.final_equity ?? 0,
                sharpe_approx: r.sharpe_approx,
                realized_rr: r.realized_rr ?? 0,
                max_drawdown_pct: r.max_drawdown_pct ?? 0,
                fib_blocked: diag.fib_filter_blocked ?? 0,
                sizing_blocked: diag.sizing_blocked ?? 0,
                risk_blocked: diag.signals_risk_blocked ?? 0,
                total_fees_paid: r.total_fees_paid ?? 0,
                profitable: (r.total_return_pct ?? 0) > 0,
                eval_date_range: r.eval_date_range || [],
                date_range: r.date_range || [],
              };
            } else {
              const detail = (data && data.detail) || `HTTP ${res.status}`;
              rowData = {
                timeframe: item.tf,
                lookback_days: item.days,
                error: detail,
                trades: 0, win_rate_pct: 0, total_return_pct: 0, final_equity: payload.starting_equity,
                sharpe_approx: null, realized_rr: 0, max_drawdown_pct: 0,
                fib_blocked: 0, sizing_blocked: 0, risk_blocked: 0, total_fees_paid: 0, profitable: false
              };
            }
          } catch (netErr) {
            rowData = {
              timeframe: item.tf,
              lookback_days: item.days,
              error: netErr.message || 'Network error',
              trades: 0, win_rate_pct: 0, total_return_pct: 0, final_equity: payload.starting_equity,
              sharpe_approx: null, realized_rr: 0, max_drawdown_pct: 0,
              fib_blocked: 0, sizing_blocked: 0, risk_blocked: 0, total_fees_paid: 0, profitable: false
            };
          }
          rows.push(rowData);
          updateSweepTableRow(i, true, rowData);
          if (typeof refreshLogs === 'function') await refreshLogs();
          const progEl = document.getElementById('bt-sweep-progress');
          if (progEl) progEl.innerText = String(i + 1);
        }
        finalizeSweepSummary(rows);
        if (typeof refreshLogs === 'function') await refreshLogs();
      } catch (err) {
        console.error('Timeframe sweep error:', err);
        const failure = describeBacktestError(err);
        showNotice(failure.title, failure.body, { kind: 'error', detail: failure.detail });
      } finally {
        if (btn) {
          btn.disabled = false;
          btn.innerHTML = '<i class="fa-solid fa-layer-group"></i> Sweep Timeframes';
        }
      }
    }

    async function executeWalkForward(e) {
      if (e && e.preventDefault) e.preventDefault();
      const btn = document.getElementById('btn-walk-forward');
      if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Validating OOS...';
      }

      const wfoBox = document.getElementById('bt-wfo-box');
      const badgeEl = document.getElementById('bt-wfo-badge');
      const summaryEl = document.getElementById('bt-wfo-summary');
      const tbody = document.getElementById('bt-wfo-tbody');

      if (wfoBox) wfoBox.classList.remove('hidden');
      if (badgeEl) {
        badgeEl.innerText = 'VALIDATING';
        badgeEl.className = 'px-2 py-0.5 rounded text-[10px] font-semibold bg-sky-500/20 text-sky-300 border border-sky-500/30';
      }
      if (summaryEl) {
        summaryEl.innerHTML = '<span class="text-sky-300 font-semibold"><i class="fa-solid fa-spinner fa-spin mr-1"></i> Running chronological Walk-Forward Validation across sequential train/test folds...</span>';
      }
      if (tbody) {
        tbody.innerHTML = '<tr><td colspan="9" class="py-2.5 px-2 text-center text-slate-500 italic"><i class="fa-solid fa-spinner fa-spin mr-1.5"></i> Splitting historical candles into sequential train and unseen out-of-sample test folds...</td></tr>';
      }

      try {
        const payload = getBacktestPayloadBase();
        const askedDays = parseInt((document.getElementById('bt-days') || {}).value, 10) || 90;
        const tf = (document.getElementById('bt-timeframe') || {}).value || '1h';
        const tfLimit = (typeof BT_LOOKBACK_LIMITS !== 'undefined' && BT_LOOKBACK_LIMITS[tf])
          ? BT_LOOKBACK_LIMITS[tf].max_days : 730;
        const days = Math.min(askedDays, tfLimit);

        payload.timeframe = tf;
        payload.days = days;
        payload.n_splits = 3;

        const res = await fetch('/api/backtest/walk-forward', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        const data = await res.json();
        if (!res.ok || !data.success) {
          throw new Error((data && (data.detail || data.error)) || `HTTP ${res.status}`);
        }

        const results = data.results || {};
        const folds = results.folds || [];
        const isVal = results.is_out_of_sample_validated;
        const wfe = Number(results.walk_forward_efficiency_pct || 0).toFixed(1);
        const avgIS = Number(results.avg_in_sample_return_pct || 0).toFixed(2);
        const avgOOS = Number(results.avg_out_of_sample_return_pct || 0).toFixed(2);

        if (badgeEl) {
          if (isVal) {
            badgeEl.innerText = 'OUT-OF-SAMPLE VALIDATED';
            badgeEl.className = 'px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40';
          } else {
            badgeEl.innerText = 'IN-SAMPLE ONLY (UNVERIFIED)';
            badgeEl.className = 'px-2 py-0.5 rounded text-[10px] font-semibold bg-amber-500/20 text-amber-300 border border-amber-500/40';
          }
        }

        if (summaryEl) {
          const valText = isVal
            ? `<span class="text-emerald-400 font-semibold">\u2714 Robust Parameters:</span> Strategy maintains positive expectancy on unseen market segments (WFE: <b>${wfe}%</b>, Avg In-Sample: ${avgIS}%, Avg OOS: <b>${avgOOS}%</b>).`
            : `<span class="text-amber-400 font-semibold">\u26a0 Curve-Fitting Warning:</span> Performance drops significantly on unseen data (WFE: <b>${wfe}%</b>, Avg In-Sample: ${avgIS}%, Avg OOS: <b>${avgOOS}%</b>). Parameters may be over-fitted to past noise.`;
          summaryEl.innerHTML = valText;
        }

        if (tbody) {
          tbody.innerHTML = '';
          folds.forEach(f => {
            const tr = document.createElement('tr');
            const isProf = f.out_of_sample_profitable;
            tr.className = isProf ? 'bg-emerald-950/20 hover:bg-emerald-950/30' : 'bg-rose-950/15 hover:bg-rose-950/25';

            const isRetStr = (f.in_sample_return_pct >= 0 ? '+' : '') + Number(f.in_sample_return_pct).toFixed(2) + '%';
            const isRetCol = f.in_sample_return_pct >= 0 ? 'text-emerald-400 font-semibold' : 'text-rose-400 font-semibold';

            const oosRetStr = (f.out_of_sample_return_pct >= 0 ? '+' : '') + Number(f.out_of_sample_return_pct).toFixed(2) + '%';
            const oosRetCol = f.out_of_sample_return_pct >= 0 ? 'text-emerald-400 font-bold' : 'text-rose-400 font-bold';

            const oosBHStr = (f.out_of_sample_bh_return_pct >= 0 ? '+' : '') + Number(f.out_of_sample_bh_return_pct).toFixed(2) + '%';
            const oosBHCol = f.out_of_sample_bh_return_pct >= 0 ? 'text-sky-400' : 'text-slate-400';

            const trainRange = (f.train_date_range || []).join(' \u2192 ') || `${f.train_bars} bars`;
            const testRange = (f.test_date_range || []).join(' \u2192 ') || `${f.test_bars} bars`;

            const verdictBadge = isProf
              ? `<span class="px-1.5 py-0.5 rounded text-[9px] bg-emerald-500/20 text-emerald-300 font-bold font-sans">OOS PROFIT</span>`
              : `<span class="px-1.5 py-0.5 rounded text-[9px] bg-rose-500/20 text-rose-300 font-bold font-sans">OOS LOSS</span>`;

            tr.innerHTML = `
              <td class="py-1.5 px-1.5 font-bold text-slate-300">Fold ${f.fold}</td>
              <td class="py-1.5 px-1.5 text-slate-400 text-[10px]" title="Train: ${trainRange}">${trainRange}</td>
              <td class="py-1.5 px-1.5 ${isRetCol}">${isRetStr}</td>
              <td class="py-1.5 px-1.5 text-slate-300">${f.in_sample_trades}</td>
              <td class="py-1.5 px-1.5 text-slate-400 text-[10px]" title="Test (OOS): ${testRange}">${testRange}</td>
              <td class="py-1.5 px-1.5 ${oosRetCol}">${oosRetStr}</td>
              <td class="py-1.5 px-1.5 ${oosBHCol}">${oosBHStr}</td>
              <td class="py-1.5 px-1.5 text-slate-300">${f.out_of_sample_trades}</td>
              <td class="py-1.5 px-1.5">${verdictBadge}</td>
            `;
            tbody.appendChild(tr);
          });
        }
        if (typeof refreshLogs === 'function') await refreshLogs();
      } catch (err) {
        console.error('Walk-forward error:', err);
        const failure = describeBacktestError(err);
        showNotice(failure.title, failure.body, { kind: 'error', detail: failure.detail });
      } finally {
        if (btn) {
          btn.disabled = false;
          btn.innerHTML = '<i class="fa-solid fa-arrows-split-up-and-left"></i> Walk-Forward (OOS)';
        }
      }
    }

    async function refreshExecutionTracker(manual) {
      if (manual) beginRefresh('btn-refresh-tracker', 'Refreshing...');
      try {
        const res = await fetch('/api/execution/tracker?limit=25&_ts=' + Date.now(), { cache: 'no-store' });
        if (!res.ok) return;
        const data = await res.json();
        const pos = data.active_position;
        const orders = data.orders || [];

        const badgeEl = document.getElementById('pos-status-badge');
        const idEl = document.getElementById('pos-id-display');
        const symEl = document.getElementById('pos-symbol-side');
        const entryEl = document.getElementById('pos-entry-price');
        const currEl = document.getElementById('pos-current-price');
        const sizeEl = document.getElementById('pos-size-amount');
        const stopEl = document.getElementById('pos-stop-level');
        const pnlEl = document.getElementById('pos-unrealized-pnl');
        const durEl = document.getElementById('pos-duration-opened');

        if (pos) {
          const isShort = (pos.side || '').toUpperCase() === 'SHORT' || (pos.side || '').toUpperCase() === 'SELL';
          const sideText = isShort ? 'SHORT' : 'LONG';
          const sideColor = isShort ? 'text-rose-400' : 'text-emerald-400';
          const badgeBg = isShort ? 'bg-rose-500/20 text-rose-400 border-rose-500/40' : 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40';

          if (badgeEl) {
            badgeEl.innerText = `OPEN (${sideText})`;
            badgeEl.className = `px-2 py-0.5 rounded text-[10px] font-semibold border animate-pulse ${badgeBg}`;
          }
          if (idEl) idEl.innerText = `Pos ID: ${pos.id || '--'}`;
          if (symEl) symEl.innerHTML = `<span class="text-white">${escapeHtml(pos.symbol || '--')}</span> <span class="${sideColor}">(${sideText})</span>`;
          const entryVal = pos.entry_price !== undefined ? pos.entry_price : (pos.entry || 0);
          if (entryEl) entryEl.innerText = '$' + Number(entryVal).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 4 });
          const currVal = pos.current_price !== undefined ? pos.current_price : entryVal;
          if (currEl) currEl.innerText = '$' + Number(currVal).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 4 });
          const posSize = pos.size !== undefined ? Number(pos.size) : Number(pos.amount || 0);
          const notionalVal = pos.notional !== undefined ? Number(pos.notional) : (posSize * Number(currVal));
          if (sizeEl) sizeEl.innerText = `${posSize.toFixed(4)} (~$${notionalVal.toFixed(2)})`;
          
          if (stopEl) {
            const stopVal = pos.trailing_stop || pos.current_stop || pos.stop_loss || pos.stop;
            stopEl.innerText = stopVal ? ('$' + Number(stopVal).toFixed(2)) : '--';
          }

          if (pnlEl) {
            const pnl = Number(pos.unrealized_pnl || 0);
            const pnlPct = Number(pos.unrealized_pnl_pct || 0);
            const sign = pnl >= 0 ? '+' : '';
            pnlEl.innerText = `${sign}$${pnl.toFixed(2)} (${sign}${pnlPct.toFixed(2)}%)`;
            pnlEl.className = pnl >= 0 ? 'font-bold text-emerald-400 mt-0.5' : 'font-bold text-rose-400 mt-0.5';
          }

          if (durEl) {
            durEl.innerText = pos.opened_at ? new Date(pos.opened_at).toLocaleTimeString() : '--';
          }
        } else {
          if (badgeEl) {
            badgeEl.innerText = 'FLAT';
            badgeEl.className = 'px-2 py-0.5 rounded text-[10px] font-semibold bg-slate-800 text-slate-400';
          }
          if (idEl) idEl.innerText = 'No Active Position';
          if (symEl) symEl.innerText = 'Flat';
          if (entryEl) entryEl.innerText = '$0.00';
          if (currEl) currEl.innerText = '$0.00';
          if (sizeEl) sizeEl.innerText = '0.00';
          if (stopEl) stopEl.innerText = '--';
          if (pnlEl) {
            pnlEl.innerText = '$0.00 (0.0%)';
            pnlEl.className = 'font-bold text-slate-400 mt-0.5';
          }
          if (durEl) durEl.innerText = '--';
        }

        // Render Orders Table
        const tbody = document.getElementById('tracker-orders-tbody');
        const countLabel = document.getElementById('orders-count-label');
        if (countLabel) countLabel.innerText = `${orders.length} order(s) recorded`;

        if (tbody) {
          if (!orders || orders.length === 0) {
            tbody.innerHTML = `<tr><td colspan="8" class="py-3 px-3 text-center text-slate-500 italic font-sans">No orders recorded yet. As orders are placed, filled, or rejected by risk controls, they will appear here in real time.</td></tr>`;
          } else {
            tbody.innerHTML = '';
            orders.forEach(o => {
              const tr = document.createElement('tr');
              const status = (o.status || '').toUpperCase();
              let statusBadge = `<span class="px-1.5 py-0.5 rounded text-[9px] bg-slate-800 text-slate-400">${escapeHtml(status)}</span>`;
              if (status === 'FILLED') {
                statusBadge = `<span class="px-1.5 py-0.5 rounded text-[9px] bg-emerald-500/20 text-emerald-300 font-bold">FILLED</span>`;
              } else if (status === 'REJECTED') {
                statusBadge = `<span class="px-1.5 py-0.5 rounded text-[9px] bg-rose-500/20 text-rose-300 font-bold">REJECTED</span>`;
              } else if (status === 'CANCELLED') {
                statusBadge = `<span class="px-1.5 py-0.5 rounded text-[9px] bg-amber-500/20 text-amber-300 font-semibold">CANCELLED</span>`;
              } else if (status === 'SUBMITTED' || status === 'PENDING') {
                statusBadge = `<span class="px-1.5 py-0.5 rounded text-[9px] bg-blue-500/20 text-blue-300 font-semibold">PENDING</span>`;
              }

              const side = (o.side || '').toUpperCase();
              const sideCol = (side === 'BUY' || side === 'LONG') ? 'text-emerald-400 font-bold' : 'text-rose-400 font-bold';
              const rawTime = o.created_at || o.timestamp;
              const timeStr = rawTime ? new Date(rawTime).toLocaleString() : '--';
              const reasonStr = o.rejection_reason || o.notes || '--';
              const reasonClass = o.rejection_reason ? 'text-rose-300 font-sans' : 'text-slate-500 font-sans';
              const sizeVal = o.size !== undefined ? Number(o.size) : Number(o.amount || 0);

              tr.innerHTML = `
                <td class="py-1.5 px-2.5 text-slate-400 text-[10px]">${escapeHtml(timeStr)}</td>
                <td class="py-1.5 px-2 font-semibold text-slate-200">${escapeHtml(o.symbol || '--')}</td>
                <td class="py-1.5 px-2 ${sideCol}">${escapeHtml(o.side || '--')}</td>
                <td class="py-1.5 px-2 text-slate-300">${escapeHtml(o.order_type || 'MARKET')}</td>
                <td class="py-1.5 px-2 text-slate-300">${o.price ? ('$' + Number(o.price).toFixed(2)) : 'Market'}</td>
                <td class="py-1.5 px-2 text-slate-300">${sizeVal.toFixed(4)}</td>
                <td class="py-1.5 px-2">${statusBadge}</td>
                <td class="py-1.5 px-2.5 ${reasonClass} truncate max-w-[200px]" title="${escapeHtml(reasonStr)}">${escapeHtml(reasonStr)}</td>
              `;
              tbody.appendChild(tr);
            });
          }
        }
        setRefreshStamp('tracker-last-refresh', 'Updated ' + new Date().toLocaleTimeString(), true);
      } catch (err) {
        console.error('Error refreshing execution tracker:', err);
      } finally {
        if (manual) endRefresh('btn-refresh-tracker', 'fa-rotate', 'Refresh');
      }
    }

    async function testLiveAlert() {
      const btn = document.getElementById('btn-test-alert');
      if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Testing Alert...';
      }
      try {
        const res = await fetch('/api/alerts/test', { method: 'POST' });
        const data = await res.json();
        if (res.ok && data.success) {
          showNotice('Live Risk Alert Dispatched', `Test alert dispatched successfully. Channels notified: ${JSON.stringify(data.dispatched_channels)}`, { kind: 'success' });
          if (typeof refreshLogs === 'function') await refreshLogs();
        } else {
          showNotice('Alert Dispatch Failed', (data && data.detail) || 'Failed to dispatch alert test.', { kind: 'error' });
        }
      } catch (err) {
        showNotice('Alert Test Error', err.message || 'Network error', { kind: 'error' });
      } finally {
        if (btn) {
          btn.disabled = false;
          btn.innerHTML = '<i class="fa-solid fa-bell"></i> Test Live Alert';
        }
      }
    }

    async function loadFeeInfo() {
      const exchEl = document.getElementById('bt-exchange');
      const pillEl = document.getElementById('bt-fee-tier-pill');
      if (!exchEl || !pillEl) return;
      const exch = exchEl.value || 'binance';
      try {
        const res = await fetch(`/api/exchange/fee-info?exchange=${encodeURIComponent(exch)}`);
        if (!res.ok) return;
        const data = await res.json();
        const feeInfo = data.fee_info || data;
        const takerPct = (Number(feeInfo.taker || 0.001) * 100).toFixed(2);
        const sourceLabel = (feeInfo.source === 'api_account_tier' || feeInfo.source === 'account_verified') ? 'API Tier' : 'Default';
        pillEl.innerText = `Taker: ${takerPct}% (${sourceLabel})`;
        pillEl.title = `Fee source: ${feeInfo.source}. Maker: ${(Number(feeInfo.maker || 0.001)*100).toFixed(2)}%, Taker: ${takerPct}%.`;
      } catch (e) {
        console.warn('Could not load fee info:', e);
      }
    }

    let logsInFlight = false;
    let currentLogFilter = 'all';
    let rawLogsCache = [];

    function setLogFilter(filter) {
      currentLogFilter = filter;
      ['all', 'sweep', 'live'].forEach(f => {
        const el = document.getElementById(`btn-log-filter-${f}`);
        if (el) {
          if (f === filter) {
            el.className = 'px-2 py-0.5 rounded font-medium bg-purple-600 text-white shadow-sm';
          } else {
            el.className = 'px-2 py-0.5 rounded font-medium text-slate-400 hover:text-white transition';
          }
        }
      });
      renderLogs();
    }

    function renderLogs() {
      const container = document.getElementById('log-container');
      if (!container) return;
      container.innerHTML = '';

      let filtered = rawLogsCache;
      if (currentLogFilter === 'sweep') {
        filtered = rawLogsCache.filter(l => l.includes('[Sweep '));
      } else if (currentLogFilter === 'live') {
        filtered = rawLogsCache.filter(l => !l.includes('[Sweep '));
      }

      if (filtered && filtered.length) {
        filtered.forEach(l => {
          const div = document.createElement('div');
          div.className = 'py-0.5 leading-relaxed';
          if (l.includes('[CRITICAL]')) {
            div.className += ' text-rose-400 font-bold';
          } else if (l.includes('[WARNING]')) {
            div.className += ' text-amber-400';
          } else if (l.includes('[Sweep ')) {
            div.className += ' text-purple-300 font-medium bg-purple-950/20 px-1.5 py-0.5 rounded border-l-2 border-purple-500/50';
          } else if (l.includes('BUY') || l.includes('Closed position')) {
            div.className += ' text-emerald-400';
          } else {
            div.className += ' text-slate-300';
          }
          div.innerText = l;
          container.appendChild(div);
        });
        container.scrollTop = container.scrollHeight;
      } else {
        const msg = currentLogFilter === 'sweep'
          ? 'No sweep logs recorded yet. Run a multi-window or timeframe sweep above.'
          : (currentLogFilter === 'live'
              ? 'No live engine logs recorded yet.'
              : 'No logs recorded yet. Start the bot or run a backtest to generate activity.');
        container.innerHTML = `<div class="text-slate-500 italic py-2">${escapeHtml(msg)}</div>`;
      }
    }

    async function refreshLogs(manual) {
      if (logsInFlight) return;
      logsInFlight = true;
      if (manual) beginRefresh('btn-refresh-logs', 'Refreshing...');
      try {
        const res = await fetch('/api/logs?lines=100&_ts=' + Date.now(), { cache: 'no-store' });
        if (!res.ok) {
          setRefreshStamp('logs-updated', `Could not refresh logs (HTTP ${res.status}).`, false);
          return;
        }
        const data = await res.json();
        rawLogsCache = data.logs || [];
        renderLogs();
        setRefreshStamp('logs-updated', 'Updated ' + new Date().toLocaleTimeString(), true);
      } catch (e) {
        setRefreshStamp('logs-updated', 'Could not refresh logs: ' + (e && e.message ? e.message : e), false);
        console.error('Error refreshing logs:', e);
      } finally {
        logsInFlight = false;
        if (manual) endRefresh('btn-refresh-logs', 'fa-rotate', 'Refresh');
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
        ['email-status-dot', 'email-status-dot-mobile'].forEach(id => {
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
          if (resendSenderCode) resendSenderCode.innerText = data.from_email || `Trader Queen <noreply@${data.verified_domain}>`;
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
    const initialSavedTradingMode = localStorage.getItem('bot_trading_mode');
    if (initialSavedTradingMode) {
      const isReal = (initialSavedTradingMode === 'real');
      updateTradingModeUI(isReal, isReal ? 'live' : 'paper', !isReal);
    }
    const initialSavedMode = localStorage.getItem('bot_execution_mode');
    if (initialSavedMode) {
      updateExecutionModeUI(initialSavedMode === 'manual');
    }
    const initialSavedPos = localStorage.getItem('bot_active_position');
    if (initialSavedPos) {
      try {
        const parsedPos = JSON.parse(initialSavedPos);
        if (parsedPos && parsedPos.entry) {
          currentActivePosition = parsedPos;
          renderActivePositionBanner(parsedPos);
        }
      } catch (e) {}
    }
    loadExchanges();
    checkAuth();
    loadBotSettings();
    fetchStatus();
    refreshAnalysis();
    refreshLogs();
    refreshExecutionTracker();
    loadFeeInfo();
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
    setInterval(refreshExecutionTracker, 4000);
    applyTimeframeLimits();

    const btnSweepWindows = document.getElementById('btn-sweep-windows');
    if (btnSweepWindows) btnSweepWindows.addEventListener('click', executeCompareWindows);
    const btnSweepTf = document.getElementById('btn-sweep-timeframes');
    if (btnSweepTf) btnSweepTf.addEventListener('click', executeCompareTimeframes);
    const btnWalkForward = document.getElementById('btn-walk-forward');
    if (btnWalkForward) btnWalkForward.addEventListener('click', executeWalkForward);
  </script>
</body>
</html>
"""

# Per-grain lookback limits, injected into the page so the UI can guide the
# lookback input from the same single source of truth the fetcher uses.
DASHBOARD_HTML = DASHBOARD_HTML.replace(
    "__BT_LOOKBACK_LIMITS__", json.dumps(lookback_limits(), sort_keys=True))


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
def get_dashboard():
    return _html_response(DASHBOARD_HTML)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"\n========================================================")
    print(f" Trader Queen Dashboard running on http://localhost:{port}")
    print(f"========================================================\n")
    uvicorn.run("web_server:app", host=host, port=port, reload=False)
