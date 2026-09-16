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

from fastapi import FastAPI, Header, HTTPException, Query, Response
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn
import pandas as pd
import numpy as np
import ccxt

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from config import BotConfig, StrategyConfig, RiskConfig, ExchangeConfig
from exchange_client import ExchangeClient
from indicators import ema, rsi, atr, adx, fib_retracement_levels
from strategy import Signal, add_indicators, generate_signal, _near_fib_support
from risk_manager import RiskManager
from backtester import fetch_historical, run_backtest, summarize
from bot import TradingBot
from auth import AuthService, init_auth_db

logger = logging.getLogger("web")
init_auth_db()

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

        def _run_loop():
            logger.info("Bot thread started.")
            try:
                while self.running:
                    step_ok = self.bot.step()
                    self.last_step_time = datetime.now(timezone.utc).isoformat()
                    if not step_ok:
                        logger.warning("Bot step returned False — halting runner loop.")
                        self.running = False
                        break
                    for _ in range(self.cfg.poll_interval_sec):
                        if not self.running or self.bot.kill_switch_active():
                            break
                        time.sleep(1)
                    if self.bot.kill_switch_active():
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
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        return True

    def trigger_kill(self):
        with open(self.cfg.kill_switch_file, "w") as f:
            f.write("HALT")
        self.stop()
        return True


runner = BotRunner()


# --- Pydantic Request Models ---
class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    username_or_email: str
    password: str


class ForgotPasswordRequest(BaseModel):
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


class BacktestRequest(BaseModel):
    exchange: str = "binance"
    symbol: str = "BTC/USDT"
    timeframe: str = "15m"
    days: int = 30
    starting_equity: float = 1000.0
    use_fib_filter: bool = True
    risk_per_trade_pct: float = 0.01


# --- Auth Dependency Helper ---
def get_current_user(authorization: Optional[str] = Header(None)) -> Optional[Dict[str, Any]]:
    if not authorization:
        return None
    token = authorization.replace("Bearer ", "").strip()
    return AuthService.get_user_from_token(token)


# --- ZIP Packaging Helper ---
def create_bot_zip() -> str:
    zip_path = Path(__file__).resolve().parent / "BOT.zip"
    exclude_dirs = {".git", ".pytest_cache", "__pycache__", ".venv", "data_cache", "logs", ".vercel"}
    exclude_files = {"users.db", "bot.log", "web.log", "KILL", "BOT.zip"}

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(Path(__file__).resolve().parent):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for file in files:
                if file in exclude_files or file.endswith(".pyc"):
                    continue
                file_path = Path(root) / file
                arcname = file_path.relative_to(Path(__file__).resolve().parent)
                # Avoid nesting redundant copies
                if "crypto_bot" in arcname.parts and len(arcname.parts) > 1:
                    continue
                zf.write(file_path, arcname)
    return str(zip_path)


# ============================================================================
# API ENDPOINTS: AUTHENTICATION & USER MANAGEMENT
# ============================================================================
@app.post("/api/auth/register")
def register(req: RegisterRequest):
    res = AuthService.register(req.username, req.email, req.password)
    if not res["success"]:
        raise HTTPException(status_code=400, detail=res["message"])
    return res


@app.get("/api/auth/activate")
@app.post("/api/auth/activate")
def activate_account(token: str = Query(...)):
    res = AuthService.activate_account(token)
    if not res["success"]:
        raise HTTPException(status_code=400, detail=res["message"])
    return res


@app.post("/api/auth/login")
def login(req: LoginRequest):
    res = AuthService.login(req.username_or_email, req.password)
    if not res["success"]:
        raise HTTPException(status_code=400, detail=res["message"])
    return res


@app.get("/api/auth/me")
def get_profile(authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    if not user:
        return {"authenticated": False}
    keys = AuthService.get_exchange_keys(user["id"])
    return {"authenticated": True, "user": user, "saved_exchanges": keys}


@app.post("/api/auth/forgot-password")
def forgot_password(req: ForgotPasswordRequest):
    res = AuthService.request_password_reset(req.email)
    return res


@app.post("/api/auth/reset-password")
def reset_password(req: ResetPasswordRequest):
    res = AuthService.reset_password(req.token, req.new_password)
    if not res["success"]:
        raise HTTPException(status_code=400, detail=res["message"])
    return res


@app.get("/api/auth/latest-email")
def get_latest_email(email: Optional[str] = None):
    """Convenience endpoint to retrieve the activation/reset link for testing."""
    preview = AuthService.get_latest_email_preview(email)
    if not preview:
        return {"found": False, "message": "No emails generated yet"}
    return {"found": True, "email": preview}


# ============================================================================
# API ENDPOINTS: EXCHANGE API KEY MANAGEMENT
# ============================================================================
@app.get("/api/exchanges")
def get_exchanges():
    return {"exchanges": EXCHANGES_META}


@app.post("/api/exchange/test")
def test_exchange(req: TestExchangeRequest):
    """Test connection and credential validity using CCXT."""
    if not hasattr(ccxt, req.exchange_id):
        raise HTTPException(status_code=400, detail=f"Unsupported exchange '{req.exchange_id}'")

    exchange_class = getattr(ccxt, req.exchange_id)
    opts = {
        "apiKey": req.api_key.strip(),
        "secret": req.api_secret.strip(),
        "enableRateLimit": True,
        "timeout": 8000,
    }
    if req.api_passphrase:
        opts["password"] = req.api_passphrase.strip()

    exchange = exchange_class(opts)
    if req.testnet and hasattr(exchange, "set_sandbox_mode"):
        try:
            exchange.set_sandbox_mode(True)
        except Exception:
            pass

    try:
        # Check balance to verify trade/read credentials
        balance = exchange.fetch_balance()
        free_balances = {k: v for k, v in balance.get("free", {}).items() if v and float(v) > 0}
        return {
            "success": True,
            "message": f"Successfully authenticated with {req.exchange_id}! Credentials are valid.",
            "testnet": req.testnet,
            "balances": free_balances or {"INFO": "Connected successfully (0 balance found)"}
        }
    except ccxt.AuthenticationError as e:
        return {"success": False, "message": f"Authentication Error: {str(e)}"}
    except Exception as e:
        # Some exchanges don't allow balance fetch on testnet without special endpoints
        return {"success": False, "message": f"Connection check failed: {str(e)}"}


@app.post("/api/exchange/save")
def save_exchange_credentials(req: SaveExchangeKeyRequest, authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    # If authenticated, store in DB per user
    if user:
        AuthService.save_exchange_key(
            user["id"], req.exchange_id, req.api_key, req.api_secret, req.api_passphrase or "", req.testnet
        )

    # Also update .env and active BotConfig so the live bot can use it immediately
    env_file = Path(".env")
    lines = []
    if env_file.exists():
        with open(env_file, "r") as f:
            lines = f.readlines()

    updates = {
        "EXCHANGE_ID": req.exchange_id,
        "EXCHANGE_API_KEY": req.api_key,
        "EXCHANGE_API_SECRET": req.api_secret,
        "EXCHANGE_API_PASSPHRASE": req.api_passphrase or "",
        "TESTNET": "true" if req.testnet else "false",
    }

    new_lines = []
    written = set()
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

    with open(env_file, "w") as f:
        f.writelines(new_lines)

    # Refresh runner exchange config
    runner.cfg = BotConfig()
    return {
        "success": True,
        "message": f"API keys for {req.exchange_id} saved and applied to active configuration.",
        "exchange_id": req.exchange_id,
        "testnet": req.testnet,
    }


# ============================================================================
# API ENDPOINTS: CODE DOWNLOAD
# ============================================================================
@app.get("/api/download/bot")
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

    return {
        "running": runner.running,
        "mode": cfg.mode,
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


@app.post("/api/bot/start")
def start_bot():
    if runner.running:
        return {"success": False, "message": "Bot is already running"}
    started = runner.start()
    return {"success": started, "message": "Bot started in " + runner.cfg.mode + " mode"}


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


@app.get("/api/analysis")
def get_analysis(symbol: Optional[str] = None, timeframe: Optional[str] = None, exchange: Optional[str] = None):
    req_symbol = symbol or runner.cfg.exchange.symbol
    req_timeframe = timeframe or runner.cfg.exchange.timeframe
    req_exchange = exchange or runner.cfg.exchange.exchange_id

    try:
        ex_cfg = ExchangeConfig(exchange_id=req_exchange, symbol=req_symbol, timeframe=req_timeframe)
        client = ExchangeClient(ex_cfg, mode="paper", fee_pct=0.001, slippage_pct=0.0005, starting_equity=1000.0)
        df = client.fetch_ohlcv(limit=100)

        if df is None or len(df) < 2:
            raise HTTPException(status_code=500, detail="Insufficient market data returned from exchange")

        strat_cfg = runner.cfg.strategy
        df = add_indicators(df, strat_cfg)
        curr = df.iloc[-1]

        price = float(curr["close"])
        ema_fast_val = float(curr["ema_fast"])
        ema_slow_val = float(curr["ema_slow"])
        rsi_val = float(curr["rsi"])
        adx_val = float(curr["adx"])
        atr_val = float(curr["atr"])
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
                "ema_fast": round(float(row["ema_fast"]), 2),
                "ema_slow": round(float(row["ema_slow"]), 2),
            })

        return {
            "symbol": req_symbol,
            "exchange": req_exchange,
            "timeframe": req_timeframe,
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
    except Exception as e:
        logger.exception("Analysis fetch failed")
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
    log_file = Path("bot.log")
    if not log_file.exists():
        return {"logs": ["No logs recorded yet. Start the bot to generate activity."]}
    try:
        with open(log_file, "r") as f:
            all_lines = f.readlines()
            return {"logs": [line.strip() for line in all_lines[-lines:]]}
    except Exception as e:
        return {"logs": [f"Error reading log file: {e}"]}


# ============================================================================
# HTML PAGES & DASHBOARD
# ============================================================================
@app.get("/activate", response_class=HTMLResponse)
def activate_page(token: str = Query(...)):
    res = AuthService.activate_account(token)
    status_icon = "fa-check-circle text-emerald-400" if res["success"] else "fa-times-circle text-rose-400"
    title = "Account Activated!" if res["success"] else "Activation Failed"

    html = f"""<!DOCTYPE html>
    <html lang="en" class="dark">
    <head>
      <meta charset="UTF-8">
      <title>{title}</title>
      <script src="https://cdn.tailwindcss.com"></script>
      <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    </head>
    <body class="bg-slate-950 text-slate-100 flex items-center justify-center min-h-screen p-4 font-sans">
      <div class="max-w-md w-full bg-slate-900 border border-slate-800 rounded-2xl p-8 text-center shadow-2xl space-y-4">
        <i class="fa-solid {status_icon} text-5xl"></i>
        <h1 class="text-2xl font-bold">{title}</h1>
        <p class="text-sm text-slate-400">{res['message']}</p>
        <div class="pt-4">
          <a href="/" class="inline-block px-6 py-2.5 bg-emerald-600 hover:bg-emerald-500 font-semibold text-white rounded-xl shadow transition">
            Proceed to Dashboard
          </a>
        </div>
      </div>
    </body>
    </html>"""
    return HTMLResponse(content=html, status_code=200)


@app.get("/reset-password", response_class=HTMLResponse)
def reset_password_page(token: str = Query(...)):
    html = f"""<!DOCTYPE html>
    <html lang="en" class="dark">
    <head>
      <meta charset="UTF-8">
      <title>Reset Your Password</title>
      <script src="https://cdn.tailwindcss.com"></script>
      <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    </head>
    <body class="bg-slate-950 text-slate-100 flex items-center justify-center min-h-screen p-4 font-sans">
      <div class="max-w-md w-full bg-slate-900 border border-slate-800 rounded-2xl p-8 shadow-2xl space-y-5">
        <div class="text-center">
          <i class="fa-solid fa-key text-4xl text-amber-400 mb-2"></i>
          <h1 class="text-2xl font-bold">Set New Password</h1>
          <p class="text-xs text-slate-400 mt-1">Please enter your new account password below.</p>
        </div>
        <form onsubmit="handleReset(event)" class="space-y-4 text-xs">
          <input type="hidden" id="reset-token" value="{token}">
          <div>
            <label class="block text-slate-400 mb-1">New Password (min 6 characters)</label>
            <input type="password" id="new-pw" required minlength="6" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white">
          </div>
          <div>
            <label class="block text-slate-400 mb-1">Confirm New Password</label>
            <input type="password" id="confirm-pw" required minlength="6" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white">
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
      </script>
    </body>
    </html>"""
    return HTMLResponse(content=html, status_code=200)


# Main Dashboard HTML with integrated Auth & Exchange Key manager
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
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
</head>
<body class="bg-darkbg text-slate-100 min-h-screen font-sans antialiased">

  <!-- Top Navigation -->
  <header class="border-b border-bordercol bg-cardbg/90 backdrop-blur sticky top-0 z-40">
    <div class="max-w-7xl mx-auto px-4 py-2.5 flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <div class="h-9 w-9 rounded-lg bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center text-emerald-400 font-bold">
          <i class="fa-solid fa-chart-line"></i>
        </div>
        <div>
          <h1 class="text-base font-bold tracking-tight text-white flex items-center gap-2">
            Crypto Trading Bot
            <span id="badge-mode" class="text-[10px] px-2 py-0.5 rounded-full font-bold bg-blue-500/20 text-blue-400 border border-blue-500/30">PAPER</span>
          </h1>
          <p class="text-[11px] text-slate-400">EMA Cross + RSI + ADX + Fib Retracement + Risk Layer</p>
        </div>
      </div>

      <!-- Quick Action Controls -->
      <div class="flex items-center space-x-2.5 text-xs">
        <!-- Download ZIP button -->
        <a href="/api/download/bot" class="px-3 py-1.5 bg-indigo-600/30 hover:bg-indigo-600 text-indigo-300 hover:text-white border border-indigo-500/40 font-semibold rounded-lg transition flex items-center gap-1.5" title="Download Complete Bot Package">
          <i class="fa-solid fa-download"></i>
          <span>Download BOT (.zip)</span>
        </a>

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

        <!-- Auth User Menu Button -->
        <div id="auth-buttons">
          <button onclick="openAuthModal('login')" class="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-bordercol font-semibold rounded-lg transition">
            Login
          </button>
        </div>
        <div id="user-badge" class="hidden items-center gap-2 bg-slate-900 border border-bordercol px-2.5 py-1 rounded-lg">
          <i class="fa-solid fa-user-circle text-emerald-400"></i>
          <span id="user-name-display" class="font-semibold text-slate-200"></span>
          <button onclick="logout()" class="text-slate-400 hover:text-rose-400 ml-1" title="Logout"><i class="fa-solid fa-arrow-right-from-bracket"></i></button>
        </div>
      </div>
    </div>
  </header>

  <!-- Main Container -->
  <main class="max-w-7xl mx-auto px-4 py-5 space-y-5">

    <!-- KPI Cards Grid -->
    <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
      <div class="bg-cardbg border border-bordercol rounded-xl p-3">
        <div class="text-slate-400 text-[11px] font-medium">Portfolio Equity</div>
        <div class="text-lg font-bold mt-0.5 text-white" id="val-equity">$1,000.00</div>
        <div class="text-[10px] text-slate-400 mt-0.5">Peak: <span id="val-peak">$1,000.00</span></div>
      </div>

      <div class="bg-cardbg border border-bordercol rounded-xl p-3">
        <div class="text-slate-400 text-[11px] font-medium">Paper Cash</div>
        <div class="text-lg font-bold mt-0.5 text-emerald-400" id="val-cash">$1,000.00</div>
        <div class="text-[10px] text-slate-400 mt-0.5">Position: <span id="val-pos">0.000000</span></div>
      </div>

      <div class="bg-cardbg border border-bordercol rounded-xl p-3">
        <div class="text-slate-400 text-[11px] font-medium">Max Drawdown</div>
        <div class="text-lg font-bold mt-0.5 text-slate-200" id="val-drawdown">0.0%</div>
        <div class="text-[10px] text-slate-400 mt-0.5">Limit: 15.0% Kill</div>
      </div>

      <div class="bg-cardbg border border-bordercol rounded-xl p-3">
        <div class="text-slate-400 text-[11px] font-medium">Daily Loss</div>
        <div class="text-lg font-bold mt-0.5 text-slate-200" id="val-daily-loss">0.0%</div>
        <div class="text-[10px] text-slate-400 mt-0.5">Limit: 3.0% Circuit</div>
      </div>

      <div class="bg-cardbg border border-bordercol rounded-xl p-3">
        <div class="text-slate-400 text-[11px] font-medium">Loss Streak</div>
        <div class="text-lg font-bold mt-0.5 text-slate-200" id="val-streak">0 / 4</div>
        <div class="text-[10px] text-slate-400 mt-0.5">Cooloff at 4</div>
      </div>

      <div class="bg-cardbg border border-bordercol rounded-xl p-3">
        <div class="text-slate-400 text-[11px] font-medium">Live Signal</div>
        <div class="text-lg font-bold mt-0.5 text-yellow-400" id="val-signal">HOLD</div>
        <div class="text-[10px] text-slate-400 mt-0.5" id="val-symbol-tag">BTC/USDT 15m</div>
      </div>
    </div>

    <!-- Active Position Banner -->
    <div id="position-banner" class="hidden bg-emerald-950/40 border border-emerald-500/40 rounded-xl p-3.5 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <div class="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></div>
        <div>
          <span class="font-bold text-emerald-400 text-xs">ACTIVE POSITION:</span>
          <span class="text-xs text-slate-300 ml-2" id="pos-details"></span>
        </div>
      </div>
      <span class="text-[10px] text-emerald-300 font-semibold px-2 py-0.5 bg-emerald-900/60 rounded">IN MARKET</span>
    </div>

    <!-- ========================================================================= -->
    <!-- EXCHANGE API KEYS CONFIGURATION SECTION                                   -->
    <!-- ========================================================================= -->
    <div class="bg-cardbg border border-bordercol rounded-xl p-5 space-y-4">
      <div class="flex items-center justify-between border-b border-bordercol pb-3">
        <div class="flex items-center gap-2">
          <i class="fa-solid fa-key text-amber-400"></i>
          <div>
            <h2 class="font-bold text-sm text-white">Exchange API Keys & Credential Management</h2>
            <p class="text-[11px] text-slate-400">Select an exchange first, then enter your API credentials with trade permissions.</p>
          </div>
        </div>
        <span class="text-[10px] px-2 py-1 bg-amber-500/10 text-amber-400 border border-amber-500/30 rounded font-medium">
          <i class="fa-solid fa-shield-halved"></i> Trade-only keys (No withdrawal)
        </span>
      </div>

      <form id="exchange-key-form" onsubmit="saveExchangeKey(event)" class="grid grid-cols-1 md:grid-cols-4 gap-4 text-xs">
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
          <input type="password" id="key-api-secret" required placeholder="Paste API Secret" class="w-full bg-slate-900 border border-bordercol rounded-lg px-3 py-2 text-white focus:border-emerald-500">
        </div>

        <!-- 4. Passphrase (Dynamic) -->
        <div id="passphrase-container">
          <label class="block text-slate-300 font-semibold mb-1" id="passphrase-label">4. Passphrase</label>
          <input type="password" id="key-api-passphrase" placeholder="Passphrase (if needed)" class="w-full bg-slate-900 border border-bordercol rounded-lg px-3 py-2 text-white focus:border-emerald-500">
        </div>

        <!-- Options & Action Buttons -->
        <div class="md:col-span-4 flex flex-wrap items-center justify-between gap-3 pt-2 border-t border-bordercol/50">
          <div class="flex items-center gap-4">
            <label class="flex items-center gap-2 cursor-pointer">
              <input type="checkbox" id="key-testnet" checked class="rounded bg-slate-900 border-bordercol text-emerald-500">
              <span class="text-slate-300 font-medium">Use Testnet / Sandbox Mode (Recommended first)</span>
            </label>
          </div>

          <div class="flex items-center gap-2">
            <button type="button" onclick="testExchangeConnection()" id="btn-test-key" class="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-bordercol font-semibold rounded-lg transition flex items-center gap-1.5">
              <i class="fa-solid fa-plug"></i> Test Connection
            </button>
            <button type="submit" id="btn-save-key" class="px-5 py-2 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold rounded-lg shadow transition flex items-center gap-1.5">
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

    <!-- Live Analysis & Backtester 2-col layout -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-5">
      <!-- Live Market Analysis (2 cols) -->
      <div class="lg:col-span-2 bg-cardbg border border-bordercol rounded-xl p-5 space-y-4">
        <div class="flex items-center justify-between border-b border-bordercol pb-3">
          <div class="flex items-center gap-2">
            <i class="fa-solid fa-radar text-emerald-400"></i>
            <h2 class="font-bold text-sm text-white">Live Market Analysis & Technical Indicators</h2>
          </div>
          <button onclick="refreshAnalysis()" class="text-xs text-slate-400 hover:text-white transition flex items-center gap-1">
            <i class="fa-solid fa-rotate"></i> Refresh
          </button>
        </div>

        <div class="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
          <div class="bg-slate-900/80 p-3 rounded-lg border border-bordercol/60">
            <div class="text-slate-400">Market Price</div>
            <div class="text-base font-bold text-white mt-0.5" id="an-price">Loading...</div>
            <div class="text-[10px] text-slate-400 mt-1" id="an-exchange">Exchange: binance</div>
          </div>

          <div class="bg-slate-900/80 p-3 rounded-lg border border-bordercol/60">
            <div class="text-slate-400">EMA Trend (12 / 26)</div>
            <div class="text-base font-bold mt-0.5" id="an-trend">--</div>
            <div class="text-[10px] text-slate-400 mt-1" id="an-emas">Fast: -- | Slow: --</div>
          </div>

          <div class="bg-slate-900/80 p-3 rounded-lg border border-bordercol/60">
            <div class="text-slate-400">RSI (14)</div>
            <div class="text-base font-bold mt-0.5" id="an-rsi">--</div>
            <div class="text-[10px] text-slate-400 mt-1" id="an-rsi-status">Neutral</div>
          </div>

          <div class="bg-slate-900/80 p-3 rounded-lg border border-bordercol/60">
            <div class="text-slate-400">ADX Trend Strength</div>
            <div class="text-base font-bold mt-0.5" id="an-adx">--</div>
            <div class="text-[10px] text-slate-400 mt-1" id="an-adx-status">Thresh: 20</div>
          </div>
        </div>

        <!-- Automatic Fibonacci Zone Breakdown -->
        <div class="bg-slate-900/80 p-3.5 rounded-lg border border-bordercol/60">
          <div class="flex items-center justify-between text-xs mb-2">
            <span class="font-semibold text-slate-300 flex items-center gap-1.5">
              <i class="fa-solid fa-layer-group text-blue-400"></i> Rolling Fibonacci Retracement Support Zone (50 bars)
            </span>
            <span id="an-fib-confluence" class="px-2 py-0.5 rounded text-[10px] font-bold bg-slate-800 text-slate-400">CHECKING</span>
          </div>
          <div class="grid grid-cols-3 md:grid-cols-5 gap-2 text-center text-xs">
            <div class="bg-slate-800/60 p-2 rounded">
              <div class="text-[10px] text-slate-400">Swing High</div>
              <div class="font-semibold text-slate-200 mt-0.5" id="fib-high">--</div>
            </div>
            <div class="bg-slate-800/60 p-2 rounded">
              <div class="text-[10px] text-slate-400">38.2% Level</div>
              <div class="font-semibold text-blue-300 mt-0.5" id="fib-382">--</div>
            </div>
            <div class="bg-slate-800/60 p-2 rounded">
              <div class="text-[10px] text-slate-400">50.0% Level</div>
              <div class="font-semibold text-purple-300 mt-0.5" id="fib-500">--</div>
            </div>
            <div class="bg-slate-800/60 p-2 rounded">
              <div class="text-[10px] text-slate-400">61.8% Level</div>
              <div class="font-semibold text-amber-300 mt-0.5" id="fib-618">--</div>
            </div>
            <div class="bg-slate-800/60 p-2 rounded">
              <div class="text-[10px] text-slate-400">Swing Low</div>
              <div class="font-semibold text-slate-200 mt-0.5" id="fib-low">--</div>
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
              <option value="binance">Binance</option>
              <option value="coinbase">Coinbase</option>
              <option value="kraken">Kraken</option>
              <option value="kucoin">KuCoin</option>
              <option value="bybit">Bybit</option>
              <option value="okx">OKX</option>
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

          <button type="submit" id="btn-run-bt" class="w-full py-2 bg-purple-600 hover:bg-purple-500 text-white font-semibold rounded-lg shadow transition flex items-center justify-center gap-2">
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

  <!-- ========================================================================= -->
  <!-- AUTH MODALS (LOGIN / REGISTER / FORGOT PASSWORD / ACTIVATION NOTICE)       -->
  <!-- ========================================================================= -->
  <div id="auth-modal" class="hidden fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
    <div class="bg-slate-900 border border-slate-800 rounded-2xl max-w-md w-full p-6 shadow-2xl space-y-4 relative">
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
          <input type="text" id="login-user" required class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white">
        </div>
        <div>
          <div class="flex justify-between mb-1">
            <label class="text-slate-400">Password</label>
            <a href="javascript:void(0)" onclick="openAuthModal('forgot')" class="text-amber-400 hover:underline">Forgot password?</a>
          </div>
          <input type="password" id="login-pass" required class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white">
        </div>
        <button type="submit" class="w-full py-2.5 bg-emerald-600 hover:bg-emerald-500 font-bold text-white rounded-lg shadow transition">
          Sign In
        </button>
        <div class="text-center text-slate-400 pt-2">
          Don't have an account? <a href="javascript:void(0)" onclick="openAuthModal('register')" class="text-emerald-400 font-semibold hover:underline">Register now</a>
        </div>
      </form>

      <!-- REGISTER FORM -->
      <form id="form-register" onsubmit="handleRegister(event)" class="hidden space-y-3 text-xs">
        <div>
          <label class="block text-slate-400 mb-1">Username</label>
          <input type="text" id="reg-user" required minlength="3" placeholder="TraderPro" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white">
        </div>
        <div>
          <label class="block text-slate-400 mb-1">Email Address</label>
          <input type="email" id="reg-email" required placeholder="trader@example.com" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white">
        </div>
        <div>
          <label class="block text-slate-400 mb-1">Password (min 6 chars)</label>
          <input type="password" id="reg-pass" required minlength="6" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white">
        </div>
        <button type="submit" class="w-full py-2.5 bg-emerald-600 hover:bg-emerald-500 font-bold text-white rounded-lg shadow transition">
          Create Account
        </button>
        <div class="text-center text-slate-400 pt-2">
          Already have an account? <a href="javascript:void(0)" onclick="openAuthModal('login')" class="text-emerald-400 font-semibold hover:underline">Log in</a>
        </div>
      </form>

      <!-- FORGOT PASSWORD FORM -->
      <form id="form-forgot" onsubmit="handleForgot(event)" class="hidden space-y-3 text-xs">
        <div>
          <label class="block text-slate-400 mb-1">Registered Email Address</label>
          <input type="email" id="forgot-email" required class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white">
        </div>
        <button type="submit" class="w-full py-2.5 bg-amber-600 hover:bg-amber-500 font-bold text-white rounded-lg shadow transition">
          Send Password Reset Email
        </button>
        <div class="text-center text-slate-400 pt-2">
          <a href="javascript:void(0)" onclick="openAuthModal('login')" class="text-slate-300 hover:underline">Back to Login</a>
        </div>
      </form>

      <!-- ACTIVATION NOTICE BOX -->
      <div id="modal-notice" class="hidden p-3 rounded-lg border text-xs space-y-2"></div>
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
        sel.innerHTML = '';
        exchangesList.forEach(ex => {
          const opt = document.createElement('option');
          opt.value = ex.id;
          opt.innerText = ex.name + (ex.needs_passphrase ? ' (Passphrase required)' : '');
          sel.appendChild(opt);
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
    }

    async function testExchangeConnection() {
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
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.success) {
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

      const token = localStorage.getItem('bot_auth_token');
      const headers = { 'Content-Type': 'application/json' };
      if (token) headers['Authorization'] = 'Bearer ' + token;

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

      if (mode === 'login') {
        document.getElementById('modal-title').innerText = 'Account Login';
        document.getElementById('modal-subtitle').innerText = 'Sign in to access your bot and exchange credentials.';
        document.getElementById('form-login').classList.remove('hidden');
      } else if (mode === 'register') {
        document.getElementById('modal-title').innerText = 'Create Account';
        document.getElementById('modal-subtitle').innerText = 'Register an account. An email activation link will be sent.';
        document.getElementById('form-register').classList.remove('hidden');
      } else if (mode === 'forgot') {
        document.getElementById('modal-title').innerText = 'Reset Password';
        document.getElementById('modal-subtitle').innerText = 'Enter your email to receive a password reset message.';
        document.getElementById('form-forgot').classList.remove('hidden');
      }
    }

    function closeAuthModal() {
      document.getElementById('auth-modal').classList.add('hidden');
    }

    async function handleLogin(e) {
      e.preventDefault();
      const user = document.getElementById('login-user').value;
      const pass = document.getElementById('login-pass').value;
      const notice = document.getElementById('modal-notice');

      try {
        const res = await fetch('/api/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username_or_email: user, password: pass })
        });
        const data = await res.json();
        if (res.ok && data.success) {
          localStorage.setItem('bot_auth_token', data.token);
          closeAuthModal();
          await checkAuth();
        } else {
          notice.classList.remove('hidden');
          notice.className = 'p-3 rounded-lg border bg-rose-950/70 border-rose-500/40 text-rose-200 text-xs';
          if (data.needs_activation) {
            notice.innerHTML = `<strong>Account Not Activated</strong><br>${data.message}<br>
              <button onclick="previewActivationEmail('${data.email}')" class="mt-2 text-xs px-2.5 py-1 bg-emerald-600 hover:bg-emerald-500 text-white rounded font-semibold">
                Click here to view & activate now
              </button>`;
          } else {
            notice.innerText = data.detail || data.message || 'Login failed';
          }
        }
      } catch (err) {
        alert('Login error: ' + err);
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
          notice.className = 'p-3.5 rounded-lg border bg-emerald-950/80 border-emerald-500/40 text-emerald-200 text-xs space-y-2';
          notice.innerHTML = `
            <strong><i class="fa-solid fa-envelope-circle-check"></i> Account Registered!</strong>
            <p>An activation email message has been sent to <strong>${email}</strong> for acceptance and activation.</p>
            <div class="p-2 bg-slate-900 rounded border border-emerald-500/30 text-[11px]">
              <span>Quick Activation Link:</span><br>
              <a href="${data.activation_link}" class="text-emerald-400 font-bold underline break-all">${data.activation_link}</a>
            </div>
            <a href="${data.activation_link}" class="inline-block mt-2 px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white font-bold rounded">
              Accept & Activate Account Now
            </a>
          `;
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
        notice.className = 'p-3 rounded-lg border bg-amber-950/80 border-amber-500/40 text-amber-200 text-xs space-y-2';
        notice.innerHTML = `
          <strong><i class="fa-solid fa-envelope"></i> ${data.message}</strong>
          ${data.reset_link ? `<div class="p-2 bg-slate-900 rounded text-[11px]">Quick Reset Link: <a href="${data.reset_link}" class="text-amber-400 underline break-all">${data.reset_link}</a></div>` : ''}
        `;
      } catch (err) {
        alert('Reset error: ' + err);
      }
    }

    async function previewActivationEmail(email) {
      try {
        const res = await fetch('/api/auth/latest-email?email=' + encodeURIComponent(email));
        const data = await res.json();
        if (data.found && data.email) {
          window.location.href = `/activate?token=${data.email.token}`;
        }
      } catch (e) {
        alert('Error fetching activation link: ' + e);
      }
    }

    async function checkAuth() {
      const token = localStorage.getItem('bot_auth_token');
      const authButtons = document.getElementById('auth-buttons');
      const userBadge = document.getElementById('user-badge');

      if (!token) {
        authButtons.classList.remove('hidden');
        userBadge.classList.add('hidden');
        return;
      }

      try {
        const res = await fetch('/api/auth/me', {
          headers: { 'Authorization': 'Bearer ' + token }
        });
        const data = await res.json();
        if (data.authenticated) {
          currentUser = data.user;
          authButtons.classList.add('hidden');
          userBadge.classList.remove('hidden');
          document.getElementById('user-name-display').innerText = currentUser.username;

          // Render saved exchange keys
          const list = document.getElementById('saved-keys-list');
          if (data.saved_exchanges && data.saved_exchanges.length) {
            list.innerHTML = '';
            data.saved_exchanges.forEach(k => {
              const div = document.createElement('div');
              div.className = 'flex items-center justify-between p-2 bg-slate-900/80 rounded border border-bordercol/80';
              div.innerHTML = `
                <div class="flex items-center gap-2">
                  <span class="font-bold text-white uppercase">${k.exchange_id}</span>
                  <span class="text-[10px] px-1.5 py-0.5 rounded ${k.testnet ? 'bg-blue-500/20 text-blue-300' : 'bg-emerald-500/20 text-emerald-300'}">${k.testnet ? 'TESTNET' : 'LIVE'}</span>
                  <span class="font-mono text-slate-400 text-[11px]">${k.api_key_masked}</span>
                </div>
                <span class="text-[10px] text-slate-500">Configured</span>
              `;
              list.appendChild(div);
            });
          }
        } else {
          localStorage.removeItem('bot_auth_token');
          authButtons.classList.remove('hidden');
          userBadge.classList.add('hidden');
        }
      } catch (e) {
        console.error('Check auth error:', e);
      }
    }

    function logout() {
      localStorage.removeItem('bot_auth_token');
      currentUser = null;
      checkAuth();
    }

    // --- Status, Analysis, Backtest & Logs ---
    async function fetchStatus() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        botRunning = data.running;

        document.getElementById('badge-mode').innerText = (data.mode + (data.testnet ? ' (TESTNET)' : '')).toUpperCase();
        document.getElementById('val-symbol-tag').innerText = `${data.symbol} ${data.timeframe}`;

        const dot = document.getElementById('status-dot');
        const text = document.getElementById('status-text');
        const btnToggleText = document.getElementById('btn-toggle-text');
        const btnToggleIcon = document.getElementById('btn-toggle-icon');
        const btnToggle = document.getElementById('btn-toggle');

        if (data.running) {
          dot.className = 'inline-block w-2 h-2 rounded-full bg-emerald-400 animate-pulse';
          text.innerText = 'RUNNING (' + data.mode.toUpperCase() + ')';
          text.className = 'font-semibold text-emerald-400';
          btnToggleText.innerText = 'Stop Bot';
          btnToggleIcon.className = 'fa-solid fa-stop';
          btnToggle.className = 'px-3.5 py-1.5 bg-amber-600 hover:bg-amber-500 text-white font-semibold rounded-lg shadow transition flex items-center gap-1.5';
        } else {
          dot.className = 'inline-block w-2 h-2 rounded-full bg-slate-500';
          text.innerText = data.kill_file_exists ? 'HALTED (KILL SWITCH)' : 'IDLE';
          text.className = data.kill_file_exists ? 'font-semibold text-red-400' : 'font-semibold text-slate-400';
          btnToggleText.innerText = 'Start Bot';
          btnToggleIcon.className = 'fa-solid fa-play';
          btnToggle.className = 'px-3.5 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold rounded-lg shadow transition flex items-center gap-1.5';
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
        const data = await res.json();

        document.getElementById('an-price').innerText = '$' + data.price.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
        document.getElementById('an-exchange').innerText = `Exchange: ${data.exchange} (${data.symbol})`;

        const trendEl = document.getElementById('an-trend');
        trendEl.innerText = data.trend;
        trendEl.className = data.trend === 'BULLISH' ? 'text-base font-bold mt-0.5 text-emerald-400' : 'text-base font-bold mt-0.5 text-rose-400';
        document.getElementById('an-emas').innerText = `Fast: $${data.ema_fast} | Slow: $${data.ema_slow}`;

        document.getElementById('an-rsi').innerText = data.rsi.toFixed(1);
        const rsiStatusEl = document.getElementById('an-rsi-status');
        rsiStatusEl.innerText = data.rsi_status;
        rsiStatusEl.className = data.rsi_status === 'OVERBOUGHT' ? 'text-[10px] mt-1 text-rose-400' : (data.rsi_status === 'OVERSOLD' ? 'text-[10px] mt-1 text-emerald-400' : 'text-[10px] mt-1 text-slate-400');

        document.getElementById('an-adx').innerText = data.adx.toFixed(1);
        const adxStatusEl = document.getElementById('an-adx-status');
        adxStatusEl.innerText = data.adx_status;
        adxStatusEl.className = data.adx >= 20 ? 'text-[10px] mt-1 text-emerald-400 font-semibold' : 'text-[10px] mt-1 text-slate-400';

        const sigEl = document.getElementById('val-signal');
        sigEl.innerText = data.signal;
        sigEl.className = data.signal === 'BUY' ? 'text-lg font-bold mt-0.5 text-emerald-400 animate-pulse' : (data.signal === 'SELL' ? 'text-lg font-bold mt-0.5 text-rose-400' : 'text-lg font-bold mt-0.5 text-yellow-400');

        document.getElementById('fib-high').innerText = data.fib.swing_high ? '$' + data.fib.swing_high.toLocaleString() : '--';
        document.getElementById('fib-382').innerText = data.fib.fib_382 ? '$' + data.fib.fib_382.toLocaleString() : '--';
        document.getElementById('fib-500').innerText = data.fib.fib_500 ? '$' + data.fib.fib_500.toLocaleString() : '--';
        document.getElementById('fib-618').innerText = data.fib.fib_618 ? '$' + data.fib.fib_618.toLocaleString() : '--';
        document.getElementById('fib-low').innerText = data.fib.swing_low ? '$' + data.fib.swing_low.toLocaleString() : '--';

        const fibBadge = document.getElementById('an-fib-confluence');
        if (data.fib.near_fib_support) {
          fibBadge.innerText = 'AT PULLBACK SUPPORT';
          fibBadge.className = 'px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30';
        } else {
          fibBadge.innerText = 'OUTSIDE SUPPORT ZONE';
          fibBadge.className = 'px-2 py-0.5 rounded text-[10px] font-bold bg-slate-800 text-slate-400';
        }

        const candleList = document.getElementById('candle-list');
        candleList.innerHTML = '';
        if (data.candles && data.candles.length) {
          const prices = data.candles.map(c => c.close);
          const minP = Math.min(...prices);
          const maxP = Math.max(...prices);
          const range = maxP - minP || 1;

          data.candles.forEach(c => {
            const heightPct = Math.max(15, Math.min(100, ((c.close - minP) / range) * 80 + 15));
            const isUp = c.close >= c.open;
            const bar = document.createElement('div');
            bar.className = 'flex flex-col items-center flex-1 group relative';
            bar.innerHTML = `
              <div class="w-full ${isUp ? 'bg-emerald-500/80' : 'bg-rose-500/80'} rounded-t" style="height: ${heightPct}px" title="${c.ts} Close: $${c.close}"></div>
              <span class="text-[7px] text-slate-500 mt-0.5 opacity-60">${c.ts}</span>
            `;
            candleList.appendChild(bar);
          });
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

    // Startup
    loadExchanges();
    checkAuth();
    fetchStatus();
    refreshAnalysis();
    refreshLogs();

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
