"""
Central configuration for the bot. All tunable knobs live here so nothing
is buried inside strategy/risk/execution logic.

Secrets (API keys) are read from environment variables — never hardcode them.
Copy .env.example to .env and fill it in, or export the vars directly.
"""
import os
import socket
from dataclasses import dataclass, field
from secret_manager import safe_load_dotenv

# Safely load non-sensitive environment variables from .env
safe_load_dotenv()

# Fallback DNS resolution in case host/ISP DNS blocks crypto exchange domains
try:
    import dns.resolver
    _orig_getaddrinfo = socket.getaddrinfo
    _resolver = dns.resolver.Resolver()
    _resolver.nameservers = ["8.8.8.8", "1.1.1.1"]
    _resolver.timeout = 2.0
    _resolver.lifetime = 4.0

    def _fallback_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        try:
            return _orig_getaddrinfo(host, port, family, type, proto, flags)
        except (socket.gaierror, socket.herror) as orig_err:
            try:
                ip = _resolver.resolve(host, "A")[0].to_text()
                return _orig_getaddrinfo(ip, port, family, type, proto, flags)
            except Exception:
                raise orig_err

    socket.getaddrinfo = _fallback_getaddrinfo
except ImportError:
    pass


@dataclass
class StrategyConfig:
    """Configurable trading strategy parameters."""
    strategy_name: str = field(default_factory=lambda: os.getenv("STRATEGY", "ema_cross_fib"))
    fast_ema: int = 12
    slow_ema: int = 26
    rsi_period: int = 14
    rsi_overbought: float = 65     # don't buy into overbought / exit if too hot
    adx_period: int = 14
    adx_trend_threshold: float = 20  # only trade when a real trend exists

    # Fibonacci retracement confluence filter — computed automatically from
    # rolling swing highs/lows, no manual line-drawing required.
    use_fib_filter: bool = True
    fib_lookback: int = 50          # bars used to find the swing high/low
    fib_tolerance_pct: float = 0.005  # price counts as "at" a level within 0.5%

    # Volatility & Momentum indicator parameters
    bb_period: int = 20
    bb_std: float = 2.0
    macd_signal_period: int = 9


@dataclass
class RiskConfig:
    """Risk management: sized before profit, always."""
    risk_per_trade_pct: float = 0.01     # risk 1% of equity per trade
    stop_loss_atr_mult: float = 2.0      # stop = entry -/+ 2x ATR
    take_profit_atr_mult: float = 4.0    # 2:1 reward:risk
    max_daily_loss_pct: float = 0.03     # circuit breaker: halt for the day
    max_drawdown_pct: float = 0.15       # kill switch: halt the bot entirely
    max_consecutive_losses: int = 4      # circuit breaker: cool off


def _get_exchange_credential(env_key: str, db_key: str) -> str:
    val = os.getenv(env_key, "").strip()
    if val:
        return val
    try:
        from auth import AuthService
        latest = AuthService.get_latest_raw_exchange_key()
        if latest and latest.get(db_key):
            return str(latest[db_key]).strip()
    except Exception:
        pass
    return ""


def _get_exchange_id() -> str:
    val = os.getenv("EXCHANGE_ID", "").strip()
    if val:
        return val
    try:
        from auth import AuthService
        latest = AuthService.get_latest_raw_exchange_key()
        if latest and latest.get("exchange_id"):
            return str(latest["exchange_id"]).strip()
    except Exception:
        pass
    return "binance"


@dataclass
class ExchangeConfig:
    # Any ccxt-supported exchange id works here — see the "Supported exchanges"
    # list in README.md for the popular ones. Override via env var or secure vault.
    exchange_id: str = field(default_factory=_get_exchange_id)
    symbol: str = field(default_factory=lambda: os.getenv("SYMBOL", "BTC/USDT"))
    timeframe: str = field(default_factory=lambda: os.getenv("TIMEFRAME", "15m"))
    testnet: bool = field(default_factory=lambda: os.getenv("TESTNET", "true").lower() == "true")
    api_key: str = field(default_factory=lambda: _get_exchange_credential("EXCHANGE_API_KEY", "api_key"))
    api_secret: str = field(default_factory=lambda: _get_exchange_credential("EXCHANGE_API_SECRET", "api_secret"))
    # Some exchanges (Coinbase, KuCoin, OKX, Bitget...) also require a passphrase
    api_passphrase: str = field(default_factory=lambda: _get_exchange_credential("EXCHANGE_API_PASSPHRASE", "api_passphrase"))


@dataclass
class BotConfig:
    mode: str = field(default_factory=lambda: os.getenv("MODE", "paper"))  # "paper" | "live" | "backtest"
    execution_mode: str = field(default_factory=lambda: os.getenv("EXECUTION_MODE", "auto"))  # "auto" | "manual"
    starting_equity: float = 1000.0
    taker_fee_pct: float = 0.001       # 0.10% per fill — set to your exchange's real fee
    slippage_pct: float = 0.0005       # 0.05% assumed slippage per fill
    poll_interval_sec: int = 60
    kill_switch_file: str = field(default_factory=lambda: "/tmp/KILL" if (os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME")) else "KILL")

    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)

