"""
Central configuration for the bot. All tunable knobs live here so nothing
is buried inside strategy/risk/execution logic.

Secrets (API keys) are read from environment variables — never hardcode them.
Copy .env.example to .env and fill it in, or export the vars directly.
"""
import os
import socket
from dataclasses import dataclass, field
from typing import Optional
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
    # Price counts as "at" a level inside this band. 1% was measured accepting
    # essentially every core signal — price had to miss the level by more than a
    # full percent before the filter said no — so the floor is 0.3% and price
    # has to actually test the retracement line.
    fib_tolerance_pct: float = 0.003
    # Dynamic band: instead of a rigid percentage the tolerance scales with
    # volatility — a bar whose close is within `fib_tolerance_atr_mult * ATR`
    # of a level (0.5 * ATR by default) counts as a pullback. The percentage
    # above stays as the floor (and the fallback when ATR is missing), so a calm
    # tape cannot starve the strategy down to a handful of trades. 0 = fixed band.
    fib_tolerance_atr_mult: float = 0.5
    # Which retracement levels an entry has to test, and therefore what the
    # filter is really doing:
    #   "deep" — within the band of the 0.5 or 0.618 level (the default). These
    #            are the reversal levels a pullback actually turns on, and a
    #            test of one of them is a tight, checkable condition.
    #   "zone" — anywhere inside the 0.382-0.618 pocket. Measured against real
    #            signals this accepted ~96% of them: the pocket spans a big part
    #            of the swing range, and with the band on top it covers almost
    #            the whole price spread, so it filters almost nothing.
    #   "all"  — within the band of any of 0.382/0.5/0.618 (loosest).
    fib_level_mode: str = "deep"
    fib_touch_window: int = 3       # bars back that may hold the pullback touch
    # Bars back that may hold the EMA crossover: 1 reproduces the old strict
    # "cross on this candle" rule, 3 lets a fresh trend stay entry-eligible
    # while price works its way back into the fib zone.
    ema_cross_window: int = 3

    # Entry debounce: candles that must separate the exit of one trade from the
    # entry of the next. A fast EMA whipsaw used to chain entries on adjacent
    # bars (17:30, 17:35, 17:40 — each one a "fresh" crossover), which is a
    # single setup charged three times in fees and stop-outs. Set an int to pin
    # it, or leave None to derive it from the timeframe (8 bars on 5m/15m, 3 on
    # 1h/4h — see `default_min_bars_between_trades`), so the gap means roughly
    # the same amount of time on a fast grain as on a slow one. 0 disables it.
    min_bars_between_trades: Optional[int] = None

    # Volatility & Momentum indicator parameters
    bb_period: int = 20
    bb_std: float = 2.0
    macd_signal_period: int = 9


# --- entry debounce defaults ------------------------------------------------
# Bars that must pass between the exit of one trade and the next entry. Three
# bars is 15 minutes on 5m but three hours on 1h, so the count is keyed to the
# grain: fast charts get a longer count to buy the same amount of cool-off.
DEFAULT_MIN_BARS_BETWEEN_TRADES = {
    "1m": 8, "3m": 8, "5m": 8, "15m": 8,          # <= 15m: 8 bars
    "30m": 3, "1h": 3, "2h": 3, "4h": 3,          # >= 30m: 3 bars
    "6h": 3, "12h": 3, "1d": 3, "1w": 3,
}
FALLBACK_MIN_BARS_BETWEEN_TRADES = 3

# Bar length per grain, in minutes. Used to translate a debounce *count* into
# elapsed market time (live bot) and back again.
TIMEFRAME_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120,
    "4h": 240, "6h": 360, "12h": 720, "1d": 1440, "1w": 10080,
}


def timeframe_minutes(timeframe: str) -> float:
    """Bar length in minutes; 0.0 for a grain we do not know."""
    return float(TIMEFRAME_MINUTES.get(str(timeframe or "").strip().lower(), 0))


def default_min_bars_between_trades(timeframe: str) -> int:
    """Debounce length for a grain: 8 bars at/under 15m, 3 bars at/over 30m."""
    key = str(timeframe or "").strip().lower()
    return DEFAULT_MIN_BARS_BETWEEN_TRADES.get(key, FALLBACK_MIN_BARS_BETWEEN_TRADES)


def resolve_min_bars_between_trades(strategy_cfg, timeframe: str = None) -> int:
    """Effective debounce length: an explicit override, else the grain default."""
    raw = getattr(strategy_cfg, "min_bars_between_trades", None)
    if raw is None:
        return default_min_bars_between_trades(timeframe)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return default_min_bars_between_trades(timeframe)


@dataclass
class RiskConfig:
    """Risk management: sized before profit, always."""
    risk_per_trade_pct: float = 0.01     # risk 1% of equity per trade
    # Asymmetrical expectancy: a 30-45% win rate only pays if each winner is
    # worth several times each loser. Stop tight (1.5 ATR), target far (4 ATR =
    # 2.7R), and never take a setup below `min_reward_risk_ratio` of risk.
    stop_loss_atr_mult: float = 1.5      # stop = entry -/+ 1.5x ATR
    take_profit_atr_mult: float = 4.0    # target = 2.7R at a 1.5 ATR stop
    min_reward_risk_ratio: float = 2.0   # never accept a setup worse than 1:2
    # Let winners run instead of banking them on the first wobble: once the
    # trade is +1.5 ATR in profit the stop moves to break-even, then trails
    # `trailing_stop_atr_mult` ATR behind the highest high since entry (never
    # back into a loss). Arming late and trailing wide is the point: a trail that
    # starts tightening at +1 ATR stops ordinary noise out of a trade that was
    # working. Turn it off to fall back to the fixed ATR stop with
    # reversal-signal exits.
    trailing_stop_enabled: bool = True
    trailing_arm_atr_mult: float = 1.5   # only trail once +1.5 ATR in profit
    trailing_stop_atr_mult: float = 3.5  # wide enough to survive normal noise
    max_daily_loss_pct: float = 0.03     # circuit breaker: halt for the day
    max_drawdown_pct: float = 0.15       # kill switch: halt the bot entirely
    # A 4-loss streak is ordinary in trend trading and used to starve the
    # strategy, so the streak limit is 8 and tripping it starts a rolling
    # cooldown that lifts on its own instead of halting forever.
    max_consecutive_losses: int = 8
    loss_streak_cooldown_hours: float = 24.0   # 0 = block until a winning trade


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
