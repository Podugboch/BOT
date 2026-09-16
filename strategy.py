"""
The bot's edge: trend-following EMA crossover, filtered by RSI (don't chase
overbought moves), ADX (only trade when a real trend exists, skip chop), and
Fibonacci retracement (only buy pullbacks into a key support zone, not
random spots in the trend).

This is one well-defined, testable strategy — not "the bot figures it out."
Swap this file to change the edge; everything else (risk, execution,
backtesting) stays the same.
"""
from enum import Enum

import pandas as pd

from config import StrategyConfig
from indicators import adx, atr, ema, fib_retracement_levels, rsi


class Signal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


def add_indicators(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = ema(df["close"], cfg.fast_ema)
    df["ema_slow"] = ema(df["close"], cfg.slow_ema)
    df["rsi"] = rsi(df["close"], cfg.rsi_period)
    df["atr"] = atr(df, cfg.adx_period)
    df["adx"] = adx(df, cfg.adx_period)
    fib = fib_retracement_levels(df, cfg.fib_lookback)
    return pd.concat([df, fib], axis=1)


def _near_fib_support(curr: pd.Series, cfg: StrategyConfig) -> bool:
    """True if price is within tolerance of the 38.2/50/61.8% retracement
    of the recent swing — i.e. a real pullback, not a random entry point."""
    price = curr["close"]
    tolerance = price * cfg.fib_tolerance_pct
    levels = (curr["fib_382"], curr["fib_500"], curr["fib_618"])
    return any(pd.notna(level) and abs(price - level) <= tolerance for level in levels)


def generate_signal(df: pd.DataFrame, cfg: StrategyConfig, position_open: bool) -> Signal:
    """
    df must already have indicators (see add_indicators) and at least 2 rows.
    """
    if len(df) < 2:
        return Signal.HOLD

    prev, curr = df.iloc[-2], df.iloc[-1]

    trending = curr["adx"] >= cfg.adx_trend_threshold
    bullish_cross = prev["ema_fast"] <= prev["ema_slow"] and curr["ema_fast"] > curr["ema_slow"]
    bearish_cross = prev["ema_fast"] >= prev["ema_slow"] and curr["ema_fast"] < curr["ema_slow"]

    if not position_open:
        fib_ok = (not cfg.use_fib_filter) or _near_fib_support(curr, cfg)
        if bullish_cross and trending and curr["rsi"] < cfg.rsi_overbought and fib_ok:
            return Signal.BUY
        return Signal.HOLD

    # position open: exit on trend reversal or overheated RSI
    if bearish_cross or curr["rsi"] > cfg.rsi_overbought + 10:
        return Signal.SELL
    return Signal.HOLD
