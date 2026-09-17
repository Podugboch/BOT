"""
Configurable multi-strategy engine with technical indicators:
1. EMA Cross + Fibonacci Pullback + RSI/ADX (Default confluence edge)
2. Pure Trend Momentum (Fast/Slow EMA Cross)
3. RSI Overbought / Oversold Mean Reversion
4. MACD Signal Line Crossover
5. Bollinger Bands Volatility Breakout
"""
from enum import Enum
from typing import List, Dict, Any

import pandas as pd

from config import StrategyConfig
from indicators import adx, atr, ema, fib_retracement_levels, rsi, bollinger_bands, macd


class Signal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


STRATEGIES: List[Dict[str, Any]] = [
    {
        "id": "ema_cross_fib",
        "name": "EMA Cross + Fib Retracement + RSI/ADX",
        "short_name": "EMA + Fib Pullback",
        "description": "Trend-following EMA 12/26 crossover filtered by ADX trend strength, RSI safety bounds, and automatic Fibonacci retracement pullback support.",
        "recommended": True,
    },
    {
        "id": "pure_ema_cross",
        "name": "Pure Trend Momentum (EMA Cross)",
        "short_name": "Pure EMA Cross",
        "description": "Fast/Slow EMA crossover with ADX trend filter. Triggers on trend shifts without requiring exact Fibonacci pullback confluence.",
        "recommended": False,
    },
    {
        "id": "rsi_reversion",
        "name": "RSI Oversold / Overbought Reversion",
        "short_name": "RSI Mean Reversion",
        "description": "Contrarian momentum: enters when RSI crosses up from oversold (<= 30), exits when RSI hits overbought (>= 70).",
        "recommended": False,
    },
    {
        "id": "macd_crossover",
        "name": "MACD Signal Line Crossover",
        "short_name": "MACD Cross",
        "description": "Standard momentum crossover: enters when MACD line crosses above Signal line with RSI filter; exits on bearish cross.",
        "recommended": False,
    },
    {
        "id": "bollinger_breakout",
        "name": "Bollinger Bands Volatility Breakout",
        "short_name": "Bollinger Breakout",
        "description": "Volatility expansion: enters on candle close breaking above upper Bollinger Band with trending ADX; exits below midline.",
        "recommended": False,
    },
]


def add_indicators(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = ema(df["close"], cfg.fast_ema)
    df["ema_slow"] = ema(df["close"], cfg.slow_ema)
    df["rsi"] = rsi(df["close"], cfg.rsi_period)
    df["atr"] = atr(df, cfg.adx_period)
    df["adx"] = adx(df, cfg.adx_period)
    fib = fib_retracement_levels(df, cfg.fib_lookback)
    bb = bollinger_bands(df["close"], getattr(cfg, "bb_period", 20), getattr(cfg, "bb_std", 2.0))
    m = macd(df["close"], cfg.fast_ema, cfg.slow_ema, getattr(cfg, "macd_signal_period", 9))
    return pd.concat([df, fib, bb, m], axis=1)


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
    strat = getattr(cfg, "strategy_name", "ema_cross_fib")

    # 1. Pure Trend Momentum (EMA Cross without Fib requirement)
    if strat == "pure_ema_cross":
        trending = curr["adx"] >= cfg.adx_trend_threshold
        bullish_cross = prev["ema_fast"] <= prev["ema_slow"] and curr["ema_fast"] > curr["ema_slow"]
        bearish_cross = prev["ema_fast"] >= prev["ema_slow"] and curr["ema_fast"] < curr["ema_slow"]

        if not position_open:
            if bullish_cross and trending and curr["rsi"] < cfg.rsi_overbought:
                return Signal.BUY
            return Signal.HOLD
        if bearish_cross or curr["rsi"] > cfg.rsi_overbought + 10:
            return Signal.SELL
        return Signal.HOLD

    # 2. RSI Oversold / Overbought Mean Reversion
    elif strat == "rsi_reversion":
        prev_rsi = float(prev["rsi"]) if pd.notna(prev["rsi"]) else 50.0
        curr_rsi = float(curr["rsi"]) if pd.notna(curr["rsi"]) else 50.0

        if not position_open:
            if (prev_rsi <= 30 and curr_rsi > prev_rsi) or curr_rsi <= 28:
                return Signal.BUY
            return Signal.HOLD
        if curr_rsi >= 70:
            return Signal.SELL
        return Signal.HOLD

    # 3. MACD Signal Line Crossover
    elif strat == "macd_crossover":
        prev_macd, curr_macd = prev.get("macd", 0), curr.get("macd", 0)
        prev_sig, curr_sig = prev.get("macd_signal", 0), curr.get("macd_signal", 0)
        bullish_cross = prev_macd <= prev_sig and curr_macd > curr_sig
        bearish_cross = prev_macd >= prev_sig and curr_macd < curr_sig

        if not position_open:
            if bullish_cross and curr["rsi"] < cfg.rsi_overbought:
                return Signal.BUY
            return Signal.HOLD
        if bearish_cross:
            return Signal.SELL
        return Signal.HOLD

    # 4. Bollinger Bands Volatility Breakout
    elif strat == "bollinger_breakout":
        upper = curr.get("bb_upper", curr["close"])
        mid = curr.get("bb_mid", curr["close"])
        trending = curr["adx"] >= cfg.adx_trend_threshold

        if not position_open:
            if curr["close"] > upper and trending and curr["rsi"] < 75:
                return Signal.BUY
            return Signal.HOLD
        if curr["close"] < mid or curr["rsi"] > 80:
            return Signal.SELL
        return Signal.HOLD

    # Default / "ema_cross_fib": Trend-following EMA Cross + RSI + ADX + Fib Retracement
    else:
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

