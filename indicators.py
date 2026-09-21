"""
Technical analysis primitives. Pure pandas/numpy — no extra TA dependency,
so behavior is transparent and easy to unit test.
"""
import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    period = max(1, int(period))
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    period = max(1, int(period))
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    all_gains = (avg_loss == 0) & (avg_gain > 0)
    flat = (avg_loss == 0) & (avg_gain == 0)
    rsi_series = rsi_series.mask(all_gains, 100.0).mask(flat, 50.0)
    return rsi_series.fillna(50)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — used for volatility-scaled stops and position sizing."""
    period = max(1, int(period))
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index — used as a trend-strength filter (avoid chop)."""
    period = max(1, int(period))
    high, low = df["high"], df["low"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = atr(df, period)
    tr_safe = tr.replace(0, np.nan)
    plus_di = (100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / tr_safe).fillna(0)
    minus_di = (100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / tr_safe).fillna(0)

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = (((plus_di - minus_di).abs() / di_sum) * 100).fillna(0)
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0)


def fib_retracement_levels(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    """
    Rolling Fibonacci retracement levels — fully automatic, no manual swing
    picking. For each bar, the "swing" is the highest high and lowest low
    over the trailing `lookback` bars; retracement levels are measured down
    from that swing high, which is the standard convention for pullback
    support in an uptrend.

    Returns a DataFrame with swing_high, swing_low, and the 38.2/50/61.8%
    levels. Values are NaN until `lookback` bars of history exist.
    """
    swing_high = df["high"].rolling(lookback).max()
    swing_low = df["low"].rolling(lookback).min()
    span = swing_high - swing_low
    return pd.DataFrame({
        "swing_high": swing_high,
        "swing_low": swing_low,
        "fib_382": swing_high - span * 0.382,
        "fib_500": swing_high - span * 0.5,
        "fib_618": swing_high - span * 0.618,
    })


def bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands volatility bands (middle SMA, upper, lower)."""
    middle = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return pd.DataFrame({
        "bb_mid": middle.bfill().ffill(),
        "bb_upper": upper.bfill().ffill(),
        "bb_lower": lower.bfill().ffill(),
    })


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal_period: int = 9) -> pd.DataFrame:
    """Moving Average Convergence Divergence (MACD line, signal line, histogram)."""
    fast_line = ema(series, fast)
    slow_line = ema(series, slow)
    macd_line = fast_line - slow_line
    sig_line = ema(macd_line, signal_period)
    hist = macd_line - sig_line
    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": sig_line,
        "macd_hist": hist,
    })

