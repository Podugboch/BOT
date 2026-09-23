"""
Configurable multi-strategy engine with technical indicators:
1. EMA Cross + Fibonacci Pullback + RSI/ADX (Default confluence edge)
2. Pure Trend Momentum (Fast/Slow EMA Cross)
3. RSI Overbought / Oversold Mean Reversion
4. MACD Signal Line Crossover
5. Bollinger Bands Volatility Breakout
"""
import math
from enum import Enum
from typing import List, Dict, Any, Optional

import pandas as pd

from config import StrategyConfig, resolve_min_bars_between_trades
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


FIB_LEVEL_COLUMNS = ("fib_382", "fib_500", "fib_618")
FIB_LEVEL_LABELS = {"fib_382": "0.382", "fib_500": "0.5", "fib_618": "0.618"}
# Deeper retracements only — the pullbacks that matter in an established trend.
FIB_DEEP_LEVEL_COLUMNS = ("fib_500", "fib_618")
_FIB_DEEP_ALIASES = ("deep", "deep_only", "deep-only", "0.5/0.618", "05_618", "0.5_0.618")
# The golden pocket: price has to be *between* the 0.382 and 0.618 retracements,
# not merely within a band of one of them.
FIB_ZONE_ALIASES = ("zone", "golden", "golden_zone", "golden-zone", "pocket", "0.382-0.618")
# Fallback half-band around a fib level when the config asks for 0 (or sends
# nonsense). Kept in step with `StrategyConfig.fib_tolerance_pct`: a floor of
# 1% accepted almost every core signal, which is the bug this value fixes.
DEFAULT_FIB_TOLERANCE_PCT = 0.003
DEFAULT_FIB_TOLERANCE_ATR_MULT = 0.5


def _record(diagnostics: Optional[Dict[str, Any]], **stages) -> None:
    """Attach pipeline-stage detail for callers that need to explain a HOLD."""
    if diagnostics is not None:
        diagnostics.update(stages)


def _num(value) -> str:
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "n/a"


def _core_reason(cross: bool, trending: bool, rsi_ok: bool, curr: pd.Series,
                 cfg: StrategyConfig, label: str,
                 cross_text: Optional[str] = None) -> str:
    """Stage-1 detail: which core-indicator condition is met (or is not)."""
    if not cross:
        return cross_text or f"no fresh {label} trigger on this candle"
    if not trending:
        return f"ADX {_num(curr.get('adx'))} below the {cfg.adx_trend_threshold} trend threshold"
    if not rsi_ok:
        return f"RSI {_num(curr.get('rsi'))} at/above the {cfg.rsi_overbought} overbought limit"
    return f"{label} trigger, trend strength and RSI bounds all satisfied"


def _core_miss(cross: bool, trending: bool, rsi_ok: bool, cross_key: str) -> Optional[str]:
    """Machine-readable stage-1 rejection key, or None when the stage passed."""
    if not cross:
        return cross_key
    if not trending:
        return "adx_filter"
    if not rsi_ok:
        return "rsi_overbought"
    return None


def _recent_bullish_cross(df: pd.DataFrame, window: int):
    """(crossed, bars_ago) for the newest bullish EMA cross within `window` bars.

    bars_ago == 0 means the cross happened on the current candle. Allowing a
    small window is what decouples the entry from the old "the crossover bar
    must itself touch the fib zone" rule: the trend can still be fresh a few
    bars later while price works back into the zone.
    """
    window = max(1, int(window or 1))
    rows = df.tail(window + 1)
    if len(rows) < 2 or "ema_fast" not in rows.columns or "ema_slow" not in rows.columns:
        return False, None
    fast = rows["ema_fast"].to_numpy(dtype=float)
    slow = rows["ema_slow"].to_numpy(dtype=float)
    for offset in range(1, len(rows)):
        index = len(rows) - offset
        if fast[index - 1] <= slow[index - 1] and fast[index] > slow[index]:
            return True, offset - 1
    return False, None


def _ema_cross_text(crossed: bool, bars_ago, still_bullish: bool, window: int) -> str:
    if not crossed:
        return (f"no EMA crossover in the last {window} bar(s)" if window > 1
                else "no fresh EMA crossover on this candle")
    if not still_bullish:
        return "EMA fast line has fallen back below the slow line since the cross"
    when = "on this candle" if not bars_ago else f"{bars_ago} bar(s) ago"
    return f"EMA bullish crossover {when} (trend still active)"


def _ema_cross_window(cfg: StrategyConfig) -> int:
    """Bars back that may still hold the EMA crossover (1 == strict)."""
    try:
        window = int(getattr(cfg, "ema_cross_window", 1) or 1)
    except (TypeError, ValueError):
        return 1
    return max(1, window)


def _fib_tolerance_pct(cfg: StrategyConfig) -> float:
    """Zero/negative/missing tolerance would degrade the filter into an exact
    price match (which almost never happens) — fall back to a sane band."""
    try:
        tolerance = float(getattr(cfg, "fib_tolerance_pct", DEFAULT_FIB_TOLERANCE_PCT))
    except (TypeError, ValueError):
        return DEFAULT_FIB_TOLERANCE_PCT
    return tolerance if tolerance > 0 else DEFAULT_FIB_TOLERANCE_PCT


def _fib_tolerance_atr_mult(cfg: StrategyConfig) -> float:
    """ATR multiple for the dynamic band; 0 (or nonsense) disables it."""
    try:
        mult = float(getattr(cfg, "fib_tolerance_atr_mult", DEFAULT_FIB_TOLERANCE_ATR_MULT))
    except (TypeError, ValueError):
        return 0.0
    return mult if mult > 0 else 0.0


def _fib_mode(cfg: StrategyConfig) -> str:
    """Normalised fib mode: 'zone', 'deep' or 'all'."""
    mode = str(getattr(cfg, "fib_level_mode", "deep") or "deep").strip().lower()
    if mode in FIB_ZONE_ALIASES:
        return "zone"
    return "deep" if mode in _FIB_DEEP_ALIASES else "all"


def _fib_level_columns(cfg: StrategyConfig) -> tuple:
    """Which retracement columns count as confluence for this config."""
    return FIB_DEEP_LEVEL_COLUMNS if _fib_mode(cfg) == "deep" else FIB_LEVEL_COLUMNS


def _fib_band_pct(row: pd.Series, cfg: StrategyConfig) -> float:
    """Effective half-band around a fib level, as a fraction of price.

    The band scales with volatility — `fib_tolerance_atr_mult * ATR / price` —
    instead of being a rigid percentage, so a 1h candle with a 1.5% ATR is not
    judged by a calm-market yardstick. It is *floored* at `fib_tolerance_pct`
    (and falls back to it whenever ATR is missing, NaN or zero) so a quiet tape
    can never shrink the band toward an impossible exact-price match.
    """
    band = _fib_tolerance_pct(cfg)
    mult = _fib_tolerance_atr_mult(cfg)
    if mult <= 0:
        return band
    try:
        price = float(row["close"])
        atr_value = float(row["atr"])
    except (KeyError, TypeError, ValueError):
        return band
    if not (math.isfinite(price) and math.isfinite(atr_value)) or price <= 0 or atr_value <= 0:
        return band
    return max(band, mult * atr_value / price)


def _fib_band_text(band: float, cfg: StrategyConfig) -> str:
    """Human note for the run report: say when the band is ATR-driven."""
    mult = _fib_tolerance_atr_mult(cfg)
    if mult > 0 and band > _fib_tolerance_pct(cfg) + 1e-12:
        return f"band \u00b1{band * 100:.2f}% ({mult:g}\u00d7ATR)"
    return f"band \u00b1{band * 100:.2f}%"


def _fib_zone_bounds(row: pd.Series):
    """(lower, upper) prices of the 0.382-0.618 retracement zone, or None.

    Levels are measured down from the swing high, so the 0.618 level sits below
    the 0.382 level: the zone is [fib_618, fib_382].
    """
    try:
        upper = float(row["fib_382"])
        lower = float(row["fib_618"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(upper) and math.isfinite(lower)) or lower <= 0 or upper <= 0:
        return None
    return (lower, upper) if lower <= upper else (upper, lower)


def _in_fib_zone(row: pd.Series, cfg: StrategyConfig) -> bool:
    """True when price has pulled back *into* the 0.382-0.618 zone.

    This is what stops entries at the top of the crossover candle: sitting near
    the 0.382 level is a pullback, being above it is buying the breakout. The
    ATR band still softens the two edges so a wick that just pokes out is not
    disqualified.
    """
    bounds = _fib_zone_bounds(row)
    if bounds is None:
        return False
    try:
        price = float(row["close"])
    except (KeyError, TypeError, ValueError):
        return False
    if not math.isfinite(price) or price <= 0:
        return False
    lower, upper = bounds
    band = _fib_band_pct(row, cfg) * price       # band as an absolute price
    return (lower - band) <= price <= (upper + band)


def _fib_levels_available(rows: pd.DataFrame) -> bool:
    """False while the rolling swing window has not warmed up yet (no fib data)."""
    return any(col in rows.columns and pd.notna(rows[col]).any() for col in FIB_LEVEL_COLUMNS)


def _bar_at_fib_level(row: pd.Series, cfg: StrategyConfig) -> bool:
    """True if this bar's close sits inside the percentage band around any of
    the 38.2/50/61.8% levels — a zone, never an exact-match requirement."""
    if _fib_mode(cfg) == "zone":
        return _in_fib_zone(row, cfg)
    nearest = _nearest_fib_level(row, cfg)
    return nearest is not None and nearest[2] <= _fib_band_pct(row, cfg)


def _nearest_fib_level(row: pd.Series, cfg: StrategyConfig):
    """(label, value, distance_pct) for the closest of the three fib levels."""
    try:
        price = float(row["close"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(price) or price <= 0:
        return None

    nearest = None
    for col in _fib_level_columns(cfg):
        level = row.get(col)
        if level is None:
            continue
        try:
            level = float(level)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(level):
            continue
        distance_pct = abs(price - level) / price
        if nearest is None or distance_pct < nearest[2]:
            nearest = (FIB_LEVEL_LABELS.get(col, col), level, distance_pct)
    return nearest


def _near_fib_support(curr: pd.Series, cfg: StrategyConfig) -> bool:
    """True if price is within tolerance of the levels selected by `fib_level_mode`
    (0.5/0.618 by default) — i.e. a real pullback, not a random entry point."""
    return _bar_at_fib_level(curr, cfg)


def _fib_pullback_ok(df: pd.DataFrame, cfg: StrategyConfig) -> bool:
    """Pullback confluence over the last `fib_touch_window` bars.

    A single-bar test starves the strategy: the bar that produces a fresh EMA
    cross is rarely the same bar that tags a fib level. Allowing a recent touch
    keeps the "pullback into support" meaning while letting the core signal
    fire. `fib_touch_window = 1` reproduces the original single-bar behaviour.
    """
    return _fib_pullback_detail(df, cfg)["pass"]


def _fib_pullback_detail(df: pd.DataFrame, cfg: StrategyConfig) -> Dict[str, Any]:
    """Stage-2 verdict: did price reach a fib zone, and if not, exactly why."""
    window = max(1, int(getattr(cfg, "fib_touch_window", 1) or 1))
    rows = df.tail(window)

    if not _fib_levels_available(rows):
        return {"pass": True, "level": None, "distance_pct": None,
                "reason": "no Fibonacci levels available yet (rolling swing not warmed up)"}

    if _fib_mode(cfg) == "zone":
        return _fib_zone_detail(rows, cfg)

    hits, nearest = [], None
    for _, row in rows.iterrows():
        candidate = _nearest_fib_level(row, cfg)
        if candidate is None:
            continue
        if nearest is None or candidate[2] < nearest[2]:
            nearest = candidate
        # The band is per-bar: it widens with that bar's ATR.
        band = _fib_band_pct(row, cfg)
        if candidate[2] <= band:
            hits.append((candidate, band))

    if hits:
        (label, value, distance_pct), band = min(hits, key=lambda hit: hit[0][2])
        return {"pass": True, "level": label, "distance_pct": distance_pct,
                "reason": f"price tagged the Fibonacci {label} zone "
                          f"(${value:,.2f}, {distance_pct * 100:.2f}% away, "
                          f"{_fib_band_text(band, cfg)})"}
    if nearest is None:
        return {"pass": False, "level": None, "distance_pct": None,
                "reason": "no Fibonacci levels available for this window"}

    label, value, distance_pct = nearest
    band = _fib_band_pct(rows.iloc[-1], cfg)
    return {"pass": False, "level": label, "distance_pct": distance_pct,
            "reason": f"Price outside the Fibonacci {label} zone "
                      f"(nearest level ${value:,.2f} is {distance_pct * 100:.2f}% away, "
                      f"{_fib_band_text(band, cfg)})"}


def _fib_zone_detail(rows: pd.DataFrame, cfg: StrategyConfig) -> Dict[str, Any]:
    """Verdict for golden-pocket mode: price must be inside 0.382-0.618.

    Buying *at* the crossover candle means buying above the 0.382 level — the
    breakout, not the pullback. Only a retracement into the zone counts, and the
    reason string says which side of it the price missed on.
    """
    hit = None
    last_bounds, last_price = None, None
    for _, row in rows.iterrows():
        bounds = _fib_zone_bounds(row)
        if bounds is None:
            continue
        last_bounds = bounds
        try:
            last_price = float(row["close"])
        except (KeyError, TypeError, ValueError):
            continue
        if _in_fib_zone(row, cfg):
            hit = (bounds, last_price)

    if last_bounds is None:
        return {"pass": False, "level": None, "distance_pct": None,
                "reason": "no Fibonacci levels available for this window"}

    lower, upper = last_bounds
    zone_text = f"zone ${lower:,.2f}-${upper:,.2f} (0.618-0.382)"
    if hit:
        bounds, _price = hit
        return {"pass": True, "level": "0.382-0.618", "distance_pct": None,
                "reason": f"price pulled back into the Fibonacci {zone_text}"}

    if last_price is None:
        return {"pass": False, "level": None, "distance_pct": None,
                "reason": "no Fibonacci levels available for this window"}
    side = "above the 0.382 level (no pullback yet)" if last_price > upper \
        else "below the 0.618 level (too deep)"
    return {"pass": False, "level": "0.382-0.618", "distance_pct": None,
            "reason": f"Price outside the Fibonacci pullback {zone_text} — "
                      f"${last_price:,.2f} is {side}"}


def _evaluate_signal(
    df: pd.DataFrame,
    cfg: StrategyConfig,
    position_open: bool,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Signal:
    """
    Raw per-candle decision — the indicator rules only, no trade bookkeeping.

    `generate_signal` wraps this with the entry debounce; call this directly only
    when you want the untouched core decision.

    df must already have indicators (see add_indicators) and at least 2 rows.

    Pass a `diagnostics` dict to have the entry pipeline stages recorded in it,
    which is how the backtester and the live bot explain why a signal did not
    turn into a trade. Keys set on entry evaluations:

        core_signal   "BUY" when the core indicators fired, else None
        core_trigger  stage 1: bool
        core_reason   human-readable stage-1 detail
        fib_evaluated stage 2 ran at all (filter enabled + core trigger)
        fib_pass      stage 2: bool, or None when not evaluated
        fib_reason    human-readable stage-2 detail
        rejected_by   machine-readable rule key, or None when nothing blocked
    """
    if len(df) < 2:
        return Signal.HOLD

    prev, curr = df.iloc[-2], df.iloc[-1]
    strat = getattr(cfg, "strategy_name", "ema_cross_fib")

    # 1. Pure Trend Momentum (EMA Cross without Fib requirement)
    if strat == "pure_ema_cross":
        trending = bool(curr["adx"] >= cfg.adx_trend_threshold)
        cross_window = _ema_cross_window(cfg)
        crossed, bars_ago = _recent_bullish_cross(df, cross_window)
        still_bullish = bool(curr["ema_fast"] > curr["ema_slow"])
        bullish_cross = bool(crossed and still_bullish)
        cross_text = _ema_cross_text(crossed, bars_ago, still_bullish, cross_window)
        bearish_cross = prev["ema_fast"] >= prev["ema_slow"] and curr["ema_fast"] < curr["ema_slow"]

        if not position_open:
            rsi_ok = bool(curr["rsi"] < cfg.rsi_overbought)
            core_setup = bool(bullish_cross and trending and rsi_ok)
            _record(diagnostics,
                    core_signal="BUY" if core_setup else None,
                    core_trigger=core_setup,
                    core_reason=_core_reason(bullish_cross, trending, rsi_ok, curr, cfg, "EMA",
                                             cross_text),
                    ema_cross_bars_ago=bars_ago,
                    fib_evaluated=False, fib_pass=None, fib_reason=None,
                    rejected_by=_core_miss(bullish_cross, trending, rsi_ok, "no_ema_cross"))
            if core_setup:
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
            core_setup = bool((prev_rsi <= 30 and curr_rsi > prev_rsi) or curr_rsi <= 28)
            _record(diagnostics,
                    core_signal="BUY" if core_setup else None,
                    core_trigger=core_setup,
                    core_reason=(f"RSI {curr_rsi:.1f} turning up from oversold" if core_setup
                                 else f"RSI {curr_rsi:.1f} not oversold (needs <= 30 and rising, or <= 28)"),
                    fib_evaluated=False, fib_pass=None, fib_reason=None,
                    rejected_by=None if core_setup else "no_rsi_trigger")
            if core_setup:
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
            rsi_ok = bool(curr["rsi"] < cfg.rsi_overbought)
            core_setup = bool(bullish_cross and rsi_ok)
            _record(diagnostics,
                    core_signal="BUY" if core_setup else None,
                    core_trigger=core_setup,
                    core_reason=(_core_reason(True, True, rsi_ok, curr, cfg, "MACD") if bullish_cross
                                 else "no fresh MACD over signal-line crossover on this candle"),
                    fib_evaluated=False, fib_pass=None, fib_reason=None,
                    rejected_by=None if core_setup else ("rsi_overbought" if bullish_cross else "no_macd_cross"))
            if core_setup:
                return Signal.BUY
            return Signal.HOLD
        if bearish_cross:
            return Signal.SELL
        return Signal.HOLD

    # 4. Bollinger Bands Volatility Breakout
    elif strat == "bollinger_breakout":
        upper = curr.get("bb_upper", curr["close"])
        mid = curr.get("bb_mid", curr["close"])
        trending = bool(curr["adx"] >= cfg.adx_trend_threshold)

        if not position_open:
            breakout = bool(curr["close"] > upper)
            rsi_ok = bool(curr["rsi"] < 75)
            core_setup = bool(breakout and trending and rsi_ok)
            _record(diagnostics,
                    core_signal="BUY" if core_setup else None,
                    core_trigger=core_setup,
                    core_reason=(_core_reason(breakout, trending, rsi_ok, curr, cfg, "Bollinger breakout")
                                 if breakout else "close is not above the upper Bollinger Band"),
                    fib_evaluated=False, fib_pass=None, fib_reason=None,
                    rejected_by=(None if core_setup else
                                 ("no_bb_breakout" if not breakout else
                                  ("adx_filter" if not trending else "rsi_overbought"))))
            if core_setup:
                return Signal.BUY
            return Signal.HOLD
        if curr["close"] < mid or curr["rsi"] > 80:
            return Signal.SELL
        return Signal.HOLD

    # Default / "ema_cross_fib": Trend-following EMA Cross + RSI + ADX + Fib Retracement
    else:
        trending = bool(curr["adx"] >= cfg.adx_trend_threshold)
        # The crossover no longer has to land on the exact bar that tags the fib
        # zone: it stays entry-eligible for `ema_cross_window` bars as long as the
        # EMA fast line is still above the slow line. `ema_cross_window = 1`
        # reproduces the strict "cross on this candle" rule.
        cross_window = _ema_cross_window(cfg)
        crossed, bars_ago = _recent_bullish_cross(df, cross_window)
        still_bullish = bool(curr["ema_fast"] > curr["ema_slow"])
        bullish_cross = bool(crossed and still_bullish)
        cross_text = _ema_cross_text(crossed, bars_ago, still_bullish, cross_window)
        bearish_cross = prev["ema_fast"] >= prev["ema_slow"] and curr["ema_fast"] < curr["ema_slow"]

        if not position_open:
            rsi_ok = bool(curr["rsi"] < cfg.rsi_overbought)
            core_setup = bool(bullish_cross and trending and rsi_ok)

            # The Fibonacci zone filter is evaluated ONLY when enabled — with the
            # checkbox off the loop never touches fib levels at all and the trade
            # fires on the core EMA/ADX/RSI signal alone.
            fib_ok = True
            fib_detail = None
            if core_setup and getattr(cfg, "use_fib_filter", True):
                fib_detail = _fib_pullback_detail(df, cfg)
                fib_ok = fib_detail["pass"]
                # No fib data yet (rolling swing not warmed up) -> do not block.

            _record(diagnostics,
                    core_signal="BUY" if core_setup else None,
                    core_trigger=core_setup,
                    core_reason=_core_reason(bullish_cross, trending, rsi_ok, curr, cfg, "EMA",
                                             cross_text),
                    ema_cross_bars_ago=bars_ago,
                    fib_evaluated=fib_detail is not None,
                    fib_pass=None if fib_detail is None else fib_detail["pass"],
                    fib_reason=None if fib_detail is None else fib_detail["reason"],
                    rejected_by=(_core_miss(bullish_cross, trending, rsi_ok, "no_ema_cross") if not core_setup
                                 else (None if fib_ok else "fib_zone")))

            if core_setup and fib_ok:
                return Signal.BUY
            if core_setup and not fib_ok and diagnostics is not None:
                diagnostics["fib_blocked"] = diagnostics.get("fib_blocked", 0) + 1
            return Signal.HOLD

        # position open: exit on trend reversal or overheated RSI
        if bearish_cross or curr["rsi"] > cfg.rsi_overbought + 10:
            return Signal.SELL
        return Signal.HOLD


#: Exact wording of the entry-debounce rejection, shared by every caller so the
#: backtest report, the API payload and the live log all read the same.
DEBOUNCE_REASON = "Signal debounce: waiting for cooldown period between setups"


def generate_signal(
    df: pd.DataFrame,
    cfg: StrategyConfig,
    position_open: bool,
    diagnostics: Optional[Dict[str, Any]] = None,
    bars_since_last_trade: Optional[int] = None,
    timeframe: Optional[str] = None,
) -> Signal:
    """The strategy's decision for the last candle in `df`, debounce included.

    Identical to `_evaluate_signal` except for one entry-level rule: a BUY is
    suppressed when fewer than `min_bars_between_trades` candles have elapsed
    since the previous trade *exit*. Without it a fast EMA whipsaw re-triggers
    on adjacent bars (17:30, 17:35, 17:40), so one setup is paid for three times
    in fees and stop-outs.

    `bars_since_last_trade` is the caller's job to know — it is the distance in
    candles between this bar and the last exit, and `None` (no previous trade,
    e.g. a one-shot probe) means there is nothing to debounce. `timeframe`
    supplies the default when `cfg.min_bars_between_trades` is None; pass None
    for both to skip the check entirely. Exits are never debounced.
    """
    sig = _evaluate_signal(df, cfg, position_open, diagnostics)
    if sig != Signal.BUY or position_open:
        return sig

    required = resolve_min_bars_between_trades(cfg, timeframe)
    if required <= 0 or bars_since_last_trade is None:
        return sig
    try:
        elapsed = int(bars_since_last_trade)
    except (TypeError, ValueError):
        return sig
    if elapsed >= required:
        return sig

    # Blocked, and only now: the core trigger fired and every other filter
    # passed, so the debounce is the one rule that turned the trade away.
    _record(diagnostics,
            core_signal=(diagnostics or {}).get("core_signal") or "BUY",
            debounce_bars_elapsed=elapsed,
            debounce_bars_required=required,
            debounce_reason=DEBOUNCE_REASON,
            rejected_by="signal_debounce")
    return Signal.HOLD
