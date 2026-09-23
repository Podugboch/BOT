"""
Parameter sweep for the 1h edge.

Runs the same backtest engine over one data window while varying only the
Fibonacci filter (off / fixed band / ATR-scaled band, all vs deep levels) and
the ATR trailing stop, then prints one comparison table. The point is to answer
"does filtering entries near the 0.5/0.618 pullback beat the core signal's win
rate *without* emptying the trade sample" with numbers instead of opinion.

Real data (needs exchange access):

    python optimize_1h.py --source exchange --symbol BTC/USDT --days 730

Offline (synthetic regime-switching series; models bull/bear/chop):

    python optimize_1h.py --source synthetic --candles 17520
"""
import argparse
import logging

import numpy as np
import pandas as pd

from backtester import fetch_historical, run_backtest
from config import BotConfig

log = logging.getLogger("optimize")


def synthetic_1h_series(candles: int = 17_520, seed: int = 7, start: float = 30_000.0) -> pd.DataFrame:
    """Hourly candles with alternating bull, bear and consolidation regimes.

    Not a market replica — a fair, deterministic stand-in that exercises the
    same regime changes the sweep is meant to survive, so the harness can be
    sanity-checked without network access. Always re-run on real candles before
    trusting any number.
    """
    rng = np.random.default_rng(seed)
    closes = [start]
    regime, left = None, 0
    while len(closes) < candles:
        if left <= 0:
            regime = rng.choice(["bull", "bear", "chop"], p=[0.4, 0.3, 0.3])
            left = int(rng.integers(500, 2_200))
        left -= 1
        drift = {"bull": 0.00035, "bear": -0.00030, "chop": 0.0}[regime]
        vol = {"bull": 0.012, "bear": 0.016, "chop": 0.007}[regime]
        closes.append(max(1.0, closes[-1] * float(np.exp(rng.normal(drift, vol)))))

    close = np.array(closes[:candles], dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    span = np.abs(rng.normal(0, 0.006, candles))
    return pd.DataFrame({
        "ts": pd.date_range("2023-09-01", periods=candles, freq="1h"),
        "open": open_,
        "high": np.maximum(open_, close) * (1 + span),
        "low": np.minimum(open_, close) * (1 - span),
        "close": close,
        "volume": rng.uniform(50, 500, candles),
    })


def _config(label: str, use_fib: bool, atr_mult: float, level_mode: str, trail: bool,
            trail_atr: float = 3.5, floor_pct: float = 0.01, arm_atr: float = 1.5,
            debounce: int = None) -> tuple:
    cfg = BotConfig()
    cfg.exchange.timeframe = "1h"
    cfg.strategy.use_fib_filter = use_fib
    cfg.strategy.fib_tolerance_atr_mult = atr_mult
    cfg.strategy.fib_tolerance_pct = floor_pct
    cfg.strategy.fib_level_mode = level_mode
    cfg.risk.trailing_stop_enabled = trail
    cfg.risk.trailing_stop_atr_mult = trail_atr
    cfg.risk.trailing_arm_atr_mult = arm_atr
    cfg.strategy.min_bars_between_trades = debounce
    return label, cfg


def sweep_configs() -> list:
    """The A/B matrix: the fib knobs only ever change the *entry* and the trail
    knobs only ever change the *exit*, so the two effects stay separable. The
    first row is the shipped default (golden zone, arm +1.5 ATR, trail 3.5 ATR)."""
    return [
        _config("DEFAULT zone/0.5xATR/3.5/3bar",    True,  0.5, "zone", True),
        _config("defaults, no debounce (0 bars)",   True,  0.5, "zone", True, debounce=0),
        _config("zone, old trail (2.0, arm 1.0)",   True,  0.5, "zone", True, trail_atr=2.0, arm_atr=1.0),
        _config("zone, no trail",                   True,  0.5, "zone", False),
        _config("fib OFF (core signal)",            False, 0.0, "all",  True),
        _config("fib fixed 1% band, all",           True,  0.0, "all",  True),
        _config("fib 0.5xATR, all levels",          True,  0.5, "all",  True),
        _config("fib 0.5xATR, deep 0.5/0.618",      True,  0.5, "deep", True),
        _config("zone 3.0 ATR trail",               True,  0.5, "zone", True, trail_atr=3.0),
    ]


def run_sweep(df: pd.DataFrame) -> list:
    rows = []
    for label, cfg in sweep_configs():
        res = run_backtest(df, cfg)
        rows.append({
            "config": label,
            "trades": res["num_trades"],
            "win_rate": res["win_rate_pct"],
            "realized_rr": res["realized_rr"],
            "return_pct": res["total_return_pct"],
            "max_dd": res["max_drawdown_pct"],
            "fib_blocked": res["fib_blocked_signals"],
            "exits": res["exit_reasons"],
        })
    return rows


def print_table(rows: list) -> None:
    header = f"{'config':32s} {'trades':>7s} {'win%':>7s} {'R:R':>6s} {'return%':>9s} {'maxDD%':>8s} {'fibblk':>7s}"
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['config']:32s} {row['trades']:7d} {row['win_rate']:7.2f} "
              f"{row['realized_rr']:6.2f} {row['return_pct']:9.2f} {row['max_dd']:8.2f} "
              f"{row['fib_blocked']:7d}")
    print("\nexit mix per config:")
    for row in rows:
        mix = ", ".join(f"{k}={v}" for k, v in sorted(row["exits"].items())) or "none"
        print(f"  {row['config']:32s} {mix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=("exchange", "synthetic"), default="synthetic")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--days", type=int, default=730, help="exchange lookback (1..730)")
    parser.add_argument("--candles", type=int, default=17_520, help="synthetic candle count")
    parser.add_argument("--verbose", action="store_true",
                        help="show circuit-breaker/cooldown log lines (off by default so the table stays readable)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING if args.verbose else logging.ERROR,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    if args.source == "exchange":
        df = fetch_historical(args.exchange, args.symbol, args.timeframe, args.days)
        if df is None or df.empty:
            raise SystemExit(f"No candles returned for {args.symbol} {args.timeframe}.")
        label = f"{args.symbol} {args.timeframe}, {len(df):,} candles, {args.days}d"
    else:
        df = synthetic_1h_series(args.candles)
        label = f"synthetic 1h, {len(df):,} candles (regime-switching)"

    print(f"\nSweep window: {label}")
    print(f"{df['ts'].iloc[0]} -> {df['ts'].iloc[-1]}")
    print_table(run_sweep(df))


if __name__ == "__main__":
    main()
