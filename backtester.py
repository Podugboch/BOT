"""
Honest backtest engine: walks bar-by-bar (no lookahead — signal at bar i only
sees data up to bar i), applies the same fees and slippage the live bot would
pay, and runs every trade through the real RiskManager (position sizing,
drawdown kill switch, daily-loss circuit breaker).

Usage:
    python backtester.py --symbol BTC/USDT --timeframe 15m --days 180
"""
import argparse
import logging

import ccxt
import numpy as np
import pandas as pd

from config import BotConfig
from risk_manager import RiskManager
from strategy import Signal, add_indicators, generate_signal

log = logging.getLogger("backtest")


def fetch_historical(exchange_id: str, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """Paginated historical OHLCV pull via ccxt."""
    ex_clean = exchange_id.strip().lower()
    if ex_clean not in ccxt.exchanges:
        raise ValueError(f"'{exchange_id}' is not a valid ccxt exchange id.")
    days = max(1, min(days, 730))
    exchange = getattr(ccxt, ex_clean)({"enableRateLimit": True})
    ms_per_bar = exchange.parse_timeframe(timeframe) * 1000
    since = exchange.milliseconds() - days * 24 * 60 * 60 * 1000

    all_rows = []
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        if not batch:
            break
        all_rows.extend(batch)
        since = batch[-1][0] + ms_per_bar
        if len(batch) < 1000 or since >= exchange.milliseconds():
            break

    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df.drop_duplicates(subset="ts").reset_index(drop=True)


def run_backtest(df: pd.DataFrame, cfg: BotConfig) -> dict:
    df = add_indicators(df, cfg.strategy)
    risk = RiskManager(cfg.risk, cfg.starting_equity)

    equity_curve = []
    trades = []
    position = None

    warmup = max(cfg.strategy.slow_ema, cfg.strategy.rsi_period, cfg.strategy.adx_period,
                 cfg.strategy.fib_lookback) + 2

    last_day = None
    for i in range(warmup, len(df)):
        window = df.iloc[: i + 1]
        row = df.iloc[i]
        price = row["close"]

        # Reset daily circuit breaker on UTC date change
        current_day = row["ts"].date() if hasattr(row["ts"], "date") else None
        if last_day is not None and current_day != last_day:
            risk.reset_day()
        last_day = current_day

        can_trade, _ = risk.can_trade()

        if position is None and can_trade:
            sig = generate_signal(window, cfg.strategy, position_open=False)
            if sig == Signal.BUY:
                atr_val = row["atr"] if (pd.notna(row["atr"]) and row["atr"] > 0) else (price * 0.02)
                stop, target = risk.stop_and_target(price, atr_val, "BUY")
                size = risk.position_size(price, stop)
                if size > 0:
                    fill = price * (1 + cfg.slippage_pct)
                    fee = fill * size * cfg.taker_fee_pct
                    position = {"entry": fill, "size": size, "stop": stop,
                                "target": target, "fee_paid": fee}

        elif position is not None:
            hit_stop = price <= position["stop"]
            hit_target = price >= position["target"]
            sig = generate_signal(window, cfg.strategy, position_open=True)

            if hit_stop or hit_target or sig == Signal.SELL:
                exit_ref = position["stop"] if hit_stop else price
                exit_fill = exit_ref * (1 - cfg.slippage_pct)
                exit_fee = exit_fill * position["size"] * cfg.taker_fee_pct
                pnl = (exit_fill - position["entry"]) * position["size"] - position["fee_paid"] - exit_fee

                risk.record_trade_result(pnl)
                trades.append({
                    "entry": position["entry"], "exit": exit_fill, "size": position["size"],
                    "pnl": pnl, "reason": "stop" if hit_stop else ("target" if hit_target else "signal"),
                })
                position = None

        equity_curve.append({"ts": row["ts"], "equity": risk.state.equity})

    return summarize(trades, equity_curve, cfg.starting_equity, cfg.exchange.timeframe)


def summarize(trades: list, equity_curve: list, starting_equity: float, timeframe: str) -> dict:
    eq = pd.DataFrame(equity_curve)
    final_equity = eq["equity"].iloc[-1] if len(eq) else starting_equity
    returns = eq["equity"].pct_change().dropna() if len(eq) > 1 else pd.Series(dtype=float)

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]

    running_max = eq["equity"].cummax() if len(eq) else pd.Series([starting_equity])
    drawdown = ((eq["equity"] - running_max) / running_max) if len(eq) else pd.Series([0])

    bars_per_year = {"1m": 525600, "5m": 105120, "15m": 35040, "1h": 8760, "4h": 2190, "1d": 365}
    ann_factor = np.sqrt(bars_per_year.get(timeframe, 35040))
    sharpe = (returns.mean() / returns.std() * ann_factor) if returns.std() else 0

    return {
        "starting_equity": round(starting_equity, 2),
        "final_equity": round(final_equity, 2),
        "total_return_pct": round((final_equity / starting_equity - 1) * 100, 2) if starting_equity > 0 else 0.0,
        "num_trades": len(trades),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 2) if trades else 0,
        "avg_win": round(float(np.mean([t["pnl"] for t in wins])), 2) if wins else 0,
        "avg_loss": round(float(np.mean([t["pnl"] for t in losses])), 2) if losses else 0,
        "max_drawdown_pct": round(float(drawdown.min()) * 100, 2) if len(drawdown) else 0,
        "sharpe_approx": round(float(sharpe), 2),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Backtest the strategy against historical data.")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--days", type=int, default=180)
    args = parser.parse_args()

    cfg = BotConfig()
    cfg.exchange.exchange_id = args.exchange
    cfg.exchange.symbol = args.symbol
    cfg.exchange.timeframe = args.timeframe

    log.info("Fetching %s days of %s %s from %s...", args.days, args.symbol, args.timeframe, args.exchange)
    hist = fetch_historical(args.exchange, args.symbol, args.timeframe, args.days)
    log.info("Fetched %d bars. Running backtest...", len(hist))

    results = run_backtest(hist, cfg)
    print("\n--- Backtest Results ---")
    for k, v in results.items():
        print(f"{k:20s}: {v}")
