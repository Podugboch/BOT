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
from exchange_client import (
    MAX_DAYS,
    MIN_PAUSE_SEC,
    MS_PER_DAY,
    chunk_limit,
    clamp_lookback_days,
    fetch_ohlcv_window,
    lookback_limit_note,
    to_frame,
)
from risk_manager import RiskManager
from strategy import Signal, add_indicators, generate_signal

log = logging.getLogger("backtest")

# Candle paging (page sizes, sleeps, retries) lives in exchange_client so the
# live client and the backtester share one implementation.

# Signal-pipeline diagnostics: how much rejection detail to keep/report.
REJECTION_LOG_LIMIT = 40      # lines kept in the returned run report
REJECTION_LOG_LINES = 12      # lines echoed to the live activity log

CORE_MISS_LABELS = {
    "no_ema_cross": "no fresh EMA crossover",
    "no_macd_cross": "no fresh MACD crossover",
    "no_rsi_trigger": "RSI not oversold",
    "no_bb_breakout": "close did not break the upper Bollinger Band",
    "adx_filter": "ADX below the trend threshold",
    "rsi_overbought": "RSI above the overbought limit",
}


def _rejection_reason(rule: str, stages: dict) -> str:
    """Human sentence for the stage that blocked an otherwise-valid signal."""
    if rule == "fib_zone":
        return stages.get("fib_reason") or "Price outside the Fibonacci zone"
    if rule == "risk_gate":
        return stages.get("risk_reason") or "Risk manager blocked new entries"
    if rule == "position_sizing":
        return stages.get("sizing_reason") or "Position sizing returned 0 (equity/risk limits)"
    return stages.get("core_reason") or CORE_MISS_LABELS.get(rule, rule)


def _note_rejection(diag: dict, ts, rule: str, stages: dict) -> None:
    """Count a blocked signal and keep an explainable line for the run report."""
    diag["rejections"][rule] = diag["rejections"].get(rule, 0) + 1
    if len(diag["rejection_log"]) >= REJECTION_LOG_LIMIT:
        diag["rejection_log_truncated"] = True
        return
    stamp = ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else str(ts)
    diag["rejection_log"].append(
        f"Timestamp: {stamp} | Core signal triggered: "
        f"{stages.get('core_signal') or 'BUY'} | Rejected by: {_rejection_reason(rule, stages)}")


def _fmt_counts(counts: dict) -> str:
    return ", ".join(f"{rule}={count}" for rule, count in sorted(counts.items())) or "none"


def format_pipeline_report(results: dict) -> str:
    """Exportable, human-readable breakdown of a backtest's signal pipeline."""
    diag = results.get("diagnostics") or {}
    start, end = (results.get("date_range") or [None, None])[:2]
    window = f"{start} -> {end}" if start and end else "n/a"
    lines = [
        "--- Signal pipeline diagnostics ---",
        f"Data window              : {window} "
        f"({results.get('candles_tested', diag.get('candles_evaluated', 0)):,} candle(s) fetched, "
        f"{results.get('candles_tested', 0) - diag.get('candles_warmup', 0):,} tested)",
        f"Candles evaluated        : {diag.get('candles_evaluated', 0):,} "
        f"(warmup {diag.get('candles_warmup', 0):,} skipped, "
        f"{diag.get('candles_in_position', 0):,} in position, "
        f"{diag.get('signals_risk_blocked', 0):,} signals blocked by risk gate)",
        f"Stage 1 core indicator   : {diag.get('core_signals_detected', 0):,} raw BUY signal(s); "
        f"misses: {_fmt_counts(diag.get('core_misses', {}))}",
        f"Stage 2 Fibonacci filter : {diag.get('fib_filter_passed', 0):,} passed, "
        f"{diag.get('fib_filter_blocked', 0):,} blocked",
        f"Stage 3 capital / sizing : {diag.get('sizing_passed', 0):,} passed, "
        f"{diag.get('sizing_blocked', 0):,} blocked",
        f"Signals rejected         : {_fmt_counts(diag.get('rejections', {}))}",
        f"Trades executed          : {diag.get('trades_executed', 0):,} "
        f"(exits: {_fmt_counts(results.get('exit_reasons', {}))})",
        f"Expectancy               : avg win ${results.get('avg_win', 0):,.2f} vs avg loss "
        f"${abs(results.get('avg_loss', 0)):,.2f} -> realized R:R "
        f"{results.get('realized_rr', 0):.2f} (win rate {results.get('win_rate_pct', 0):.1f}%)",
    ]
    rejection_log = diag.get("rejection_log") or []
    if rejection_log:
        lines.append("")
        lines.append("Rejection detail (core signal fired, no trade):")
        lines.extend(f"  {entry}" for entry in rejection_log)
        if diag.get("rejection_log_truncated"):
            lines.append(f"  ... truncated at {len(rejection_log)} line(s); "
                         f"rejection counts above cover the whole run")
    return "\n".join(lines)


def _log_pipeline_summary(diag: dict, fib_filter_enabled: bool, warmup: int) -> None:
    """Echo the pipeline outcome to the activity log (bot.log -> dashboard)."""
    for entry in diag["rejection_log"][:REJECTION_LOG_LINES]:
        log.info("%s", entry)
    hidden = len(diag["rejection_log"]) - REJECTION_LOG_LINES
    if hidden > 0:
        log.info("... %d more rejection(s) suppressed — full list in the run report", hidden)
    log.info("Backtest summary: %s candle(s) evaluated, %s raw signal(s), %s rejected (%s), %s trade(s) executed",
             f"{diag['candles_evaluated']:,}", f"{diag['core_signals_detected']:,}",
             f"{sum(diag['rejections'].values()):,}", _fmt_counts(diag["rejections"]),
             f"{diag['trades_executed']:,}")
    log.info("Backtest stages: fib filter %s (%s passed / %s blocked), sizing %s passed / %s blocked, "
             "no-signal candles by rule: %s",
             "on" if fib_filter_enabled else "off",
             f"{diag['fib_filter_passed']:,}", f"{diag['fib_filter_blocked']:,}",
             f"{diag['sizing_passed']:,}", f"{diag['sizing_blocked']:,}",
             _fmt_counts(diag["core_misses"]))

def fetch_historical(exchange_id: str, symbol: str, timeframe: str, days: int,
                     client=None, pause_sec: float = None, max_chunks: int = None) -> pd.DataFrame:
    """Historical OHLCV pull that paginates across the whole lookback window.

    A single kline request only returns a few hundred candles, so a 250-day 5m
    backtest (~72,000 candles) is stitched from hundreds of pages: we walk
    forward from the start of the window with the `since`/startTime cursor,
    sleep between requests to respect rate limits, and concatenate the slices
    into one continuous, de-duplicated, time-ordered series.

    `client` takes a pre-built ccxt exchange (used by tests); otherwise one is
    built from `exchange_id`.
    """
    requested_days = max(1, min(int(days), MAX_DAYS))
    # Grains differ in how far back venues store candles: asking MEXC for 365
    # days of 5m (~105,000 candles) answers with an empty page, not more data.
    days = clamp_lookback_days(requested_days, timeframe)
    if days != requested_days:
        log.warning("Lookback %s day(s) exceeds what %s stores for %s — requesting %s day(s) "
                    "instead. %s", requested_days, exchange_id, timeframe, days,
                    lookback_limit_note(timeframe))

    if client is None:
        ex_clean = str(exchange_id).strip().lower()
        if ex_clean not in ccxt.exchanges:
            raise ValueError(f"'{exchange_id}' is not a valid ccxt exchange id.")
        exchange = getattr(ccxt, ex_clean)({"enableRateLimit": True})
    else:
        exchange = client

    try:
        ms_per_bar = int(exchange.parse_timeframe(timeframe)) * 1000
    except Exception as err:
        raise ValueError(f"Unsupported timeframe '{timeframe}' for '{exchange_id}'") from err
    if ms_per_bar <= 0:
        raise ValueError(f"Unsupported timeframe '{timeframe}' for '{exchange_id}'")

    exchange.load_markets()
    limit = chunk_limit(exchange, symbol)
    end_ms = exchange.milliseconds()
    since_ms = end_ms - days * MS_PER_DAY
    expected = int(days * MS_PER_DAY // ms_per_bar)
    pause = float(pause_sec) if pause_sec is not None else max(
        getattr(exchange, "rateLimit", 0) / 1000.0, MIN_PAUSE_SEC)
    if max_chunks is None:
        # Generous: some venues serve half of what they advertise (MEXC), so
        # budget 3x the nominal page count plus headroom for retries.
        max_chunks = max(50, int(expected / limit) * 3 + 25)

    log.info("Fetching %s %s from %s: %s day(s) (~%s candles) in pages of %s, %.2fs pause",
             symbol, timeframe, getattr(exchange, "id", exchange_id), days,
             f"{expected:,}", limit, pause)

    rows, pages, largest_page, effective_since = fetch_ohlcv_window(
        exchange, symbol, timeframe, since_ms, end_ms, limit, pause, max_chunks)

    df = to_frame(rows)
    if df.empty:
        # Not silent: say why, and what to do about it, rather than returning a
        # bare empty frame that the caller can only report as "insufficient bars".
        log.error("No %s candles returned for %s from %s. %s",
                  timeframe, symbol, getattr(exchange, "id", exchange_id),
                  lookback_limit_note(timeframe))
        return df

    # Compare against the window actually served, not the one requested, so a
    # graceful deep-start fallback does not masquerade as missing pages.
    served_days = (end_ms - effective_since) / MS_PER_DAY if effective_since else days
    expected_served = max(1, int(served_days * MS_PER_DAY // ms_per_bar))
    coverage = len(df) / expected_served * 100 if expected_served else 0.0
    if effective_since and effective_since > since_ms:
        log.warning("Venue stores no %s candles before %s — covered %.0f of the %s day(s) "
                    "requested. %s", timeframe,
                    df["ts"].iloc[0].strftime("%Y-%m-%d %H:%M UTC"), served_days, days,
                    lookback_limit_note(timeframe))
    # Report the page size actually served, not just the one we asked for:
    # venues like MEXC advertise 1000 and hand back 500.
    log.info("Fetched %s candles across %s days for %s (%s) in %d page(s) of up to %d — %s to %s UTC (%.0f%% of the nominal %s)",
             f"{len(df):,}", f"{served_days:.0f}", symbol, timeframe, pages, largest_page,
             df["ts"].iloc[0].strftime("%Y-%m-%d %H:%M"), df["ts"].iloc[-1].strftime("%Y-%m-%d %H:%M"),
             coverage, f"{expected_served:,}")
    if coverage < 80:
        log.warning("Only %.0f%% of the expected candles came back for %s — the exchange may "
                    "not list this symbol that far back, or pages were skipped", coverage, symbol)
    return df


def run_backtest(df: pd.DataFrame, cfg: BotConfig) -> dict:
    candles_tested = int(len(df))
    date_range = _date_range(df)
    df = add_indicators(df, cfg.strategy)
    risk = RiskManager(cfg.risk, cfg.starting_equity)

    # Is the Fibonacci zone filter actually in play for this run? When the UI
    # checkbox is off there is no fib evaluation anywhere in the loop below:
    # entries are decided by the core EMA/ADX/RSI signal alone.
    strategy_id = (getattr(cfg.strategy, "strategy_name", "ema_cross_fib") or "ema_cross_fib").strip().lower()
    fib_filter_enabled = strategy_id == "ema_cross_fib" and bool(cfg.strategy.use_fib_filter)

    # Per-stage tally of the entry pipeline, so a run can explain itself:
    # stage 1 core trigger -> stage 2 Fibonacci filter -> stage 3 sizing/capital.
    diag = {
        "candles_evaluated": 0,
        "candles_in_position": 0,
        "signals_risk_blocked": 0,
        "core_signals_detected": 0,
        "fib_filter_passed": 0,
        "fib_filter_blocked": 0,
        "sizing_passed": 0,
        "sizing_blocked": 0,
        "trades_executed": 0,
        "rejections": {},          # post-trigger: fib_zone / position_sizing / risk_gate
        "core_misses": {},         # stage 1 False, keyed by rule
        "rejection_log": [],
        "rejection_log_truncated": False,
    }

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

        # `now` is the candle's own timestamp, so a losing-streak cooldown is
        # measured in market time and expires mid-backtest exactly as it would
        # live — instead of vetoing every remaining signal.
        can_trade, risk_reason = risk.can_trade(now=row["ts"])
        diag["candles_evaluated"] += 1

        if position is None:
            # Evaluate the full entry pipeline every flat candle, even when the
            # risk gate is shut, so the run report can say what actually blocked.
            stages: dict = {}
            sig = generate_signal(window, cfg.strategy, position_open=False, diagnostics=stages)

            if not stages.get("core_trigger"):
                miss = stages.get("rejected_by")
                if miss:
                    diag["core_misses"][miss] = diag["core_misses"].get(miss, 0) + 1
            else:
                diag["core_signals_detected"] += 1
                if stages.get("fib_evaluated"):
                    diag["fib_filter_passed" if stages.get("fib_pass") else "fib_filter_blocked"] += 1

                blocked_by = None
                if sig != Signal.BUY:
                    # Stage 2 blocked it (fib zone, or a strategy-level filter).
                    blocked_by = stages.get("rejected_by") or "fib_zone"
                elif not can_trade:
                    # Stage 2 passed but the risk gate is shut for this candle.
                    diag["signals_risk_blocked"] += 1
                    stages["risk_reason"] = risk_reason
                    blocked_by = "risk_gate"

                if blocked_by is None:
                    atr_val = row["atr"] if (pd.notna(row["atr"]) and row["atr"] > 0) else (price * 0.02)
                    stop, target = risk.stop_and_target(price, atr_val, "BUY")
                    size = risk.position_size(price, stop)
                    if size > 0:
                        diag["sizing_passed"] += 1
                        fill = price * (1 + cfg.slippage_pct)
                        fee = fill * size * cfg.taker_fee_pct
                        position = {"entry": fill, "size": size, "stop": stop,
                                    "target": target, "fee_paid": fee, "initial_stop": stop,
                                    "opened_ts": row["ts"]}
                        diag["trades_executed"] += 1
                    else:
                        diag["sizing_blocked"] += 1
                        stages["sizing_reason"] = (
                            f"Position sizing returned 0 (equity ${risk.state.equity:,.2f}, "
                            f"risk {cfg.risk.risk_per_trade_pct:.2%} per trade)")
                        blocked_by = "position_sizing"

                if blocked_by is not None:
                    _note_rejection(diag, row["ts"], blocked_by, stages)

        elif position is not None:
            diag["candles_in_position"] += 1
            atr_val = row["atr"] if (pd.notna(row["atr"]) and row["atr"] > 0) else (price * 0.02)

            # Trail first, then test the exit against the ratcheted stop, so a
            # winner that has already run is protected at a profit instead of
            # being given back to the original ATR stop. The bar's high feeds
            # the trail, so the stop follows the highest high since entry.
            risk.update_trailing_stop(position, price, atr_val, "BUY", peak_price=row["high"])
            hit_stop = price <= position["stop"]
            hit_target = price >= position["target"]
            running = risk.is_running(position, position.get("peak", price), atr_val, "BUY")
            sig = generate_signal(window, cfg.strategy, position_open=True)
            # "Let it run": once the trailing stop is armed the trade is in
            # profit, so a reversal signal no longer closes it early — the trail
            # (already above entry) manages the exit.
            signal_exit = sig == Signal.SELL and not running

            if hit_stop or hit_target or signal_exit:
                exit_ref = position["stop"] if hit_stop else price
                exit_fill = exit_ref * (1 - cfg.slippage_pct)
                exit_fee = exit_fill * position["size"] * cfg.taker_fee_pct
                pnl = (exit_fill - position["entry"]) * position["size"] - position["fee_paid"] - exit_fee

                reason = "stop" if hit_stop else ("target" if hit_target else "signal")
                if hit_stop and position.get("trail_armed"):
                    reason = "trail"
                risk.record_trade_result(pnl, now=row["ts"])
                trades.append({
                    "entry": position["entry"], "exit": exit_fill, "size": position["size"],
                    "pnl": pnl, "reason": reason,
                    "stop": position["stop"], "initial_stop": position.get("initial_stop"),
                })
                position = None

        equity_curve.append({"ts": row["ts"], "equity": risk.state.equity})

    _log_pipeline_summary(diag, fib_filter_enabled, warmup)

    results = summarize(trades, equity_curve, cfg.starting_equity, cfg.exchange.timeframe)
    results["fib_filter_enabled"] = fib_filter_enabled
    results["fib_blocked_signals"] = diag["fib_filter_blocked"]
    # Echo the knobs the run was measured with, so a sweep of configs is
    # self-describing without having to keep the request alongside the result.
    results["fib_tolerance_atr_mult"] = float(
        getattr(cfg.strategy, "fib_tolerance_atr_mult", 0.0) or 0.0)
    results["fib_level_mode"] = str(getattr(cfg.strategy, "fib_level_mode", "all") or "all")
    results["trailing_stop_enabled"] = risk.trailing_enabled()
    results["trailing_stop_atr_mult"] = float(
        getattr(cfg.risk, "trailing_stop_atr_mult", 0.0) or 0.0)
    results["diagnostics"] = diag
    # Flat aliases so the API/UI can read the headline numbers without digging.
    results["candles_evaluated"] = diag["candles_evaluated"]
    results["raw_signals"] = diag["core_signals_detected"]
    results["signals_rejected"] = dict(diag["rejections"])
    results["trades_executed"] = diag["trades_executed"]
    # Payload the frontend renders directly: how much data was actually tested
    # and which window it covered.
    results["candles_tested"] = candles_tested
    results["date_range"] = date_range
    diag["candles_warmup"] = warmup
    diag["report"] = format_pipeline_report(results)
    return results


def _date_range(df: pd.DataFrame) -> list:
    """[first, last] candle timestamps as ISO strings, for the run summary."""
    if df is None or len(df) == 0 or "ts" not in getattr(df, "columns", []):
        return [None, None]
    stamps = []
    for value in (df["ts"].iloc[0], df["ts"].iloc[-1]):
        stamps.append(value.isoformat() if hasattr(value, "isoformat") else str(value))
    return stamps


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

    avg_win = abs(float(np.mean([t["pnl"] for t in wins]))) if wins else 0.0
    avg_loss = abs(float(np.mean([t["pnl"] for t in losses]))) if losses else 0.0
    exit_reasons: dict = {}
    for trade in trades:
        key = trade.get("reason", "signal")
        exit_reasons[key] = exit_reasons.get(key, 0) + 1

    return {
        "starting_equity": round(starting_equity, 2),
        "final_equity": round(final_equity, 2),
        "total_return_pct": round((final_equity / starting_equity - 1) * 100, 2) if starting_equity > 0 else 0.0,
        "num_trades": len(trades),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 2) if trades else 0,
        "avg_win": round(float(np.mean([t["pnl"] for t in wins])), 2) if wins else 0,
        "avg_loss": round(float(np.mean([t["pnl"] for t in losses])), 2) if losses else 0,
        # The number that decides whether a 30-45% win rate pays: a winner has to
        # be worth more than a loser on average, not merely the same.
        "realized_rr": round(avg_win / avg_loss, 2) if avg_loss > 0 else 0,
        "exit_reasons": exit_reasons,
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
    parser.add_argument("--no-fib", action="store_true",
                        help="trade the core EMA/ADX/RSI signal alone")
    parser.add_argument("--fib-mode", choices=("all", "deep"), default=None,
                        help="fib confluence levels: all (0.382/0.5/0.618) or deep (0.5/0.618)")
    parser.add_argument("--fib-atr-mult", type=float, default=None,
                        help="dynamic fib band as a multiple of ATR (0 = fixed percentage band)")
    parser.add_argument("--no-trail", action="store_true", help="disable the ATR trailing stop")
    parser.add_argument("--trail-atr", type=float, default=None,
                        help="trailing-stop distance in ATRs (default 2.0)")
    args = parser.parse_args()

    cfg = BotConfig()
    cfg.exchange.exchange_id = args.exchange
    cfg.exchange.symbol = args.symbol
    cfg.exchange.timeframe = args.timeframe
    cfg.strategy.use_fib_filter = not args.no_fib
    if args.fib_mode is not None:
        cfg.strategy.fib_level_mode = args.fib_mode
    if args.fib_atr_mult is not None:
        cfg.strategy.fib_tolerance_atr_mult = args.fib_atr_mult
    cfg.risk.trailing_stop_enabled = not args.no_trail
    if args.trail_atr is not None:
        cfg.risk.trailing_stop_atr_mult = args.trail_atr

    hist = fetch_historical(args.exchange, args.symbol, args.timeframe, args.days)
    if hist.empty:
        raise SystemExit(f"No historical candles returned for {args.symbol} {args.timeframe}.")
    log.info("Running the backtest loop over %s candle(s)...", f"{len(hist):,}")

    results = run_backtest(hist, cfg)
    print("\n--- Backtest Results ---")
    for k, v in results.items():
        if k == "diagnostics":
            continue
        print(f"{k:20s}: {v}")
    print()
    print(results["diagnostics"]["report"])
    print("\n(Tip: the rejection detail is also written to the activity log, "
          "and the full list lives at results['diagnostics']['rejection_log'].)")
