"""
Main run loop. Ties strategy + risk + exchange together and keeps running
until told to stop (kill switch file, drawdown kill switch, or Ctrl+C).

Run:
    python bot.py                 # paper mode (default, config.py)
    EXCHANGE_API_KEY=... EXCHANGE_API_SECRET=... python bot.py   # for live, after editing mode

Stop immediately at any time:
    touch KILL
"""
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

from config import BotConfig, timeframe_minutes
from exchange_client import ExchangeClient
from risk_manager import RiskManager
from strategy import DEBOUNCE_REASON, Signal, add_indicators, generate_signal

try:
    from execution_tracker import get_execution_tracker
except ImportError:
    get_execution_tracker = None

try:
    from alerts import get_alert_dispatcher
except ImportError:
    get_alert_dispatcher = None

log = logging.getLogger("bot")


def _pipeline_rejection(stages: dict) -> str:
    """Why a fired core signal did not become a trade (stage 2 or 3)."""
    if stages.get("fib_evaluated") and not stages.get("fib_pass"):
        return stages.get("fib_reason") or "Price outside the Fibonacci zone"
    return stages.get("core_reason") or "blocked after the core trigger"


def setup_logging():
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    log_path = Path("/tmp/bot.log") if (os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME")) else Path("bot.log")

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    try:
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        has_file = any(isinstance(h, logging.FileHandler) for h in root.handlers)
        if not has_file:
            root.addHandler(file_handler)
    except Exception:
        # Gracefully proceed if running in a read-only or serverless environment
        pass

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    has_stream = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers)
    if not has_stream:
        root.addHandler(stream_handler)

    log.setLevel(logging.INFO)
    logging.getLogger("web").setLevel(logging.INFO)


setup_logging()


class TradingBot:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.risk = RiskManager(cfg.risk, cfg.starting_equity)
        self.exchange = ExchangeClient(
            cfg.exchange, cfg.mode, cfg.taker_fee_pct, cfg.slippage_pct, cfg.starting_equity
        )
        self.position = None
        self.last_day = datetime.now(timezone.utc).date()
        self.last_exit_ts = None
        self.tracker = get_execution_tracker() if get_execution_tracker else None
        self.alerts = get_alert_dispatcher() if get_alert_dispatcher else None

    def kill_switch_active(self) -> bool:
        return os.path.exists(self.cfg.kill_switch_file) or getattr(self.risk.state, "kill_switch_triggered", False)

    def deactivate_kill_switch(self) -> bool:
        """Deactivate the kill switch by removing the trigger file and clearing risk manager latch."""
        if os.path.exists(self.cfg.kill_switch_file):
            try:
                os.remove(self.cfg.kill_switch_file)
                log.info("Kill switch file '%s' removed.", self.cfg.kill_switch_file)
            except Exception as e:
                log.warning("Could not remove kill switch file '%s': %s", self.cfg.kill_switch_file, e)

        self.risk.reset_kill_switch(reset_peak=True)
        if self.alerts:
            self.alerts.send_alert(
                "KILL_SWITCH_RESET",
                "Kill Switch Deactivated",
                "Emergency kill switch has been cleared. System returned to IDLE / ready state.",
                level="info"
            )
        log.info("Emergency kill switch deactivated. Bot is ready.")
        return True

    @staticmethod
    def _last_bar_ts(df):
        """Bar timestamp of the newest candle, falling back to wall-clock time."""
        try:
            if df is not None and len(df) > 0 and "ts" in df:
                stamp = pd.Timestamp(df["ts"].iloc[-1])
                return stamp.tz_convert("UTC").tz_localize(None) if stamp.tzinfo else stamp
        except Exception:
            pass
        return pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)

    def _mark_exit(self, df=None) -> None:
        """Remember when the bot went flat, for the next entry's debounce."""
        self.last_exit_ts = self._last_bar_ts(df)

    def _bars_since_last_exit(self, df=None):
        """Candles since the last exit, or None when there is no prior trade.

        None means "nothing to debounce" — the caller (and the strategy) treats
        that as no restriction, which is what a first-ever entry should be.
        """
        if self.last_exit_ts is None:
            return None
        bar_minutes = timeframe_minutes(self.cfg.exchange.timeframe)
        if bar_minutes <= 0:
            return None
        try:
            gap = self._last_bar_ts(df) - self.last_exit_ts
            return max(0, int(gap.total_seconds() // (bar_minutes * 60)))
        except Exception:
            return None

    def maybe_reset_day(self):
        today = datetime.now(timezone.utc).date()
        if today != self.last_day:
            self.risk.reset_day()
            self.last_day = today
            log.info("New UTC trading day — daily loss counter reset.")

    def sync_live_balance(self) -> dict:
        """Synchronize the bot's risk equity with the live exchange wallet balance.
        In live mode with authenticated API keys, dynamically updates RiskManager equity.
        """
        try:
            wallet = self.exchange.fetch_wallet_balance(self.cfg.exchange.symbol)
            if self.cfg.mode == "live" and wallet.get("is_live"):
                total_val = float(wallet.get("total_quote", 0.0) or 0.0)
                if total_val > 0:
                    old_equity = self.risk.state.equity
                    self.risk.state.equity = total_val
                    if abs(old_equity - total_val) > 1.0:
                        log.info("Live wallet balance sync: equity updated $%.2f -> $%.2f (%s free: %.2f)",
                                 old_equity, total_val, wallet.get("quote_currency"), wallet.get("free_quote", 0.0))
            return wallet
        except Exception as e:
            log.warning("Could not sync live wallet balance: %s", e)
            return self.exchange.fetch_wallet_balance(self.cfg.exchange.symbol)

    def step(self) -> bool:
        """One decision cycle. Returns False only if an active kill switch is triggered."""
        if self.kill_switch_active():
            if os.path.exists(self.cfg.kill_switch_file):
                log.critical("Kill switch file '%s' detected — halting.", self.cfg.kill_switch_file)
            else:
                log.critical("Risk manager kill switch triggered — halting.")
            return False

        self.maybe_reset_day()
        self.sync_live_balance()

        try:
            df = self.exchange.fetch_ohlcv(limit=200)
        except Exception:
            log.exception("Market data fetch failed — skipping this cycle, will retry.")
            return True

        if df is None or len(df) < 2:
            log.warning("Insufficient market data returned (%s bars) — skipping this cycle.", 0 if df is None else len(df))
            return True

        try:
            df = add_indicators(df, self.cfg.strategy)
            price = float(df["close"].iloc[-1])
        except Exception as e:
            log.warning("Indicator calculation failed: %s — skipping this cycle, will retry.", e)
            return True

        can_trade, reason = self.risk.can_trade()

        # In manual execution mode, monitor market and signals without auto-placing orders
        if getattr(self.cfg, "execution_mode", "auto") == "manual":
            try:
                stages: dict = {}
                sig = generate_signal(df, self.cfg.strategy, position_open=bool(self.position),
                                      diagnostics=stages,
                                      bars_since_last_trade=self._bars_since_last_exit(df),
                                      timeframe=self.cfg.exchange.timeframe)
                rsi_val = float(df["rsi"].iloc[-1]) if ("rsi" in df and pd.notna(df["rsi"].iloc[-1])) else 0.0
                adx_val = float(df["adx"].iloc[-1]) if ("adx" in df and pd.notna(df["adx"].iloc[-1])) else 0.0
                if self.position is not None:
                    atr_now = float(df["atr"].iloc[-1]) if ("atr" in df and pd.notna(df["atr"].iloc[-1])) else (price * 0.02)
                    bar_high = float(df["high"].iloc[-1]) if ("high" in df and pd.notna(df["high"].iloc[-1])) else price
                    self.risk.update_trailing_stop(self.position, price, atr_now, "BUY", peak_price=bar_high)
                    hit_stop = price <= self.position["stop"]
                    hit_target = price >= self.position["target"]
                    pnl_approx = (price - self.position["entry"]) * self.position["size"]
                    if hit_stop:
                        log.warning("MANUAL MODE: Stop loss reached (%.2f <= %.2f). Awaiting manual sell.", price, self.position["stop"])
                    elif hit_target:
                        log.info("MANUAL MODE: Target price reached (%.2f >= %.2f). Awaiting manual sell.", price, self.position["target"])
                    elif sig == Signal.SELL:
                        log.info("MANUAL MODE: Sell signal triggered at %.2f. Awaiting manual order.", price)
                    else:
                        log.info("MANUAL MODE: Position holding [%s %s] price=$%.2f | entry=$%.2f | stop=$%.2f | target=$%.2f | unrealized PnL=$%+.2f",
                                 self.cfg.exchange.symbol, self.cfg.exchange.timeframe, price, self.position["entry"], self.position["stop"], self.position["target"], pnl_approx)
                else:
                    if sig == Signal.BUY:
                        log.info("MANUAL MODE: Buy signal triggered at %.2f for %s. Awaiting manual order.", price, self.cfg.exchange.symbol)
                    elif stages.get("rejected_by") == "signal_debounce":
                        log.info("MANUAL MODE: %s (last trade closed too recently).", DEBOUNCE_REASON)
                    else:
                        log.info("MANUAL MODE [%s %s]: price=$%.2f | signal=%s | RSI=%.1f | ADX=%.1f (monitoring only)",
                                 self.cfg.exchange.symbol, self.cfg.exchange.timeframe, price, sig.name, rsi_val, adx_val)
            except Exception as e:
                log.warning("Manual mode monitoring error: %s — will retry on next cycle.", e)
            return True

        try:
            if self.position is None:
                if not can_trade:
                    log.warning("Not trading: %s", reason)
                    return True
                self._maybe_enter(df, price)
            else:
                self._maybe_exit(df, price)
        except Exception as e:
            log.warning("Transient error during trade decision/execution: %s — will retry on next cycle.", e)
            return True

        return True

    def manual_buy(self, custom_size: float = 0.0) -> dict:
        """Place a manual BUY order."""
        if self.kill_switch_active():
            raise RuntimeError(f"Kill switch active ({self.cfg.kill_switch_file}). Operation aborted.")
        if self.position is not None:
            raise RuntimeError("Position already open. Please close your current position before opening a new one.")
        can_trade, reason = self.risk.can_trade()
        if not can_trade:
            raise RuntimeError(f"Risk check blocked trade: {reason}")

        df = self.exchange.fetch_ohlcv(limit=50)
        if df is None or len(df) < 2:
            raise RuntimeError("Could not fetch current market price from exchange.")
        df = add_indicators(df, self.cfg.strategy)
        price = float(df["close"].iloc[-1])
        atr_val = float(df["atr"].iloc[-1]) if "atr" in df.columns and pd.notna(df["atr"].iloc[-1]) else (price * 0.02)

        stop, target = self.risk.stop_and_target(price, atr_val, "BUY")
        if custom_size > 0:
            max_size = (self.risk.state.equity * 0.99) / price
            if custom_size > max_size:
                raise RuntimeError(f"Requested size {custom_size:.6f} exceeds maximum allowed by available equity ({max_size:.6f}).")
            size = custom_size
        else:
            size = self.risk.position_size(price, stop)

        # Normalize size to exchange lot precision
        size = self.exchange.amount_to_precision(size, self.cfg.exchange.symbol)

        if size <= 0:
            if self.tracker:
                self.tracker.record_order(self.cfg.exchange.symbol, self.cfg.exchange.timeframe, "BUY", 0.0, price,
                                          mode=self.cfg.mode, status="REJECTED", rejection_reason="Calculated position size is 0")
            if self.alerts:
                self.alerts.send_alert("ORDER_REJECTED", f"Manual BUY Rejected ({self.cfg.exchange.symbol})", "Calculated position size is 0", level="warning")
            raise RuntimeError("Calculated position size is 0. Check equity or risk settings.")

        valid_notional, notional_msg = self.risk.validate_min_notional(price, size)
        if not valid_notional:
            if self.tracker:
                self.tracker.record_order(self.cfg.exchange.symbol, self.cfg.exchange.timeframe, "BUY", size, price,
                                          mode=self.cfg.mode, status="REJECTED", rejection_reason=notional_msg)
            if self.alerts:
                self.alerts.send_alert("ORDER_REJECTED", f"Manual BUY Rejected ({self.cfg.exchange.symbol})", notional_msg, level="warning")
            raise RuntimeError(notional_msg)

        # Available wallet balance verification for manual buy
        wallet = self.exchange.fetch_wallet_balance(self.cfg.exchange.symbol)
        notional_req = size * price
        free_quote = wallet.get("free_quote", 0.0)
        if (self.cfg.mode == "live" or wallet.get("is_live")) and free_quote < notional_req:
            msg = (f"Insufficient wallet balance: available {free_quote:.2f} {wallet.get('quote_currency', 'USDT')} "
                   f"< required {notional_req:.2f} for size {size:.6f}")
            if self.tracker:
                self.tracker.record_order(self.cfg.exchange.symbol, self.cfg.exchange.timeframe, "BUY", size, price,
                                          mode=self.cfg.mode, status="REJECTED", rejection_reason=msg)
            if self.alerts:
                self.alerts.send_alert("INSUFFICIENT_FUNDS", f"Insufficient Funds ({self.cfg.exchange.symbol})", msg, level="critical")
            raise RuntimeError(msg)

        try:
            fill = self.exchange.place_order("BUY", size, price)
        except Exception as e:
            if self.tracker:
                self.tracker.record_order(self.cfg.exchange.symbol, self.cfg.exchange.timeframe, "BUY", size, price,
                                          mode=self.cfg.mode, status="REJECTED", rejection_reason=str(e))
            raise

        self.position = {
            "entry": fill["price"],
            "size": size,
            "stop": stop,
            "target": target,
            "fee_paid": fill["fee"],
            "initial_stop": stop,
            "manual": True,
        }
        if self.tracker:
            self.tracker.record_order(self.cfg.exchange.symbol, self.cfg.exchange.timeframe, "BUY", size, fill["price"],
                                      fee=fill["fee"], mode=self.cfg.mode, status="FILLED")
            self.tracker.record_position_opened(self.cfg.exchange.symbol, self.cfg.exchange.timeframe, "BUY", size,
                                                fill["price"], stop, target, fill["fee"], self.cfg.mode, initial_stop=stop)
        if self.alerts:
            self.alerts.send_alert("EXECUTION", f"Position Opened: MANUAL BUY {self.cfg.exchange.symbol}",
                                   f"Filled at ${fill['price']:,.2f} | Size: {size:.6f} | Stop: ${stop:,.2f} | Target: ${target:,.2f}",
                                   level="info")
        log.info("MANUAL BUY EXECUTED: entry=%.2f stop=%.2f target=%.2f size=%.6f (R:R 1:%.2f)",
                 fill["price"], stop, target, size, self.risk.reward_risk_ratio())
        return {
            "success": True,
            "action": "BUY",
            "price": fill["price"],
            "size": size,
            "stop": stop,
            "target": target,
            "fee": fill["fee"],
            "mode": fill.get("mode", self.cfg.mode),
        }

    def manual_sell(self) -> dict:
        """Place a manual SELL / close position order."""
        if self.position is None:
            raise RuntimeError("No active open position to close.")

        df = None
        try:
            df = self.exchange.fetch_ohlcv(limit=10)
            price = float(df["close"].iloc[-1]) if (df is not None and len(df) > 0) else self.position["entry"]
        except Exception:
            try:
                price = self.exchange.current_price()
            except Exception:
                price = self.position["entry"]

        sell_size = self.position["size"]
        if self.cfg.mode == "live":
            wallet = self.exchange.fetch_wallet_balance(self.cfg.exchange.symbol)
            free_base = wallet.get("free_base", 0.0)
            if 0 < free_base < sell_size:
                log.info("Adjusting manual sell size from %.6f to available free base %.6f", sell_size, free_base)
                sell_size = free_base
        sell_size = self.exchange.amount_to_precision(sell_size, self.cfg.exchange.symbol)

        fill = self.exchange.place_order("SELL", sell_size, price)
        pnl = ((fill["price"] - self.position["entry"]) * sell_size
               - self.position["fee_paid"] - fill["fee"])
        self.risk.record_trade_result(pnl)
        closed_position = dict(self.position)
        self.position = None
        self._mark_exit(df)
        if self.tracker:
            self.tracker.record_order(self.cfg.exchange.symbol, self.cfg.exchange.timeframe, "SELL", sell_size,
                                      fill["price"], fee=fill["fee"], mode=self.cfg.mode, status="FILLED")
            self.tracker.record_position_closed(fill["price"], fill["fee"], pnl, "manual")
        if self.alerts:
            self.alerts.send_alert("EXECUTION", f"Position Closed: MANUAL SELL ({self.cfg.exchange.symbol})",
                                   f"Closed at ${fill['price']:,.2f} with PnL ${pnl:+,.2f} | Equity: ${self.risk.state.equity:,.2f}",
                                   level="info")
        log.info("MANUAL SELL EXECUTED: pnl=%+.2f equity=%.2f", pnl, self.risk.state.equity)
        return {
            "success": True,
            "action": "SELL",
            "price": fill["price"],
            "size": sell_size,
            "pnl": round(pnl, 2),
            "equity": round(self.risk.state.equity, 2),
            "fee": fill["fee"],
            "mode": fill.get("mode", self.cfg.mode),
        }

    def _maybe_enter(self, df, price):
        stages = {}
        sig = generate_signal(df, self.cfg.strategy, position_open=False, diagnostics=stages,
                              bars_since_last_trade=self._bars_since_last_exit(df),
                              timeframe=self.cfg.exchange.timeframe)
        rsi_val = float(df["rsi"].iloc[-1]) if ("rsi" in df and pd.notna(df["rsi"].iloc[-1])) else 0.0
        adx_val = float(df["adx"].iloc[-1]) if ("adx" in df and pd.notna(df["adx"].iloc[-1])) else 0.0
        trend = "BULLISH" if ("ema_fast" in df and "ema_slow" in df and df["ema_fast"].iloc[-1] > df["ema_slow"].iloc[-1]) else "BEARISH"

        if sig != Signal.BUY:
            if stages.get("core_trigger"):
                log.info("Core signal triggered: BUY | Rejected by: %s", _pipeline_rejection(stages))
                log.info("Signal pipeline [%s %s]: core_trigger=%s | fib_filter=%s | sizing=%s",
                         self.cfg.exchange.symbol, self.cfg.exchange.timeframe, stages.get("core_trigger"),
                         "not evaluated" if not stages.get("fib_evaluated") else bool(stages.get("fib_pass")),
                         "not reached")
            else:
                log.debug("Signal pipeline [%s %s]: core_trigger=False (%s)",
                          self.cfg.exchange.symbol, self.cfg.exchange.timeframe,
                          stages.get("core_reason", "no core trigger"))
            log.info("Market check [%s %s]: price=$%.2f | signal=%s | trend=%s | RSI=%.1f | ADX=%.1f | equity=$%.2f",
                     self.cfg.exchange.symbol, self.cfg.exchange.timeframe, price, sig.name, trend, rsi_val, adx_val, self.risk.state.equity)
            return

        atr_val = float(df["atr"].iloc[-1]) if "atr" in df and pd.notna(df["atr"].iloc[-1]) else (price * 0.02)
        stop, target = self.risk.stop_and_target(price, atr_val, "BUY")
        size = self.risk.position_size(price, stop)

        # Normalize size to exchange lot precision
        size = self.exchange.amount_to_precision(size, self.cfg.exchange.symbol)

        if size <= 0:
            log.warning("Core signal triggered: BUY | Rejected by: position sizing returned 0 "
                        "(equity $%.2f, risk %.2f%% per trade, stop distance $%.2f)",
                        self.risk.state.equity, self.risk.cfg.risk_per_trade_pct * 100, abs(price - stop))
            if self.tracker:
                self.tracker.record_order(self.cfg.exchange.symbol, self.cfg.exchange.timeframe, "BUY", 0.0, price,
                                          mode=self.cfg.mode, status="REJECTED", rejection_reason="Position sizing returned 0")
            if self.alerts:
                self.alerts.send_alert("ORDER_REJECTED", f"Order Sizing Rejected ({self.cfg.exchange.symbol})",
                                       f"Position sizing returned 0 (Equity: ${self.risk.state.equity:.2f}, Risk: {self.risk.cfg.risk_per_trade_pct:.2%})", level="warning")
            log.info("Signal pipeline [%s %s]: core_trigger=True | fib_filter=%s | sizing=False",
                     self.cfg.exchange.symbol, self.cfg.exchange.timeframe,
                     "not evaluated" if not stages.get("fib_evaluated") else bool(stages.get("fib_pass")))
            return

        # Minimum notional validation
        min_notional = 5.0
        try:
            if hasattr(self.exchange, "load_market_limits"):
                limits = self.exchange.load_market_limits(self.cfg.exchange.symbol)
                min_notional = limits.get("min_notional", 5.0)
        except Exception:
            pass

        valid_notional, notional_msg = self.risk.validate_min_notional(price, size, min_notional=min_notional)
        if not valid_notional:
            log.warning("Core signal triggered: BUY | Rejected by: %s", notional_msg)
            if self.tracker:
                self.tracker.record_order(self.cfg.exchange.symbol, self.cfg.exchange.timeframe, "BUY", size, price,
                                          mode=self.cfg.mode, status="REJECTED", rejection_reason=notional_msg)
            if self.alerts:
                self.alerts.send_alert("ORDER_REJECTED", f"Min Notional Rejected ({self.cfg.exchange.symbol})",
                                       notional_msg, level="warning",
                                       details={"symbol": self.cfg.exchange.symbol, "size": size, "notional": round(price * size, 2), "min_notional": min_notional})
            return

        # Order book spread & liquidity guard
        max_spread = getattr(self.risk.cfg, "max_spread_pct", 0.35)
        spread_info = self.exchange.fetch_order_book_spread(self.cfg.exchange.symbol)
        if spread_info.get("valid") and spread_info.get("spread_pct", 0.0) > max_spread:
            msg = (f"Order book spread too wide: {spread_info['spread_pct']:.3f}% "
                   f"(limit: {max_spread:.3f}%, Bid: {spread_info['bid']}, Ask: {spread_info['ask']})")
            log.warning("Core signal triggered: BUY | Rejected by: %s", msg)
            if self.tracker:
                self.tracker.record_order(self.cfg.exchange.symbol, self.cfg.exchange.timeframe, "BUY", size, price,
                                          mode=self.cfg.mode, status="REJECTED", rejection_reason=msg)
            if self.alerts:
                self.alerts.send_alert("ORDER_REJECTED", f"Spread Too Wide ({self.cfg.exchange.symbol})", msg, level="warning",
                                       details={"symbol": self.cfg.exchange.symbol, "spread_pct": spread_info["spread_pct"], "max_spread_pct": max_spread})
            return

        # Available quote wallet balance verification
        wallet = self.exchange.fetch_wallet_balance(self.cfg.exchange.symbol)
        notional_req = size * price
        free_quote = wallet.get("free_quote", 0.0)
        if (self.cfg.mode == "live" or wallet.get("is_live")) and free_quote < notional_req:
            msg = (f"Insufficient wallet balance: available {free_quote:.2f} {wallet.get('quote_currency', 'USDT')} "
                   f"< required {notional_req:.2f} for size {size:.6f}")
            log.warning("Core signal triggered: BUY | Rejected by: %s", msg)
            if self.tracker:
                self.tracker.record_order(self.cfg.exchange.symbol, self.cfg.exchange.timeframe, "BUY", size, price,
                                          mode=self.cfg.mode, status="REJECTED", rejection_reason=msg)
            if self.alerts:
                self.alerts.send_alert("INSUFFICIENT_FUNDS", f"Insufficient Funds ({self.cfg.exchange.symbol})", msg, level="critical",
                                       details={"symbol": self.cfg.exchange.symbol, "free_quote": free_quote, "required": notional_req})
            return

        try:
            fill = self.exchange.place_order("BUY", size, price)
        except Exception as e:
            if self.tracker:
                self.tracker.record_order(self.cfg.exchange.symbol, self.cfg.exchange.timeframe, "BUY", size, price,
                                          mode=self.cfg.mode, status="REJECTED", rejection_reason=str(e))
            raise

        self.position = {"entry": fill["price"], "size": size, "stop": stop,
                          "target": target, "fee_paid": fill["fee"], "initial_stop": stop}
        if self.tracker:
            self.tracker.record_order(self.cfg.exchange.symbol, self.cfg.exchange.timeframe, "BUY", size, fill["price"],
                                      fee=fill["fee"], mode=self.cfg.mode, status="FILLED")
            self.tracker.record_position_opened(self.cfg.exchange.symbol, self.cfg.exchange.timeframe, "BUY", size,
                                                fill["price"], stop, target, fill["fee"], self.cfg.mode, initial_stop=stop)
        if self.alerts:
            self.alerts.send_alert("EXECUTION", f"Position Opened: BUY {self.cfg.exchange.symbol}",
                                   f"Filled at ${fill['price']:,.2f} | Stop: ${stop:,.2f} | Target: ${target:,.2f} | Size: {size:.6f}",
                                   level="info")
        log.info("Signal pipeline [%s %s]: core_trigger=True | fib_filter=%s | sizing=True -> order placed",
                 self.cfg.exchange.symbol, self.cfg.exchange.timeframe,
                 "not evaluated" if not stages.get("fib_evaluated") else bool(stages.get("fib_pass")))
        log.info("OPENED POSITION: entry=%.2f stop=%.2f target=%.2f size=%.6f (Signal: %s, R:R 1:%.2f, "
                 "stop %.2f ATR / target %.2f ATR, trailing from +%.2f ATR)",
                  fill["price"], stop, target, size, sig.name, self.risk.reward_risk_ratio(),
                  abs(fill["price"] - stop) / atr_val if atr_val else 0.0,
                  abs(target - fill["price"]) / atr_val if atr_val else 0.0,
                  float(getattr(self.risk.cfg, "trailing_arm_atr_mult", 1.0) or 0.0))

    def _maybe_exit(self, df, price):
        atr_val = float(df["atr"].iloc[-1]) if ("atr" in df and pd.notna(df["atr"].iloc[-1])) else (price * 0.02)
        bar_high = float(df["high"].iloc[-1]) if ("high" in df and pd.notna(df["high"].iloc[-1])) else price

        previous_stop = self.position["stop"]
        self.risk.update_trailing_stop(self.position, price, atr_val, "BUY", peak_price=bar_high)
        if self.position["stop"] > previous_stop:
            log.info("TRAILING STOP raised to %.2f (peak %.2f, entry %.2f) — letting the winner run",
                     self.position["stop"], self.position.get("peak", price), self.position["entry"])

        hit_stop = price <= self.position["stop"]
        hit_target = price >= self.position["target"]
        running = self.risk.is_running(self.position, self.position.get("peak", price), atr_val, "BUY")
        sig = generate_signal(df, self.cfg.strategy, position_open=True)
        signal_exit = sig == Signal.SELL and not running

        if not (hit_stop or hit_target or signal_exit):
            pnl_approx = (price - self.position["entry"]) * self.position["size"]
            if self.tracker:
                self.tracker.update_open_position(price, self.position["stop"], trailing_stop=self.position.get("stop"), peak_price=self.position.get("peak", price))
            log.info("Holding position [%s %s]: price=$%.2f | entry=$%.2f | stop=$%.2f | target=$%.2f | unrealized PnL=$%+.2f%s",
                     self.cfg.exchange.symbol, self.cfg.exchange.timeframe, price, self.position["entry"], self.position["stop"], self.position["target"], pnl_approx,
                     " | trailing (signal exit suppressed while the trend runs)" if running and sig == Signal.SELL else "")
            return

        sell_size = self.position["size"]
        if self.cfg.mode == "live":
            wallet = self.exchange.fetch_wallet_balance(self.cfg.exchange.symbol)
            free_base = wallet.get("free_base", 0.0)
            if 0 < free_base < sell_size:
                log.info("Adjusting exit sell size from %.6f to available free base %.6f", sell_size, free_base)
                sell_size = free_base
        sell_size = self.exchange.amount_to_precision(sell_size, self.cfg.exchange.symbol)

        fill = self.exchange.place_order("SELL", sell_size, price)
        pnl = ((fill["price"] - self.position["entry"]) * sell_size
               - self.position["fee_paid"] - fill["fee"])
        reason = "stop" if hit_stop else ("target" if hit_target else "signal")
        if hit_stop and self.position.get("trail_armed"):
            reason = "trail"
        self.risk.record_trade_result(pnl)
        if self.tracker:
            self.tracker.record_order(self.cfg.exchange.symbol, self.cfg.exchange.timeframe, "SELL", sell_size,
                                      fill["price"], fee=fill["fee"], mode=self.cfg.mode, status="FILLED")
            self.tracker.record_position_closed(fill["price"], fill["fee"], pnl, reason)
        if self.alerts:
            self.alerts.send_alert("EXECUTION", f"Position Closed: {reason.upper()} ({self.cfg.exchange.symbol})",
                                   f"Closed at ${fill['price']:,.2f} with PnL ${pnl:+,.2f} | Total Equity: ${self.risk.state.equity:,.2f}",
                                   level="info", details={"symbol": self.cfg.exchange.symbol, "exit": fill["price"], "pnl": round(pnl, 2), "reason": reason, "equity": round(self.risk.state.equity, 2)})
        log.info("CLOSED POSITION (%s): entry=%.2f exit=%.2f pnl=%+.2f equity=%.2f%s",
                  reason, self.position["entry"], fill["price"], pnl, self.risk.state.equity,
                  "" if reason != "trail" else f" (trailing stop {self.position['stop']:.2f})")
        self.position = None
        self._mark_exit(df)

    def run(self):
        log.info("Bot starting: mode=%s symbol=%s timeframe=%s equity=%.2f",
                  self.cfg.mode, self.cfg.exchange.symbol, self.cfg.exchange.timeframe,
                  self.cfg.starting_equity)
        try:
            while self.step():
                for _ in range(self.cfg.poll_interval_sec):
                    if self.kill_switch_active():
                        break
                    time.sleep(1)
        except KeyboardInterrupt:
            log.info("Stopped by user (Ctrl+C).")
        log.info("Bot stopped. Final equity: %.2f", self.risk.state.equity)


if __name__ == "__main__":
    setup_logging()
    cfg = BotConfig()

    if cfg.mode == "live":
        log.warning("LIVE MODE — real funds will be traded. Ctrl+C within 10s to abort.")
        time.sleep(10)

    TradingBot(cfg).run()
