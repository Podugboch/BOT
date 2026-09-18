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
import pandas as pd

from config import BotConfig
from exchange_client import ExchangeClient
from risk_manager import RiskManager
from strategy import Signal, add_indicators, generate_signal

log = logging.getLogger("bot")


def setup_logging():
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler("bot.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    has_file = any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "").endswith("bot.log") for h in root.handlers)
    if not has_file:
        root.addHandler(file_handler)

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

    def kill_switch_active(self) -> bool:
        return os.path.exists(self.cfg.kill_switch_file)

    def maybe_reset_day(self):
        today = datetime.now(timezone.utc).date()
        if today != self.last_day:
            self.risk.reset_day()
            self.last_day = today
            log.info("New UTC trading day — daily loss counter reset.")

    def step(self) -> bool:
        """One decision cycle. Returns False if the bot should stop."""
        if self.kill_switch_active():
            log.critical("Kill switch file '%s' detected — halting.", self.cfg.kill_switch_file)
            return False

        self.maybe_reset_day()

        try:
            df = self.exchange.fetch_ohlcv(limit=200)
        except Exception:
            log.exception("Market data fetch failed — skipping this cycle, will retry.")
            return True

        if df is None or len(df) < 2:
            log.warning("Insufficient market data returned (%s bars) — skipping this cycle.", 0 if df is None else len(df))
            return True

        df = add_indicators(df, self.cfg.strategy)
        price = float(df["close"].iloc[-1])
        can_trade, reason = self.risk.can_trade()

        # In manual execution mode, monitor market and signals without auto-placing orders
        if getattr(self.cfg, "execution_mode", "auto") == "manual":
            sig = generate_signal(df, self.cfg.strategy, position_open=bool(self.position))
            rsi_val = float(df["rsi"].iloc[-1]) if ("rsi" in df and pd.notna(df["rsi"].iloc[-1])) else 0.0
            adx_val = float(df["adx"].iloc[-1]) if ("adx" in df and pd.notna(df["adx"].iloc[-1])) else 0.0
            if self.position is not None:
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
                else:
                    log.info("MANUAL MODE [%s %s]: price=$%.2f | signal=%s | RSI=%.1f | ADX=%.1f (monitoring only)",
                             self.cfg.exchange.symbol, self.cfg.exchange.timeframe, price, sig.name, rsi_val, adx_val)
            return True

        try:
            if self.position is None:
                if not can_trade:
                    log.warning("Not trading: %s", reason)
                    return True
                self._maybe_enter(df, price)
            else:
                self._maybe_exit(df, price)
        except Exception:
            log.exception("Error during trade decision/execution — halting to avoid unknown state.")
            return False

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
        size = custom_size if custom_size > 0 else self.risk.position_size(price, stop)
        if size <= 0:
            raise RuntimeError("Calculated position size is 0. Check equity or risk settings.")

        fill = self.exchange.place_order("BUY", size, price)
        self.position = {
            "entry": fill["price"],
            "size": size,
            "stop": stop,
            "target": target,
            "fee_paid": fill["fee"],
            "manual": True,
        }
        log.info("MANUAL BUY EXECUTED: entry=%.2f stop=%.2f target=%.2f size=%.6f", fill["price"], stop, target, size)
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

        df = self.exchange.fetch_ohlcv(limit=10)
        price = float(df["close"].iloc[-1]) if (df is not None and len(df) > 0) else self.position["entry"]

        fill = self.exchange.place_order("SELL", self.position["size"], price)
        pnl = ((fill["price"] - self.position["entry"]) * self.position["size"]
               - self.position["fee_paid"] - fill["fee"])
        self.risk.record_trade_result(pnl)
        closed_position = dict(self.position)
        self.position = None
        log.info("MANUAL SELL EXECUTED: pnl=%+.2f equity=%.2f", pnl, self.risk.state.equity)
        return {
            "success": True,
            "action": "SELL",
            "price": fill["price"],
            "size": closed_position["size"],
            "pnl": round(pnl, 2),
            "equity": round(self.risk.state.equity, 2),
            "fee": fill["fee"],
            "mode": fill.get("mode", self.cfg.mode),
        }

    def _maybe_enter(self, df, price):
        sig = generate_signal(df, self.cfg.strategy, position_open=False)
        rsi_val = float(df["rsi"].iloc[-1]) if ("rsi" in df and pd.notna(df["rsi"].iloc[-1])) else 0.0
        adx_val = float(df["adx"].iloc[-1]) if ("adx" in df and pd.notna(df["adx"].iloc[-1])) else 0.0
        trend = "BULLISH" if ("ema_fast" in df and "ema_slow" in df and df["ema_fast"].iloc[-1] > df["ema_slow"].iloc[-1]) else "BEARISH"

        if sig != Signal.BUY:
            log.info("Market check [%s %s]: price=$%.2f | signal=%s | trend=%s | RSI=%.1f | ADX=%.1f | equity=$%.2f",
                     self.cfg.exchange.symbol, self.cfg.exchange.timeframe, price, sig.name, trend, rsi_val, adx_val, self.risk.state.equity)
            return

        atr_val = float(df["atr"].iloc[-1]) if "atr" in df and pd.notna(df["atr"].iloc[-1]) else (price * 0.02)
        stop, target = self.risk.stop_and_target(price, atr_val, "BUY")
        size = self.risk.position_size(price, stop)
        if size <= 0:
            log.warning("Signal is BUY at $%.2f, but position sizing calculated 0 (check equity/risk)", price)
            return
        fill = self.exchange.place_order("BUY", size, price)
        self.position = {"entry": fill["price"], "size": size, "stop": stop,
                          "target": target, "fee_paid": fill["fee"]}
        log.info("OPENED POSITION: entry=%.2f stop=%.2f target=%.2f size=%.6f (Signal: %s)",
                  fill["price"], stop, target, size, sig.name)

    def _maybe_exit(self, df, price):
        hit_stop = price <= self.position["stop"]
        hit_target = price >= self.position["target"]
        sig = generate_signal(df, self.cfg.strategy, position_open=True)

        if not (hit_stop or hit_target or sig == Signal.SELL):
            pnl_approx = (price - self.position["entry"]) * self.position["size"]
            log.info("Holding position [%s %s]: price=$%.2f | entry=$%.2f | stop=$%.2f | target=$%.2f | unrealized PnL=$%+.2f",
                     self.cfg.exchange.symbol, self.cfg.exchange.timeframe, price, self.position["entry"], self.position["stop"], self.position["target"], pnl_approx)
            return

        fill = self.exchange.place_order("SELL", self.position["size"], price)
        pnl = ((fill["price"] - self.position["entry"]) * self.position["size"]
               - self.position["fee_paid"] - fill["fee"])
        self.risk.record_trade_result(pnl)
        log.info("CLOSED POSITION (%s): entry=%.2f exit=%.2f pnl=%+.2f equity=%.2f",
                  "stop" if hit_stop else ("target" if hit_target else "signal"),
                  self.position["entry"], fill["price"], pnl, self.risk.state.equity)
        self.position = None

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
