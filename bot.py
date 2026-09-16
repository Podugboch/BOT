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

from config import BotConfig
from exchange_client import ExchangeClient
from risk_manager import RiskManager
from strategy import Signal, add_indicators, generate_signal

log = logging.getLogger("bot")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler("bot.log"), logging.StreamHandler(sys.stdout)],
    )


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

    def _maybe_enter(self, df, price):
        sig = generate_signal(df, self.cfg.strategy, position_open=False)
        if sig != Signal.BUY:
            return
        atr_val = float(df["atr"].iloc[-1])
        stop, target = self.risk.stop_and_target(price, atr_val, "BUY")
        size = self.risk.position_size(price, stop)
        if size <= 0:
            return
        fill = self.exchange.place_order("BUY", size, price)
        self.position = {"entry": fill["price"], "size": size, "stop": stop,
                          "target": target, "fee_paid": fill["fee"]}
        log.info("Opened position: entry=%.2f stop=%.2f target=%.2f size=%.6f",
                  fill["price"], stop, target, size)

    def _maybe_exit(self, df, price):
        hit_stop = price <= self.position["stop"]
        hit_target = price >= self.position["target"]
        sig = generate_signal(df, self.cfg.strategy, position_open=True)

        if not (hit_stop or hit_target or sig == Signal.SELL):
            return

        fill = self.exchange.place_order("SELL", self.position["size"], price)
        pnl = ((fill["price"] - self.position["entry"]) * self.position["size"]
               - self.position["fee_paid"] - fill["fee"])
        self.risk.record_trade_result(pnl)
        log.info("Closed position (%s): pnl=%.2f equity=%.2f",
                  "stop" if hit_stop else ("target" if hit_target else "signal"),
                  pnl, self.risk.state.equity)
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
