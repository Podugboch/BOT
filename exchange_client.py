"""
Thin wrapper around ccxt (https://github.com/ccxt/ccxt), which supports 100+
exchanges through one API. Handles both real order placement (live) and a
simulated fill engine (paper) that charges the same fees/slippage a real
order would, so paper results are a fair preview of live performance.
"""
import logging

import ccxt
import pandas as pd

from config import ExchangeConfig

log = logging.getLogger("exchange")


class ExchangeClient:
    def __init__(self, cfg: ExchangeConfig, mode: str, fee_pct: float,
                 slippage_pct: float, starting_equity: float):
        self.cfg = cfg
        self.mode = mode
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct

        if not hasattr(ccxt, cfg.exchange_id):
            raise ValueError(
                f"'{cfg.exchange_id}' is not a known ccxt exchange id. "
                f"Run `python -c \"import ccxt; print(ccxt.exchanges)\"` to see all supported ids."
            )

        exchange_class = getattr(ccxt, cfg.exchange_id)
        exchange_opts = {
            "apiKey": cfg.api_key,
            "secret": cfg.api_secret,
            "enableRateLimit": True,
        }
        if cfg.api_passphrase:
            exchange_opts["password"] = cfg.api_passphrase  # ccxt's field name for passphrase
        self.exchange = exchange_class(exchange_opts)
        if cfg.testnet and hasattr(self.exchange, "set_sandbox_mode"):
            try:
                self.exchange.set_sandbox_mode(True)
            except Exception:
                pass

        # paper trading book-keeping
        self.paper_cash = starting_equity
        self.paper_position = 0.0

    def fetch_ohlcv(self, limit: int = 200) -> pd.DataFrame:
        raw = self.exchange.fetch_ohlcv(self.cfg.symbol, timeframe=self.cfg.timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        return df

    def current_price(self) -> float:
        return self.exchange.fetch_ticker(self.cfg.symbol)["last"]

    def place_order(self, side: str, size: float, ref_price: float) -> dict:
        if self.mode == "live":
            return self._place_live_order(side, size)
        return self._simulate_fill(side, size, ref_price)

    # -- live --------------------------------------------------------------
    def _place_live_order(self, side: str, size: float) -> dict:
        try:
            order = self.exchange.create_order(self.cfg.symbol, "market", side.lower(), size)
            fill_price = order.get("average") or order.get("price") or self.current_price()
            fee = sum(f.get("cost", 0) for f in order.get("fees", []) or [])
            log.info("LIVE %s order placed: id=%s price=%.2f", side, order.get("id"), fill_price)
            return {"side": side, "size": size, "price": fill_price, "fee": fee, "mode": "live"}
        except Exception:
            log.exception("Live order execution failed")
            raise

    # -- paper ---------------------------------------------------------------
    def _simulate_fill(self, side: str, size: float, ref_price: float) -> dict:
        """Fill against ref_price with the same slippage direction a real market
        order would suffer (worse price, never better)."""
        slip = ref_price * self.slippage_pct * (1 if side == "BUY" else -1)
        fill_price = ref_price + slip
        fee = fill_price * size * self.fee_pct

        if side == "BUY":
            self.paper_cash -= fill_price * size + fee
            self.paper_position += size
        else:
            self.paper_cash += fill_price * size - fee
            self.paper_position -= size

        log.info("PAPER %s %.6f @ %.2f (fee %.4f)", side, size, fill_price, fee)
        return {"side": side, "size": size, "price": fill_price, "fee": fee, "mode": "paper"}
