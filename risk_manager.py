"""
Risk first, profit second. This module is the only place that decides
how big a trade is and whether trading is allowed to happen at all.
"""
import logging
from dataclasses import dataclass

from config import RiskConfig

log = logging.getLogger("risk")


@dataclass
class RiskState:
    equity: float
    peak_equity: float
    daily_start_equity: float
    consecutive_losses: int = 0
    trading_halted_today: bool = False
    kill_switch_triggered: bool = False


class RiskManager:
    def __init__(self, cfg: RiskConfig, starting_equity: float):
        self.cfg = cfg
        self.state = RiskState(
            equity=starting_equity,
            peak_equity=starting_equity,
            daily_start_equity=starting_equity,
        )

    def position_size(self, entry_price: float, stop_price: float) -> float:
        """Risk a fixed % of equity per trade, sized by stop distance (not a fixed $ bet)."""
        if self.state.equity <= 0 or entry_price <= 0:
            return 0.0
        risk_amount = self.state.equity * self.cfg.risk_per_trade_pct
        stop_distance = abs(entry_price - stop_price)
        if stop_distance <= 0:
            return 0.0
        size = risk_amount / stop_distance
        # Ensure spot order value does not exceed available equity
        max_size = (self.state.equity * 0.99) / entry_price
        return max(0.0, min(size, max_size))

    def stop_and_target(self, entry_price: float, atr_value: float, side: str) -> tuple[float, float]:
        if side == "BUY":
            stop = entry_price - self.cfg.stop_loss_atr_mult * atr_value
            target = entry_price + self.cfg.take_profit_atr_mult * atr_value
        else:
            stop = entry_price + self.cfg.stop_loss_atr_mult * atr_value
            target = entry_price - self.cfg.take_profit_atr_mult * atr_value
        return stop, target

    def record_trade_result(self, pnl: float) -> None:
        self.state.equity += pnl
        self.state.peak_equity = max(self.state.peak_equity, self.state.equity)
        self.state.consecutive_losses = 0 if pnl > 0 else self.state.consecutive_losses + 1

    def reset_day(self) -> None:
        self.state.daily_start_equity = self.state.equity
        self.state.trading_halted_today = False

    def can_trade(self) -> tuple[bool, str]:
        s, c = self.state, self.cfg

        if s.equity <= 0:
            return False, "Circuit breaker: equity fully depleted (<= 0)"

        drawdown = (s.peak_equity - s.equity) / s.peak_equity if s.peak_equity > 0 else 0
        if drawdown >= c.max_drawdown_pct:
            s.kill_switch_triggered = True
            return False, f"Kill switch: max drawdown {drawdown:.1%} reached"

        if s.kill_switch_triggered:
            return False, "Kill switch previously triggered"

        daily_loss = (
            (s.daily_start_equity - s.equity) / s.daily_start_equity
            if s.daily_start_equity > 0 else 0
        )
        if daily_loss >= c.max_daily_loss_pct:
            s.trading_halted_today = True

        if s.trading_halted_today:
            return False, f"Circuit breaker: daily loss limit hit ({daily_loss:.1%})"

        if s.consecutive_losses >= c.max_consecutive_losses:
            return False, f"Circuit breaker: {s.consecutive_losses} consecutive losses"

        return True, "OK"
