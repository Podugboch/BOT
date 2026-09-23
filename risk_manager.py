"""
Risk first, profit second. This module is the only place that decides
how big a trade is and whether trading is allowed to happen at all.

Two behaviours matter more than the rest here:

* **Expectancy.** With a 30-45% win rate the winners have to be worth several
  times the losers, so stops sit tight (1.5 ATR), targets sit far (>= 2R) and a
  trailing stop takes over once the trade is in profit.
* **Circuit breakers that recover.** A losing streak is normal in trend
  trading; it triggers a rolling cooldown with a published resume time instead
  of silently vetoing every later signal forever.
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from config import RiskConfig

log = logging.getLogger("risk")


def _as_utc(now=None) -> datetime:
    """Wall clock by default; the backtester passes the candle timestamp so a
    cooldown is measured in market time, not in seconds of CPU time."""
    if now is None:
        return datetime.now(timezone.utc)
    if isinstance(now, datetime):
        return now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(now))
    except ValueError:
        return datetime.now(timezone.utc)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


MIN_TAKE_PROFIT_ATR_MULT = 2.5   # a winner must be able to reach this far


def _fmt_until(when: datetime) -> str:
    return _as_utc(when).strftime("%Y-%m-%d %H:%M UTC")


def _fmt_span(start: datetime, end: datetime) -> str:
    minutes = max(0, int((_as_utc(end) - _as_utc(start)).total_seconds() // 60))
    hours, mins = divmod(minutes, 60)
    return f"in {hours}h {mins:02d}m" if hours else f"in {mins}m"


def reward_risk_atr_mult(cfg) -> float:
    """Target size in ATR multiples for this config.

    Floored three ways: the configured `take_profit_atr_mult`,
    `MIN_TAKE_PROFIT_ATR_MULT` (a winner has to be able to run at least 2.5 ATR
    or a trailing stop has nothing to protect), and `min_reward_risk_ratio` x the
    stop distance. Whichever is widest wins, so no combination of settings can
    quietly shrink the primary target into a micro-gain.
    """
    target_mult = float(getattr(cfg, "take_profit_atr_mult", 4.0) or 0.0)
    stop_mult = max(0.0, float(getattr(cfg, "stop_loss_atr_mult", 1.5) or 0.0))
    min_rr = float(getattr(cfg, "min_reward_risk_ratio", 2.0) or 0.0)
    # Never accept a payoff below `min_rr` x risk: with a sub-50% win rate a
    # 1:1 setup loses money even when the signal itself is decent.
    return max(target_mult, MIN_TAKE_PROFIT_ATR_MULT, min_rr * stop_mult)


def reward_risk_ratio(cfg) -> float:
    """Configured reward:risk, e.g. 2.67 for a 1.5 ATR stop and a 4 ATR target."""
    stop_mult = max(0.0, float(getattr(cfg, "stop_loss_atr_mult", 1.5) or 0.0))
    return reward_risk_atr_mult(cfg) / stop_mult if stop_mult > 0 else 0.0


@dataclass
class RiskState:
    equity: float
    peak_equity: float
    daily_start_equity: float
    consecutive_losses: int = 0
    trading_halted_today: bool = False
    kill_switch_triggered: bool = False
    cooldown_until: Optional[datetime] = None   # rolling pause after a losing streak
    cooldown_losses: int = 0                    # streak size that tripped it


class RiskManager:
    def __init__(self, cfg: RiskConfig, starting_equity: float):
        self.cfg = cfg
        self.state = RiskState(
            equity=starting_equity,
            peak_equity=starting_equity,
            daily_start_equity=starting_equity,
        )

    # -- sizing ------------------------------------------------------------
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

    # -- stop / target -----------------------------------------------------
    def reward_risk_ratio(self) -> float:
        """Realised target-to-stop multiple under the current config."""
        return reward_risk_ratio(self.cfg)

    def _reward_atr_mult(self) -> float:
        return reward_risk_atr_mult(self.cfg)

    def stop_and_target(self, entry_price: float, atr_value: float, side: str) -> tuple[float, float]:
        stop_distance = float(getattr(self.cfg, "stop_loss_atr_mult", 1.5) or 0.0) * atr_value
        reward_distance = self._reward_atr_mult() * atr_value
        if str(side).upper() == "BUY":
            stop = entry_price - stop_distance
            target = entry_price + reward_distance
        else:
            stop = entry_price + stop_distance
            target = entry_price - reward_distance
        return stop, target

    # -- trailing stop -----------------------------------------------------
    def trailing_enabled(self) -> bool:
        """Is the ATR trailing stop in play at all for this config?"""
        return bool(getattr(self.cfg, "trailing_stop_enabled", True))

    def is_running(self, position: dict, price: float, atr_value: float, side: str = "BUY") -> bool:
        """True once the trade is in profit past the trailing arm threshold.

        This is the "let it run" gate: while it holds, a reversal signal is not
        enough to close the trade, because the trailing stop (which by then sits
        above the entry) is what protects the gain.
        """
        if not self.trailing_enabled():
            return False
        arm_mult = float(getattr(self.cfg, "trailing_arm_atr_mult", 1.0) or 0.0)
        if arm_mult <= 0 or atr_value <= 0 or not position:
            return False
        entry = float(position.get("entry", price))
        if str(side).upper() == "BUY":
            return (price - entry) >= arm_mult * atr_value
        return (entry - price) >= arm_mult * atr_value

    def update_trailing_stop(self, position: dict, price: float, atr_value: float,
                             side: str = "BUY", peak_price: Optional[float] = None) -> float:
        """Ratchet the stop behind the best price once the trade is in profit.

        The stop only ever moves in the trade's favour (to break-even first, then
        behind the peak), so a winner can run past the fixed target without
        handing the gain back, while noise below the arm threshold cannot stop
        the trade out early.

        `peak_price` is the bar's extreme (high on a long) so the trail follows
        the *highest high since entry* rather than the close; callers that have
        no candle handy can omit it and the close is used. Returns the current
        stop unchanged when the trailing stop is disabled.
        """
        if not position:
            return 0.0
        if not self.trailing_enabled():
            return float(position.get("stop", 0.0))
        trail_mult = float(getattr(self.cfg, "trailing_stop_atr_mult", 2.0) or 0.0)
        if trail_mult <= 0 or atr_value <= 0:
            return float(position.get("stop", 0.0))

        entry = float(position.get("entry", price))
        try:
            extreme = float(peak_price) if peak_price is not None else float(price)
        except (TypeError, ValueError):
            extreme = float(price)

        # Once armed the stop first moves to break-even, then trails the best
        # price. It never dips below break-even: a trail that tightens the stop
        # into a smaller loss just converts winners into losers.
        if str(side).upper() == "BUY":
            peak = max(float(position.get("peak", entry)), float(price), extreme)
            position["peak"] = peak
            if not self.is_running(position, peak, atr_value, side):
                return float(position["stop"])
            trailed = max(peak - trail_mult * atr_value, entry)
            if trailed > float(position["stop"]):
                position["stop"] = trailed
                position["trail_armed"] = True
            return float(position["stop"])

        trough = min(float(position.get("trough", entry)), float(price), extreme)
        position["trough"] = trough
        if not self.is_running(position, trough, atr_value, side):
            return float(position["stop"])
        trailed = min(trough + trail_mult * atr_value, entry)
        if trailed < float(position["stop"]):
            position["stop"] = trailed
            position["trail_armed"] = True
        return float(position["stop"])

    # -- circuit breakers --------------------------------------------------
    def record_trade_result(self, pnl: float, now=None) -> None:
        now = _as_utc(now)
        self.state.equity += pnl
        self.state.peak_equity = max(self.state.peak_equity, self.state.equity)

        if pnl > 0:
            streak = self.state.consecutive_losses
            self.state.consecutive_losses = 0
            if self.state.cooldown_until:
                log.info("Winning trade (PnL %+.2f) — losing-streak cooldown cleared early, "
                         "entries resumed", pnl)
                self.state.cooldown_until = None
                self.state.cooldown_losses = 0
            elif streak:
                log.info("Winning trade (PnL %+.2f) — %d-loss streak reset", pnl, streak)
            return

        self.state.consecutive_losses += 1
        limit = int(getattr(self.cfg, "max_consecutive_losses", 8) or 0)
        if limit > 0 and self.state.consecutive_losses >= limit:
            self._trip_cooldown(now)

    def _trip_cooldown(self, now: datetime) -> None:
        hours = float(getattr(self.cfg, "loss_streak_cooldown_hours", 24.0) or 0.0)
        self.state.cooldown_losses = self.state.consecutive_losses
        if hours <= 0:
            self.state.cooldown_until = None
            log.warning("Circuit breaker: %d consecutive losses — new entries blocked until a "
                        "winning trade resets the streak (loss_streak_cooldown_hours=0)",
                        self.state.cooldown_losses)
            return
        if self.state.cooldown_until and _as_utc(self.state.cooldown_until) > now:
            return                                     # already cooling off
        self.state.cooldown_until = now + timedelta(hours=hours)
        log.warning("Circuit breaker tripped: %d consecutive losses — pausing NEW entries until "
                    "%s (%g hour cooldown). The bot resumes automatically; it is not halted "
                    "permanently, and a winning trade clears it early.",
                    self.state.cooldown_losses, _fmt_until(self.state.cooldown_until), hours)

    def _clear_cooldown(self, now: datetime) -> None:
        expired = self.state.cooldown_until
        self.state.cooldown_until = None
        self.state.cooldown_losses = 0
        self.state.consecutive_losses = 0
        log.info("Circuit breaker cooldown expired at %s — trading resumed (loss streak reset)",
                 _fmt_until(now) if expired is None else _fmt_until(expired))

    def cooldown_note(self, now=None, peek: bool = False) -> str:
        """Human status of the losing-streak cooldown, '' when not cooling off.

        Also the place an expired cooldown is noticed and lifted, so a backtest
        walking months of candles resumes exactly when the pause runs out.
        Pass `peek=True` for read-only callers (the status API) that must not
        mutate risk state.
        """
        now = _as_utc(now)
        until = self.state.cooldown_until
        if not until:
            return ""
        until = _as_utc(until)
        if now >= until:
            if peek:
                return ""
            self._clear_cooldown(now)
            return ""
        return (f"cooling off after {self.state.cooldown_losses} consecutive losses — entries "
                f"resume {_fmt_until(until)} ({_fmt_span(now, until)})")

    def reset_day(self) -> None:
        self.state.daily_start_equity = self.state.equity
        self.state.trading_halted_today = False

    def can_trade(self, now=None) -> tuple[bool, str]:
        s, c = self.state, self.cfg
        now = _as_utc(now)

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
            resume = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            return False, (f"Circuit breaker: daily loss limit hit ({daily_loss:.1%}) — entries "
                           f"resume {_fmt_until(resume)} at the next UTC day roll")

        note = self.cooldown_note(now)
        if note:
            return False, f"Circuit breaker: {note}"

        limit = int(getattr(self.cfg, "max_consecutive_losses", 8) or 0)
        if limit > 0 and s.consecutive_losses >= limit:
            hours = float(getattr(self.cfg, "loss_streak_cooldown_hours", 24.0) or 0.0)
            resets = (f"resumes after the {hours:g}h cooldown or on the next winning trade"
                      if hours > 0 else "blocked until a winning trade resets the streak")
            return False, f"Circuit breaker: {s.consecutive_losses} consecutive losses — {resets}"

        return True, "OK"
