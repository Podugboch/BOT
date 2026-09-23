"""
Comprehensive test suite for Crypto Trading Bot.
Covers indicators, strategy signals, risk management, execution, and backtesting.
"""
import os
import re
import shutil
import subprocess
import tempfile
import json
from datetime import datetime, timedelta, timezone

import ccxt
import pytest
import numpy as np
import pandas as pd

import backtester
import exchange_client
from config import BotConfig, StrategyConfig, RiskConfig, ExchangeConfig
from indicators import ema, rsi, atr, adx, fib_retracement_levels
from strategy import Signal, add_indicators, generate_signal, _near_fib_support, _fib_pullback_ok
from risk_manager import RiskManager
from exchange_client import ExchangeClient
from backtester import run_backtest, summarize, fetch_historical
from bot import TradingBot


@pytest.fixture
def sample_ohlcv():
    """Generates 120 bars of sample OHLCV data."""
    np.random.seed(42)
    n = 120
    dates = pd.date_range("2026-01-01", periods=n, freq="15min")
    close = 50000 + np.cumsum(np.random.normal(10, 50, n))
    high = close + np.random.uniform(20, 100, n)
    low = close - np.random.uniform(20, 100, n)
    open_p = close + np.random.normal(0, 20, n)
    volume = np.random.uniform(50, 500, n)
    return pd.DataFrame({
        "ts": dates,
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


# ============================================================================
# INDICATORS TESTS
# ============================================================================
class TestIndicators:
    def test_ema(self, sample_ohlcv):
        series = sample_ohlcv["close"]
        res = ema(series, 12)
        assert len(res) == len(series)
        assert not res.isna().any()
        # Last EMA should reflect recent prices closely
        assert abs(res.iloc[-1] - series.iloc[-1]) < 1000

    def test_rsi_bounds(self, sample_ohlcv):
        res = rsi(sample_ohlcv["close"], 14)
        assert len(res) == len(sample_ohlcv)
        assert (res >= 0).all() and (res <= 100).all()

    def test_rsi_all_gains(self):
        series = pd.Series([10.0 + i for i in range(30)])
        res = rsi(series, 14)
        assert res.iloc[-1] > 95.0

    def test_atr(self, sample_ohlcv):
        res = atr(sample_ohlcv, 14)
        assert len(res) == len(sample_ohlcv)
        assert (res > 0).all()

    def test_adx(self, sample_ohlcv):
        res = adx(sample_ohlcv, 14)
        assert len(res) == len(sample_ohlcv)
        assert (res >= 0).all()

    def test_fib_retracement(self, sample_ohlcv):
        fib = fib_retracement_levels(sample_ohlcv, lookback=50)
        assert "swing_high" in fib.columns
        assert "swing_low" in fib.columns
        assert "fib_382" in fib.columns
        assert "fib_500" in fib.columns
        assert "fib_618" in fib.columns
        # After 50 bars, levels should be valid and ordered: high > 382 > 500 > 618 > low
        valid_row = fib.dropna().iloc[-1]
        assert valid_row["swing_high"] >= valid_row["fib_382"]
        assert valid_row["fib_382"] >= valid_row["fib_500"]
        assert valid_row["fib_500"] >= valid_row["fib_618"]
        assert valid_row["fib_618"] >= valid_row["swing_low"]


# ============================================================================
# STRATEGY TESTS
# ============================================================================
class TestStrategy:
    def test_add_indicators(self, sample_ohlcv):
        cfg = StrategyConfig()
        df = add_indicators(sample_ohlcv, cfg)
        for col in ["ema_fast", "ema_slow", "rsi", "atr", "adx", "fib_382", "fib_500", "fib_618"]:
            assert col in df.columns

    def test_signal_bullish_cross_generates_buy(self):
        cfg = StrategyConfig(use_fib_filter=False, adx_trend_threshold=20)
        # Construct 2 rows: prev fast <= slow, curr fast > slow, trending adx, low rsi
        df = pd.DataFrame([
            {"ema_fast": 100.0, "ema_slow": 100.0, "rsi": 50.0, "adx": 25.0, "close": 100.0},
            {"ema_fast": 102.0, "ema_slow": 100.5, "rsi": 55.0, "adx": 26.0, "close": 102.0},
        ])
        sig = generate_signal(df, cfg, position_open=False)
        assert sig == Signal.BUY

    def test_signal_overbought_blocks_buy(self):
        cfg = StrategyConfig(use_fib_filter=False, rsi_overbought=65)
        df = pd.DataFrame([
            {"ema_fast": 100.0, "ema_slow": 100.0, "rsi": 70.0, "adx": 25.0, "close": 100.0},
            {"ema_fast": 102.0, "ema_slow": 100.5, "rsi": 72.0, "adx": 26.0, "close": 102.0},
        ])
        sig = generate_signal(df, cfg, position_open=False)
        assert sig == Signal.HOLD

    def test_signal_bearish_cross_exits_position(self):
        cfg = StrategyConfig()
        df = pd.DataFrame([
            {"ema_fast": 102.0, "ema_slow": 100.0, "rsi": 50.0, "adx": 25.0, "close": 102.0},
            {"ema_fast": 99.0, "ema_slow": 100.0, "rsi": 45.0, "adx": 25.0, "close": 99.0},
        ])
        sig = generate_signal(df, cfg, position_open=True)
        assert sig == Signal.SELL

    def test_fib_support_filter(self):
        cfg = StrategyConfig(fib_tolerance_pct=0.01)
        curr_matching = pd.Series({"close": 100.0, "fib_382": 100.5, "fib_500": 95.0, "fib_618": 90.0})
        assert _near_fib_support(curr_matching, cfg) is True

        curr_far = pd.Series({"close": 100.0, "fib_382": 110.0, "fib_500": 90.0, "fib_618": 80.0})
        assert _near_fib_support(curr_far, cfg) is False


# ============================================================================
# RISK MANAGER TESTS
# ============================================================================
class TestRiskManager:
    def test_position_sizing(self):
        cfg = RiskConfig(risk_per_trade_pct=0.01)
        risk = RiskManager(cfg, starting_equity=1000.0)
        # Entry 100, Stop 95 -> risk = $10, distance = $5 -> size = 2
        size = risk.position_size(entry_price=100.0, stop_price=95.0)
        assert size == 2.0

    def test_position_size_spot_cap(self):
        cfg = RiskConfig(risk_per_trade_pct=0.01)
        risk = RiskManager(cfg, starting_equity=1000.0)
        # Extremely small stop distance -> uncapped size would be $10 / 0.001 = 10000
        # But max spot size is (1000 * 0.99) / 100 = 9.9
        size = risk.position_size(entry_price=100.0, stop_price=99.999)
        assert size == pytest.approx(9.9, rel=1e-3)

    def test_stops_and_targets(self):
        cfg = RiskConfig(stop_loss_atr_mult=2.0, take_profit_atr_mult=4.0)
        risk = RiskManager(cfg, starting_equity=1000.0)
        stop, target = risk.stop_and_target(entry_price=100.0, atr_value=5.0, side="BUY")
        assert stop == 90.0
        assert target == 120.0

    def test_drawdown_kill_switch(self):
        cfg = RiskConfig(max_drawdown_pct=0.15)
        risk = RiskManager(cfg, starting_equity=1000.0)
        assert risk.can_trade()[0] is True

        # Lose $160 (16% drawdown)
        risk.record_trade_result(-160.0)
        can_trade, reason = risk.can_trade()
        assert can_trade is False
        assert "Kill switch" in reason

    def test_daily_loss_circuit_breaker(self):
        cfg = RiskConfig(max_daily_loss_pct=0.03)
        risk = RiskManager(cfg, starting_equity=1000.0)
        # Lose $35 (3.5%)
        risk.record_trade_result(-35.0)
        can_trade, reason = risk.can_trade()
        assert can_trade is False
        assert "Circuit breaker: daily loss" in reason

        # Reset day should restore trading
        risk.reset_day()
        can_trade, reason = risk.can_trade()
        assert can_trade is True
        assert reason == "OK"

    def test_consecutive_losses_circuit_breaker(self):
        cfg = RiskConfig(max_consecutive_losses=3)
        risk = RiskManager(cfg, starting_equity=1000.0)
        risk.record_trade_result(-5.0)
        risk.record_trade_result(-5.0)
        assert risk.can_trade()[0] is True
        risk.record_trade_result(-5.0)  # 3rd consecutive loss
        can_trade, reason = risk.can_trade()
        assert can_trade is False
        assert "3 consecutive losses" in reason


# ============================================================================
# EXCHANGE CLIENT (PAPER MODE) TESTS
# ============================================================================
class TestLossStreakCooldown:
    """A losing streak must cool the bot off, not silence it forever."""

    @staticmethod
    def _trip(risk, losses=8, start=None):
        """Trip the losing-streak breaker without touching the daily-loss one."""
        start = start or datetime(2026, 3, 14, 12, 0, tzinfo=timezone.utc)
        trip_time = start
        for i in range(losses):
            trip_time = start + timedelta(minutes=i)
            risk.record_trade_result(-1.0, now=trip_time)   # 8 x $1 = 0.8% of $1000
        return trip_time

    def test_streak_limit_is_raised_above_a_normal_run_of_losses(self):
        """Four losses in a row is ordinary in trend trading — no longer a halt."""
        cfg = RiskConfig()
        assert cfg.max_consecutive_losses == 8
        assert cfg.loss_streak_cooldown_hours > 0
        risk = RiskManager(cfg, starting_equity=1000.0)

        for _ in range(4):
            risk.record_trade_result(-5.0, now=datetime(2026, 3, 14, tzinfo=timezone.utc))
        can_trade, reason = risk.can_trade(now=datetime(2026, 3, 14, tzinfo=timezone.utc))
        assert can_trade is True
        assert reason == "OK"

    def test_cooldown_blocks_then_resumes_by_itself(self):
        cfg = RiskConfig()
        risk = RiskManager(cfg, starting_equity=1000.0)
        tripped_at = self._trip(risk)

        can_trade, reason = risk.can_trade(now=tripped_at + timedelta(minutes=30))
        assert can_trade is False
        assert "8 consecutive losses" in reason
        assert "resume" in reason
        # The exact resume timestamp is published, not just "blocked".
        assert (tripped_at + timedelta(hours=24)).strftime("%Y-%m-%d %H:%M UTC") in reason

        # ...and it lifts on its own once the cooldown window passes.
        can_trade, reason = risk.can_trade(now=tripped_at + timedelta(hours=25))
        assert can_trade is True
        assert reason == "OK"
        assert risk.state.consecutive_losses == 0
        assert risk.state.cooldown_until is None

    def test_tripping_the_breaker_logs_the_expiry_timestamp(self, caplog):
        cfg = RiskConfig()
        risk = RiskManager(cfg, starting_equity=1000.0)
        start = datetime(2026, 3, 14, 12, 0, tzinfo=timezone.utc)

        with caplog.at_level("WARNING", logger="risk"):
            tripped_at = self._trip(risk, start=start)

        text = " ".join(record.getMessage() for record in caplog.records)
        assert "Circuit breaker tripped: 8 consecutive losses" in text
        assert (tripped_at + timedelta(hours=24)).strftime("%Y-%m-%d %H:%M UTC") in text
        assert "resumes automatically" in text
        assert risk.state.cooldown_until == tripped_at + timedelta(hours=24)

    def test_winning_trade_clears_the_cooldown_early(self):
        cfg = RiskConfig()
        risk = RiskManager(cfg, starting_equity=1000.0)
        tripped_at = self._trip(risk)
        assert risk.can_trade(now=tripped_at + timedelta(hours=1))[0] is False

        risk.record_trade_result(+25.0, now=tripped_at + timedelta(hours=1))
        can_trade, reason = risk.can_trade(now=tripped_at + timedelta(hours=1))
        assert can_trade is True
        assert reason == "OK"
        assert risk.state.cooldown_until is None

    def test_zero_cooldown_keeps_the_old_hard_halt(self):
        cfg = RiskConfig(loss_streak_cooldown_hours=0.0)
        risk = RiskManager(cfg, starting_equity=1000.0)
        tripped_at = self._trip(risk)

        can_trade, reason = risk.can_trade(now=tripped_at + timedelta(days=30))
        assert can_trade is False
        assert "until a winning trade" in reason

        risk.record_trade_result(+10.0, now=tripped_at + timedelta(days=30))
        assert risk.can_trade(now=tripped_at + timedelta(days=30))[0] is True

    def test_backtest_rejections_carry_the_cooldown_expiry(self):
        """The run report must say when trading resumes, not just 'circuit breaker'."""
        res = run_backtest(TestPipelineDiagnostics._series(), TestPipelineDiagnostics._diag_cfg())
        lines = res["diagnostics"]["rejection_log"]

        assert any("Circuit breaker" in line and "resume" in line for line in lines)

    def test_backtest_resumes_after_the_cooldown_mid_run(self):
        """More trades than the old 4-loss hard halt would ever allow."""
        cfg = TestPipelineDiagnostics._diag_cfg()
        res = run_backtest(TestPipelineDiagnostics._series(), cfg)
        diag = res["diagnostics"]

        assert diag["rejections"].get("risk_gate", 0) < diag["trades_executed"]
        assert res["trades_executed"] > 10


class TestRewardRiskBalance:
    """Stop 1.5 ATR, target >= 2R, and a trail that protects a running winner."""

    @staticmethod
    def _risk(**overrides):
        return RiskManager(RiskConfig(**overrides), starting_equity=1000.0)

    def test_defaults_take_at_least_two_and_a_half_times_the_risk(self):
        cfg = RiskConfig()
        assert cfg.stop_loss_atr_mult == 1.5
        assert cfg.take_profit_atr_mult >= 3.0          # >= 3 ATR as requested
        assert cfg.take_profit_atr_mult / cfg.stop_loss_atr_mult >= 2.5
        assert cfg.min_reward_risk_ratio >= 2.0

        risk = self._risk()
        stop, target = risk.stop_and_target(entry_price=100.0, atr_value=4.0, side="BUY")
        assert stop == pytest.approx(100.0 - 1.5 * 4.0)
        assert target - 100.0 >= 2.0 * (100.0 - stop)
        assert risk.reward_risk_ratio() == pytest.approx(cfg.take_profit_atr_mult / 1.5)

    def test_thin_targets_are_lifted_to_the_minimum_reward_risk(self):
        """A 1:1.2 setup would bleed with a 30-45% win rate — it gets widened."""
        risk = self._risk(stop_loss_atr_mult=2.0, take_profit_atr_mult=2.5,
                          min_reward_risk_ratio=2.0)
        stop, target = risk.stop_and_target(entry_price=100.0, atr_value=5.0, side="BUY")
        risk_distance = 100.0 - stop
        assert risk_distance == pytest.approx(10.0)                  # 2.0 ATR stop
        assert target - 100.0 == pytest.approx(2.0 * risk_distance)  # lifted to 1:2

    def test_sell_side_is_mirrored(self):
        risk = self._risk()
        stop, target = risk.stop_and_target(entry_price=100.0, atr_value=4.0, side="SELL")
        assert stop > 100.0 > target
        assert (target - 100.0) <= -2.0 * (stop - 100.0)

    def test_trailing_stop_waits_for_profit_then_only_tightens(self):
        risk = self._risk()
        position = {"entry": 100.0, "stop": 94.0, "initial_stop": 94.0}

        # Not in profit enough to arm: the original stop stands.
        assert risk.update_trailing_stop(position, 102.0, 4.0) == 94.0
        assert risk.is_running(position, 102.0, 4.0) is False

        # +1 ATR arms the trail and pulls the stop up to break-even...
        assert risk.update_trailing_stop(position, 104.0, 4.0) == pytest.approx(100.0)
        assert risk.is_running(position, 104.0, 4.0) is True

        # ...then it trails 2.0 ATR behind the best price.
        assert risk.update_trailing_stop(position, 120.0, 4.0) == pytest.approx(112.0)
        assert position["peak"] == 120.0
        # A pullback never loosens the stop again.
        assert risk.update_trailing_stop(position, 108.0, 4.0) == pytest.approx(112.0)
        assert position["trail_armed"] is True

    def test_backtest_lets_winners_run_and_reports_realized_rr(self):
        """Realised R:R must beat the ~1.1 the fix was written for."""
        cfg = TestPipelineDiagnostics._diag_cfg()
        res = run_backtest(TestPipelineDiagnostics._series(), cfg)

        assert res["num_trades"] > 5
        # Was ~1.1 (wins barely covering losses) before the stop/trail rebalance.
        assert res["realized_rr"] > 1.5
        assert abs(res["avg_win"]) > abs(res["avg_loss"])
        # The trail (not an early reversal exit) is now a real exit path.
        assert "trail" in res["exit_reasons"]
        assert set(res["exit_reasons"]) <= {"stop", "target", "trail", "signal"}
        assert "Expectancy" in res["diagnostics"]["report"]
        assert "realized R:R" in res["diagnostics"]["report"]
        # Fewer reversal-signal exits than before the change (they used to be the
        # single biggest exit reason and kept winners no bigger than losers).
        assert res["exit_reasons"].get("signal", 0) <= res["exit_reasons"].get("trail", 0)


def web_server_dashboard_html() -> str:
    """The dashboard page exactly as served — lets the UI wiring be asserted
    without a browser or a live HTTP socket."""
    import web_server
    return web_server.DASHBOARD_HTML


class TestDashboardRefreshWiring:
    """The UI Refresh buttons must always do something visible (the deployed
    build silently swallowed failed/cached refreshes)."""

    def test_refresh_buttons_are_wired_to_the_handlers(self):
        html = web_server_dashboard_html()

        for handler in ("refreshAnalysis", "refreshLogs"):
            assert f'onclick="{handler}(' in html
        # Both are explicit buttons, so a stray form wrapper can never turn a
        # click into a page submit.
        for button_id in ('id="btn-refresh-analysis"', 'id="btn-refresh-logs"'):
            assert button_id in html
        for stamp_id in ('id="an-updated"', 'id="logs-updated"'):
            assert stamp_id in html

    def test_dashboard_never_reuses_a_cached_api_response(self):
        html = web_server_dashboard_html()

        # A cached GET is the classic reason a Refresh button appears dead once
        # the app sits behind a CDN or a browser cache.
        assert html.count("cache: 'no-store'") >= 2
        assert "&_ts=" in html

    def test_failed_refresh_is_visible_instead_of_console_only(self):
        html = web_server_dashboard_html()
        assert "Could not refresh" in html

    def test_api_responses_are_marked_no_store(self):
        import web_server
        from fastapi import Response

        response = Response()
        web_server.get_logs(lines=5, response=response)
        assert "no-store" in response.headers.get("Cache-Control", "")


class TestExchangeClientPaper:
    def test_paper_execution_fees_and_slippage(self):
        cfg = ExchangeConfig(exchange_id="binance", symbol="BTC/USDT")
        client = ExchangeClient(
            cfg=cfg,
            mode="paper",
            fee_pct=0.001,      # 0.1%
            slippage_pct=0.0005, # 0.05%
            starting_equity=1000.0
        )
        # BUY 0.01 BTC at ref_price 50,000
        # Slippage: 50000 * 1.0005 = 50025.0
        # Order cost: 50025 * 0.01 = 500.25
        # Fee: 500.25 * 0.001 = 0.50025
        fill = client.place_order("BUY", size=0.01, ref_price=50000.0)
        assert fill["price"] == 50025.0
        assert fill["fee"] == pytest.approx(0.50025)
        assert client.paper_position == 0.01
        assert client.paper_cash == pytest.approx(1000.0 - 500.25 - 0.50025)

        # SELL 0.01 BTC at ref_price 52,000
        # Slippage: 52000 * 0.9995 = 51974.0
        # Order gross: 51974 * 0.01 = 519.74
        # Fee: 519.74 * 0.001 = 0.51974
        sell_fill = client.place_order("SELL", size=0.01, ref_price=52000.0)
        assert sell_fill["price"] == 51974.0
        assert client.paper_position == 0.0
        expected_cash = (1000.0 - 500.25 - 0.50025) + 519.74 - 0.51974
        assert client.paper_cash == pytest.approx(expected_cash)


# ============================================================================
# BACKTESTER TESTS
# ============================================================================
class TestBacktester:
    def test_run_backtest_with_trades(self):
        np.random.seed(42)
        n = 800
        dates = pd.date_range("2026-01-01", periods=n, freq="15min")
        # Generate wave data to trigger crosses
        trend = 50000 + np.sin(np.linspace(0, 30, n)) * 3000
        noise = np.random.normal(0, 50, n)
        close = trend + noise
        df = pd.DataFrame({
            "ts": dates,
            "open": close - 20,
            "high": close + 100,
            "low": close - 100,
            "close": close,
            "volume": np.random.uniform(10, 100, n),
        })

        cfg = BotConfig()
        cfg.strategy.use_fib_filter = False
        cfg.strategy.adx_trend_threshold = 10

        res = run_backtest(df, cfg)
        assert "starting_equity" in res
        assert "final_equity" in res
        assert "total_return_pct" in res
        assert "win_rate_pct" in res
        assert "sharpe_approx" in res
        assert res["num_trades"] > 0


# ============================================================================
# TRADING BOT TESTS
# ============================================================================
class TestTradingBot:
    def test_kill_switch_file(self):
        cfg = BotConfig()
        cfg.kill_switch_file = "TEST_KILL_FILE"
        bot = TradingBot(cfg)

        assert bot.kill_switch_active() is False
        try:
            with open(cfg.kill_switch_file, "w") as f:
                f.write("kill")
            assert bot.kill_switch_active() is True
            assert bot.step() is False
        finally:
            if os.path.exists(cfg.kill_switch_file):
                os.remove(cfg.kill_switch_file)

    def test_manual_mode_execution(self, sample_ohlcv):
        cfg = BotConfig()
        cfg.execution_mode = "manual"
        bot = TradingBot(cfg)
        bot.exchange.fetch_ohlcv = lambda *args, **kwargs: sample_ohlcv
        # In manual mode, bot.step() should run without opening auto position
        res = bot.step()
        assert res is True
        assert bot.position is None

        # Manual buy should open position
        buy_res = bot.manual_buy(custom_size=0.01)
        assert buy_res["success"] is True
        assert bot.position is not None
        assert bot.position["size"] == 0.01

        # Manual sell should close position
        sell_res = bot.manual_sell()
        assert sell_res["success"] is True
        assert bot.position is None

    def test_transient_enter_error_does_not_halt_bot(self, sample_ohlcv):
        cfg = BotConfig()
        cfg.execution_mode = "auto"
        bot = TradingBot(cfg)
        bot.exchange.fetch_ohlcv = lambda *args, **kwargs: sample_ohlcv

        # Simulate transient error (e.g., CCXT NetworkError / order timeout) during entry
        def mock_maybe_enter(df, price):
            raise RuntimeError("CCXT: Network timeout while placing entry order")

        bot._maybe_enter = mock_maybe_enter
        res = bot.step()
        assert res is True
        assert bot.position is None

    def test_transient_exit_error_does_not_halt_bot(self, sample_ohlcv):
        cfg = BotConfig()
        cfg.execution_mode = "auto"
        bot = TradingBot(cfg)
        bot.exchange.fetch_ohlcv = lambda *args, **kwargs: sample_ohlcv
        bot.position = {
            "entry": 50000.0,
            "size": 0.01,
            "stop": 49000.0,
            "target": 52000.0,
            "fee_paid": 0.5,
        }

        # Simulate transient error during exit
        def mock_maybe_exit(df, price):
            raise RuntimeError("CCXT: Temporary 502 Bad Gateway during exit order")

        bot._maybe_exit = mock_maybe_exit
        res = bot.step()
        assert res is True
        # Position remains open to allow retry on next cycle
        assert bot.position is not None

    def test_transient_indicator_error_does_not_halt_bot(self):
        cfg = BotConfig()
        bot = TradingBot(cfg)
        # Return valid OHLCV dataframe that fails in indicator calculation
        df = pd.DataFrame({"ts": [1, 2], "open": [1, 2], "high": [1, 2], "low": [1, 2], "close": ["invalid", "data"], "volume": [1, 2]})
        bot.exchange.fetch_ohlcv = lambda *args, **kwargs: df

        res = bot.step()
        assert res is True

    def test_risk_manager_kill_switch_halts_step(self):
        cfg = BotConfig()
        bot = TradingBot(cfg)
        assert bot.kill_switch_active() is False

        bot.risk.state.kill_switch_triggered = True
        assert bot.kill_switch_active() is True
        assert bot.step() is False


# ============================================================================
# MULTI-STRATEGY & SETTINGS ENDPOINT TESTS
# ============================================================================
    def test_live_entry_pipeline_logs_the_rejection_reason(self, caplog):
        """A fired core signal that is filtered out must say why in the activity log."""
        cfg = BotConfig()
        bot = TradingBot(cfg)
        df = _pipeline_cross_frame()

        with caplog.at_level("INFO", logger="bot"):
            bot._maybe_enter(df, float(df["close"].iloc[-1]))

        text = " ".join(record.getMessage() for record in caplog.records)
        assert "Core signal triggered: BUY | Rejected by: Price outside the Fibonacci" in text
        assert "core_trigger=True" in text and "sizing=not reached" in text
        assert bot.position is None

    def test_live_entry_pipeline_logs_sizing_rejection(self, caplog):
        cfg = BotConfig()
        cfg.strategy.use_fib_filter = False
        cfg.risk.risk_per_trade_pct = 0.0            # sizing collapses to 0
        bot = TradingBot(cfg)
        df = _pipeline_cross_frame()

        with caplog.at_level("INFO", logger="bot"):
            bot._maybe_enter(df, float(df["close"].iloc[-1]))

        text = " ".join(record.getMessage() for record in caplog.records)
        assert "Rejected by: position sizing returned 0" in text
        assert "sizing=False" in text
        assert bot.position is None

class TestNewIndicatorsAndMultiStrategy:
    def test_bollinger_bands(self, sample_ohlcv):
        from indicators import bollinger_bands
        bb = bollinger_bands(sample_ohlcv["close"], period=20, num_std=2.0)
        assert "bb_mid" in bb.columns
        assert "bb_upper" in bb.columns
        assert "bb_lower" in bb.columns
        valid_row = bb.iloc[-1]
        assert valid_row["bb_upper"] >= valid_row["bb_mid"] >= valid_row["bb_lower"]

    def test_macd(self, sample_ohlcv):
        from indicators import macd
        m = macd(sample_ohlcv["close"], fast=12, slow=26, signal_period=9)
        assert "macd" in m.columns
        assert "macd_signal" in m.columns
        assert "macd_hist" in m.columns

    def test_pure_ema_cross_signal(self):
        cfg = StrategyConfig(strategy_name="pure_ema_cross", adx_trend_threshold=15)
        df = pd.DataFrame([
            {"ema_fast": 100.0, "ema_slow": 101.0, "rsi": 45.0, "adx": 20.0, "close": 100.0},
            {"ema_fast": 102.0, "ema_slow": 101.0, "rsi": 48.0, "adx": 22.0, "close": 102.0},
        ])
        sig = generate_signal(df, cfg, position_open=False)
        assert sig == Signal.BUY

    def test_rsi_reversion_signal(self):
        cfg = StrategyConfig(strategy_name="rsi_reversion")
        # Oversold bounce: prev <= 30, curr > prev
        df_buy = pd.DataFrame([
            {"rsi": 25.0, "close": 100.0},
            {"rsi": 29.0, "close": 101.0},
        ])
        sig_buy = generate_signal(df_buy, cfg, position_open=False)
        assert sig_buy == Signal.BUY

        # Overbought exit
        df_sell = pd.DataFrame([
            {"rsi": 68.0, "close": 110.0},
            {"rsi": 72.0, "close": 112.0},
        ])
        sig_sell = generate_signal(df_sell, cfg, position_open=True)
        assert sig_sell == Signal.SELL

    def test_settings_api_and_5m_timeframe(self):
        from fastapi.testclient import TestClient
        from web_server import app, AVAILABLE_TIMEFRAMES

        assert "5m" in AVAILABLE_TIMEFRAMES

        client = TestClient(app)
        # GET settings
        get_res = client.get("/api/bot/settings")
        assert get_res.status_code == 200
        data = get_res.json()
        assert "5m" in data["available_timeframes"]
        assert any(s["id"] == "pure_ema_cross" for s in data["available_strategies"])

        # POST settings with 5m and manual mode
        post_res = client.post("/api/bot/settings", json={
            "execution_mode": "manual",
            "strategy": "pure_ema_cross",
            "timeframe": "5m",
            "symbol": "BTC/USDT",
        })
        assert post_res.status_code == 200
        post_data = post_res.json()
        assert post_data["execution_mode"] == "manual"
        assert post_data["timeframe"] == "5m"
        assert post_data["strategy"] == "pure_ema_cross"

        # Check status reflects settings
        st_res = client.get("/api/status")
        assert st_res.status_code == 200
        st_data = st_res.json()
        assert st_data["execution_mode"] == "manual"
        assert st_data["timeframe"] == "5m"
        assert st_data["strategy"] == "pure_ema_cross"


# ============================================================================
# FIBONACCI ZONE FILTER — STARVATION FIXES
# ============================================================================
class TestFibZoneFilter:
    """Covers the three starvation fixes: rolling swing anchors, a percentage
    band instead of an exact match, and a toggle that skips the check."""

    @staticmethod
    def _cross(extra_curr=None, extra_prev=None):
        """A clean core BUY setup: bullish EMA cross, trending ADX, tame RSI."""
        prev = {"ema_fast": 100.0, "ema_slow": 100.0, "rsi": 50.0, "adx": 25.0, "close": 100.0}
        curr = {"ema_fast": 102.0, "ema_slow": 100.5, "rsi": 55.0, "adx": 26.0, "close": 102.0}
        prev.update(extra_prev or {})
        curr.update(extra_curr or {})
        return pd.DataFrame([prev, curr])

    def test_swing_anchors_follow_the_trailing_window(self):
        """The swing high/low must roll forward with the last N bars — an old
        spike that leaves the window must stop anchoring the levels."""
        n = 40
        high = np.full(n, 101.0)
        low = np.full(n, 99.0)
        high[5] = 200.0                      # one-off spike early in the series
        df = pd.DataFrame({
            "high": high, "low": low, "close": (high + low) / 2.0,
        })
        fib = fib_retracement_levels(df, lookback=20)

        # Bar 19 still sees the spike (its 20-bar window covers index 5)...
        assert fib["swing_high"].iloc[19] == 200.0
        # ...but by bar 25 the spike has aged out of the trailing window.
        assert fib["swing_high"].iloc[25] == pytest.approx(101.0)
        assert fib["swing_high"].iloc[-1] == pytest.approx(101.0)
        # Anchors equal the trailing extremes on every warmed-up bar, i.e. the
        # window slides with the data instead of being pinned to its start.
        expected = df["high"].rolling(20).max()
        assert fib["swing_high"].dropna().equals(expected.dropna())

    def test_fib_levels_accept_a_percentage_band_not_exact_match(self):
        cfg = StrategyConfig()               # fib_tolerance_pct = 0.01 (1%)
        # Price 0.4% away from the 38.2% level is inside the band.
        assert _near_fib_support(pd.Series({"close": 100.0, "fib_382": 100.4,
                                            "fib_500": 95.0, "fib_618": 90.0}), cfg) is True
        # 1.4% away is outside it.
        assert _near_fib_support(pd.Series({"close": 100.0, "fib_382": 101.4,
                                            "fib_500": 95.0, "fib_618": 90.0}), cfg) is False
        # The band scales with price, so a $50000 asset gets a $500 band.
        assert _near_fib_support(pd.Series({"close": 50000.0, "fib_382": 50200.0,
                                            "fib_500": 49000.0, "fib_618": 48000.0}), cfg) is True

    def test_default_tolerance_is_widened_and_tunable(self):
        """0.5% starved the strategy — the default band is now 1.0% and can be
        opened up to ~1.5% for choppier books."""
        cfg = StrategyConfig()
        assert cfg.fib_tolerance_pct == pytest.approx(0.01)

        inside = pd.Series({"close": 100.0, "fib_382": 100.9, "fib_500": 95.0, "fib_618": 90.0})
        outside = pd.Series({"close": 100.0, "fib_382": 101.4, "fib_500": 95.0, "fib_618": 90.0})
        assert _near_fib_support(inside, cfg) is True
        assert _near_fib_support(outside, cfg) is False
        assert _near_fib_support(outside, StrategyConfig(fib_tolerance_pct=0.015)) is True
        # A nonsense tolerance never degrades back into an exact-price match.
        assert _near_fib_support(inside, StrategyConfig(fib_tolerance_pct=-1.0)) is True

    def test_zero_tolerance_does_not_degrade_to_exact_match(self):
        cfg = StrategyConfig(fib_tolerance_pct=0.0)
        assert _near_fib_support(pd.Series({"close": 100.0, "fib_382": 100.4,
                                            "fib_500": 90.0, "fib_618": 80.0}), cfg) is True

    def test_missing_or_nan_fib_levels_never_block(self):
        cfg = StrategyConfig()
        # NaN levels (rolling swing not warmed up) are not "support".
        assert _near_fib_support(pd.Series({"close": 100.0, "fib_382": np.nan,
                                            "fib_500": np.nan, "fib_618": np.nan}), cfg) is False
        # A frame with no fib columns at all must not raise either.
        assert _near_fib_support(pd.Series({"close": 100.0}), cfg) is False
        # And with no fib data available the entry falls back to the core signal.
        sig = generate_signal(self._cross(), cfg, position_open=False)
        assert sig == Signal.BUY

    def test_recent_pullback_touch_satisfies_the_window(self):
        far, near = 130.0, 100.0
        rows = [
            {"close": far, "fib_382": 100.2, "fib_500": 95.0, "fib_618": 90.0},
            {"close": near, "fib_382": 100.2, "fib_500": 95.0, "fib_618": 90.0},  # tagged 2 bars back
            {"close": far, "fib_382": 100.2, "fib_500": 95.0, "fib_618": 90.0},
            {"close": far, "fib_382": 100.2, "fib_500": 95.0, "fib_618": 90.0},
        ]
        df = pd.DataFrame(rows)
        assert _fib_pullback_ok(df, StrategyConfig(fib_touch_window=3)) is True
        # fib_touch_window = 1 reproduces the old single-bar behaviour.
        assert _fib_pullback_ok(df, StrategyConfig(fib_touch_window=1)) is False
        # The signal bar itself is not at a level — only the window makes it pass.
        assert _near_fib_support(df.iloc[-1], StrategyConfig()) is False

    @staticmethod
    def _pullback_frame():
        """Cross 2 bars back, EMA trend still intact, price back in the fib zone."""
        return pd.DataFrame([
            {"close": 101.0, "ema_fast": 100.0, "ema_slow": 100.0, "rsi": 50.0, "adx": 25.0,
             "fib_382": 102.2, "fib_500": 95.0, "fib_618": 90.0},
            {"close": 102.0, "ema_fast": 102.0, "ema_slow": 100.5, "rsi": 55.0, "adx": 26.0,
             "fib_382": 102.2, "fib_500": 95.0, "fib_618": 90.0},
            {"close": 103.0, "ema_fast": 103.0, "ema_slow": 101.0, "rsi": 56.0, "adx": 27.0,
             "fib_382": 102.2, "fib_500": 95.0, "fib_618": 90.0},
            {"close": 102.0, "ema_fast": 104.0, "ema_slow": 102.0, "rsi": 55.0, "adx": 26.0,
             "fib_382": 102.2, "fib_500": 95.0, "fib_618": 90.0},
        ])

    def test_cross_stays_entry_eligible_across_recent_bars(self):
        """The crossover must not have to land on the exact bar that tags the
        fib zone: a cross a couple of bars back still counts while the trend holds."""
        df = self._pullback_frame()
        cfg = StrategyConfig()
        assert cfg.ema_cross_window == 3

        stages = {}
        assert generate_signal(df, cfg, position_open=False, diagnostics=stages) == Signal.BUY
        assert stages["core_trigger"] is True
        assert stages["ema_cross_bars_ago"] == 2
        assert stages["fib_pass"] is True

        # ema_cross_window = 1 reproduces the old strict "cross on this candle" rule.
        strict = StrategyConfig(ema_cross_window=1)
        strict_stages = {}
        assert generate_signal(df, strict, position_open=False, diagnostics=strict_stages) == Signal.HOLD
        assert strict_stages["core_trigger"] is False
        assert strict_stages["rejected_by"] == "no_ema_cross"
        assert "no fresh EMA crossover on this candle" in strict_stages["core_reason"]

    def test_cross_that_flipped_back_below_is_not_an_entry(self):
        """A window looks back for freshness only — the trend must still be up."""
        df = self._pullback_frame()
        df.loc[3, ["ema_fast", "ema_slow"]] = [101.0, 101.5]

        stages = {}
        assert generate_signal(df, StrategyConfig(), position_open=False, diagnostics=stages) == Signal.HOLD
        assert stages["core_trigger"] is False
        assert "fallen back below the slow line" in stages["core_reason"]

    def test_disabled_filter_uses_core_signals_only(self):
        """Same core cross, price far from every fib level: the filter blocks
        it, and toggling the checkbox off lets the trade through."""
        curr = {"close": 102.0, "fib_382": 130.0, "fib_500": 120.0, "fib_618": 110.0}
        df = self._cross(extra_curr=curr)

        blocked = StrategyConfig(use_fib_filter=True)
        diagnostics = {}
        sig = generate_signal(df, blocked, position_open=False, diagnostics=diagnostics)
        assert sig == Signal.HOLD
        assert diagnostics.get("fib_blocked") == 1

        allowed = StrategyConfig(use_fib_filter=False)
        diagnostics_off = {}
        sig = generate_signal(df, allowed, position_open=False, diagnostics=diagnostics_off)
        assert sig == Signal.BUY
        # The fib stage is recorded as "not evaluated" — the levels were never
        # consulted, so the trade rode on the core signal alone.
        assert diagnostics_off["core_trigger"] is True
        assert diagnostics_off["fib_evaluated"] is False
        assert diagnostics_off["fib_pass"] is None
        assert diagnostics_off["fib_reason"] is None

    def test_backtest_reports_and_skips_fib_when_toggled_off(self):
        n = 800
        trend = 50000 + np.sin(np.linspace(0, 30, n)) * 3000
        close = trend + np.random.normal(0, 50, n)
        df = pd.DataFrame({
            "ts": pd.date_range("2026-01-01", periods=n, freq="15min"),
            "open": close - 20, "high": close + 100, "low": close - 100,
            "close": close, "volume": np.random.uniform(10, 100, n),
        })

        off = BotConfig()
        off.strategy.use_fib_filter = False
        off.strategy.adx_trend_threshold = 10
        res_off = run_backtest(df.copy(), off)
        assert res_off["fib_filter_enabled"] is False
        assert res_off["fib_blocked_signals"] == 0

        on = BotConfig()
        on.strategy.use_fib_filter = True
        on.strategy.adx_trend_threshold = 10
        res_on = run_backtest(df.copy(), on)
        assert res_on["fib_filter_enabled"] is True

        # Strict single-bar fib check can only ever produce fewer trades.
        strict = BotConfig()
        strict.strategy.use_fib_filter = True
        strict.strategy.adx_trend_threshold = 10
        strict.strategy.fib_touch_window = 1
        res_strict = run_backtest(df.copy(), strict)
        assert res_strict["num_trades"] <= res_off["num_trades"]
        assert res_on["num_trades"] >= res_strict["num_trades"]

    def test_backtest_endpoint_honors_the_ui_checkbox(self, monkeypatch):
        """The 'Enable Fibonacci Zone Filter' checkbox must reach the engine.

        Drives the real /api/backtest handler (request model -> config -> engine)
        with the historical fetch stubbed out, so no exchange network is needed.
        """
        import web_server

        n = 800
        trend = 50000 + np.sin(np.linspace(0, 30, n)) * 3000
        close = trend + np.random.normal(0, 50, n)
        hist = pd.DataFrame({
            "ts": pd.date_range("2026-01-01", periods=n, freq="15min"),
            "open": close - 20, "high": close + 100, "low": close - 100,
            "close": close, "volume": np.random.uniform(10, 100, n),
        })
        monkeypatch.setattr(web_server, "fetch_historical", lambda *a, **k: hist.copy())

        base = {"exchange": "binance", "symbol": "BTC/USDT", "timeframe": "15m",
                "days": 30, "starting_equity": 1000.0}

        off = web_server.run_backtest_endpoint(
            web_server.BacktestRequest(**base, use_fib_filter=False))
        assert off["success"] is True
        assert off["results"]["fib_filter_enabled"] is False
        assert off["results"]["fib_blocked_signals"] == 0

        on = web_server.run_backtest_endpoint(
            web_server.BacktestRequest(**base, use_fib_filter=True))
        assert on["success"] is True
        assert on["results"]["fib_filter_enabled"] is True
        assert on["results"]["num_trades"] <= off["results"]["num_trades"]

    def test_backtest_endpoint_carries_the_1h_tuning_knobs(self, monkeypatch):
        """Deep-level mode, the ATR band and the trail toggle must all reach the
        engine through the real handler — not just exist as request fields."""
        import web_server

        n = 800
        close = 50000 + np.sin(np.linspace(0, 30, n)) * 3000 + np.random.normal(0, 50, n)
        hist = pd.DataFrame({
            "ts": pd.date_range("2026-01-01", periods=n, freq="1h"),
            "open": close - 20, "high": close + 100, "low": close - 100,
            "close": close, "volume": np.random.uniform(10, 100, n),
        })
        monkeypatch.setattr(web_server, "fetch_historical", lambda *a, **k: hist.copy())

        res = web_server.run_backtest_endpoint(web_server.BacktestRequest(
            exchange="binance", symbol="BTC/USDT", timeframe="1h", days=730,
            use_fib_filter=True, fib_level_mode="deep", fib_tolerance_atr_mult=0.5,
            fib_tolerance_pct=0.001, trailing_stop_enabled=False,
            trailing_stop_atr_mult=2.0))["results"]

        assert res["fib_level_mode"] == "deep"
        assert res["fib_tolerance_atr_mult"] == pytest.approx(0.5)
        assert res["trailing_stop_enabled"] is False
        assert res["trailing_stop_atr_mult"] == pytest.approx(2.0)
        assert "trail" not in res["exit_reasons"]      # the toggle really applied

    def test_endpoint_accepts_enable_fib_filter_alias(self, monkeypatch):
        """The UI checkbox is labelled 'Enable Fibonacci Zone Filter', so the
        payload may arrive as `enable_fib_filter`; it must override config.py."""
        import web_server

        n = 800
        trend = 50000 + np.sin(np.linspace(0, 30, n)) * 3000
        close = trend + np.random.normal(0, 50, n)
        hist = pd.DataFrame({
            "ts": pd.date_range("2026-01-01", periods=n, freq="15min"),
            "open": close - 20, "high": close + 100, "low": close - 100,
            "close": close, "volume": np.random.uniform(10, 100, n),
        })
        monkeypatch.setattr(web_server, "fetch_historical", lambda *a, **k: hist.copy())

        base = {"exchange": "binance", "symbol": "BTC/USDT", "timeframe": "15m",
                "days": 30, "starting_equity": 1000.0}

        # config.py ships with use_fib_filter = True, so an explicit False here
        # can only come from the request payload.
        assert BotConfig().strategy.use_fib_filter is True
        req_off = web_server.BacktestRequest(**base, enable_fib_filter=False)
        assert req_off.use_fib_filter is False
        off = web_server.run_backtest_endpoint(req_off)
        assert off["results"]["fib_filter_enabled"] is False

        on = web_server.run_backtest_endpoint(
            web_server.BacktestRequest(**base, enable_fib_filter=True))
        assert on["results"]["fib_filter_enabled"] is True

        # An explicit `use_fib_filter` still wins over a stale alias.
        both = web_server.BacktestRequest(**base, use_fib_filter=False, enable_fib_filter=True)
        assert both.use_fib_filter is False


class TestBacktestResultPayload:
    """Keys the frontend renders straight out of the /api/backtest response."""

    def test_results_report_candles_tested_and_date_range(self):
        df = _wave_ohlcv(n=400)
        res = run_backtest(df.copy(), BotConfig())

        assert res["candles_tested"] == len(df) == 400
        assert res["date_range"] == [df["ts"].iloc[0].isoformat(), df["ts"].iloc[-1].isoformat()]
        assert res["date_range"][0] < res["date_range"][1]
        assert "Data window" in res["diagnostics"]["report"]

    def test_date_range_is_present_in_the_http_payload(self, monkeypatch):
        import web_server

        hist = _wave_ohlcv(n=400)
        monkeypatch.setattr(web_server, "fetch_historical", lambda *a, **k: hist.copy())
        payload = web_server.run_backtest_endpoint(web_server.BacktestRequest(
            exchange="binance", symbol="BTC/USDT", timeframe="15m", days=5))

        assert payload["success"] is True
        assert payload["results"]["candles_tested"] == 400
        assert payload["bars_analyzed"] == 400
        assert payload["results"]["date_range"][0] == hist["ts"].iloc[0].isoformat()

    def test_date_range_handles_missing_timestamps(self):
        assert backtester._date_range(pd.DataFrame({"close": [1.0, 2.0]})) == [None, None]
        assert backtester._date_range(pd.DataFrame()) == [None, None]


# ============================================================================
# HISTORICAL CANDLE FETCHING — PAGINATION
# ============================================================================
class _FakeExchange:
    """ccxt-shaped exchange that serves synthetic klines in capped pages.

    Mirrors the real-world shapes the fetcher has to survive: an advertised
    page size, an actual (smaller) page size — MEXC spot advertises 1000 but
    serves at most 500 — optional overlapping pages, empty payloads, transient
    network errors and a venue that ignores `since`.
    """

    TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}

    def __init__(self, advertised=1000, per_page=500, days_of_history=400, stall=False,
                 fail_times=0, overlap=0, empty_before_history=False):
        self.id = "fake"
        self.rateLimit = 50                     # ms, like MEXC
        self.features = {"spot": {"fetchOHLCV": {"limit": advertised}}}
        self._advertised = advertised
        self._per_page = per_page
        self._now = 1_700_000_000_000
        self._first_ms = self._now - days_of_history * 24 * 60 * 60 * 1000
        self._stall = stall
        self._fail_times = fail_times
        self._overlap = overlap
        # MEXC answers a `since` older than its stored range with an empty list
        # instead of snapping to the earliest candle it has.
        self._empty_before_history = empty_before_history
        self.calls = []

    def market(self, symbol):
        return {"limits": {"ohlcv": {"max": self._advertised}}}

    def load_markets(self):
        return {}

    def parse_timeframe(self, timeframe):
        return self.TF_SECONDS[timeframe]

    def milliseconds(self):
        return self._now

    def fetch_ohlcv(self, symbol, timeframe=None, since=None, limit=None, **kwargs):
        self.calls.append({"symbol": symbol, "since": since, "limit": limit})
        if self._fail_times > 0:
            self._fail_times -= 1
            raise ccxt.NetworkError("temporary failure")

        ms = self.TF_SECONDS[timeframe] * 1000
        if self._empty_before_history and since is not None and since < self._first_ms:
            return []
        if self._stall:
            base = self._first_ms
            return [[base + i * ms, 10.0, 11.0, 9.0, 10.5, 1.0] for i in range(self._per_page)]

        start = max(since if since is not None else self._first_ms, self._first_ms)
        # Real venues snap `since` onto the candle grid (Binance returns the
        # bar containing it), so a +1ms cursor re-serves one bar per page.
        start -= (start - self._first_ms) % ms
        page = self._per_page if limit is None else min(self._per_page, limit)
        rows = []
        for k in range(self._overlap):          # re-serve a few older candles
            rows.append([start - (self._overlap - k) * ms, 10.0, 11.0, 9.0, 10.5, 1.0])
        ts = start
        for _ in range(page):
            if ts > self._now:
                break
            rows.append([ts, 10.0, 11.0, 9.0, 10.5, 1.0])
            ts += ms
        return rows


class TestHistoricalPagination:
    """Backtester data fetch must cover the whole lookback window, not one page."""

    def test_walks_past_a_capped_page(self):
        """MEXC regression: advertise 1000, serve 500 — the old loop stopped at 500."""
        ex = _FakeExchange(advertised=1000, per_page=500)
        df = fetch_historical("mexc", "BTC/USDT", "5m", 20, client=ex, pause_sec=0)

        assert len(df) > 5000                      # was 500 before the fix
        assert len(ex.calls) >= 12                 # ~5760 candles / 500 per page
        assert df["ts"].is_unique
        assert df["ts"].is_monotonic_increasing
        window_ms = 20 * 24 * 60 * 60 * 1000
        assert df["ts"].iloc[0].value / 1_000_000 >= ex.milliseconds() - window_ms

    def test_each_request_respects_the_exchange_page_limit(self):
        ex = _FakeExchange(advertised=300, per_page=300)     # Coinbase-like
        fetch_historical("coinbase", "BTC/USDT", "5m", 2, client=ex, pause_sec=0)
        assert ex.calls
        assert all(call["limit"] <= 300 for call in ex.calls)

    def test_pauses_between_pages_to_avoid_rate_limits(self, monkeypatch):
        slept = []
        monkeypatch.setattr(exchange_client.time, "sleep", slept.append)
        ex = _FakeExchange(per_page=500)

        fetch_historical("mexc", "BTC/USDT", "5m", 20, client=ex, pause_sec=0.3)

        assert len(slept) == len(ex.calls) - 1        # never sleeps before page 1
        assert all(pause == pytest.approx(0.3) for pause in slept)

    def test_pause_defaults_to_a_floor_over_exchange_ratelimit(self, monkeypatch):
        slept = []
        monkeypatch.setattr(exchange_client.time, "sleep", slept.append)
        ex = _FakeExchange(per_page=500)              # rateLimit 50ms -> floor wins

        fetch_historical("mexc", "BTC/USDT", "5m", 5, client=ex)

        assert slept and all(pause >= exchange_client.MIN_PAUSE_SEC for pause in slept)

    def test_retries_transient_network_errors(self, monkeypatch):
        monkeypatch.setattr(exchange_client.time, "sleep", lambda _s: None)
        ex = _FakeExchange(per_page=500, fail_times=2)

        df = fetch_historical("mexc", "BTC/USDT", "5m", 5, client=ex, pause_sec=0)

        assert not df.empty
        assert len(ex.calls) == 5                     # 3 pages + 2 retries
        assert len(df) > 1000

    def test_overlapping_pages_are_deduplicated(self):
        ex = _FakeExchange(per_page=500, overlap=5)
        df = fetch_historical("mexc", "BTC/USDT", "5m", 20, client=ex, pause_sec=0)

        assert df["ts"].is_unique                     # overlap must not duplicate bars
        assert df["ts"].is_monotonic_increasing
        assert len(df) > 5000

    def test_empty_payload_stops_cleanly_when_history_runs_out(self):
        ex = _FakeExchange(per_page=500, days_of_history=1)
        df = fetch_historical("mexc", "BTC/USDT", "5m", 30, client=ex, pause_sec=0)

        assert 0 < len(df) < 1000                     # only ~1 day exists
        assert len(ex.calls) <= 5

    def test_venue_ignoring_since_cannot_loop_forever(self):
        ex = _FakeExchange(per_page=500, stall=True)
        df = fetch_historical("mexc", "BTC/USDT", "5m", 30, client=ex, pause_sec=0)

        assert len(ex.calls) == 1                     # detected no progress, stopped at once
        assert len(df) == 500

    def test_logs_candle_count_and_lookback_window(self, caplog):
        ex = _FakeExchange(per_page=500)
        with caplog.at_level("INFO", logger="backtest"):
            fetch_historical("mexc", "BTC/USDT", "5m", 10, client=ex, pause_sec=0)

        text = " ".join(record.getMessage() for record in caplog.records)
        assert "candles across 10 days for BTC/USDT" in text
        assert "2,881" in text                        # 10 days of 5m bars, page count reported
        assert "page(s)" in text

    def test_unknown_exchange_and_timeframe_are_rejected(self):
        with pytest.raises(ValueError):
            fetch_historical("not_a_real_venue", "BTC/USDT", "5m", 1)
        with pytest.raises(ValueError):
            fetch_historical("mexc", "BTC/USDT", "7m", 1, client=_FakeExchange())


# ============================================================================
# SIGNAL PIPELINE DIAGNOSTICS
# ============================================================================
def _pipeline_cross_frame():
    """Two-row frame holding a clean core BUY cross with fib levels far away."""
    return pd.DataFrame([
        {"ts": pd.Timestamp("2026-03-14 11:45"), "open": 100.0, "high": 101.0, "low": 99.0,
         "close": 100.0, "volume": 1.0, "ema_fast": 100.0, "ema_slow": 100.0, "rsi": 50.0,
         "adx": 25.0, "atr": 2.0, "fib_382": 130.0, "fib_500": 120.0, "fib_618": 110.0},
        {"ts": pd.Timestamp("2026-03-14 12:00"), "open": 101.0, "high": 103.0, "low": 100.0,
         "close": 102.0, "volume": 1.0, "ema_fast": 102.0, "ema_slow": 100.5, "rsi": 55.0,
         "adx": 26.0, "atr": 2.0, "fib_382": 130.0, "fib_500": 120.0, "fib_618": 110.0},
    ])


def _wave_ohlcv(n=800, seed=7):
    """Trending-wave series that produces EMA crosses (same shape as above)."""
    dates = pd.date_range("2026-01-01", periods=n, freq="15min")
    trend = 50000 + np.sin(np.linspace(0, 30, n)) * 3000
    close = trend + np.random.default_rng(seed).normal(0, 50, n)
    return pd.DataFrame({
        "ts": dates, "open": close - 20, "high": close + 100, "low": close - 100,
        "close": close, "volume": np.random.default_rng(seed + 1).uniform(10, 100, n),
    })


class TestPipelineDiagnostics:
    """Each entry-pipeline stage must be reported, and every blocked signal explained."""

    @staticmethod
    def _series(n=1500, seed=3):
        """Deterministic series that produces passes, fib rejections and risk blocks."""
        rng = np.random.default_rng(seed)
        t = np.arange(n)
        close = 30000 * np.exp(np.cumsum(0.0004 + 0.0003 * np.sin(t / 300.0) + rng.normal(0, 0.004, n)))
        return pd.DataFrame({
            "ts": pd.date_range("2024-01-01", periods=n, freq="15min"),
            "open": np.concatenate([[close[0]], close[:-1]]),
            "high": close * (1 + np.abs(rng.normal(0, 0.003, n))),
            "low": close * (1 - np.abs(rng.normal(0, 0.003, n))),
            "close": close, "volume": rng.uniform(50, 500, n),
        })

    @staticmethod
    def _diag_cfg():
        cfg = BotConfig()
        cfg.strategy.adx_trend_threshold = 10
        cfg.strategy.fib_touch_window = 1      # strict single-bar fib check -> real rejections
        return cfg

    @staticmethod
    def _cross(extra_curr=None):
        prev = {"ema_fast": 100.0, "ema_slow": 100.0, "rsi": 50.0, "adx": 25.0, "close": 100.0}
        curr = {"ema_fast": 102.0, "ema_slow": 100.5, "rsi": 55.0, "adx": 26.0, "close": 102.0}
        curr.update(extra_curr or {})
        return pd.DataFrame([prev, curr])

    def test_stage1_blocked_by_adx(self):
        df = self._cross({"adx": 12.0})          # no trend strength
        stages = {}
        assert generate_signal(df, StrategyConfig(), position_open=False, diagnostics=stages) == Signal.HOLD
        assert stages["core_trigger"] is False
        assert stages["core_signal"] is None
        assert stages["rejected_by"] == "adx_filter"
        assert "ADX" in stages["core_reason"]

    def test_stage1_blocked_by_rsi(self):
        df = self._cross({"rsi": 71.0})          # overbought
        stages = {}
        generate_signal(df, StrategyConfig(), position_open=False, diagnostics=stages)
        assert stages["core_trigger"] is False
        assert stages["rejected_by"] == "rsi_overbought"
        assert "RSI" in stages["core_reason"]

    def test_stage1_blocked_by_missing_cross(self):
        df = self._cross({"ema_fast": 99.9, "ema_slow": 100.0})
        stages = {}
        generate_signal(df, StrategyConfig(), position_open=False, diagnostics=stages)
        assert stages["core_trigger"] is False
        assert stages["rejected_by"] == "no_ema_cross"

    def test_stage2_reports_which_fib_zone_rejected(self):
        df = self._cross({"close": 102.0, "fib_382": 130.0, "fib_500": 120.0, "fib_618": 110.0})
        stages = {}
        assert generate_signal(df, StrategyConfig(), position_open=False, diagnostics=stages) == Signal.HOLD
        assert stages["core_trigger"] is True
        assert stages["fib_evaluated"] is True
        assert stages["fib_pass"] is False
        assert stages["rejected_by"] == "fib_zone"
        assert "outside the Fibonacci 0.618 zone" in stages["fib_reason"]

    def test_stage2_records_the_zone_when_it_passes(self):
        df = self._cross({"close": 102.0, "fib_382": 102.2, "fib_500": 95.0, "fib_618": 90.0})
        stages = {}
        assert generate_signal(df, StrategyConfig(), position_open=False, diagnostics=stages) == Signal.BUY
        assert stages["fib_pass"] is True
        assert "Fibonacci 0.382 zone" in stages["fib_reason"]

    def test_backtest_counts_every_stage(self):
        res = run_backtest(self._series(), self._diag_cfg())
        diag = res["diagnostics"]

        assert diag["candles_evaluated"] == 1500 - diag["candles_warmup"]
        assert diag["core_signals_detected"] == diag["fib_filter_passed"] + diag["fib_filter_blocked"]
        assert diag["fib_filter_passed"] == (diag["sizing_passed"] + diag["sizing_blocked"]
                                             + diag["signals_risk_blocked"])
        # A winning trade now rides the trailing stop instead of being banked on
        # the first pullback, so the final position can still be open at the end
        # of the window: entries taken >= closed trades.
        assert diag["trades_executed"] >= res["num_trades"]
        assert diag["trades_executed"] - res["num_trades"] <= 1
        assert res["raw_signals"] == diag["core_signals_detected"]
        assert res["trades_executed"] == diag["trades_executed"]
        # Every signal that fired is accounted for: executed or rejected by a named rule.
        assert diag["fib_filter_blocked"] > 0 and diag["trades_executed"] > 0
        assert diag["rejections"].get("fib_zone", 0) == diag["fib_filter_blocked"]
        assert diag["sizing_passed"] == diag["trades_executed"]
        # Raw signals = executed + rejected: nothing is dropped silently.
        assert sum(diag["rejections"].values()) + diag["trades_executed"] == diag["core_signals_detected"]

    def test_rejection_log_has_the_requested_format(self):
        res = run_backtest(self._series(), self._diag_cfg())
        log_lines = res["diagnostics"]["rejection_log"]

        assert log_lines, "expected at least one explained rejection"
        for line in log_lines:
            assert line.startswith("Timestamp: ")
            assert "| Core signal triggered: BUY | Rejected by: " in line
        assert any("Fibonacci" in line for line in log_lines)

    def test_run_report_lists_the_four_summary_items(self):
        res = run_backtest(self._series(), self._diag_cfg())
        report = res["diagnostics"]["report"]

        for expected in ["Candles evaluated", "Stage 1 core indicator",
                         "Stage 2 Fibonacci filter", "Signals rejected", "Trades executed"]:
            assert expected in report
        assert "Rejection detail" in report

    def test_summary_is_echoed_to_the_activity_log(self, caplog):
        cfg = self._diag_cfg()
        with caplog.at_level("INFO", logger="backtest"):
            run_backtest(self._series(), cfg)
        text = " ".join(record.getMessage() for record in caplog.records)

        assert "Backtest summary:" in text
        assert "candle(s) evaluated" in text
        assert "Rejected by:" in text                  # per-signal explanation lines
        assert "Backtest stages:" in text

    def test_sizing_stage_can_be_the_blocker(self):
        cfg = BotConfig()
        cfg.strategy.use_fib_filter = False            # isolate the sizing stage
        cfg.strategy.adx_trend_threshold = 10
        cfg.risk.risk_per_trade_pct = 0.0              # sizing always returns 0
        res = run_backtest(_wave_ohlcv(), cfg)
        diag = res["diagnostics"]

        assert diag["core_signals_detected"] > 0
        assert diag["trades_executed"] == 0
        assert diag["sizing_blocked"] == diag["core_signals_detected"]
        assert diag["rejections"]["position_sizing"] == diag["core_signals_detected"]
        assert any("Position sizing returned 0" in line for line in diag["rejection_log"])

    def test_risk_gate_is_reported_as_the_blocker(self):
        cfg = BotConfig()
        cfg.strategy.use_fib_filter = False
        cfg.strategy.adx_trend_threshold = 10
        cfg.starting_equity = 0.0                      # equity depleted -> no trading
        res = run_backtest(_wave_ohlcv(), cfg)
        diag = res["diagnostics"]

        assert diag["core_signals_detected"] > 0
        assert diag["trades_executed"] == 0
        assert diag["rejections"]["risk_gate"] == diag["core_signals_detected"]
        assert any("depleted" in line for line in diag["rejection_log"])


# ============================================================================
# 1H TUNING — EXTENDED LOOKBACK, DYNAMIC FIB BAND, ATR TRAILING STOP
# ============================================================================
class TestExtendedLookback:
    """A 1h run has to be able to span ~2 years of bull, bear and chop."""

    def test_request_accepts_a_two_year_lookback(self):
        from web_server import BacktestRequest, MAX_DAYS

        assert BacktestRequest(days=365).days == 365
        assert BacktestRequest(days=730).days == 730
        assert MAX_DAYS == 730

    def test_endpoint_clamps_a_bad_lookback_instead_of_failing(self):
        from web_server import _clamp_days

        assert _clamp_days(730) == 730
        assert _clamp_days(365) == 365
        assert _clamp_days(9_999) == 730          # capped, not rejected
        assert _clamp_days(0) == 1
        assert _clamp_days(-5) == 1

    def test_ui_offers_the_two_year_window(self):
        html = web_server_dashboard_html()

        assert 'id="bt-days"' in html
        assert 'max="730"' in html
        # 365/730 must be reachable in one click, not typed by hand.
        assert "setBacktestDays(730)" in html
        assert "setBacktestDays(365)" in html


class TestDynamicFibBand:
    """The fib tolerance scales with ATR instead of being a rigid percentage."""

    @staticmethod
    def _row(close, level, atr=None):
        row = {"close": close, "fib_382": level, "fib_500": level, "fib_618": level}
        if atr is not None:
            row["atr"] = atr
        return pd.Series(row)

    def test_band_widens_with_atr_beyond_the_percentage_floor(self):
        cfg = StrategyConfig(fib_tolerance_pct=0.01, fib_tolerance_atr_mult=0.5)
        # Price 1.4% away from the level: outside the 1% floor...
        far = self._row(100.0, 101.4, atr=6.0)     # 0.5 * 6 / 100 = 3% band
        assert _near_fib_support(far, cfg) is True
        # ...and outside the ATR band too when the ATR is small.
        calm = self._row(100.0, 101.4, atr=0.5)    # band stays at the 1% floor
        assert _near_fib_support(calm, cfg) is False

    def test_band_never_goes_tighter_than_the_fixed_floor(self):
        """A tiny ATR must not shrink the band toward an exact-price match."""
        cfg = StrategyConfig(fib_tolerance_pct=0.01, fib_tolerance_atr_mult=0.5)
        tight = self._row(100.0, 100.4, atr=0.01)   # 0.5*0.01/100 = 0.005%
        assert _near_fib_support(tight, cfg) is True

    def test_zero_multiplier_restores_the_fixed_percentage_band(self):
        cfg = StrategyConfig(fib_tolerance_pct=0.01, fib_tolerance_atr_mult=0.0)
        assert _near_fib_support(self._row(100.0, 101.4, atr=6.0), cfg) is False

    def test_missing_atr_falls_back_to_the_percentage_band(self):
        cfg = StrategyConfig(fib_tolerance_pct=0.01, fib_tolerance_atr_mult=0.5)
        assert _near_fib_support(self._row(100.0, 100.4), cfg) is True     # no atr key
        assert _near_fib_support(self._row(100.0, 101.4), cfg) is False
        # A NaN ATR (warmup) must not raise or silently widen the band.
        assert _near_fib_support(self._row(100.0, 100.4, atr=np.nan), cfg) is True

    def test_reason_text_says_when_the_band_is_atr_driven(self):
        from strategy import _fib_pullback_detail

        # 0.618 is the nearest level (3% away) and a 4% ATR band admits it.
        df = pd.DataFrame([{"close": 100.0, "atr": 8.0, "fib_382": 110.0,
                            "fib_500": 105.0, "fib_618": 103.0}])
        detail = _fib_pullback_detail(df, StrategyConfig(fib_tolerance_pct=0.01,
                                                         fib_tolerance_atr_mult=0.5))
        assert detail["pass"] is True
        assert "0.618" in detail["reason"]
        assert "\u00d7ATR" in detail["reason"]      # the wider ATR band is named


class TestFibLevelSelection:
    """Deep-level mode focuses entries on the 0.5/0.618 pullbacks."""

    def test_deep_mode_ignores_the_shallow_0_382_level(self):
        cfg_deep = StrategyConfig(fib_level_mode="deep", fib_tolerance_atr_mult=0.0)
        cfg_all = StrategyConfig(fib_level_mode="all", fib_tolerance_atr_mult=0.0)
        row = pd.Series({"close": 100.0, "fib_382": 100.2, "fib_500": 120.0, "fib_618": 130.0})

        assert _near_fib_support(row, cfg_all) is True    # 0.382 is close enough
        assert _near_fib_support(row, cfg_deep) is False  # ...but not in deep mode
        # The deep levels still count.
        deep_row = pd.Series({"close": 100.0, "fib_382": 100.2, "fib_500": 100.3, "fib_618": 130.0})
        assert _near_fib_support(deep_row, cfg_deep) is True

    def test_default_mode_is_unchanged(self):
        assert StrategyConfig().fib_level_mode == "all"

    def test_unknown_mode_does_not_silently_disable_the_filter(self):
        cfg = StrategyConfig(fib_level_mode="nonsense", fib_tolerance_atr_mult=0.0)
        row = pd.Series({"close": 100.0, "fib_382": 100.2, "fib_500": 120.0, "fib_618": 130.0})
        assert _near_fib_support(row, cfg) is True


class TestATRTrailingStopOption:
    """The trailing stop is optional and follows the highest high since entry."""

    @staticmethod
    def _risk(**overrides):
        return RiskManager(RiskConfig(**overrides), starting_equity=1000.0)

    def test_default_trail_distance_is_two_atr(self):
        assert RiskConfig().trailing_stop_atr_mult == 2.0
        assert RiskConfig().trailing_stop_enabled is True

    def test_trail_follows_the_bar_high_not_just_the_close(self):
        risk = self._risk()
        position = {"entry": 100.0, "stop": 94.0, "initial_stop": 94.0}
        # The bar wicked to 125 but closed at 118: the trail must use the high.
        stop = risk.update_trailing_stop(position, 118.0, 4.0, "BUY", peak_price=125.0)
        assert position["peak"] == 125.0
        assert stop == pytest.approx(125.0 - 2.0 * 4.0)

    def test_disabling_the_trail_leaves_the_fixed_stop_alone(self):
        risk = self._risk(trailing_stop_enabled=False)
        position = {"entry": 100.0, "stop": 94.0, "initial_stop": 94.0}

        assert risk.update_trailing_stop(position, 130.0, 4.0, "BUY", peak_price=132.0) == 94.0
        assert risk.is_running(position, 130.0, 4.0) is False
        assert "peak" not in position

    def test_disabled_trail_lets_a_reversal_signal_close_the_trade(self):
        """With no trail, the reversal exit is the only thing protecting a winner."""
        cfg = TestPipelineDiagnostics._diag_cfg()
        cfg.risk.trailing_stop_enabled = False
        res = run_backtest(TestPipelineDiagnostics._series(), cfg)

        assert res["trailing_stop_enabled"] is False
        assert "trail" not in res["exit_reasons"]
        assert res["exit_reasons"].get("signal", 0) > 0


class TestOneHourBacktestReporting:
    """A 1h run reports the knobs it was measured with, for sweep comparisons."""

    @staticmethod
    def _series(candles=4_000, seed=11):
        rng = np.random.default_rng(seed)
        t = np.arange(candles)
        close = 30000 * np.exp(np.cumsum(0.0002 + 0.0004 * np.sin(t / 400.0)
                                          + rng.normal(0, 0.010, candles)))
        return pd.DataFrame({
            "ts": pd.date_range("2024-01-01", periods=candles, freq="1h"),
            "open": np.concatenate([[close[0]], close[:-1]]),
            "high": close * (1 + np.abs(rng.normal(0, 0.006, candles))),
            "low": close * (1 - np.abs(rng.normal(0, 0.006, candles))),
            "close": close, "volume": rng.uniform(50, 500, candles),
        })

    @staticmethod
    def _cfg(**strategy_overrides):
        cfg = BotConfig()
        cfg.exchange.timeframe = "1h"
        cfg.strategy.adx_trend_threshold = 12
        cfg.strategy.fib_touch_window = 2
        for key, value in strategy_overrides.items():
            setattr(cfg.strategy, key, value)
        return cfg

    def test_run_echoes_the_fib_and_trail_settings(self):
        res = run_backtest(self._series(), self._cfg(fib_level_mode="deep"))

        assert res["fib_level_mode"] == "deep"
        assert res["fib_tolerance_atr_mult"] == pytest.approx(0.5)
        assert res["trailing_stop_enabled"] is True
        assert res["trailing_stop_atr_mult"] == pytest.approx(2.0)

    def test_deep_levels_keep_a_real_sample_and_the_trail_exits_are_used(self):
        res = run_backtest(self._series(), self._cfg(fib_level_mode="deep"))

        assert res["num_trades"] >= 5            # the sample is not eliminated
        assert res["exit_reasons"].get("trail", 0) > 0
        assert res["realized_rr"] > 1.0

    def test_atr_scaled_band_and_fixed_band_are_both_measurable(self):
        """The sweep compares them, so each must be independently runnable."""
        dynamic = run_backtest(self._series(), self._cfg(fib_tolerance_atr_mult=0.5))
        fixed = run_backtest(self._series(), self._cfg(fib_tolerance_atr_mult=0.0))

        assert dynamic["fib_tolerance_atr_mult"] == pytest.approx(0.5)
        assert fixed["fib_tolerance_atr_mult"] == 0.0
        # The ATR band can only ever admit at least as many entries as the floor.
        assert dynamic["fib_blocked_signals"] <= fixed["fib_blocked_signals"]


class TestBacktestControlWiring:
    """The new 1h controls must exist AND be sent, or the UI silently lies."""

    def test_new_controls_exist_in_the_form(self):
        html = web_server_dashboard_html()

        for control in ('id="bt-fib-depth"', 'id="bt-fib-atr"', 'id="bt-fib-floor"',
                        'id="bt-trail"', 'id="bt-trail-atr"', 'id="bt-res-rr"'):
            assert control in html, control
        assert '<option value="deep"' in html
        assert "0.5 / 0.618" in html

    def test_controls_are_sent_in_the_backtest_payload(self):
        html = web_server_dashboard_html()

        for field in ("fib_level_mode:", "fib_tolerance_atr_mult:", "fib_tolerance_pct:",
                      "trailing_stop_enabled:", "trailing_stop_atr_mult:"):
            assert field in html, field
        # The floor is typed in percent, the API wants a fraction.
        assert "/ 100" in html

    def test_request_defaults_match_the_documented_1h_tuning(self):
        from web_server import BacktestRequest

        req = BacktestRequest()
        assert req.fib_tolerance_atr_mult == 0.5
        assert req.fib_level_mode == "all"
        assert req.trailing_stop_enabled is True
        assert req.trailing_stop_atr_mult == 2.0

    def test_results_panel_reports_realized_rr_and_exits(self):
        html = web_server_dashboard_html()

        assert "Realized R:R" in html
        assert "bt-res-exits" in html
        assert "bt-res-rr" in html


# ============================================================================
# DASHBOARD JAVASCRIPT — REAL ENGINE, NOT GREP
# ============================================================================
class TestLookbackGranularityLimits:
    """Venues store different amounts per grain; asking deeper is not more data."""

    def test_limits_match_what_venues_actually_store(self):
        from exchange_client import MAX_DAYS, max_lookback_days

        assert max_lookback_days("5m") == 60
        assert max_lookback_days("15m") == 90
        assert max_lookback_days("1h") == 730        # years of 1h are stored
        assert max_lookback_days("4h") == 730
        assert max_lookback_days("1d") == 730
        assert max_lookback_days("7m") <= 120        # unknown grain -> conservative

    def test_clamp_never_exceeds_either_cap(self):
        from exchange_client import MAX_DAYS, clamp_lookback_days

        assert clamp_lookback_days(365, "5m") == 60
        assert clamp_lookback_days(365, "15m") == 90
        assert clamp_lookback_days(365, "1h") == 365
        assert clamp_lookback_days(730, "1h") == 730
        assert clamp_lookback_days(9_999, "1h") == MAX_DAYS
        assert clamp_lookback_days(0, "1h") == 1
        assert clamp_lookback_days(None, "1h") >= 1

    def test_note_explains_the_five_minute_trap(self):
        from exchange_client import lookback_limit_note, lookback_limits

        note = lookback_limit_note("5m")
        assert "60 days" in note
        assert "1h" in note                          # points at a grain that works
        assert "730" in lookback_limit_note("1h")
        # The table the dashboard is built from must cover the UI's options.
        limits = lookback_limits()
        for grain in ("5m", "15m", "1h", "4h", "1d"):
            assert limits[grain]["max_days"] >= 1
            assert limits[grain]["note"]


class TestDeepStartFallback:
    """A `since` deeper than the venue's storage must degrade, never blow up."""

    @staticmethod
    def _mexc_like(days_of_history=60, per_page=500):
        return _FakeExchange(advertised=1000, per_page=per_page,
                             days_of_history=days_of_history, empty_before_history=True)

    def test_empty_deep_page_falls_back_to_the_earliest_available(self):
        from exchange_client import MS_PER_DAY, fetch_ohlcv_window

        ex = self._mexc_like(days_of_history=60)
        end_ms = ex.milliseconds()
        since_ms = end_ms - 365 * MS_PER_DAY          # far deeper than stored

        rows, pages, _largest, effective_since = fetch_ohlcv_window(
            ex, "BTC/USDT", "5m", since_ms, end_ms, limit=500, pause_sec=0.0, max_chunks=40)

        assert rows, "the fallback must return the history the venue does have"
        assert effective_since is not None
        assert effective_since > since_ms             # pulled back to real data
        # The window served is the ~60 days the venue stores, not the 365 asked.
        served_days = (end_ms - effective_since) / MS_PER_DAY
        assert 55 <= served_days <= 61
        first_candle = min(int(row[0]) for row in rows)
        assert first_candle >= ex._first_ms

    def test_fallback_is_bisected_not_guessed(self):
        """It must converge on the boundary, not step back in fixed jumps."""
        from exchange_client import find_earliest_available

        ex = self._mexc_like(days_of_history=90)
        end_ms = ex.milliseconds()
        earliest = find_earliest_available(
            ex, "BTC/USDT", "5m", end_ms - 365 * 24 * 60 * 60 * 1000, end_ms,
            limit=500, pause_sec=0.0, probes=20)

        assert earliest is not None
        # Converged onto the venue's true first candle (a fixed step-back search
        # would be days away), never earlier than data that actually exists.
        assert ex._first_ms <= earliest <= ex._first_ms + 5 * 60 * 1000
        assert len(ex.calls) <= 25                     # bounded probe cost

    def test_venue_with_no_data_at_all_returns_empty_without_raising(self):
        from exchange_client import fetch_ohlcv_window

        class _Silent(_FakeExchange):
            def fetch_ohlcv(self, symbol, timeframe=None, since=None, limit=None, **kwargs):
                self.calls.append({"symbol": symbol, "since": since, "limit": limit})
                return []

        ex = _Silent(days_of_history=1)
        rows, pages, _largest, effective = fetch_ohlcv_window(
            ex, "BTC/USDT", "5m", ex.milliseconds() - 86_400_000, ex.milliseconds(),
            limit=500, pause_sec=0.0, max_chunks=5)

        assert rows == []
        assert effective is None

    def test_backtester_fetch_degrades_instead_of_failing(self):
        """The 365-day 5m request that used to come back empty now returns data."""
        from exchange_client import MS_PER_DAY

        ex = self._mexc_like(days_of_history=60)
        df = fetch_historical("mexc", "BTC/USDT", "5m", 365, client=ex, pause_sec=0.0)

        assert not df.empty
        covered_days = (df["ts"].iloc[-1] - df["ts"].iloc[0]).total_seconds() / 86_400
        assert 50 <= covered_days <= 61
        # It really did try the deep start first, then walked a shallower one.
        assert any(call["since"] is not None for call in ex.calls)

    def test_backtester_clamps_a_grain_it_knows_cannot_go_that_deep(self, caplog):
        ex = self._mexc_like(days_of_history=60)
        with caplog.at_level("WARNING", logger="backtest"):
            fetch_historical("mexc", "BTC/USDT", "5m", 365, client=ex, pause_sec=0.0)

        text = " ".join(r.getMessage() for r in caplog.records)
        assert "5m" in text and "60" in text           # the grain limit is explained


class TestBacktestLookbackLimitsInUI:
    """The dashboard must warn/cap instead of requesting data nobody stores."""

    def test_limits_are_injected_as_valid_json(self):
        import json
        from exchange_client import lookback_limits

        html = web_server_dashboard_html()
        match = re.search(r"const BT_LOOKBACK_LIMITS = (\{.*?\});", html)
        assert match, "the limits table was not injected into the page"
        assert json.loads(match.group(1)) == lookback_limits()

    def test_inputs_are_wired_for_the_warning(self):
        html = web_server_dashboard_html()

        assert 'id="bt-days-note"' in html
        assert 'onchange="applyTimeframeLimits()"' in html
        assert "setBacktestDays" in html

    def test_quick_pick_buttons_respect_the_grain_limit(self, tmp_path):
        """Clicking '730d' on 5m must land on 60, not send an impossible request."""
        out = self._run_limits_harness(tmp_path, {
            "bt-timeframe": "5m", "bt-days": "90", "bt-fib-depth": "all",
            "bt-fib-atr": "0.5", "bt-fib-floor": "1.0", "bt-trail-atr": "2.0",
            "bt-exchange": "mexc", "bt-symbol": "BTC/USDT", "bt-equity": "1000",
            "bt-use-fib": True, "bt-trail": True,
        }, click_days=730)

        assert out["daysInput"] == "60"
        assert "capped" in out["note"].lower() or "60" in out["note"]

    def test_deep_request_is_capped_before_it_is_posted(self, tmp_path):
        out = self._run_limits_harness(tmp_path, {
            "bt-timeframe": "5m", "bt-days": "365", "bt-fib-depth": "deep",
            "bt-fib-atr": "0.5", "bt-fib-floor": "1.0", "bt-trail-atr": "2.0",
            "bt-exchange": "mexc", "bt-symbol": "BTC/USDT", "bt-equity": "1000",
            "bt-use-fib": True, "bt-trail": True,
        })

        assert out["posted"]["days"] == 60            # capped to the grain limit
        assert out["posted"]["timeframe"] == "5m"
        assert "60" in out["note"]

    @staticmethod
    def _run_limits_harness(tmp_path, form, click_days=None):
        node = shutil.which("node")
        if not node:
            pytest.skip("node is not installed; JS runtime check skipped")
        (tmp_path / "fn.js").write_text(_backtest_js_source())
        (tmp_path / "form.json").write_text(json.dumps(form))
        (tmp_path / "api.json").write_text(json.dumps(_BACKTEST_API_PAYLOAD))
        (tmp_path / "harness.js").write_text(
            "globalThis.__CLICK_DAYS = "
            + (str(click_days) if click_days is not None else "null") + ";\n"
            + _JS_BACKTEST_HARNESS)
        proc = subprocess.run(
            [node, str(tmp_path / "harness.js"), str(tmp_path / "fn.js"),
             str(tmp_path / "form.json"), str(tmp_path / "api.json")],
            capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, f"node failed:\n{proc.stdout}\n{proc.stderr}"
        return json.loads(proc.stdout.strip().splitlines()[-1])

def _dashboard_script() -> str:
    """The dashboard's main inline <script>, exactly as served."""
    import web_server

    scripts = re.findall(r"<script>(.*?)</script>", web_server.DASHBOARD_HTML, re.S)
    return max(scripts, key=len)


def _dashboard_function(name: str) -> str:
    """Source of one declared function from the dashboard script.

    Brace-matched from the signature; enough to lift a handler out of the page
    and run it on its own.
    """
    src = _dashboard_script()
    match = re.search(rf"(?:async\s+)?function\s+{re.escape(name)}\s*\(", src)
    assert match, f"{name} is not declared in the dashboard script"
    start = match.start()          # keep the `async` keyword, or `await` is illegal
    open_brace = src.index("{", start)
    depth = 0
    for index in range(open_brace, len(src)):
        if src[index] == "{":
            depth += 1
        elif src[index] == "}":
            depth -= 1
            if depth == 0:
                return src[start:index + 1]
    raise AssertionError(f"unbalanced braces while extracting {name}")


def _backtest_js_source() -> str:
    """The backtest handler plus every helper it calls, as one runnable script.

    The limits table is lifted from the served page, so the test runs the real
    injected values rather than a copy that can drift from the server's.
    """
    limits = re.search(r"const BT_LOOKBACK_LIMITS = (\{.*?\});", web_server_dashboard_html())
    assert limits, "the limits table is not in the page"
    return (
        "const BT_LOOKBACK_LIMITS = " + limits.group(1) + ";\n"
        + _dashboard_function("setBacktestDays") + "\n"
        + _dashboard_function("backtestDayLimit") + "\n"
        + _dashboard_function("applyTimeframeLimits") + "\n"
        + _dashboard_function("executeBacktest")
    )


# Runs the extracted handler in a real engine with a stubbed DOM and fetch, so
# "cannot access X before initialization" and friends fail the test instead of
# only surfacing as a browser alert on the deployed dashboard.
_JS_BACKTEST_HARNESS = r"""
const fs = require('fs');
const [fnPath, formPath, apiPath] = process.argv.slice(2);
const form = JSON.parse(fs.readFileSync(formPath, 'utf8'));
const apiPayload = JSON.parse(fs.readFileSync(apiPath, 'utf8'));

const elements = {};
function makeEl(id) {
  const classes = new Set(['hidden']);
  let text = '';
  let val = '';
  return {
    id,
    // A real DOM stringifies innerText; the stub does too, so the assertions
    // see what a browser would show.
    get innerText() { return text; },
    set innerText(v) { text = String(v); },
    get value() { return val; },
    set value(v) { val = String(v); },
    innerHTML: '', checked: false,
    disabled: false, className: '',
    classList: {
      add: (c) => classes.add(c),
      remove: (c) => classes.delete(c),
      contains: (c) => classes.has(c),
    },
  };
}
const el = (id) => (elements[id] = elements[id] || makeEl(id));

globalThis.document = { getElementById: el, createElement: () => makeEl('x') };
globalThis.alert = (m) => { throw new Error('alert(): ' + m); };
globalThis.fetch = async (url, opts) => {
  globalThis.__posted = JSON.parse(opts.body);
  return { ok: true, status: 200, json: async () => apiPayload };
};

for (const [id, val] of Object.entries(form)) {
  if (typeof val === 'boolean') el(id).checked = val;
  else el(id).value = val;
}

const executeBacktest = new Function(
  fs.readFileSync(fnPath, 'utf8')
  + '\nreturn { executeBacktest, setBacktestDays, applyTimeframeLimits };')();

// Optionally drive a quick-pick button (90d/365d/730d) before submitting.
if (globalThis.__CLICK_DAYS !== null && globalThis.__CLICK_DAYS !== undefined) {
  executeBacktest.setBacktestDays(globalThis.__CLICK_DAYS);
}

executeBacktest.executeBacktest({ preventDefault() {} }).then(() => {
  console.log(JSON.stringify({
    posted: globalThis.__posted,
    rr: el('bt-res-rr').innerText,
    winrate: el('bt-res-winrate').innerText,
    trades: el('bt-res-trades').innerText,
    fibNote: el('bt-res-fib-note').innerText,
    note: el('bt-days-note').innerText,
    daysInput: el('bt-days').value,
    daysMax: el('bt-days').max,
    resultsHidden: el('bt-results-box').classList.contains('hidden'),
    buttonDisabled: el('btn-run-bt').disabled,
  }));
}).catch((err) => {
  console.log('FAILED: ' + err.name + ': ' + err.message);
  process.exit(1);
});
"""

_BACKTEST_API_PAYLOAD = {
    "success": True,
    "results": {
        "total_return_pct": 5.43, "num_trades": 145, "win_rate_pct": 34.48,
        "final_equity": 1054.3, "sharpe_approx": 0.9, "avg_win": 32.1,
        "avg_loss": -15.5, "max_drawdown_pct": -10.99, "realized_rr": 2.07,
        "exit_reasons": {"trail": 73, "stop": 32, "target": 17, "signal": 23},
        "candles_tested": 17_520, "date_range": ["2023-09-01", "2025-08-30"],
        "raw_signals": 120, "fib_filter_enabled": True, "fib_blocked_signals": 60,
        "fib_level_mode": "deep", "fib_tolerance_atr_mult": 0.5,
        "trailing_stop_enabled": True, "trailing_stop_atr_mult": 2.0,
        "diagnostics": {"report": "--- pipeline report ---", "fib_filter_blocked": 60,
                        "sizing_blocked": 0, "signals_risk_blocked": 3,
                        "candles_evaluated": 17_520},
    },
}

_BACKTEST_FORM_VALUES = {
    "bt-exchange": "binance", "bt-symbol": "BTC/USDT", "bt-timeframe": "1h",
    "bt-days": "730", "bt-equity": "1000",
    "bt-use-fib": True, "bt-fib-depth": "deep", "bt-fib-atr": "0.5",
    "bt-fib-floor": "0.1", "bt-trail": True, "bt-trail-atr": "2.0",
}


class TestDashboardBacktestRuntime:
    """executeBacktest must actually run in a JS engine, not merely exist.

    The whole results panel is rendered by this one handler, so a single
    use-before-declaration error (a `const` helper in the temporal dead zone)
    silently kills every field on the panel behind an alert().
    """

    @staticmethod
    def _run_harness(tmp_path, function_name="executeBacktest"):
        node = shutil.which("node")
        if not node:
            pytest.skip("node is not installed; JS runtime check skipped")
        (tmp_path / "fn.js").write_text(_backtest_js_source())
        (tmp_path / "form.json").write_text(json.dumps(_BACKTEST_FORM_VALUES))
        (tmp_path / "api.json").write_text(json.dumps(_BACKTEST_API_PAYLOAD))
        (tmp_path / "harness.js").write_text(_JS_BACKTEST_HARNESS)
        proc = subprocess.run(
            [node, str(tmp_path / "harness.js"), str(tmp_path / "fn.js"),
             str(tmp_path / "form.json"), str(tmp_path / "api.json")],
            capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, f"node failed:\n{proc.stdout}\n{proc.stderr}"
        return json.loads(proc.stdout.strip().splitlines()[-1])

    def test_handler_runs_without_a_reference_error(self, tmp_path):
        out = self._run_harness(tmp_path)

        # The results box is revealed and every headline field was written.
        assert out["resultsHidden"] is False
        assert out["rr"] == "2.07 : 1"
        assert out["winrate"] == "34.48%"
        assert out["trades"] == "145"
        assert "trail=73" in out["fibNote"] or "Fibonacci Zone Filter" in out["fibNote"]
        assert out["buttonDisabled"] is False        # the button is re-enabled

    def test_handler_posts_every_form_field_cleanly(self, tmp_path):
        """Item 3: the payload must gather and post all form values."""
        posted = self._run_harness(tmp_path)["posted"]

        assert posted["exchange"] == "binance"
        assert posted["symbol"] == "BTC/USDT"
        assert posted["timeframe"] == "1h"
        assert posted["days"] == 730                 # lookback reaches two years
        assert posted["starting_equity"] == 1000.0
        assert posted["use_fib_filter"] is True
        assert posted["fib_level_mode"] == "deep"
        assert posted["fib_tolerance_atr_mult"] == 0.5
        assert posted["fib_tolerance_pct"] == pytest.approx(0.001)   # 0.1% typed
        assert posted["trailing_stop_enabled"] is True
        assert posted["trailing_stop_atr_mult"] == 2.0
        # Numbers must arrive as numbers, not strings, or pydantic 422s.
        for key in ("days", "starting_equity", "fib_tolerance_atr_mult",
                    "fib_tolerance_pct", "trailing_stop_atr_mult", "risk_per_trade_pct"):
            assert isinstance(posted[key], (int, float)), key

    def test_no_setter_is_referenced_before_its_declaration(self):
        """A `const`/`let` used above its own declaration is a TDZ ReferenceError."""
        src = _dashboard_function("executeBacktest")
        declared = set(re.findall(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)", src))

        for name in declared:
            first_decl = src.index(name)
            # Skip the declaration itself: look for a use strictly before it.
            for match in re.finditer(rf"\b{re.escape(name)}\b", src):
                if match.start() < first_decl:
                    raise AssertionError(f"{name} is used before it is declared")
        # The helper that used to break the panel is declared, and early.
        assert "const setIf" in src
        assert src.index("const setIf") < src.index("setIf('bt-res-rr'")

    def test_every_referenced_element_id_exists_or_is_guarded(self):
        """A getElementById typo silently no-ops; catch the real ones."""
        import web_server

        declared = set(re.findall(r'id="([^"]+)"', web_server.DASHBOARD_HTML))
        used = set(re.findall(r"getElementById\(\s*'([^']+)'\s*\)", _dashboard_script()))
        # Checked with `||` fallbacks / null guards inside the script.
        guarded = {"configured-keys-list"}

        assert used - declared - guarded == set()
