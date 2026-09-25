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
from unittest import mock
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
        # The default asks for a real test of the 0.5/0.618 pullback levels.
        cfg = StrategyConfig()
        at_500 = pd.Series({"close": 95.05, "fib_382": 110.0, "fib_500": 95.0, "fib_618": 85.0})
        at_382_only = pd.Series({"close": 100.0, "fib_382": 100.5, "fib_500": 90.0, "fib_618": 80.0})
        above = pd.Series({"close": 120.0, "fib_382": 110.0, "fib_500": 90.0, "fib_618": 80.0})

        assert _near_fib_support(at_500, cfg) is True
        assert _near_fib_support(at_382_only, cfg) is False   # 0.382 alone is not a pullback level
        assert _near_fib_support(above, cfg) is False         # breakout, no pullback yet


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

    def test_reset_kill_switch_clears_latch_and_resets_baseline(self):
        cfg = RiskConfig(max_drawdown_pct=0.15)
        risk = RiskManager(cfg, starting_equity=1000.0)
        risk.record_trade_result(-160.0)
        assert risk.can_trade()[0] is False
        assert risk.state.kill_switch_triggered is True

        risk.reset_kill_switch(reset_peak=True)
        assert risk.state.kill_switch_triggered is False
        assert risk.state.peak_equity == 840.0
        assert risk.state.equity == 840.0
        assert risk.state.trading_halted_today is False
        can_trade, reason = risk.can_trade()
        assert can_trade is True
        assert reason == "OK"


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
        cfg = TestPipelineDiagnostics._diag_cfg()
        cfg.strategy.use_fib_filter = False      # isolate the risk gate from the fib filter
        cfg.strategy.min_bars_between_trades = 0  # isolate from entry debounce
        res = run_backtest(TestPipelineDiagnostics._series(), cfg)
        lines = res["diagnostics"]["rejection_log"]

        assert any("Circuit breaker" in line and "resume" in line for line in lines)

    def test_backtest_resumes_after_the_cooldown_mid_run(self):
        """More trades than the old 4-loss hard halt would ever allow."""
        cfg = TestPipelineDiagnostics._diag_cfg()
        cfg.strategy.use_fib_filter = False      # isolate the risk gate from the fib filter
        cfg.strategy.min_bars_between_trades = 0  # isolate from entry debounce
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

        # +1 ATR of profit is NOT enough any more: the original stop stands.
        assert risk.update_trailing_stop(position, 104.0, 4.0) == 94.0
        assert risk.is_running(position, 104.0, 4.0) is False

        # +1.5 ATR arms the trail and pulls the stop up to break-even...
        assert risk.update_trailing_stop(position, 106.0, 4.0) == pytest.approx(100.0)
        assert risk.is_running(position, 106.0, 4.0) is True

        # ...then it trails 3.5 ATR behind the best price.
        assert risk.update_trailing_stop(position, 120.0, 4.0) == pytest.approx(106.0)
        assert position["peak"] == 120.0
        # A pullback never loosens the stop again.
        assert risk.update_trailing_stop(position, 108.0, 4.0) == pytest.approx(106.0)
        assert position["trail_armed"] is True

    def test_backtest_lets_winners_run_and_reports_realized_rr(self):
        """Realised R:R must beat the ~1.1 the fix was written for."""
        cfg = TestPipelineDiagnostics._diag_cfg()
        cfg.strategy.use_fib_filter = False      # this is about the trail, not the filter
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

    def test_entry_debounce_measures_bars_since_the_last_exit(self, sample_ohlcv):
        """The live cooldown is a bar count derived from the exit's candle."""
        cfg = BotConfig()
        cfg.exchange.timeframe = "15m"                    # the fixture's bar spacing
        bot = TradingBot(cfg)
        df = add_indicators(sample_ohlcv, cfg.strategy)

        assert bot._bars_since_last_exit(df) is None      # never traded -> nothing to debounce

        bot._mark_exit(df.iloc[:-5])                      # last exit five candles back
        assert bot._bars_since_last_exit(df) == 5

        bot._mark_exit(df)                                # exited on the newest candle
        assert bot._bars_since_last_exit(df) == 0         # inside any cooldown

    def test_live_entry_is_suppressed_while_the_cooldown_runs(self, sample_ohlcv, monkeypatch):
        """A 5m whipsaw cannot re-enter on the candle it just exited on."""
        import strategy

        monkeypatch.setattr(strategy, "_evaluate_signal",
                            lambda df, cfg, position_open, diagnostics=None: Signal.BUY)
        cfg = BotConfig()
        cfg.exchange.timeframe = "15m"                    # 8-bar default for this grain
        bot = TradingBot(cfg)
        bot.exchange.fetch_ohlcv = lambda *args, **kwargs: sample_ohlcv
        df = add_indicators(sample_ohlcv, cfg.strategy)
        price = float(df["close"].iloc[-1])

        bot._maybe_enter(df, price)                       # first entry is never debounced
        assert bot.position is not None

        bot.position = None
        bot._mark_exit(df)                                # stop-out on this very candle
        bot._maybe_enter(df, price)
        assert bot.position is None                       # suppressed by the cooldown

        # Eight candles later the setup is tradeable again.
        bot.last_exit_ts = bot._last_bar_ts(df.iloc[:-8])
        bot._maybe_enter(df, price)
        assert bot.position is not None

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

    def test_deactivate_kill_switch(self, tmp_path):
        cfg = BotConfig()
        kill_file = str(tmp_path / "TEST_KILL_FILE")
        cfg.kill_switch_file = kill_file
        bot = TradingBot(cfg)

        with open(kill_file, "w") as f:
            f.write("HALT")
        bot.risk.state.kill_switch_triggered = True

        assert bot.kill_switch_active() is True
        bot.deactivate_kill_switch()
        assert bot.kill_switch_active() is False
        assert not os.path.exists(kill_file)
        assert bot.risk.state.kill_switch_triggered is False


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

    def test_settings_api_and_5m_timeframe(self, monkeypatch, tmp_path):
        from pathlib import Path
        from fastapi.testclient import TestClient
        from web_server import app, AVAILABLE_TIMEFRAMES, runner

        fake_env = tmp_path / ".test_env"
        fake_env.write_text("STRATEGY=ema_cross_fib\n")
        orig_path = Path
        monkeypatch.setattr("web_server.Path", lambda p: fake_env if p == ".env" else orig_path(p))
        monkeypatch.setenv("STRATEGY", os.getenv("STRATEGY", "ema_cross_fib"))
        monkeypatch.setenv("EXECUTION_MODE", os.getenv("EXECUTION_MODE", "auto"))
        monkeypatch.setenv("TIMEFRAME", os.getenv("TIMEFRAME", "15m"))
        monkeypatch.setenv("SYMBOL", os.getenv("SYMBOL", "BTC/USDT"))
        orig_strat = runner.cfg.strategy.strategy_name
        orig_mode = runner.cfg.execution_mode
        orig_tf = runner.cfg.exchange.timeframe
        orig_sym = runner.cfg.exchange.symbol
        try:
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
        finally:
            runner.cfg.strategy.strategy_name = orig_strat
            runner.cfg.execution_mode = orig_mode
            runner.cfg.exchange.timeframe = orig_tf
            runner.cfg.exchange.symbol = orig_sym

    def test_api_reset_kill_and_manual_sticky_dock(self, tmp_path):
        from fastapi.testclient import TestClient
        from web_server import app, runner, DASHBOARD_HTML

        assert 'id="manual-sticky-dock"' in DASHBOARD_HTML
        assert 'id="btn-sticky-buy"' in DASHBOARD_HTML
        assert 'id="btn-sticky-sell"' in DASHBOARD_HTML
        assert 'id="kill-switch-active-banner"' in DASHBOARD_HTML
        assert 'id="btn-reset-kill"' in DASHBOARD_HTML
        assert 'deactivateKillSwitch()' in DASHBOARD_HTML

        client = TestClient(app)
        old_kill_file = runner.cfg.kill_switch_file
        test_kill_file = str(tmp_path / "KILL")
        runner.cfg.kill_switch_file = test_kill_file
        try:
            with open(test_kill_file, "w") as f:
                f.write("HALT")
            assert os.path.exists(test_kill_file)

            res = client.post("/api/bot/reset_kill")
            assert res.status_code == 200
            assert res.json()["success"] is True
            assert not os.path.exists(test_kill_file)
        finally:
            runner.cfg.kill_switch_file = old_kill_file
            if os.path.exists(test_kill_file):
                os.remove(test_kill_file)

    def test_execution_mode_persistence_and_dedicated_endpoint(self, monkeypatch, tmp_path):
        from fastapi.testclient import TestClient
        from web_server import app, runner, DASHBOARD_HTML

        assert 'id="btn-exec-auto"' in DASHBOARD_HTML
        assert 'id="btn-exec-manual"' in DASHBOARD_HTML
        assert '/api/bot/execution_mode' in DASHBOARD_HTML

        client = TestClient(app)
        orig_mode = runner.cfg.execution_mode
        try:
            # 1. Switch to manual via dedicated endpoint
            res = client.post("/api/bot/execution_mode", json={"execution_mode": "manual"})
            assert res.status_code == 200
            assert res.json()["execution_mode"] == "manual"
            assert runner.cfg.execution_mode == "manual"
            assert client.get("/api/status").json()["execution_mode"] == "manual"

            # 2. Start bot - verify runner.start() does NOT reset execution_mode to auto
            started = runner.start()
            try:
                assert runner.cfg.execution_mode == "manual"
                if runner.bot:
                    assert runner.bot.cfg.execution_mode == "manual"
                assert client.get("/api/status").json()["execution_mode"] == "manual"
            finally:
                runner.stop()

            assert runner.cfg.execution_mode == "manual"

            # 3. Switching environment mode (paper / live) does NOT reset execution_mode
            m_res = client.post("/api/bot/mode", json={"trading_mode": "real"})
            assert m_res.status_code == 200
            assert runner.cfg.execution_mode == "manual"

            t_res = client.post("/api/bot/mode", json={"trading_mode": "testnet"})
            assert t_res.status_code == 200
            assert runner.cfg.execution_mode == "manual"

            # 4. Partial settings update does NOT reset execution_mode
            p_res = client.post("/api/bot/settings", json={"fast_ema": 14})
            assert p_res.status_code == 200
            assert runner.cfg.execution_mode == "manual"

            # 5. Switch back to auto
            a_res = client.post("/api/bot/execution_mode", json={"execution_mode": "auto"})
            assert a_res.status_code == 200
            assert a_res.json()["execution_mode"] == "auto"
            assert runner.cfg.execution_mode == "auto"
            assert client.get("/api/status").json()["execution_mode"] == "auto"
        finally:
            runner.cfg.execution_mode = orig_mode
            if runner.bot:
                runner.bot.cfg.execution_mode = orig_mode


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
        # Band semantics are tested in level mode; zone mode measures a range.
        cfg = StrategyConfig(fib_level_mode="all", fib_tolerance_pct=0.01)   # a 1% band
        # Price 0.4% away from the 38.2% level is inside the band.
        assert _near_fib_support(pd.Series({"close": 100.0, "fib_382": 100.4,
                                            "fib_500": 95.0, "fib_618": 90.0}), cfg) is True
        # 1.4% away is outside it.
        assert _near_fib_support(pd.Series({"close": 100.0, "fib_382": 101.4,
                                            "fib_500": 95.0, "fib_618": 90.0}), cfg) is False
        # The band scales with price, so a $50000 asset gets a $500 band.
        assert _near_fib_support(pd.Series({"close": 50000.0, "fib_382": 50200.0,
                                            "fib_500": 49000.0, "fib_618": 48000.0}), cfg) is True

    def test_default_tolerance_is_tight_and_tunable(self):
        """A 1% floor accepted nearly every core signal, so the shipped band is
        0.3% and price has to actually test the retracement line."""
        from strategy import DEFAULT_FIB_TOLERANCE_PCT

        cfg = StrategyConfig()
        assert cfg.fib_tolerance_pct == pytest.approx(0.003)
        assert DEFAULT_FIB_TOLERANCE_PCT == pytest.approx(0.003)

        level = {"fib_level_mode": "all"}
        inside = pd.Series({"close": 100.0, "fib_382": 100.2, "fib_500": 95.0, "fib_618": 90.0})
        outside = pd.Series({"close": 100.0, "fib_382": 100.9, "fib_500": 95.0, "fib_618": 90.0})
        assert _near_fib_support(inside, StrategyConfig(**level)) is True
        assert _near_fib_support(outside, StrategyConfig(**level)) is False
        # Still tunable: a 1% band re-admits the bar the default turned away.
        assert _near_fib_support(outside, StrategyConfig(fib_tolerance_pct=0.01, **level)) is True
        # A nonsense tolerance never degrades back into an exact-price match.
        assert _near_fib_support(inside, StrategyConfig(fib_tolerance_pct=-1.0, **level)) is True

    def test_zero_tolerance_does_not_degrade_to_exact_match(self):
        cfg = StrategyConfig(fib_tolerance_pct=0.0)
        assert _near_fib_support(pd.Series({"close": 100.0, "fib_382": 110.0,
                                            "fib_500": 100.2, "fib_618": 90.0}), cfg) is True

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
        far, near = 130.0, 95.2      # near tags the 0.5 level (95.0) on the default set
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
        """Cross 2 bars back, EMA trend intact, price back at the 0.5 fib level."""
        return pd.DataFrame([
            {"close": 101.0, "ema_fast": 100.0, "ema_slow": 100.0, "rsi": 50.0, "adx": 25.0,
             "fib_382": 110.0, "fib_500": 101.9, "fib_618": 95.0},
            {"close": 102.0, "ema_fast": 102.0, "ema_slow": 100.5, "rsi": 55.0, "adx": 26.0,
             "fib_382": 110.0, "fib_500": 101.9, "fib_618": 95.0},
            {"close": 103.0, "ema_fast": 103.0, "ema_slow": 101.0, "rsi": 56.0, "adx": 27.0,
             "fib_382": 110.0, "fib_500": 101.9, "fib_618": 95.0},
            {"close": 102.0, "ema_fast": 104.0, "ema_slow": 102.0, "rsi": 55.0, "adx": 26.0,
             "fib_382": 110.0, "fib_500": 101.9, "fib_618": 95.0},
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
        # The arm threshold rides along in the result too, so a run says when the
        # trail was allowed to start tightening (the shipped default is +1.5 ATR).
        assert res["trailing_arm_atr_mult"] == pytest.approx(1.5)
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
        assert generate_signal(df, StrategyConfig(fib_level_mode="zone"), position_open=False,
                               diagnostics=stages) == Signal.HOLD
        assert stages["core_trigger"] is True
        assert stages["fib_evaluated"] is True
        assert stages["fib_pass"] is False
        assert stages["rejected_by"] == "fib_zone"
        # Price is below the whole zone, and the reason says which side missed.
        assert "pullback zone" in stages["fib_reason"]
        assert "0.382" in stages["fib_reason"]
        assert "too deep" in stages["fib_reason"]

    def test_stage2_rejects_an_entry_above_the_zone(self):
        """The crossover candle itself is above 0.382 — buying there is the bug."""
        df = self._cross({"close": 108.0, "fib_382": 100.0, "fib_500": 95.0, "fib_618": 90.0})
        stages = {}
        assert generate_signal(df, StrategyConfig(fib_level_mode="zone"), position_open=False,
                               diagnostics=stages) == Signal.HOLD
        assert stages["core_trigger"] is True
        assert stages["fib_pass"] is False
        assert "no pullback yet" in stages["fib_reason"]

    def test_stage2_records_the_zone_when_it_passes(self):
        df = self._cross({"close": 102.0, "fib_382": 102.2, "fib_500": 95.0, "fib_618": 90.0})
        stages = {}
        assert generate_signal(df, StrategyConfig(fib_level_mode="zone"), position_open=False,
                               diagnostics=stages) == Signal.BUY
        assert stages["fib_pass"] is True
        assert "pulled back into the Fibonacci" in stages["fib_reason"]
        assert "0.618-0.382" in stages["fib_reason"]

    def test_backtest_counts_every_stage(self):
        res = run_backtest(self._series(), self._diag_cfg())
        diag = res["diagnostics"]

        assert diag["candles_evaluated"] == 1500 - diag["candles_warmup"]
        assert diag["core_signals_detected"] == diag["fib_filter_passed"] + diag["fib_filter_blocked"]
        # Everything that cleared the fib filter was then sized, failed sizing,
        # hit the risk gate, or was held back by the entry debounce.
        assert diag["fib_filter_passed"] == (diag["sizing_passed"] + diag["sizing_blocked"]
                                             + diag["signals_risk_blocked"]
                                             + diag["debounce_blocked"])
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
        assert diag["rejections"].get("signal_debounce", 0) == diag["debounce_blocked"]
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
        detail = _fib_pullback_detail(df, StrategyConfig(fib_level_mode="all",
                                                         fib_tolerance_pct=0.01,
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

    def test_default_mode_is_the_key_levels_not_the_blanket_zone(self):
        """The whole point: a real test of 0.5/0.618, not "anywhere near the pocket"."""
        assert StrategyConfig().fib_level_mode == "deep"
        cfg = StrategyConfig()

        # Only the 0.382 level is nearby — the shallow breakout level the zone
        # mode used to accept, and exactly what this filter is meant to refuse.
        at_382_only = pd.Series({"close": 100.0, "fib_382": 100.2, "fib_500": 90.0, "fib_618": 85.0})
        at_500 = pd.Series({"close": 91.9, "fib_382": 100.0, "fib_500": 92.0, "fib_618": 85.0})
        at_618 = pd.Series({"close": 85.1, "fib_382": 100.0, "fib_500": 93.0, "fib_618": 85.0})
        above = pd.Series({"close": 108.0, "fib_382": 100.0, "fib_500": 95.0, "fib_618": 90.0})

        assert _near_fib_support(at_382_only, cfg) is False
        assert _near_fib_support(at_500, cfg) is True
        assert _near_fib_support(at_618, cfg) is True
        assert _near_fib_support(above, cfg) is False

    def test_zone_mode_is_alias_tolerant(self):
        for alias in ("zone", "golden", "golden_zone", "0.382-0.618", "pocket"):
            cfg = StrategyConfig(fib_level_mode=alias)
            above = pd.Series({"close": 108.0, "fib_382": 100.0, "fib_500": 95.0, "fib_618": 90.0})
            assert _near_fib_support(above, cfg) is False, alias

    def test_unknown_mode_does_not_silently_disable_the_filter(self):
        cfg = StrategyConfig(fib_level_mode="nonsense", fib_tolerance_atr_mult=0.0)
        row = pd.Series({"close": 100.0, "fib_382": 100.2, "fib_500": 120.0, "fib_618": 130.0})
        assert _near_fib_support(row, cfg) is True


class TestFibFilterBlocksSignals:
    """A filter that blocks nothing is not a filter.

    The shipped default used to be the 0.382-0.618 pocket with a 1% floor, which
    measured against real signals accepted essentially all of them (`Fib
    blocked: 0`) — the pocket covers a large part of the swing range and the
    band on top of it covered the rest. The default is now a tight test of the
    0.5/0.618 levels, and these tests pin the *effect*, not just the settings.
    """

    @staticmethod
    def _series():
        return TestOneHourBacktestReporting._series()

    _runs: dict = {}

    @classmethod
    def _run(cls, **overrides):
        """One shared 1h backtest per config — a run of this fixture is ~3s."""
        key = tuple(sorted(overrides.items()))
        if key not in cls._runs:
            cfg = BotConfig()
            cfg.exchange.timeframe = "1h"
            for name, value in overrides.items():
                setattr(cfg.strategy, name, value)
            cls._runs[key] = run_backtest(cls._series(), cfg)
        return cls._runs[key]

    def test_shipped_defaults_are_the_tight_key_levels(self):
        assert StrategyConfig().fib_level_mode == "deep"
        assert StrategyConfig().fib_tolerance_pct == pytest.approx(0.003)
        from web_server import BacktestRequest

        assert BacktestRequest().fib_level_mode == "deep"
        assert BacktestRequest().fib_tolerance_pct == pytest.approx(0.003)

    def test_the_old_blanket_zone_blocked_nothing(self):
        """Reproduces the reported symptom, so the fix has something to beat."""
        unfiltered = self._run(use_fib_filter=False)
        old = self._run(fib_level_mode="zone", fib_tolerance_pct=0.01)

        assert old["fib_blocked_signals"] == 0
        assert old["num_trades"] == unfiltered["num_trades"]     # a filter in name only

    def test_one_hour_run_with_the_filter_on_actually_filters(self):
        """The acceptance check: 1h, filter ON -> 'Fib blocked > 0', fewer trades."""
        unfiltered = self._run(use_fib_filter=False)
        filtered = self._run()                                   # shipped defaults

        assert filtered["fib_filter_enabled"] is True
        assert filtered["fib_level_mode"] == "deep"
        assert filtered["fib_blocked_signals"] > 0
        assert filtered["num_trades"] < unfiltered["num_trades"]  # a smaller, chosen subset
        # The stage-2 tally and the rejection log agree with the headline number.
        diag = filtered["diagnostics"]
        assert diag["fib_filter_blocked"] == filtered["fib_blocked_signals"]
        assert diag["rejections"].get("fib_zone", 0) == diag["fib_filter_blocked"]
        assert any("Fibonacci" in line for line in diag["rejection_log"])
        assert "Stage 2 Fibonacci filter" in backtester.format_pipeline_report(filtered)

    def test_deep_mode_blocks_more_than_the_blanket_zone(self):
        """The whole fix in one comparison: same window, only the level set differs."""
        zone = self._run(fib_level_mode="zone", fib_tolerance_pct=0.003)
        deep = self._run(fib_level_mode="deep", fib_tolerance_pct=0.003)
        broad = self._run(fib_level_mode="all", fib_tolerance_pct=0.003)

        assert deep["fib_blocked_signals"] > zone["fib_blocked_signals"]
        assert deep["num_trades"] < zone["num_trades"]
        # "all" is the loosest of the three: it also accepts the 0.382 level.
        assert broad["fib_blocked_signals"] <= deep["fib_blocked_signals"]

    def test_a_half_percent_miss_is_no_longer_a_touch(self):
        """0.5% away from the level: inside the old 1% floor, outside the new 0.3%."""
        from strategy import _fib_pullback_detail

        row = pd.Series({"close": 100.0, "atr": 0.0, "fib_382": 120.0,
                         "fib_500": 100.5, "fib_618": 80.0})
        frame = pd.DataFrame([row])

        tight = _fib_pullback_detail(frame, StrategyConfig(fib_tolerance_atr_mult=0.0))
        loose = _fib_pullback_detail(frame, StrategyConfig(fib_tolerance_pct=0.01,
                                                           fib_tolerance_atr_mult=0.0))

        assert tight["pass"] is False
        assert "outside" in tight["reason"].lower()
        assert loose["pass"] is True


class TestPrematureStopOut:
    """A trail that arms at +1 ATR and hugs 2 ATR back stops good trades out.

    The fix is two-sided: arm later (+1.5 ATR, so an ordinary pullback on the
    way up is just noise) and trail wider (3.5 ATR, so the primary target stays
    reachable instead of being clipped into a micro-gain).
    """

    @staticmethod
    def _risk(**overrides):
        return RiskManager(RiskConfig(**overrides), starting_equity=1000.0)

    def test_stop_does_not_move_at_all_before_the_arm_threshold(self):
        risk = self._risk()
        position = {"entry": 100.0, "stop": 94.0, "initial_stop": 94.0}

        # +1.4 ATR of profit is still not enough to arm (1.5 ATR required).
        for price in (100.0, 102.0, 104.0, 105.6):
            assert risk.update_trailing_stop(position, price, 4.0) == 94.0, price
            assert risk.is_running(position, price, 4.0) is False, price
        assert "trail_armed" not in position

    def test_a_wide_trail_keeps_the_target_reachable(self):
        """At +3 ATR with a 3.5 ATR trail the stop is still only break-even."""
        risk = self._risk()
        position = {"entry": 100.0, "stop": 94.0, "initial_stop": 94.0}
        stop = risk.update_trailing_stop(position, 112.0, 4.0)     # +3 ATR

        assert stop == pytest.approx(100.0)                        # not tightened
        # Only past +3.5 ATR does the trail lock in profit...
        assert risk.update_trailing_stop(position, 115.0, 4.0) == pytest.approx(101.0)
        # ...and it is still below the 4 ATR primary target (116.0).
        assert position["stop"] < 100.0 + 4.0 * 4.0

    def test_an_old_style_trail_would_have_clipped_the_same_trade(self):
        """Same path, previous 1.0 ATR arm / 2.0 ATR trail: stopped out early."""
        old = self._risk(trailing_arm_atr_mult=1.0, trailing_stop_atr_mult=2.0)
        new = self._risk()
        path = [104.0, 108.0, 110.0]
        pullback = 101.0                       # an ordinary wobble on the way up

        old_position = {"entry": 100.0, "stop": 94.0, "initial_stop": 94.0}
        new_position = {"entry": 100.0, "stop": 94.0, "initial_stop": 94.0}
        for price in path:
            old.stop = old.update_trailing_stop(old_position, price, 4.0)
            new.stop = new.update_trailing_stop(new_position, price, 4.0)

        # Old: peak 110 - 2*4 -> stop at 102, hit by a pullback to 101.
        assert old_position["stop"] == pytest.approx(102.0)
        assert pullback <= old_position["stop"]
        # New: still at break-even, so the same pullback leaves the trade alone.
        assert new_position["stop"] == pytest.approx(100.0)
        assert pullback > new_position["stop"]

    def test_take_profit_cannot_be_configured_below_two_and_a_half_atr(self):
        from risk_manager import MIN_TAKE_PROFIT_ATR_MULT, reward_risk_atr_mult

        assert MIN_TAKE_PROFIT_ATR_MULT == 2.5
        # Even a deliberately thin target is lifted to the floor.
        thin = RiskConfig(take_profit_atr_mult=1.0, min_reward_risk_ratio=0.0)
        assert reward_risk_atr_mult(thin) == pytest.approx(2.5)

        risk = RiskManager(thin, starting_equity=1000.0)
        stop, target = risk.stop_and_target(entry_price=100.0, atr_value=4.0, side="BUY")
        assert target - 100.0 >= 2.5 * 4.0

    def test_shipped_defaults_target_at_least_two_and_a_half_atr(self):
        from risk_manager import reward_risk_atr_mult

        cfg = RiskConfig()
        atr = 4.0
        risk = RiskManager(cfg, starting_equity=1000.0)
        _stop, target = risk.stop_and_target(entry_price=100.0, atr_value=atr, side="BUY")

        assert reward_risk_atr_mult(cfg) >= 2.5
        assert target - 100.0 >= 2.5 * atr

    def test_default_backtest_reaches_targets_and_arms_late(self):
        """End to end: winners get to targets, and losses are not the trail."""
        cfg = TestPipelineDiagnostics._diag_cfg()
        res = run_backtest(TestPipelineDiagnostics._series(), cfg)

        exits = res["exit_reasons"]
        assert exits.get("target", 0) > 0            # targets are reached
        # A late, wide trail must not be the dominant exit reason any more.
        assert exits.get("trail", 0) <= exits.get("target", 0) + exits.get("stop", 0)
        assert res["realized_rr"] > 1.5


class TestATRTrailingStopOption:
    """The trailing stop is optional and follows the highest high since entry."""

    @staticmethod
    def _risk(**overrides):
        return RiskManager(RiskConfig(**overrides), starting_equity=1000.0)

    def test_default_trail_arms_late_and_trails_wide(self):
        """A 1 ATR arm / 2 ATR trail stopped ordinary noise out of good trades."""
        cfg = RiskConfig()
        assert cfg.trailing_arm_atr_mult == 1.5
        assert cfg.trailing_stop_atr_mult == 3.5
        assert RiskConfig().trailing_stop_enabled is True

    def test_trail_follows_the_bar_high_not_just_the_close(self):
        risk = self._risk()
        position = {"entry": 100.0, "stop": 94.0, "initial_stop": 94.0}
        # The bar wicked to 125 but closed at 118: the trail must use the high,
        # and 3.5 ATR behind that is 111 after the +1.5 ATR arm.
        stop = risk.update_trailing_stop(position, 118.0, 4.0, "BUY", peak_price=125.0)
        assert position["peak"] == 125.0
        assert stop == pytest.approx(125.0 - 3.5 * 4.0)

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


class TestEntryDebounce:
    """A whipsaw must not chain entries on adjacent candles.

    The reported symptom was a BUY at 17:30, another at 17:35 and another at
    17:40 on a 5m chart — one setup, paid for three times over. The gate lives
    in `generate_signal` and is measured from the previous *exit*, so these
    tests drive the strategy directly and then prove the backtest loop feeds it
    the bar distance it needs.
    """

    def test_defaults_scale_with_the_grain(self):
        """3 bars is 15 minutes on 5m but 3 hours on 1h, so the count is keyed."""
        from config import FALLBACK_MIN_BARS_BETWEEN_TRADES, default_min_bars_between_trades

        for grain in ("1m", "3m", "5m", "15m"):
            assert default_min_bars_between_trades(grain) == 8, grain
        for grain in ("30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"):
            assert default_min_bars_between_trades(grain) == 3, grain
        # Case/whitespace tolerant, and an unknown grain still debounces.
        assert default_min_bars_between_trades(" 5M ") == 8
        assert default_min_bars_between_trades("3d") == FALLBACK_MIN_BARS_BETWEEN_TRADES
        assert default_min_bars_between_trades(None) == FALLBACK_MIN_BARS_BETWEEN_TRADES
        # The shipped config leaves it unset, i.e. "derive it from the grain".
        assert StrategyConfig().min_bars_between_trades is None

    def test_resolve_prefers_an_explicit_override(self):
        from config import resolve_min_bars_between_trades

        assert resolve_min_bars_between_trades(StrategyConfig(), "5m") == 8
        assert resolve_min_bars_between_trades(StrategyConfig(min_bars_between_trades=12), "5m") == 12
        # An explicit 0 means "no debounce", not "fall back to the default".
        assert resolve_min_bars_between_trades(StrategyConfig(min_bars_between_trades=0), "5m") == 0
        assert resolve_min_bars_between_trades(StrategyConfig(min_bars_between_trades=-4), "5m") == 0

    def test_a_signal_inside_the_cooldown_is_suppressed_with_the_exact_reason(self):
        from strategy import DEBOUNCE_REASON

        df = TestPipelineDiagnostics._cross()          # a candle that really is a BUY
        cfg = StrategyConfig()
        assert generate_signal(df, cfg, position_open=False, timeframe="1h") == Signal.BUY

        for gap in (0, 1, 2):
            stages = {}
            sig = generate_signal(df, cfg, position_open=False, diagnostics=stages,
                                  bars_since_last_trade=gap, timeframe="1h")
            assert sig == Signal.HOLD, gap
            assert stages["rejected_by"] == "signal_debounce"
            assert DEBOUNCE_REASON == "Signal debounce: waiting for cooldown period between setups"
            assert stages["debounce_reason"] == DEBOUNCE_REASON
            # The core trigger still fired — the debounce is what turned it away.
            assert stages["core_signal"] == "BUY"
            assert stages["core_trigger"] is True
            assert stages["debounce_bars_elapsed"] == gap
            assert stages["debounce_bars_required"] == 3     # 1h default

    def test_the_cooldown_clears_once_enough_bars_have_passed(self):
        df = TestPipelineDiagnostics._cross()
        cfg = StrategyConfig()

        for gap in (3, 4, 50):                                # >= required
            stages = {}
            assert generate_signal(df, cfg, position_open=False, diagnostics=stages,
                                   bars_since_last_trade=gap, timeframe="1h") == Signal.BUY
            assert stages.get("rejected_by") is None

        # Same candle and same distance, slower grain: 3 bars is not enough on 5m.
        stages = {}
        assert generate_signal(df, cfg, position_open=False, diagnostics=stages,
                               bars_since_last_trade=3, timeframe="5m") == Signal.HOLD
        assert stages["debounce_bars_required"] == 8

        # An explicit override beats the grain table in both directions.
        assert generate_signal(df, StrategyConfig(min_bars_between_trades=1), position_open=False,
                               bars_since_last_trade=1, timeframe="5m") == Signal.BUY
        assert generate_signal(df, StrategyConfig(min_bars_between_trades=20), position_open=False,
                               bars_since_last_trade=19, timeframe="1h") == Signal.HOLD

    def test_no_prior_trade_and_a_disabled_cooldown_never_block(self):
        df = TestPipelineDiagnostics._cross()

        # None = the bot has never traded, so there is nothing to debounce.
        assert generate_signal(df, StrategyConfig(), position_open=False,
                               bars_since_last_trade=None, timeframe="1h") == Signal.BUY
        # 0 = explicitly off, at every distance.
        for gap in (0, 1, 7):
            assert generate_signal(df, StrategyConfig(min_bars_between_trades=0), position_open=False,
                                   bars_since_last_trade=gap, timeframe="1h") == Signal.BUY

    def test_managing_an_open_position_is_never_debounced(self):
        """The cooldown gates entries; it must not touch exit management."""
        stages = {}
        generate_signal(TestPipelineDiagnostics._cross(), StrategyConfig(), position_open=True,
                        diagnostics=stages, bars_since_last_trade=0, timeframe="1h")
        assert stages.get("rejected_by") != "signal_debounce"

    def test_backtest_never_re_enters_inside_the_cooldown(self):
        """End to end: the loop counts bars from the exit, and the gap is honoured."""
        series = TestPipelineDiagnostics._series()

        # Baseline: with the debounce off, entries are allowed to crowd up.
        off = TestPipelineDiagnostics._diag_cfg()
        off.exchange.timeframe = "15m"
        off.strategy.min_bars_between_trades = 0
        base = run_backtest(series, off)
        assert base["min_bars_between_trades"] == 0
        assert base["debounce_blocked"] == 0
        assert base["min_entry_gap_bars"] < 8          # the chaining we are fixing

        on = TestPipelineDiagnostics._diag_cfg()
        on.exchange.timeframe = "15m"                  # 8-bar default for this grain
        guarded = run_backtest(series, on)

        assert guarded["min_bars_between_trades"] == 8
        assert guarded["debounce_blocked"] > 0
        assert guarded["min_entry_gap_bars"] >= 8      # never entered inside the cooldown
        assert guarded["min_entry_gap_bars"] > base["min_entry_gap_bars"]
        assert guarded["diagnostics"]["rejections"]["signal_debounce"] == guarded["debounce_blocked"]

        log = guarded["diagnostics"]["rejection_log"]
        assert any("Signal debounce: waiting for cooldown period between setups" in line
                   for line in log), log
        assert "entry debounce" in backtester.format_pipeline_report(guarded)

    def test_api_carries_the_debounce_into_the_engine(self, monkeypatch):
        """The knob reaches run_backtest through the real handler, and is echoed."""
        import web_server

        n = 900
        close = 50000 + np.sin(np.linspace(0, 30, n)) * 3000 + np.random.normal(0, 50, n)
        hist = pd.DataFrame({
            "ts": pd.date_range("2026-01-01", periods=n, freq="15min"),
            "open": close - 20, "high": close + 100, "low": close - 100,
            "close": close, "volume": np.random.uniform(10, 100, n),
        })
        monkeypatch.setattr(web_server, "fetch_historical", lambda *a, **k: hist.copy())

        def run(**kwargs):
            return web_server.run_backtest_endpoint(web_server.BacktestRequest(
                exchange="binance", symbol="BTC/USDT", timeframe="15m", days=30, **kwargs))["results"]

        assert run(min_bars_between_trades=12)["min_bars_between_trades"] == 12
        # Omitted -> the grain default (15m serves 8 bars) is what gets reported.
        assert run()["min_bars_between_trades"] == 8
        # An explicit 0 is "off", not "fall back to the default".
        assert run(min_bars_between_trades=0)["min_bars_between_trades"] == 0


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
        assert res["trailing_stop_atr_mult"] == pytest.approx(3.5)

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
        assert req.fib_level_mode == "deep"                 # 0.5 / 0.618, not the blanket zone
        assert req.fib_tolerance_pct == pytest.approx(0.003)  # tight floor, not 1%
        assert req.trailing_stop_enabled is True
        assert req.trailing_stop_atr_mult == 3.5

    def test_results_panel_reports_realized_rr_and_exits(self):
        html = web_server_dashboard_html()

        assert "Realized R:R" in html
        assert "bt-res-exits" in html
        assert "bt-res-rr" in html

    def test_served_pages_are_never_cached(self):
        """A cached dashboard keeps serving the previous JS — including a fixed
        bug. The HTML must arrive uncacheable so a deploy actually takes effect."""
        import web_server

        for response in (web_server.get_dashboard(), web_server.activate_page(token="x")):
            cache_control = response.headers.get("Cache-Control", "")
            assert "no-store" in cache_control, cache_control
            assert response.headers.get("Pragma") == "no-cache"


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
    and run it on its own. Braces inside strings, template literals and
    comments do not count, so a handler that quotes a JSON `{` still extracts.
    """
    src = _dashboard_script()
    match = re.search(rf"(?:async\s+)?function\s+{re.escape(name)}\s*\(", src)
    assert match, f"{name} is not declared in the dashboard script"
    start = match.start()          # keep the `async` keyword, or `await` is illegal
    open_brace = src.index("{", start)
    depth = 0
    quote = None                   # "'", '"' or "`" while inside a literal
    comment = None                 # "line" or "block" while inside a comment
    index = open_brace
    while index < len(src):
        char = src[index]
        nxt = src[index + 1] if index + 1 < len(src) else ""
        if comment == "line":
            if char == "\n":
                comment = None
        elif comment == "block":
            if char == "*" and nxt == "/":
                comment = None
                index += 1
        elif quote:
            if char == "\\":
                index += 1         # the next character is escaped
            elif char == quote:
                quote = None
        elif char == "/" and nxt == "/":
            comment = "line"
            index += 1
        elif char == "/" and nxt == "*":
            comment = "block"
            index += 1
        elif char in "\"'`":
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return src[start:index + 1]
        index += 1
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
        + "let noticeTimer = null;\n"
        + _dashboard_function("escapeHtml") + "\n"
        + _dashboard_function("setBacktestDays") + "\n"
        + _dashboard_function("backtestDayLimit") + "\n"
        + _dashboard_function("applyTimeframeLimits") + "\n"
        + _dashboard_function("hideNotice") + "\n"
        + _dashboard_function("toggleNoticeDetail") + "\n"
        + _dashboard_function("showNotice") + "\n"
        + _dashboard_function("firstUsefulLine") + "\n"
        + _dashboard_function("describeBacktestError") + "\n"
        + _dashboard_function("executeBacktest") + "\n"
        + _dashboard_function("getBacktestPayloadBase") + "\n"
        + _dashboard_function("initSweepTable") + "\n"
        + _dashboard_function("updateSweepTableRow") + "\n"
        + _dashboard_function("finalizeSweepSummary") + "\n"
        + _dashboard_function("executeCompareWindows") + "\n"
        + _dashboard_function("executeCompareTimeframes")
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
  const children = [];
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
    appendChild: (c) => children.push(c),
    children,
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
  // A payload can opt into a failure envelope via `__status`, so the error
  // path runs in the same engine (and with the same stub DOM) as the happy one.
  const status = apiPayload.__status || 200;
  return { ok: status >= 200 && status < 300, status, json: async () => apiPayload };
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
    noticeHidden: el('app-notice').classList.contains('hidden'),
    noticeTitle: el('app-notice-title').innerText,
    noticeBody: el('app-notice-body').innerText,
    noticeDetail: el('app-notice-detail').innerText,
    noticeDetailHidden: el('app-notice-detail').classList.contains('hidden'),
    noticeDetailBtnHidden: el('app-notice-details-btn').classList.contains('hidden'),
    noticeIcon: el('app-notice-icon').className,
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
    def _run_harness(tmp_path, function_name="executeBacktest", payload=None,
                     form=None):
        node = shutil.which("node")
        if not node:
            pytest.skip("node is not installed; JS runtime check skipped")
        (tmp_path / "fn.js").write_text(_backtest_js_source())
        (tmp_path / "form.json").write_text(json.dumps(form or _BACKTEST_FORM_VALUES))
        (tmp_path / "api.json").write_text(json.dumps(
            _BACKTEST_API_PAYLOAD if payload is None else payload))
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


class TestInAppNotice:
    """A failed backtest is explained inside the page, not by a browser alert().

    The deployed dashboard answered a blocked exchange with
    `alert('Backtest error: ' + err.message)`, which dumped a wall of ccxt JSON
    into a modal the user can neither style nor dismiss from the page.
    """

    # Copied from the reported console output in the screenshot.
    BINANCE_451 = (
        "binance GET https://api.binance.com/api/v3/exchangeInfo 451 "
        '{"code": 0, "msg": "Service unavailable from a restricted location '
        "according to 'b. Eligibility' in https://www.binance.com/en/terms. "
        'Please contact customer service if you believe you received this message '
        'in error."}'
    )

    def test_notice_markup_is_present_and_accessible(self):
        html = web_server_dashboard_html()

        for element_id in ("app-notice", "app-notice-icon", "app-notice-title",
                           "app-notice-body", "app-notice-detail",
                           "app-notice-details-btn", "app-notice-close"):
            assert f'id="{element_id}"' in html, element_id
        assert 'role="status"' in html
        assert 'aria-live="polite"' in html
        # The card stays out of the way until it actually has news.
        assert 'id="app-notice" class="hidden' in html

    def test_backtest_handler_no_longer_calls_alert(self):
        src = _dashboard_function("executeBacktest")
        code = re.sub(r"//[^\n]*", "", src)      # comments may still say alert()

        assert "alert(" not in code
        assert "showNotice(" in src
        assert "describeBacktestError(" in src

    def test_the_whole_served_script_is_valid_javascript(self, tmp_path):
        """One bad escape kills every handler on the page, not just its own.

        The inline script is served out of a Python string, so writing a bare
        `\n` where the page needed a literal backslash-n drops a raw newline
        inside a string literal and the whole dashboard stops parsing.
        """
        node = shutil.which("node")
        if not node:
            pytest.skip("node is not installed; JS runtime check skipped")
        script = tmp_path / "dashboard.js"
        script.write_text(_dashboard_script())
        proc = subprocess.run([node, "--check", str(script)],
                              capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, proc.stderr

    def test_blocked_region_reads_as_one_short_sentence(self, tmp_path):
        """The 451 wall becomes a title, a suggestion, and a hidden detail."""
        out = TestDashboardBacktestRuntime._run_harness(
            tmp_path, payload={"__status": 500, "detail": self.BINANCE_451})

        assert out["noticeHidden"] is False
        assert out["noticeTitle"] == "Binance is not available in this region"
        assert "Kraken" in out["noticeBody"]
        assert "451" not in out["noticeBody"]          # no raw JSON up front
        assert "b. Eligibility" not in out["noticeBody"]
        assert len(out["noticeBody"]) <= 200           # a sentence, not a paragraph
        # The raw text is kept, but collapsed behind "Technical detail".
        assert "b. Eligibility" in out["noticeDetail"]
        assert out["noticeDetailHidden"] is True
        assert out["noticeDetailBtnHidden"] is False
        assert "text-rose-400" in out["noticeIcon"]
        # No stale numbers from an earlier run are left on screen.
        assert out["resultsHidden"] is True
        assert out["buttonDisabled"] is False

    def test_unknown_failure_still_says_something_readable(self, tmp_path):
        out = TestDashboardBacktestRuntime._run_harness(
            tmp_path, payload={"__status": 502, "detail": "Bad Gateway"})

        assert out["noticeHidden"] is False
        assert out["noticeTitle"] == "Backtest could not run"
        assert out["noticeBody"] == "Bad Gateway"
        assert out["noticeDetail"] == "Bad Gateway"
        assert out["resultsHidden"] is True

    def test_a_good_run_clears_any_error_card(self, tmp_path):
        """A successful run must not leave the previous failure on screen."""
        out = TestDashboardBacktestRuntime._run_harness(tmp_path)

        assert out["noticeHidden"] is True
        assert out["resultsHidden"] is False

    def test_non_json_error_page_is_still_explained(self, tmp_path):
        """A gateway that answers with HTML must not leak a JSON parse error."""
        node = shutil.which("node")
        if not node:
            pytest.skip("node is not installed; JS runtime check skipped")
        (tmp_path / "fn.js").write_text(_backtest_js_source())
        (tmp_path / "form.json").write_text(json.dumps(_BACKTEST_FORM_VALUES))
        (tmp_path / "api.json").write_text('"<html>502 Bad Gateway</html>"')
        harness = _JS_BACKTEST_HARNESS.replace(
            "json: async () => apiPayload", "json: async () => { throw new SyntaxError('Unexpected token <'); }")
        (tmp_path / "harness.js").write_text(harness)
        proc = subprocess.run(
            [node, str(tmp_path / "harness.js"), str(tmp_path / "fn.js"),
             str(tmp_path / "form.json"), str(tmp_path / "api.json")],
            capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, f"node failed:\n{proc.stdout}\n{proc.stderr}"
        out = json.loads(proc.stdout.strip().splitlines()[-1])

        assert out["noticeHidden"] is False
        assert out["noticeTitle"] == "Backtest could not run"
        assert "HTTP 200" in out["noticeBody"] or "Unexpected token" in out["noticeDetail"]
        assert out["resultsHidden"] is True


class TestBacktestEnhancements:
    """Tests for per-trade Sharpe, fee accounting, timestamp reporting, and multi-window sweeps."""

    def test_per_trade_sharpe_suppressed_below_20_trades(self):
        trades = [{"pnl": 10.0, "exit_price": 100.0, "size": 1.0, "fee": 0.2} for _ in range(10)]
        res = summarize(trades, [{"equity": 1000.0}], 1000.0, "1h")
        assert res["sharpe_approx"] is None
        assert res["num_trades"] == 10

    def test_per_trade_sharpe_calculated_at_20_trades_or_more(self):
        trades = []
        for i in range(25):
            pnl = 20.0 if i % 2 == 0 else -10.0
            trades.append({"pnl": pnl, "exit_price": 100.0, "size": 1.0, "fee": 0.2})
        res = summarize(trades, [{"equity": 1000.0}], 1000.0, "1h")
        assert res["sharpe_approx"] is not None
        assert isinstance(res["sharpe_approx"], float)
        assert res["num_trades"] == 25

    def test_backtest_reports_total_fees_paid_and_timestamps(self):
        df = _wave_ohlcv(n=500)
        cfg = BotConfig()
        cfg.strategy.min_bars_between_trades = 0
        res = run_backtest(df.copy(), cfg)
        assert "total_fees_paid" in res
        assert "eval_date_range" in res
        assert "date_range" in res
        assert "candles_warmup" in res
        assert res["taker_fee_pct"] == 0.001
        assert res["slippage_pct"] == 0.0005
        assert res["eval_date_range"][0] == df["ts"].iloc[res["candles_warmup"]].isoformat()
        assert res["eval_date_range"][1] == df["ts"].iloc[-1].isoformat()

    def test_compare_windows_endpoint(self, monkeypatch):
        import web_server
        hist = _wave_ohlcv(n=600)
        monkeypatch.setattr(web_server, "fetch_historical", lambda *a, **k: hist.copy())

        req = web_server.CompareWindowsRequest(
            exchange="binance",
            symbol="BTC/USDT",
            timeframe="1h",
            windows=[30, 90],
            starting_equity=1000.0,
        )
        res = web_server.compare_windows_endpoint(req)
        assert res["success"] is True
        assert res["mode"] == "windows"
        assert len(res["rows"]) == 2
        for row in res["rows"]:
            assert "lookback_days" in row
            assert "trades" in row
            assert "win_rate_pct" in row
            assert "total_return_pct" in row
            assert "sharpe_approx" in row
            assert "realized_rr" in row
            assert "max_drawdown_pct" in row
            assert "fib_blocked" in row
            assert "sizing_blocked" in row
            assert "total_fees_paid" in row
            assert "profitable" in row

    def test_compare_timeframes_endpoint(self, monkeypatch):
        import web_server
        hist = _wave_ohlcv(n=300)
        monkeypatch.setattr(web_server, "fetch_historical", lambda *a, **k: hist.copy())

        req = web_server.CompareTimeframesRequest(
            exchange="binance",
            symbol="BTC/USDT",
            days=30,
            timeframes=["15m", "1h"],
            starting_equity=1000.0,
        )
        res = web_server.compare_timeframes_endpoint(req)
        assert res["success"] is True
        assert res["mode"] == "timeframes"
        assert len(res["rows"]) == 2
        assert res["rows"][0]["timeframe"] == "15m"
        assert res["rows"][1]["timeframe"] == "1h"

    def test_sanitize_for_json_converts_numpy_scalars(self):
        """The sweep buttons 500'd because FastAPI cannot encode NumPy scalars."""
        from web_server import sanitize_for_json

        payload = {
            "flag": np.bool_(True),
            "count": np.int64(7),
            "ratio": np.float64(1.25),
            "nested": {"blocked": np.bool_(False), "items": [np.int64(1), np.float64(2.5)]},
            "arr": np.array([1, 2, 3]),
            "arr2d": np.zeros((2, 2)),
            "plain": "text",
        }
        clean = sanitize_for_json(payload)

        assert clean["flag"] is True
        assert clean["count"] == 7 and isinstance(clean["count"], int)
        assert clean["ratio"] == 1.25 and isinstance(clean["ratio"], float)
        assert clean["nested"]["blocked"] is False
        assert clean["nested"]["items"] == [1, 2.5]
        assert clean["arr"] == [1, 2, 3]
        assert clean["arr2d"] == [[0.0, 0.0], [0.0, 0.0]]
        assert clean["plain"] == "text"
        # Every leaf must survive a real JSON round-trip.
        assert json.loads(json.dumps(clean)) == clean
        for value in clean.values():
            assert not isinstance(value, np.generic)

    @staticmethod
    def _numpy_backtest_result(candles):
        """A backtest payload stuffed with the NumPy scalars indicators emit."""
        return {
            "starting_equity": np.float64(1000.0),
            "final_equity": np.float64(1085.5),
            "total_return_pct": np.float64(8.55),
            "num_trades": np.int64(22),
            "win_rate_pct": np.float64(45.45),
            "avg_win": np.float64(20.0),
            "avg_loss": np.float64(-10.0),
            "realized_rr": np.float64(2.0),
            "total_fees_paid": np.float64(10.5),
            "exit_reasons": {"target": np.int64(10)},
            "max_drawdown_pct": np.float64(-4.5),
            "sharpe_approx": np.float64(1.2),
            "fib_filter_enabled": np.bool_(True),
            "fib_blocked_signals": np.int64(3),
            "trailing_stop_enabled": np.bool_(True),
            "debounce_blocked": np.int64(1),
            "min_bars_between_trades": np.int64(3),
            "min_entry_gap_bars": np.int64(4),
            "candles_tested": np.int64(len(candles)),
            "candles_warmup": np.int64(50),
            "date_range": [candles["ts"].iloc[0].isoformat(), candles["ts"].iloc[-1].isoformat()],
            "eval_date_range": [candles["ts"].iloc[0].isoformat(), candles["ts"].iloc[-1].isoformat()],
            "raw_signals": np.int64(30),
            "signals_rejected": {"fib_zone": np.int64(8)},
            "trades_executed": np.int64(22),
            "diagnostics": {
                "candles_evaluated": np.int64(len(candles)),
                "signals_risk_blocked": np.int64(0),
                "fib_filter_blocked": np.int64(3),
                "sizing_blocked": np.int64(0),
                "rejections": {"fib_zone": np.int64(8)},
                "core_misses": {"no_ema_cross": np.int64(12)},
                "all_clear": np.bool_(True),
                "report": "ok",
            },
        }

    def test_backtest_endpoint_sanitizes_numpy_results(self, monkeypatch):
        import web_server
        hist = _wave_ohlcv(n=100)
        monkeypatch.setattr(web_server, "fetch_historical", lambda *a, **k: hist.copy())
        monkeypatch.setattr(web_server, "run_backtest",
                            lambda df, cfg: self._numpy_backtest_result(hist))

        req = web_server.BacktestRequest(exchange="binance", symbol="BTC/USDT",
                                         timeframe="1h", days=30)
        res = web_server.run_backtest_endpoint(req)

        assert res["success"] is True
        assert res["results"]["num_trades"] == 22
        assert res["results"]["diagnostics"]["fib_filter_blocked"] == 3
        assert res["results"]["diagnostics"]["all_clear"] is True
        assert not isinstance(res["results"]["num_trades"], np.generic)
        assert json.loads(json.dumps(res))["results"]["final_equity"] == 1085.5

    def test_compare_endpoints_sanitize_numpy_results(self, monkeypatch):
        import web_server
        dates = pd.date_range("2026-01-01", periods=1500, freq="1h")
        close = 50000 + np.sin(np.linspace(0, 30, 1500)) * 3000
        hist = pd.DataFrame({
            "ts": dates, "open": close - 20, "high": close + 100,
            "low": close - 100, "close": close,
            "volume": np.random.default_rng(3).uniform(10, 100, 1500),
        })
        monkeypatch.setattr(web_server, "fetch_historical", lambda *a, **k: hist.copy())
        monkeypatch.setattr(web_server, "run_backtest",
                            lambda df, cfg: self._numpy_backtest_result(hist))

        windows = web_server.compare_windows_endpoint(web_server.CompareWindowsRequest(
            exchange="binance", symbol="BTC/USDT", timeframe="1h",
            windows=[30, 60], starting_equity=1000.0))
        assert windows["success"] is True
        assert len(windows["rows"]) == 2
        assert windows["rows"][0]["profitable"] is True
        assert json.loads(json.dumps(windows))["rows"][0]["trades"] == 22

        timeframes = web_server.compare_timeframes_endpoint(web_server.CompareTimeframesRequest(
            exchange="binance", symbol="BTC/USDT", days=30,
            timeframes=["15m", "1h"], starting_equity=1000.0))
        assert timeframes["success"] is True
        assert [row["timeframe"] for row in timeframes["rows"]] == ["15m", "1h"]
        assert timeframes["rows"][1]["fib_blocked"] == 3
        assert json.loads(json.dumps(timeframes))["profitable_count"] == 2

    def test_dashboard_sweep_runtime_and_error_handling(self, tmp_path):
        """executeCompareWindows and executeCompareTimeframes must run cleanly in Node.js."""
        node = shutil.which("node")
        if not node:
            pytest.skip("node is not installed; JS runtime check skipped")

        src = _backtest_js_source()
        (tmp_path / "fn.js").write_text(src)
        test_js = r"""
const fs = require('fs');
const fnCode = fs.readFileSync(process.argv[2], 'utf8');

const elements = {};
function makeEl(id) {
  const classes = new Set(['hidden']);
  let text = '';
  let val = '';
  const children = [];
  return {
    id,
    get innerText() { return text; },
    set innerText(v) { text = String(v); },
    get value() { return val; },
    set value(v) { val = String(v); },
    innerHTML: '', checked: true,
    disabled: false, className: '',
    appendChild: (c) => children.push(c),
    children,
    classList: {
      add: (c) => classes.add(c),
      remove: (c) => classes.delete(c),
      contains: (c) => classes.has(c),
    },
  };
}
const el = (id) => (elements[id] = elements[id] || makeEl(id));
globalThis.document = { getElementById: el, createElement: () => makeEl('tr') };

el('bt-exchange').value = 'mexc';
el('bt-symbol').value = 'BTC/USDT';
el('bt-timeframe').value = '1h';
el('bt-days').value = '365';
el('bt-equity').value = '1000';
el('bt-fib-depth').value = 'deep';
el('bt-fib-atr').value = '0.5';
el('bt-fib-floor').value = '0.3';
el('bt-trail-atr').value = '3.5';

const sweepFunctions = new Function(
  fnCode + '\nreturn { executeCompareWindows, executeCompareTimeframes };')();

globalThis.fetch = async (url, opts) => {
  return {
    ok: true, status: 200,
    json: async () => ({
      success: true,
      results: {
        total_return_pct: 8.5,
        num_trades: 22,
        win_rate_pct: 50.0,
        final_equity: 1085.0,
        sharpe_approx: 1.2,
        realized_rr: 2.0,
        max_drawdown_pct: -4.5,
        total_fees_paid: 10.0,
        date_range: ['2023-01-01', '2024-01-01'],
        eval_date_range: ['2023-02-01', '2024-01-01'],
        diagnostics: { fib_filter_blocked: 3, sizing_blocked: 0, signals_risk_blocked: 0 }
      }
    })
  };
};

(async () => {
  await sweepFunctions.executeCompareWindows({ preventDefault() {} });
  const badgeWindows = el('bt-sweep-badge').innerText;
  const btnWindowsDisabled = el('btn-sweep-windows').disabled;

  await sweepFunctions.executeCompareTimeframes({ preventDefault() {} });
  const badgeTf = el('bt-sweep-badge').innerText;
  const btnTfDisabled = el('btn-sweep-timeframes').disabled;

  globalThis.fetch = async () => ({
    ok: false, status: 500,
    json: async () => ({ success: false, detail: 'binance 451 restricted location' })
  });

  await sweepFunctions.executeCompareWindows({ preventDefault() {} });
  const badgeErr = el('bt-sweep-badge').innerText;
  const noticeTitle = el('app-notice-title').innerText;

  console.log(JSON.stringify({
    badgeWindows, btnWindowsDisabled,
    badgeTf, btnTfDisabled,
    badgeErr, noticeTitle
  }));
})().catch(err => {
  console.error('FAILED:', err);
  process.exit(1);
});
"""
        (tmp_path / "test.js").write_text(test_js)
        proc = subprocess.run([node, str(tmp_path / "test.js"), str(tmp_path / "fn.js")],
                              capture_output=True, text=True, timeout=30)
        assert proc.returncode == 0, f"node failed:\n{proc.stdout}\n{proc.stderr}"
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        assert out["badgeWindows"] == "ROBUST REGIME"
        assert out["btnWindowsDisabled"] is False
        assert out["badgeTf"] == "ROBUST REGIME"
        assert out["btnTfDisabled"] is False
        assert out["badgeErr"] == "EXECUTION ERROR"
        assert "is not available in this region" in out["noticeTitle"]

    def test_compare_windows_skips_windows_exceeding_history_on_short_timeframes(self, monkeypatch):
        """15m only stores ~90d: 365d and 730d must be skipped without failing the request."""
        import web_server
        hist = _wave_ohlcv(n=500)
        monkeypatch.setattr(web_server, "fetch_historical", lambda *a, **k: hist.copy())

        req = web_server.CompareWindowsRequest(
            exchange="binance",
            symbol="BTC/USDT",
            timeframe="15m",
            windows=[90, 365, 730],
            starting_equity=1000.0,
        )
        res = web_server.compare_windows_endpoint(req)
        assert res["success"] is True
        assert len(res["rows"]) == 3
        # 90d runs
        assert res["rows"][0]["lookback_days"] == 90
        assert "error" not in res["rows"][0]
        # 365d skipped
        assert res["rows"][1]["lookback_days"] == 365
        assert res["rows"][1]["error"] == "Not enough history for 15m (max ~90d)"
        assert res["rows"][1]["trades"] == 0
        # 730d skipped
        assert res["rows"][2]["lookback_days"] == 730
        assert res["rows"][2]["error"] == "Not enough history for 15m (max ~90d)"
        assert res["rows"][2]["trades"] == 0

        # Entire response must serialize cleanly to JSON
        encoded = json.loads(json.dumps(res))
        assert encoded["rows"][1]["error"] == "Not enough history for 15m (max ~90d)"

    def test_compare_timeframes_skips_short_timeframes_when_lookback_exceeds(self, monkeypatch):
        """When 365d is requested, 5m and 15m must be skipped while 1h runs."""
        import web_server
        hist = _wave_ohlcv(n=500)
        monkeypatch.setattr(web_server, "fetch_historical", lambda *a, **k: hist.copy())

        req = web_server.CompareTimeframesRequest(
            exchange="binance",
            symbol="BTC/USDT",
            days=365,
            timeframes=["5m", "15m", "1h"],
            starting_equity=1000.0,
        )
        res = web_server.compare_timeframes_endpoint(req)
        assert res["success"] is True
        assert len(res["rows"]) == 3
        # 5m skipped (max ~60d)
        assert res["rows"][0]["timeframe"] == "5m"
        assert res["rows"][0]["error"] == "Not enough history for 5m (max ~60d)"
        # 15m skipped (max ~90d)
        assert res["rows"][1]["timeframe"] == "15m"
        assert res["rows"][1]["error"] == "Not enough history for 15m (max ~90d)"
        # 1h runs (stores up to 730d)
        assert res["rows"][2]["timeframe"] == "1h"
        assert "error" not in res["rows"][2]
        assert res["rows"][2]["lookback_days"] == 365

        # Entire response must serialize cleanly to JSON
        encoded = json.loads(json.dumps(res))
        assert encoded["rows"][0]["error"] == "Not enough history for 5m (max ~60d)"
        assert encoded["rows"][1]["error"] == "Not enough history for 15m (max ~90d)"

    def test_breakdown_accounting_sums_to_candles_evaluated_exactly(self):
        """The miss breakdown + raw signals must sum to candles_evaluated with 0 gap."""
        from backtester import run_backtest
        res = run_backtest(TestPipelineDiagnostics._series(), TestPipelineDiagnostics._diag_cfg())
        diag = res["diagnostics"]

        assert diag["candles_in_position"] > 0
        assert "in_position" in diag["core_misses"]
        assert diag["core_misses"]["in_position"] == diag["candles_in_position"]
        total_accounted = sum(diag["core_misses"].values()) + diag["core_signals_detected"]
        assert total_accounted == diag["candles_evaluated"]
        assert total_accounted == res["candles_evaluated"]

    def test_walked_candle_range_and_post_warmup_count_logged(self, caplog):
        """Activity log must include walked first/last timestamps and post-warmup evaluated count."""
        from backtester import run_backtest
        from config import BotConfig
        cfg = BotConfig()
        cfg.strategy.use_fib_filter = False
        hist = _wave_ohlcv(n=200)

        with caplog.at_level("INFO", logger="backtest"):
            res = run_backtest(hist, cfg)

        text = " ".join(record.getMessage() for record in caplog.records)
        eval_range = res["eval_date_range"]
        eval_count = res["candles_evaluated"]

        assert "Strategy loop walked" in text
        assert f"{eval_count:,} post-warmup candle(s) evaluated for signals" in text
        assert eval_range[0] in text
        assert eval_range[1] in text

    def test_sweep_endpoints_log_walked_range_and_evaluated_candles(self, monkeypatch, caplog):
        """Compare windows and compare timeframes endpoints must log walked range and count per row."""
        import web_server
        hist = _wave_ohlcv(n=400)
        monkeypatch.setattr(web_server, "fetch_historical", lambda *a, **k: hist.copy())

        with caplog.at_level("INFO", logger="web"):
            # Compare windows
            req_win = web_server.CompareWindowsRequest(
                exchange="binance", symbol="BTC/USDT", timeframe="1h",
                windows=[30, 60], starting_equity=1000.0,
            )
            res_win = web_server.compare_windows_endpoint(req_win)
            assert res_win["success"] is True

            # Compare timeframes
            req_tf = web_server.CompareTimeframesRequest(
                exchange="binance", symbol="BTC/USDT", days=30,
                timeframes=["1h", "4h"], starting_equity=1000.0,
            )
            res_tf = web_server.compare_timeframes_endpoint(req_tf)
            assert res_tf["success"] is True

        web_text = " ".join(record.getMessage() for record in caplog.records)
        assert "Sweep window 30d: walked" in web_text
        assert "Sweep window 60d: walked" in web_text
        assert "Sweep timeframe 1h: walked" in web_text
        assert "Sweep timeframe 4h: walked" in web_text

    def test_sweep_tag_logged_and_passed_through_pipeline(self, caplog):
        """When sweep_tag is provided, both engine and web logs should be tagged with it."""
        import web_server
        import backtester
        from config import BotConfig
        cfg = BotConfig()
        df = _wave_ohlcv(n=200)

        with caplog.at_level("INFO", logger="backtest"):
            res = backtester.run_backtest(df, cfg, sweep_tag="Sweep 365d")
            assert res["sweep_tag"] == "Sweep 365d"
            assert res["diagnostics"]["sweep_tag"] == "Sweep 365d"

        bt_text = " ".join(record.getMessage() for record in caplog.records)
        assert "[Sweep 365d] Strategy loop walked" in bt_text
        assert "[Sweep 365d] Backtest summary:" in bt_text
        assert "[Sweep 365d] Backtest stages:" in bt_text

    def test_dashboard_html_contains_log_filters_and_sweep_tagging(self):
        """Dashboard HTML must include sweep log filters and sweep tagging logic."""
        import web_server
        html = web_server.DASHBOARD_HTML
        assert 'id="btn-log-filter-sweep"' in html
        assert 'id="btn-log-filter-live"' in html
        assert 'id="btn-log-filter-all"' in html
        assert "setLogFilter(" in html
        assert "renderLogs(" in html
        assert "sweep_tag: 'Sweep ' + item.label" in html
        assert "sweep_tag: 'Sweep ' + item.tf" in html


class TestProductionReadinessStructuralGaps:
    """
    Tests for the 8 structural gaps addressed prior to deploying real capital:
    #0: Ambiguity resolution & safety checks on negative-expectancy timeframes (15m/5m)
    #1: Persistent live/paper execution tracking for positions and orders (fills, rejections, cancellations)
    #2: Walk-Forward Validation (out-of-sample parameter verification)
    #3: Buy-and-Hold benchmark on every backtest and sweep row
    #4: Exchange minimum-notional sizing validation
    #5: Fee-tier transparency and per-exchange accuracy
    #6: Complete pipeline candle accounting
    #7: Automated risk alerting dispatcher
    """

    def test_item0_negative_expectancy_timeframe_guarded_in_live_mode(self):
        """Live execution on 15m/5m must be blocked without explicit override."""
        import web_server
        from fastapi import HTTPException

        # Mock runner in stopped state
        web_server.runner.running = False
        web_server.runner.cfg.mode = "live"
        web_server.runner.cfg.exchange.timeframe = "15m"

        # Attempt to start in live mode on 15m without force flag -> HTTP 400
        with pytest.raises(HTTPException) as exc_info:
            web_server.start_bot(force=False)
        assert exc_info.value.status_code == 400
        assert "negative expectancy" in exc_info.value.detail.lower()

        # Starting on 15m with force=True is allowed
        with mock.patch.object(web_server.runner, "start", return_value=True):
            res_force = web_server.start_bot(force=True)
            assert res_force["success"] is True

        # In paper mode, starting on 15m is allowed without force
        web_server.runner.running = False
        web_server.runner.cfg.mode = "paper"
        with mock.patch.object(web_server.runner, "start", return_value=True):
            res_paper = web_server.start_bot(force=False)
            assert res_paper["success"] is True

        # Status endpoint reports timeframe_warning on 15m
        status = web_server.get_status()
        assert status["timeframe_warning"] is not None
        assert "15m" in status["timeframe_warning"]

        # Reset timeframe to 1h
        web_server.runner.cfg.exchange.timeframe = "1h"
        status_1h = web_server.get_status()
        assert status_1h["timeframe_warning"] is None

    def test_item1_execution_tracker_orders_and_position_lifecycle(self, tmp_path):
        """ExecutionTracker records persistent order audits and position telemetry."""
        from execution_tracker import ExecutionTracker
        import web_server

        db_path = str(tmp_path / "test_tracker.db")
        tracker = ExecutionTracker(db_path=db_path)

        # 1. Record orders
        o1_id = tracker.record_order(symbol="BTC/USDT", timeframe="1h", side="BUY", size=0.05, price=65000.0, status="FILLED")
        o2_id = tracker.record_order(symbol="BTC/USDT", timeframe="1h", side="BUY", size=0.0001, price=65000.0, status="REJECTED", rejection_reason="Below exchange min notional $10.00")
        o3_id = tracker.record_order(symbol="BTC/USDT", timeframe="1h", side="SELL", size=0.05, price=66000.0, status="CANCELLED")

        orders = tracker.get_orders(limit=10)
        assert len(orders) == 3
        statuses = [o["status"] for o in orders]
        assert "FILLED" in statuses
        assert "REJECTED" in statuses
        assert "CANCELLED" in statuses

        # 2. Position telemetry
        pos_id = tracker.record_position_opened(
            symbol="BTC/USDT",
            timeframe="1h",
            side="BUY",
            size=0.05,
            entry_price=65000.0,
            stop=64000.0,
            target=68000.0,
            fee=3.25,
            mode="paper"
        )
        assert tracker.get_open_position() is not None

        tracker.update_open_position(current_price=66500.0, current_stop=64500.0, trailing_stop=64800.0)
        open_pos = tracker.get_open_position()
        assert open_pos["unrealized_pnl"] > 0

        # 3. Close position
        tracker.record_position_closed(exit_price=67000.0, exit_fee=3.35, realized_pnl=93.4, exit_reason="TAKE_PROFIT")
        assert tracker.get_open_position() is None
        assert len(tracker.get_closed_positions()) == 1

        # 4. Verify API endpoint
        with mock.patch("web_server.get_execution_tracker", return_value=tracker):
            res = web_server.get_execution_tracker_data()
            assert res["success"] is True
            assert len(res["orders"]) == 3
            assert res["active_position"] is None

    def test_item2_walk_forward_validation(self):
        """Walk-forward validation generates train/test folds and computes WFE."""
        import backtester
        from config import BotConfig
        import web_server

        df = _wave_ohlcv(n=250)
        cfg = BotConfig()

        wfo = backtester.run_walk_forward_validation(df, cfg, n_splits=3)
        assert wfo["success"] is True
        assert len(wfo["folds"]) == 3
        assert "walk_forward_efficiency_pct" in wfo
        assert "status" in wfo
        assert wfo["status"] in ("OUT_OF_SAMPLE_VALIDATED", "IN_SAMPLE_ONLY_UNVERIFIED")

        for f in wfo["folds"]:
            assert "in_sample_return_pct" in f
            assert "out_of_sample_return_pct" in f
            assert "out_of_sample_bh_return_pct" in f
            assert "in_sample_trades" in f
            assert "out_of_sample_trades" in f

        # Test endpoint
        with mock.patch("web_server.fetch_historical", return_value=df):
            req = web_server.WalkForwardRequest(
                exchange="binance",
                symbol="BTC/USDT",
                timeframe="1h",
                days=30,
                starting_equity=1000.0,
                n_splits=3
            )
            res = web_server.walk_forward_endpoint(req)
            assert res["success"] is True
            assert "results" in res
            assert len(res["results"]["folds"]) == 3

    def test_item3_buy_and_hold_benchmark(self):
        """Every backtest and sweep row must include zero-fee buy & hold benchmark."""
        import backtester
        from config import BotConfig
        import web_server

        df = _wave_ohlcv(n=200)
        cfg = BotConfig()

        res = backtester.run_backtest(df, cfg)
        assert "buy_and_hold_return_pct" in res
        assert "buy_and_hold_final_equity" in res
        assert isinstance(res["buy_and_hold_return_pct"], (int, float))

        # Sweep row verification
        with mock.patch("web_server.fetch_historical", return_value=df):
            req_win = web_server.CompareWindowsRequest(
                exchange="binance", symbol="BTC/USDT", timeframe="1h",
                windows=[30, 60], starting_equity=1000.0,
            )
            res_win = web_server.compare_windows_endpoint(req_win)
            for row in res_win["rows"]:
                assert "buy_and_hold_return_pct" in row

    def test_item4_exchange_minimum_notional_validation(self):
        """Computed order size must clear exchange minimum notional/amount limits."""
        from risk_manager import RiskManager
        from config import RiskConfig
        import web_server

        rm = RiskManager(RiskConfig(), starting_equity=1000.0)
        # Case 1: below min cost ($5 < $10)
        ok1, err1 = rm.validate_min_notional(entry_price=50000.0, size=0.0001, min_notional=10.0, min_amount=0.00001)
        assert ok1 is False
        assert "below exchange minimum" in err1

        # Case 2: above min cost ($50 >= $10)
        ok2, err2 = rm.validate_min_notional(entry_price=50000.0, size=0.001, min_notional=10.0, min_amount=0.00001)
        assert ok2 is True
        assert err2 == "OK"

        # Sizing validation endpoint
        req_sizing = web_server.SizingValidationRequest(
            equity=1000.0, risk_per_trade_pct=0.01, price=50000.0, stop_price=49000.0, min_notional=10.0
        )
        res = web_server.validate_sizing_endpoint(req_sizing)
        assert res["success"] is True
        assert "clears_minimum" in res
        assert "explanation" in res

    def test_item5_fee_tier_accuracy(self):
        """Fee tiers must accurately reflect exchange schedule or verified API tier."""
        from exchange_client import get_exchange_fee_info
        import web_server

        mexc_fees = get_exchange_fee_info("mexc")
        assert mexc_fees["taker"] == 0.001  # 0.10% taker
        assert mexc_fees["taker_fee_pct"] == 0.001
        assert mexc_fees["source"] == "hardcoded_default"

        binance_fees = get_exchange_fee_info("binance")
        assert binance_fees["taker"] == 0.001

        res = web_server.get_fee_info_endpoint("mexc")
        assert res["success"] is True
        assert res["fee_info"]["taker"] == 0.001

    def test_item6_pipeline_candle_accounting(self):
        """Pipeline accounting must account for open position candles as well as misses."""
        import backtester
        from config import BotConfig

        df = _wave_ohlcv(n=250)
        cfg = BotConfig()

        res = backtester.run_backtest(df, cfg)
        diag = res.get("diagnostics", {})
        candles_eval = diag.get("candles_evaluated", 0)

        # Rejection breakdown
        misses = diag.get("core_misses", {})
        total_misses = sum(misses.values())
        raw_signals = diag.get("raw_signals", 0)

        # Sum of non-signal candles + core signals evaluated
        assert total_misses + raw_signals == candles_eval

    def test_item7_alert_dispatcher(self, caplog):
        """AlertDispatcher notifies channels and handles unconfigured endpoints gracefully."""
        from alerts import AlertDispatcher
        import web_server

        dispatcher = AlertDispatcher()

        with caplog.at_level("CRITICAL", logger="alerts"):
            ok = dispatcher.send_alert(
                event_type="CIRCUIT_BREAKER",
                title="Daily loss limit hit",
                message="Daily loss limit hit (-5.0%). Trading halted.",
                level="critical",
                details={"current_drawdown_pct": 5.2},
                force=True,
            )
            assert ok is True

        log_text = " ".join(record.getMessage() for record in caplog.records)
        assert "CIRCUIT_BREAKER" in log_text

        # API test endpoint
        res = web_server.test_alert_endpoint()
        assert res["success"] is True


# ============================================================================
# CCXT ENHANCEMENTS TESTS
# ============================================================================
class TestCCXTEnhancements:
    def test_amount_and_price_precision(self):
        """Precision helpers use CCXT formatting or graceful fallbacks."""
        cfg = ExchangeConfig(exchange_id="binance", symbol="BTC/USDT")
        client = ExchangeClient(cfg=cfg, mode="paper", fee_pct=0.001, slippage_pct=0.0005, starting_equity=1000.0)

        # Fallback when markets are not loaded
        assert client.amount_to_precision(0.123456789) == 0.123457
        assert client.price_to_precision(50000.12345) == 50000.12

        # When exchange methods are present and markets loaded
        client.exchange.markets = {"BTC/USDT": {"precision": {"amount": 3, "price": 1}}}
        client.exchange.amount_to_precision = mock.MagicMock(return_value="0.123")
        client.exchange.price_to_precision = mock.MagicMock(return_value="50000.1")

        assert client.amount_to_precision(0.123456, "BTC/USDT") == 0.123
        assert client.price_to_precision(50000.123, "BTC/USDT") == 50000.1
        client.exchange.amount_to_precision.assert_called_with("BTC/USDT", 0.123456)
        client.exchange.price_to_precision.assert_called_with("BTC/USDT", 50000.123)

    def test_fetch_wallet_balance_paper_and_live(self):
        """Wallet balance reflects paper accounting or live CCXT fetch_balance."""
        cfg = ExchangeConfig(exchange_id="binance", symbol="BTC/USDT")
        client = ExchangeClient(cfg=cfg, mode="paper", fee_pct=0.001, slippage_pct=0.0005, starting_equity=1250.0)
        client.paper_position = 0.05

        # Paper mode
        bal_paper = client.fetch_wallet_balance("BTC/USDT")
        assert bal_paper["mode"] == "paper"
        assert bal_paper["is_live"] is False
        assert bal_paper["free_quote"] == 1250.0
        assert bal_paper["free_base"] == 0.05
        assert bal_paper["quote_currency"] == "USDT"
        assert bal_paper["base_currency"] == "BTC"

        # Live mode
        cfg_live = ExchangeConfig(exchange_id="binance", symbol="BTC/USDT", api_key="test_key", api_secret="test_sec")
        client_live = ExchangeClient(cfg=cfg_live, mode="live", fee_pct=0.001, slippage_pct=0.0005, starting_equity=1000.0)
        client_live.exchange.fetch_balance = mock.MagicMock(return_value={
            "free": {"USDT": 4500.50, "BTC": 0.25},
            "total": {"USDT": 5000.00, "BTC": 0.25},
        })

        bal_live = client_live.fetch_wallet_balance("BTC/USDT")
        assert bal_live["mode"] == "live"
        assert bal_live["is_live"] is True
        assert bal_live["free_quote"] == 4500.50
        assert bal_live["free_base"] == 0.25
        assert bal_live["total_quote"] == 5000.00

    def test_fetch_order_book_spread(self):
        """Order book spread extracts top-of-book and computes spread percentage."""
        cfg = ExchangeConfig(exchange_id="binance", symbol="BTC/USDT")
        client = ExchangeClient(cfg=cfg, mode="paper", fee_pct=0.001, slippage_pct=0.0005, starting_equity=1000.0)

        # Mock order book with 0.1% spread
        client.exchange.fetch_order_book = mock.MagicMock(return_value={
            "bids": [[50000.0, 1.5]],
            "asks": [[50050.0, 2.0]],
        })

        ob_data = client.fetch_order_book_spread("BTC/USDT")
        assert ob_data["valid"] is True
        assert ob_data["bid"] == 50000.0
        assert ob_data["ask"] == 50050.0
        assert ob_data["spread"] == 50.0
        assert ob_data["mid"] == 50025.0
        assert ob_data["spread_pct"] == pytest.approx(0.0999, rel=1e-2)

        # Failure / empty fallback
        client.exchange.fetch_order_book = mock.MagicMock(side_effect=Exception("Timeout"))
        client.current_price = mock.MagicMock(return_value=50000.0)
        fallback = client.fetch_order_book_spread("BTC/USDT")
        assert fallback["valid"] is False
        assert fallback["spread"] == 0.0

    def test_sync_live_balance_updates_equity(self):
        """sync_live_balance updates RiskManager equity in live mode."""
        cfg = BotConfig(mode="live")
        cfg.exchange.api_key = "live_key"
        cfg.exchange.api_secret = "live_secret"

        with mock.patch.object(ccxt.Exchange, "load_time_difference"):
            bot = TradingBot(cfg)
            assert bot.risk.state.equity == 1000.0

            with mock.patch.object(bot.exchange, "fetch_wallet_balance", return_value={
                "mode": "live",
                "is_live": True,
                "free_quote": 8500.0,
                "total_quote": 9200.0,
                "free_base": 0.1,
                "quote_currency": "USDT",
                "base_currency": "BTC",
            }):
                wallet = bot.sync_live_balance()
                assert wallet["is_live"] is True
                assert bot.risk.state.equity == 9200.0

    def test_order_book_spread_guard_blocks_entry(self):
        """Spread exceeding max_spread_pct suppresses BUY order entry."""
        from strategy import Signal

        cfg = BotConfig(mode="paper")
        cfg.risk.max_spread_pct = 0.35
        bot = TradingBot(cfg)

        df = _wave_ohlcv(n=100)
        price = 50000.0

        # Mock signal to BUY with wide spread
        with mock.patch("bot.generate_signal", return_value=Signal.BUY), \
             mock.patch.object(bot.exchange, "fetch_order_book_spread", return_value={
                 "valid": True,
                 "bid": 49700.0,
                 "ask": 50000.0,
                 "spread": 300.0,
                 "spread_pct": 0.60,
                 "mid": 49850.0,
             }), \
             mock.patch.object(bot.exchange, "place_order") as mock_place:
            bot._maybe_enter(df, price)
            mock_place.assert_not_called()
            assert bot.position is None

    def test_insufficient_wallet_balance_blocks_entry(self):
        """Insufficient free quote currency suppresses/rejects BUY order."""
        from strategy import Signal

        cfg = BotConfig(mode="live")
        cfg.exchange.api_key = "live_key"

        with mock.patch.object(ccxt.Exchange, "load_time_difference"):
            bot = TradingBot(cfg)

            df = _wave_ohlcv(n=100)
            price = 50000.0

            with mock.patch("bot.generate_signal", return_value=Signal.BUY), \
                 mock.patch.object(bot.exchange, "fetch_order_book_spread", return_value={"valid": True, "spread_pct": 0.05}), \
                 mock.patch.object(bot.exchange, "fetch_wallet_balance", return_value={
                     "mode": "live",
                     "is_live": True,
                     "free_quote": 5.0,
                     "total_quote": 5.0,
                     "quote_currency": "USDT",
                 }), \
                 mock.patch.object(bot.exchange, "place_order") as mock_place:
                bot._maybe_enter(df, price)
                mock_place.assert_not_called()
                assert bot.position is None

            # manual_buy raises RuntimeError
            with mock.patch.object(bot.exchange, "fetch_ohlcv", return_value=df), \
                 mock.patch.object(bot.exchange, "fetch_wallet_balance", return_value={
                     "mode": "live",
                     "is_live": True,
                     "free_quote": 5.0,
                     "total_quote": 5.0,
                     "quote_currency": "USDT",
                 }):
                with pytest.raises(RuntimeError, match="Insufficient wallet balance"):
                    bot.manual_buy(custom_size=0.01)

    def test_typed_ccxt_exceptions_on_live_order(self):
        """_place_live_order catches typed CCXT exceptions and sends alerts."""
        cfg = ExchangeConfig(exchange_id="binance", symbol="BTC/USDT", api_key="k", api_secret="s")
        with mock.patch.object(ccxt.Exchange, "load_time_difference"):
            client = ExchangeClient(cfg=cfg, mode="live", fee_pct=0.001, slippage_pct=0.0005, starting_equity=1000.0)

            # InsufficientFunds
            client.exchange.create_order = mock.MagicMock(side_effect=ccxt.InsufficientFunds("Account has insufficient balance"))
            with pytest.raises(ccxt.InsufficientFunds):
                client._place_live_order("BUY", 0.01)

            # InvalidOrder
            client.exchange.create_order = mock.MagicMock(side_effect=ccxt.InvalidOrder("Filter failure: LOT_SIZE"))
            with pytest.raises(ccxt.InvalidOrder):
                client._place_live_order("BUY", 0.01)

            # RateLimitExceeded
            client.exchange.create_order = mock.MagicMock(side_effect=ccxt.RateLimitExceeded("Too many requests"))
            with pytest.raises(ccxt.RateLimitExceeded):
                client._place_live_order("BUY", 0.01)

    def test_api_exchange_wallet_endpoint(self):
        """GET /api/exchange/wallet returns wallet balances and spread info."""
        import web_server

        res = web_server.get_exchange_wallet_endpoint()
        assert res["success"] is True
        assert "wallet" in res
        assert "spread" in res
        assert "limits" in res
        assert "max_spread_pct" in res
        assert res["wallet"]["mode"] in ("paper", "live")

