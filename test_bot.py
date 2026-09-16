"""
Comprehensive test suite for Crypto Trading Bot.
Covers indicators, strategy signals, risk management, execution, and backtesting.
"""
import os
import shutil
import tempfile
import pytest
import numpy as np
import pandas as pd

from config import BotConfig, StrategyConfig, RiskConfig, ExchangeConfig
from indicators import ema, rsi, atr, adx, fib_retracement_levels
from strategy import Signal, add_indicators, generate_signal, _near_fib_support
from risk_manager import RiskManager
from exchange_client import ExchangeClient
from backtester import run_backtest, summarize
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
