# Trader Queen

EMA-crossover trend strategy, filtered by RSI, ADX, and an automatic
Fibonacci retracement zone, with a full risk layer (position sizing, stops,
drawdown kill switch, circuit breakers), fee/slippage-aware paper trading,
and a backtester — built on `ccxt` so it works with any supported exchange.

## How each design principle shows up in the code

| Principle | Where |
|---|---|
| Clear, testable strategy | `strategy.py` — EMA crossover + RSI + ADX + Fibonacci retracement filter, nothing vague |
| Backtesting before risking money | `backtester.py` — walks bar-by-bar, no lookahead |
| Risk management first | `risk_manager.py` — position sizing, stops, drawdown kill switch, daily-loss & losing-streak circuit breakers |
| Fee/slippage awareness | modeled in both `backtester.py` and `exchange_client.py` — same cost assumptions in both places |
| Error handling & monitoring | `bot.py` — every cycle is wrapped in try/except, logs to `bot.log` and console |
| Kill switch | `touch KILL` in the project directory stops the bot on its next cycle |
| Security | API key/secret only ever read from environment variables, never hardcoded; use trade-only keys |
| Adaptability without overfitting | ADX filter skips choppy/non-trending markets instead of forcing trades everywhere |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in keys only if/when you go live — not needed for paper or backtest
```

## Supported exchanges

The bot runs on [ccxt](https://github.com/ccxt/ccxt), which supports 100+ exchanges
through one interface — switching is a config change, not new code. Popular ones:

| Exchange | `EXCHANGE_ID` | Needs passphrase? |
|---|---|---|
| Binance | `binance` | no |
| Coinbase Advanced | `coinbase` | yes |
| Kraken | `kraken` | no |
| KuCoin | `kucoin` | yes |
| Bybit | `bybit` | no |
| OKX | `okx` | yes |
| Bitget | `bitget` | yes |
| Gate.io | `gateio` | no |
| MEXC | `mexc` | no |
| Bitfinex | `bitfinex` | no |
| Crypto.com | `cryptocom` | no |

Set it in `.env` (or export directly):

```bash
EXCHANGE_ID=coinbase SYMBOL=ETH/USD python bot.py
```

For a full list of every exchange ccxt supports:
```bash
python -c "import ccxt; print(ccxt.exchanges)"
```

Not every exchange supports every feature (testnet/sandbox mode, all order
types) — check `python -c "import ccxt; print(ccxt.binance().has)"` (swap in
your exchange id) if something errors on a specific exchange.

## 1. Backtest first (do this before anything else)

```bash
python backtester.py --symbol BTC/USDT --timeframe 15m --days 180
```

Pulls real historical OHLCV via ccxt, runs the strategy bar-by-bar with fees
and slippage applied, and prints win rate, total return, max drawdown, and an
approximate Sharpe ratio. Try multiple symbols, timeframes, and date ranges —
if it only looks good on one narrow window, that's overfitting, not an edge.

Long lookbacks are paginated, because every exchange caps a single kline
request (MEXC spot serves 500 candles, Kraken 720, Coinbase 300, Binance/Bybit
1000). The fetcher walks the window forward with the `since` cursor, sleeps
`max(exchange.rateLimit, 250ms)` between pages, retries transient failures with
backoff, and concatenates the slices into one ordered, de-duplicated series. So
`--days 250 --timeframe 5m` really does fetch ~72,000 candles across 144 pages
(~35s of rate-limit pacing) instead of silently backtesting a single page:

```
Fetched 72,000 candles across 250 days for BTC/USDT (5m) in 144 page(s) of up to 500 — 2023-03-09 22:13 to 2023-11-14 22:08 UTC (100% of the nominal 72,000)
```

### Why no trades? Every signal explains itself

The backtest walks the entry pipeline stage by stage — core indicator trigger,
Fibonacci filter, capital/position sizing — and tallies what happened at each
one. Whenever a core signal fires but no trade happens, the exact reason is
written to the activity log (`bot.log`, i.e. the dashboard's Live Bot Activity
Logs) and kept in the run report:

```
Timestamp: 2026-03-14 12:00 | Core signal triggered: BUY | Rejected by: Price outside the Fibonacci 0.618 zone (nearest level $46,800.00 is 0.91% away, band ±0.50%)
```

The same wording is used by the live bot (`bot.py`) for the cycle it just
evaluated. Each run ends with the summary, printed by the CLI, returned under
`results["diagnostics"]` by `/api/backtest`, and copyable from the backtester
panel's "Run report" box:

```
--- Signal pipeline diagnostics ---
Candles evaluated        : 1,448 (warmup 52 skipped, 77 in position, 9 signals blocked by risk gate)
Stage 1 core indicator   : 23 raw BUY signal(s); misses: adx_filter=3, no_ema_cross=1,345
Stage 2 Fibonacci filter : 17 passed, 6 blocked
Stage 3 capital / sizing : 8 passed, 0 blocked
Signals rejected         : fib_zone=6, risk_gate=9
Trades executed          : 8
```

Raw signals always reconcile: `executed + rejected == stage 1 triggers`, so a
run can never silently drop a signal. Rejection detail is capped
(`REJECTION_LOG_LIMIT`) so a 72,000-candle run cannot flood the log.

## 2. Paper trade

`config.py` defaults to `mode = "paper"`. Just run:

```bash
python bot.py
```

This fetches live market data and simulates fills (with the same fees/slippage
as the backtest) against a virtual balance — no real money, no API keys
required. Watch `bot.log` for at least several days across different market
conditions before considering live money.

## 3. Go live only after that (optional, higher risk)

1. On your exchange, create an API key with **trade permission only** — no withdrawal.
2. In `.env`, set `EXCHANGE_ID`, `SYMBOL`, and paste in your key/secret (and
   passphrase, if your exchange needs one — see the table above).
3. In `config.py`, set `mode = "live"`. Keep `TESTNET=true` in `.env` first if
   your exchange has a sandbox, and verify there before going fully live.
4. Start with a small `starting_equity` and conservative `risk_per_trade_pct`
   in `config.py`.
5. `python bot.py` — it prints a 10-second warning before live mode starts, so
   you have a last chance to Ctrl+C.

Stop the bot at any time, in any mode: `touch KILL` in this directory.

## How the Fibonacci filter works

No lines to draw, nothing manual. Each bar, the bot looks back
`fib_lookback` bars (50 by default), finds the highest high and lowest low
in that window, and computes the 38.2% / 50% / 61.8% retracement levels down
from that high. The anchors are a **rolling window**: they are recomputed from
each bar's own last `fib_lookback` candles, so an old swing that drops out of
the window stops pulling the levels around — nothing is pinned to the start of
the data.

A BUY signal fires when the core EMA/ADX/RSI setup is also at a pullback: price
must sit inside a **percentage band** of `fib_tolerance_pct` (1.0% by default;
tune it between roughly 1.0% and 1.5%) around one of those levels, or have
tagged one within the last `fib_touch_window` bars (3 by default; set it to 1
for the old same-bar-only behaviour). The crossover itself is likewise allowed
to stay fresh for `ema_cross_window` bars (3 by default) as long as EMA fast is
still above EMA slow — set it to 1 for the old "cross on this exact candle"
rule. Requiring both the tag *and* the crossover on one single bar is what used
to starve the strategy. A zero/negative tolerance falls back to the default
band instead of degenerating into an impossible exact price match.

Turn the filter off with the **"Enable Fibonacci Zone Filter"** checkbox in the
backtester (or `use_fib_filter = False` in `config.py` / the settings panel) to
trade the core signal alone: the check is then skipped entirely, and the backtest
results panel reports whether it ran and how many core signals it skipped. The
checkbox is sent to `POST /api/backtest` as `use_fib_filter` (the alias
`enable_fib_filter` is accepted too) and overrides `config.py` for that run.

## Reading a backtest result

Every run returns, alongside the P&L summary, the data window it was measured
on and a stage-by-stage account of the signal pipeline:

- `candles_tested` / `date_range` — how many candles were fetched and the first
  and last timestamp covered (e.g. 72,000 candles over 250 days on 5m).
- `diagnostics.report` — total candles evaluated, raw signals detected,
  rejections per filter rule (fib zone, sizing, risk gate, circuit breaker),
  and final trades executed, with the headline numbers echoed to the activity log.
- `diagnostics.rejection_log` — for each blocked signal:
  `Timestamp: 2026-03-14 12:00 | Core signal triggered: BUY | Rejected by: ...`
  so a run with no trades explains itself instead of just reporting zero.

## Historical data volume

A single ccxt `fetch_ohlcv` call only returns one batch (500 candles on MEXC
spot, up to 1,000 elsewhere), so the backtester walks the whole lookback window
in pages: `since = now - days`, advancing to the last candle's timestamp + 1ms
each round, sleeping between requests (`exchange.rateLimit / 1000`, floor 0.25s)
and retrying transient network errors. Pages are concatenated into one
continuous, de-duplicated series, and the run logs the tally, e.g.
`Fetched 72,000 candles across 250 days for BTC/USDT (5m) in 144 page(s) of up
to 500`.

## Tuning

All parameters live in `config.py` — strategy periods, risk-per-trade %,
stop/target ATR multiples, drawdown and daily-loss limits, symbol, timeframe,
fee/slippage assumptions. Change them there; nothing else needs editing.

### Risk model: sizing first, then expectancy

Size comes from the stop, not a fixed bet: each trade risks
`risk_per_trade_pct` of equity over the distance to its stop. The defaults are
set for a sub-50% win rate, where a winner has to be worth several losers:

- **Tight stop, far target** — `stop_loss_atr_mult = 1.5` with
  `take_profit_atr_mult = 4.0` (~2.7R), and no setup is taken below
  `min_reward_risk_ratio` (2.0) of risk, so a 30-45% win rate can still pay.
- **Trailing stop** — once a trade is `trailing_arm_atr_mult` (1 ATR) in
  profit the stop ratchets to break-even and then trails
  `trailing_stop_atr_mult` (1.5 ATR) behind the best price, never back below
  entry. While the trail is armed a reversal signal no longer closes the trade,
  so winners run instead of being banked for about what a loser costs.
- **Circuit breakers that recover** — the daily-loss and drawdown switches stay
  hard halts; the losing-streak breaker (`max_consecutive_losses = 8`, raised
  from 4 because a normal run of losses used to starve the strategy) instead
  starts a rolling `loss_streak_cooldown_hours` (24h) pause that lifts on its
  own — or early on the next winning trade — and publishes its resume time in
  `/api/status` and the dashboard banner.

## Honest limitations

- The strategy (EMA/RSI/ADX trend-following) is a reasonable, well-known
  starting edge — not a guaranteed-profitable one. Public, simple strategies
  tend to decay over time as more people run them.
- This uses ccxt's REST polling, not a websocket/colocated setup — fine for
  the 15m timeframe it's configured for, not suitable for arbitrage or
  market-making, which need much lower latency.
- Backtests can't perfectly replicate live order book depth/partial fills —
  treat backtest numbers as an upper bound, not a promise.
- No strategy is "safe." Only trade money you can afford to lose, and paper
  trade for real before risking live capital.
