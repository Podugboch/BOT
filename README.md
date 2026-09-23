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
Fibonacci filter, entry debounce, capital/position sizing — and tallies what
happened at each one. Whenever a core signal fires but no trade happens, the
exact reason is written to the activity log (`bot.log`, i.e. the dashboard's
Live Bot Activity Logs) and kept in the run report:

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

### When the exchange refuses the request

Failures are reported inside the dashboard, not by a browser `alert()`. A
blocked region, for instance, says exactly this — a title and two short
sentences, nothing more:

```
Binance is not available in this region
This server sits in a restricted region, so Binance refuses the request.
Try Kraken, Bybit, KuCoin or OKX instead.
```

The raw exchange text (HTTP status, ccxt message, JSON body) stays behind a
**Technical detail** toggle on the same card, so the reason is always one tap
away instead of shoved in your face. Rate limits, unsupported pairs,
unreachable venues and a too-deep lookback each get their own one-sentence
explanation.

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
must sit inside the **band** around one of the selected levels, or have tagged
one within the last `fib_touch_window` bars (3 by default; set it to 1 for the
old same-bar-only behaviour). The crossover itself is likewise allowed
to stay fresh for `ema_cross_window` bars (3 by default) as long as EMA fast is
still above EMA slow — set it to 1 for the old "cross on this exact candle"
rule. Requiring both the tag *and* the crossover on one single bar is what used
to starve the strategy.

Two knobs shape that band, both live in the backtester panel and in `config.py`:

- **Dynamic band (`fib_tolerance_atr_mult`, 0.5 by default).** Instead of a
  rigid percentage the tolerance scales with volatility: a bar within
  `0.5 × ATR` of a level counts as a pullback, so a 1h candle with a 1.5% ATR
  is not judged by a calm-market yardstick. `fib_tolerance_pct` stays as the
  **floor** (0.3% by default, was 1%), and as the fallback when ATR is missing
  or zero, so a quiet tape can never shrink the band toward an impossible
  exact-price match. Set the multiplier to 0 for the old fixed-band behaviour.

  The floor is the *smaller* of the two levers: with a 0.5 × ATR band the floor
  rarely binds, because an ATR band is usually wider than 0.3%. Lowering it
  from 1% therefore changes little on its own — what actually makes the filter
  bite is which levels count (below).
- **Which levels count (`fib_level_mode`).** This is the knob that decides
  whether the filter filters at all.
  - `deep` (the default) — price must come within the band of the **0.5 or
    0.618** level, the retracements a pullback actually turns on. Measured over
    a 1h window this turned away 3 of 8 core signals and cut 6 trades to 5,
    which is a filter doing its job.
  - `zone` — anywhere inside the **0.382-0.618 pocket**. Measured against real
    signals this accepted ~96% of them (`Fib blocked: 0`): the pocket already
    spans much of the swing range, and the band on top of it covers nearly the
    rest, so it is a filter in name only. Kept for comparison — see the sweep.
  - `all` — within the band of any of 0.382/0.5/0.618 (loosest, it also accepts
    the shallow 0.382 level).

  Switchable in the backtester's **Fib Depth** select and via
  `fib_level_mode` — see the sweep below.

A zero/negative tolerance falls back to the default band instead of degenerating
into an exact price match.

Turn the filter off with the **"Enable Fibonacci Zone Filter"** checkbox in the
backtester (or `use_fib_filter = False` in `config.py` / the settings panel) to
trade the core signal alone: the check is then skipped entirely, and the backtest
results panel reports whether it ran and how many core signals it skipped. The
checkbox is sent to `POST /api/backtest` as `use_fib_filter` (the alias
`enable_fib_filter` is accepted too) and overrides `config.py` for that run.

### Entry debounce: never re-enter on adjacent candles

A fast moving-average whipsaw used to chain entries on consecutive candles — a
BUY at 17:30, another at 17:35, another at 17:40 — which is one setup billed
three times over in fees and stop-outs. The strategy therefore enforces a
cool-off measured from the **exit** of the previous trade:

- **`min_bars_between_trades`** (strategy config / `min_bars_between_trades` in
  the backtest request) is the number of candles that must pass after an exit
  before a new BUY is allowed. Left unset — the default — it derives from the
  timeframe, because three bars is 15 minutes on 5m but three hours on 1h:
  **8 bars on 1m/5m/15m, 3 bars on 30m and slower**. Set an int to pin it, `0`
  to turn it off.
- A blocked setup is reported, never silently dropped: the core signal still
  counts as triggered, and the rejection reads `Signal debounce: waiting for
  cooldown period between setups` in the run's rejection log and in the live
  bot's cycle log. The run also echoes `min_bars_between_trades` (the resolved
  count), `debounce_blocked` (re-triggers held back) and `min_entry_gap_bars`
  (the tightest spacing actually taken), so the rule is checkable from the
  result instead of taken on faith.
- It gates **entries** only. Exit management — stop, target, trailing stop and
  the reversal signal — is never debounced, and the first trade after a restart
  or an empty history is never blocked.

## Reading a backtest result

Every run returns, alongside the P&L summary, the data window it was measured
on and a stage-by-stage account of the signal pipeline:

- `candles_tested` / `date_range` — how many candles were fetched and the first
  and last timestamp covered (e.g. 72,000 candles over 250 days on 5m).
- `diagnostics.report` — total candles evaluated, raw signals detected,
  rejections per filter rule (fib zone, entry debounce, sizing, risk gate,
  circuit breaker), and final trades executed, with the headline numbers echoed
  to the activity log.
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

### Granularity limits: how far back each timeframe really goes

Exchanges do not store every grain equally deep, and asking for more than they
keep is **not** an error — MEXC answers a `since` older than its stored range
with an **empty page**, so a 365-day 5m request (≈105,000 candles) comes back
with nothing at all rather than a short series. `exchange_client.py` therefore
carries a per-grain table of what venues actually serve:

| Timeframe | Realistic lookback | Notes |
|-----------|--------------------|-------|
| `1m` | ~7 days | shortest retention |
| `5m` | ~60 days | the classic trap: 365d returns an empty list |
| `15m` | ~90 days | |
| `30m` | ~120 days | |
| `1h` / `4h` / `1d` | up to 730 days | years of candles; use these for long runs |

Three things follow from that:

1. **Deep starts fall back instead of failing.** If a window returns nothing at
   all, the start is pulled back to the grain's known limit, and failing that the
   earliest candle the venue actually serves is *found by bisection*
   (`find_earliest_available`, ~12 requests) rather than guessed. The run then
   reports the window it really covered, e.g.
   `Venue stores no 5m candles before 2026-07-25 04:00 UTC — covered 60 of the
   365 day(s) requested.`
2. **An empty venue is explained, not silent.** When nothing can be fetched the
   log says why and what to do (`lookback_limit_note`), and `POST /api/backtest`
   returns that sentence in its 400 `detail` instead of a bare
   "insufficient bars".
3. **The dashboard warns and caps.** The backtester panel injects the same
   limits table, so choosing `5m` bounds Lookback (Days) to 60 and shows
   `5m on this venue stores ~60 days; the run will be capped to 60.` Clicking a
   quick-pick button (90d/365d/730d) on a short-retention grain lands on that
   grain's limit, and the request is capped before it is posted rather than sent
   off to return nothing.

## Tuning

All parameters live in `config.py` — strategy periods, risk-per-trade %,
stop/target ATR multiples, drawdown and daily-loss limits, symbol, timeframe,
fee/slippage assumptions. Change them there; nothing else needs editing.

### Tuning the 1h edge

The backtester's **Lookback (Days)** control goes to 730 (with 90/365/730
one-click presets), so a 1h run can span ~8,700–17,500 candles — full bull,
bear and consolidation regimes rather than a single friendly stretch. On 1h the
strategy-relevant axes are the Fibonacci band/depth and the trailing stop, so
`optimize_1h.py` sweeps them over one window and prints a comparison table:

```bash
python optimize_1h.py --source exchange --symbol BTC/USDT --days 730   # real candles
python optimize_1h.py --source synthetic --candles 17520               # offline harness check
```

It reports trades, win rate, realized R:R, return, max drawdown and the fib
rejection count per configuration, plus the exit mix (`stop`/`target`/`trail`/
`signal`). The two axes are kept separable — the fib knobs only change the
entry, the trailing knobs only change the exit — and `backtester.py
--fib-atr-mult 0` restores the fixed percentage band as the control. The same
single run can be driven from that CLI instead: `--fib-mode zone|all|deep`
selects the entry rule, `--no-fib` drops the filter, and `--trail-atr` /
`--trail-arm-atr` set the exit (or `--no-trail` disables it). The knobs are also
exposed in the backtester panel, and every run echoes them back in its JSON
(`fib_level_mode`, `fib_tolerance_atr_mult`, `trailing_stop_enabled`,
`trailing_stop_atr_mult`), so a result is self-describing.

The synthetic source exists so the harness can be exercised without exchange
access; it is a regime-switching stand-in, **not** a market replica. Any number
that matters has to come from the `--source exchange` run against real candles.

### Risk model: sizing first, then expectancy

Size comes from the stop, not a fixed bet: each trade risks
`risk_per_trade_pct` of equity over the distance to its stop. The defaults are
set for a sub-50% win rate, where a winner has to be worth several losers:

- **Tight stop, far target** — `stop_loss_atr_mult = 1.5` with
  `take_profit_atr_mult = 4.0` (~2.7R). The target is floored at
  `MIN_TAKE_PROFIT_ATR_MULT` (2.5 ATR) and at `min_reward_risk_ratio` (2.0) of
  risk, so no combination of settings can shrink the primary target into a
  micro-gain.
- **Trailing stop: arm late, trail wide** — the trail does nothing until the
  trade is `trailing_arm_atr_mult` (1.5 ATR) in profit; before that the original
  ATR stop stands, so an ordinary pullback on the way up is just noise. Once
  armed it moves to break-even and then trails `trailing_stop_atr_mult` (3.5
  ATR) behind the **highest high since entry**, never back below entry. With a
  3.5 ATR trail and a 4 ATR target the stop is still at break-even at +3 ATR, so
  the target stays reachable instead of being clipped into a micro-gain. While
  the trail is armed a reversal signal no longer closes the trade. It is
  optional — `trailing_stop_enabled = False` (or the backtester checkbox) falls
  back to the fixed ATR stop with reversal-signal exits.
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
