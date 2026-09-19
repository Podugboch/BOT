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
from that high. A BUY signal only fires if price is also within
`fib_tolerance_pct` (0.5% by default) of one of those levels — so the EMA
crossover has to line up with a real pullback into a support zone, not just
any point in the trend. Set `use_fib_filter = False` in `config.py` to turn
it off and compare.

## Tuning

All parameters live in `config.py` — strategy periods, risk-per-trade %,
stop/target ATR multiples, drawdown and daily-loss limits, symbol, timeframe,
fee/slippage assumptions. Change them there; nothing else needs editing.

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
