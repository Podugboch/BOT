"""
Thin wrapper around ccxt (https://github.com/ccxt/ccxt), which supports 100+
exchanges through one API. Handles both real order placement (live) and a
simulated fill engine (paper) that charges the same fees/slippage a real
order would, so paper results are a fair preview of live performance.

It also owns historical candle retrieval: a single `fetch_ohlcv` call only
returns one batch (500-1000 candles depending on the venue), so looking back
over dozens or hundreds of days means paginating. `paginate_ohlcv` walks the
requested window in batches, sleeping between calls to respect the exchange
rate limit, and returns one continuous, de-duplicated series.
"""
import logging
import time

import ccxt
import pandas as pd

from config import ExchangeConfig

log = logging.getLogger("exchange")

OHLCV_COLUMNS = ["ts", "open", "high", "low", "close", "volume"]

# Per-request candle caps: MEXC spot 500, Kraken 720, Coinbase 300,
# Binance/Bybit 1000, KuCoin 1500. Ask for what the venue advertises, clamped
# to something sane, and never treat a short page as the end of history.
DEFAULT_CHUNK_LIMIT = 500
MAX_CHUNK_LIMIT = 1000
MIN_CHUNK_LIMIT = 50
MIN_PAUSE_SEC = 0.25          # rate-limit cushion between page requests
MAX_CHUNK_RETRIES = 3
BACKOFF_BASE_SEC = 1.5
MAX_STALLS = 2                # consecutive no-progress pages before giving up
MAX_DAYS = 730
MS_PER_DAY = 24 * 60 * 60 * 1000


def chunk_limit(exchange, symbol: str) -> int:
    """Candles to request per page, per what the exchange advertises."""
    advertised = 0
    try:
        limits = (exchange.market(symbol).get("limits") or {}).get("ohlcv") or {}
        advertised = int(limits.get("max") or 0)
    except (ccxt.BaseError, AttributeError, KeyError, TypeError, ValueError):
        advertised = 0
    if advertised <= 0:
        try:
            spot = (exchange.features or {}).get("spot") or {}
            advertised = int((spot.get("fetchOHLCV") or {}).get("limit") or 0)
        except (AttributeError, TypeError, ValueError):
            advertised = 0
    if advertised <= 0:
        advertised = DEFAULT_CHUNK_LIMIT
    return max(MIN_CHUNK_LIMIT, min(advertised, MAX_CHUNK_LIMIT))


def fetch_page(exchange, symbol: str, timeframe: str, since_ms: int, limit: int) -> list:
    """One kline page, with backoff retries on transient network/rate limits."""
    for attempt in range(1, MAX_CHUNK_RETRIES + 2):
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
            return list(batch) if batch else []
        except ccxt.NetworkError as err:
            # NetworkError covers RequestTimeout, DDoSProtection, RateLimitExceeded
            # and ExchangeNotAvailable — all of which are worth retrying.
            if attempt > MAX_CHUNK_RETRIES:
                raise
            delay = BACKOFF_BASE_SEC ** attempt
            log.warning("OHLCV page request failed at %s (%s: %s) — retry %d/%d in %.1fs",
                        since_ms, type(err).__name__, err, attempt, MAX_CHUNK_RETRIES, delay)
            time.sleep(delay)
    return []


def paginate_ohlcv(exchange, symbol: str, timeframe: str, since_ms: int, end_ms: int,
                   limit: int, pause_sec: float, max_chunks: int) -> tuple:
    """Walk [since_ms, end_ms] page by page and concatenate every slice.

    The cursor advances to the last candle's timestamp + 1ms. Venues that snap
    `since` down to the containing candle (Binance does) would then re-serve
    that same candle forever, so a stalled page falls back to stepping one full
    bar forward — that keeps the walk honest instead of truncating the series.
    """
    ms_per_bar = int(exchange.parse_timeframe(timeframe)) * 1000
    rows: list = []
    cursor = since_ms
    pages = 0
    largest_page = 0
    stalls = 0
    last_ts = None

    for page in range(max_chunks):
        if page:
            time.sleep(pause_sec)          # stay under the venue's rate limit
        batch = fetch_page(exchange, symbol, timeframe, cursor, limit)
        if not batch:
            # A transient empty payload is common when throttled: give it one
            # more chance before calling it the end of the available history.
            time.sleep(pause_sec)
            batch = fetch_page(exchange, symbol, timeframe, cursor, limit)
            if not batch:
                break

        pages += 1
        rows.extend(batch)
        largest_page = max(largest_page, len(batch))

        newest = int(batch[-1][0])
        next_cursor = newest + 1
        if last_ts is not None and newest <= last_ts:
            # No forward progress: step a whole bar past the data we already
            # have, otherwise the same page would repeat indefinitely.
            stalls += 1
            if stalls > MAX_STALLS:
                log.warning("OHLCV pagination made no progress at %s (last candle %s) — stopping",
                            cursor, newest)
                break
            next_cursor = last_ts + ms_per_bar
        else:
            stalls = 0
            last_ts = newest

        if next_cursor <= cursor:
            log.warning("OHLCV pagination cannot advance past %s — stopping", cursor)
            break
        cursor = next_cursor
        if cursor >= end_ms:
            break
    else:
        log.warning("OHLCV pagination stopped at the %d-page safety cap — series may be truncated",
                    max_chunks)

    return rows, pages, largest_page


def to_frame(rows: list) -> pd.DataFrame:
    """Normalise raw ccxt candles into one ordered, de-duplicated DataFrame."""
    df = pd.DataFrame(rows, columns=OHLCV_COLUMNS)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    if df.empty:
        return df
    return df.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)


class ExchangeClient:
    def __init__(self, cfg: ExchangeConfig, mode: str, fee_pct: float,
                 slippage_pct: float, starting_equity: float):
        self.cfg = cfg
        self.mode = mode
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct

        if cfg.exchange_id.lower() not in ccxt.exchanges:
            raise ValueError(
                f"'{cfg.exchange_id}' is not a known ccxt exchange id. "
                f"Run `python -c \"import ccxt; print(ccxt.exchanges)\"` to see all supported ids."
            )

        exchange_class = getattr(ccxt, cfg.exchange_id.lower())
        exchange_opts = {
            "apiKey": cfg.api_key,
            "secret": cfg.api_secret,
            "enableRateLimit": True,
            "timeout": 15000,
            "options": {
                "defaultType": "spot",
                "adjustForTimeDifference": True,
                "recvWindow": 60000,
            },
        }
        if cfg.api_passphrase:
            exchange_opts["password"] = cfg.api_passphrase  # ccxt's field name for passphrase
        self.exchange = exchange_class(exchange_opts)
        if cfg.testnet and hasattr(self.exchange, "set_sandbox_mode"):
            try:
                self.exchange.set_sandbox_mode(True)
            except Exception:
                pass

        # Time difference synchronization is only required for live order signing
        if self.mode == "live" and cfg.api_key and self.exchange.has.get("fetchTime") and hasattr(self.exchange, "load_time_difference"):
            try:
                self.exchange.load_time_difference()
            except Exception as e:
                log.warning(f"Could not load time difference for {cfg.exchange_id}: {e}")

        # paper trading book-keeping
        self.paper_cash = starting_equity
        self.paper_position = 0.0

    def fetch_ohlcv(self, limit: int = 200) -> pd.DataFrame:
        """Most recent `limit` candles — one request, used by the live signal path."""
        raw = self.exchange.fetch_ohlcv(self.cfg.symbol, timeframe=self.cfg.timeframe, limit=limit)
        return to_frame(raw or [])

    def fetch_historical(self, days: int, timeframe: str = None, pause_sec: float = None,
                         max_chunks: int = None) -> pd.DataFrame:
        """Paginated candles covering the last `days` — used for backtests.

        `since` starts at now - days and the walk continues in batches up to the
        current timestamp, sleeping between requests so a long lookback cannot
        trip a rate limit or return short pages.
        """
        days = max(1, min(int(days), MAX_DAYS))
        timeframe = timeframe or self.cfg.timeframe
        ms_per_bar = int(self.exchange.parse_timeframe(timeframe)) * 1000

        self.exchange.load_markets()
        limit = chunk_limit(self.exchange, self.cfg.symbol)
        end_ms = self.exchange.milliseconds()
        since_ms = end_ms - days * MS_PER_DAY
        expected = int(days * MS_PER_DAY // ms_per_bar)
        pause = float(pause_sec) if pause_sec is not None else max(
            getattr(self.exchange, "rateLimit", 0) / 1000.0, MIN_PAUSE_SEC)
        if max_chunks is None:
            max_chunks = max(50, int(expected / limit) * 3 + 25)

        log.info("Fetching %s %s from %s: %s day(s) (~%s candles) in pages of %s, %.2fs pause",
                 self.cfg.symbol, timeframe, self.cfg.exchange_id, days,
                 f"{expected:,}", limit, pause)
        rows, pages, largest_page = paginate_ohlcv(
            self.exchange, self.cfg.symbol, timeframe, since_ms, end_ms, limit, pause, max_chunks)
        df = to_frame(rows)
        log.info("Fetched %s candles across %s days for %s (%s) in %d page(s) of up to %d",
                 f"{len(df):,}", days, self.cfg.symbol, timeframe, pages, largest_page)
        return df

    def current_price(self) -> float:
        ticker = self.exchange.fetch_ticker(self.cfg.symbol)
        price = ticker.get("last") or ticker.get("close") or ticker.get("bid") or 0.0
        return float(price)

    def place_order(self, side: str, size: float, ref_price: float) -> dict:
        if self.mode == "live":
            return self._place_live_order(side, size)
        return self._simulate_fill(side, size, ref_price)

    # -- live --------------------------------------------------------------
    def _place_live_order(self, side: str, size: float) -> dict:
        try:
            order = self.exchange.create_order(self.cfg.symbol, "market", side.lower(), size)
            fill_price = order.get("average") or order.get("price") or self.current_price()
            fee = 0.0
            if order.get("fees"):
                fee = sum(float(f.get("cost", 0) or 0) for f in order["fees"] if isinstance(f, dict))
            elif isinstance(order.get("fee"), dict):
                fee = float(order["fee"].get("cost", 0) or 0)
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
