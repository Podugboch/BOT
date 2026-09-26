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
from typing import Optional

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
MIN_PAUSE_SEC = 0.10          # rate-limit cushion between page requests
MAX_CHUNK_RETRIES = 3
BACKOFF_BASE_SEC = 1.5
MAX_STALLS = 2                # consecutive no-progress pages before giving up
MAX_DAYS = 730
MS_PER_DAY = 24 * 60 * 60 * 1000

# How far back venues actually *store* candles, per grain. This is a storage
# limit, not a request limit: MEXC keeps roughly the last 60 days of 5m, so
# asking for 365 days (105,000 candles) comes back as an empty page rather than
# an error. Requesting deeper than this is not "more data" — it is no data.
MAX_LOOKBACK_DAYS = {
    "1m": 7, "3m": 30, "5m": 60, "15m": 90, "30m": 120,
    "1h": 730, "2h": 730, "4h": 730, "6h": 730, "12h": 730, "1d": 730, "1w": 730,
}
DEFAULT_LOOKBACK_DAYS = 90     # unknown/unsupported grain: assume a 15m-like cap


def max_lookback_days(timeframe: str) -> int:
    """Deepest lookback a venue realistically serves for this grain."""
    if not timeframe:
        return DEFAULT_LOOKBACK_DAYS
    return MAX_LOOKBACK_DAYS.get(str(timeframe).strip().lower(), DEFAULT_LOOKBACK_DAYS)


def clamp_lookback_days(days, timeframe: str) -> int:
    """Clamp to the global cap and to the grain's storage limit."""
    try:
        value = int(days)
    except (TypeError, ValueError):
        value = DEFAULT_LOOKBACK_DAYS
    return max(1, min(value, MAX_DAYS, max_lookback_days(timeframe)))


def lookback_limit_note(timeframe: str) -> str:
    """Human sentence explaining what this grain can actually deliver."""
    grain = str(timeframe or "").strip().lower() or "this timeframe"
    limit = max_lookback_days(timeframe)
    if limit >= MAX_DAYS:
        return (f"{grain} can run the full {MAX_DAYS}-day window; the venue stores "
                f"years of candles at this grain.")
    return (f"{grain} candles are only stored for roughly the last {limit} days — a deeper "
            f"`since` returns an empty page, not more history. Use 1h/4h/1d for 180-365+ day runs.")


def lookback_limits() -> dict:
    """Per-grain limits as plain data, for the API and the dashboard."""
    grains = list(MAX_LOOKBACK_DAYS)
    return {
        grain: {"max_days": MAX_LOOKBACK_DAYS[grain], "note": lookback_limit_note(grain)}
        for grain in grains
    }


EXCHANGE_DEFAULT_FEES = {
    "binance": {"maker": 0.0010, "taker": 0.0010, "label": "Binance standard tier (0.10% taker, 0.075% with BNB)"},
    "mexc": {"maker": 0.0000, "taker": 0.0010, "label": "MEXC standard spot (0.00% maker / 0.10% taker)"},
    "bybit": {"maker": 0.0010, "taker": 0.0010, "label": "Bybit VIP 0 (0.10% taker)"},
    "okx": {"maker": 0.0008, "taker": 0.0010, "label": "OKX VIP 0 (0.10% taker)"},
    "gate": {"maker": 0.0009, "taker": 0.0009, "label": "Gate.io Tier 0 (0.09% taker)"},
    "kraken": {"maker": 0.0016, "taker": 0.0026, "label": "Kraken Tier 1 (0.26% taker)"},
    "coinbase": {"maker": 0.0040, "taker": 0.0060, "label": "Coinbase base tier (0.60% taker)"},
    "deribit": {"maker": 0.0000, "taker": 0.0005, "label": "Deribit standard tier (0.05% taker)"},
}


def get_exchange_fee_info(exchange_id: str, exchange_obj=None, symbol: str = None) -> dict:
    exch = str(exchange_id or "").strip().lower()
    default_meta = EXCHANGE_DEFAULT_FEES.get(exch, {"maker": 0.0010, "taker": 0.0010, "label": f"{exch} standard default (0.10% taker)"})

    # If exchange_obj is provided and authenticated, attempt to fetch live tier
    if exchange_obj and getattr(exchange_obj, "apiKey", None):
        try:
            if hasattr(exchange_obj, "fetch_trading_fee") and symbol:
                fee_data = exchange_obj.fetch_trading_fee(symbol)
                taker = float(fee_data.get("taker", default_meta["taker"]))
                maker = float(fee_data.get("maker", default_meta["maker"]))
                return {
                    "exchange": exch,
                    "symbol": symbol,
                    "maker_fee_pct": maker,
                    "taker_fee_pct": taker,
                    "source": "account_verified",
                    "is_verified": True,
                    "label": f"{exch.title()} account tier: {taker * 100:.3f}% taker, {maker * 100:.3f}% maker (API Verified)",
                }
            elif hasattr(exchange_obj, "fetch_trading_fees"):
                fees = exchange_obj.fetch_trading_fees()
                if symbol and symbol in fees:
                    fee_data = fees[symbol]
                    taker = float(fee_data.get("taker", default_meta["taker"]))
                    maker = float(fee_data.get("maker", default_meta["maker"]))
                    return {
                        "exchange": exch,
                        "symbol": symbol,
                        "maker_fee_pct": maker,
                        "taker_fee_pct": taker,
                        "source": "account_verified",
                        "is_verified": True,
                        "label": f"{exch.title()} account tier: {taker * 100:.3f}% taker, {maker * 100:.3f}% maker (API Verified)",
                    }
        except Exception as e:
            log.debug("Could not fetch authenticated trading fees: %s", e)

    return {
        "exchange": exch,
        "symbol": symbol,
        "maker": default_meta["maker"],
        "taker": default_meta["taker"],
        "maker_fee_pct": default_meta["maker"],
        "taker_fee_pct": default_meta["taker"],
        "source": "hardcoded_default",
        "is_verified": False,
        "label": f"{default_meta['label']} (Hardcoded Default · Unauthenticated)",
    }


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
        except (ccxt.NetworkError, ccxt.RateLimitExceeded) as err:
            # NetworkError and RateLimitExceeded covers RequestTimeout, DDoSProtection, RateLimitExceeded
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


def find_earliest_available(exchange, symbol: str, timeframe: str, since_ms: int, end_ms: int,
                            limit: int, pause_sec: float = MIN_PAUSE_SEC,
                            probes: int = 12) -> Optional[int]:
    """Bisect for the earliest timestamp the venue will actually serve.

    A `since` older than the venue's stored range answers with an empty page
    (MEXC does), and "no candles" there does not mean "no candles at all" —
    it means the request was too deep. Data availability is monotone in `since`
    (deep -> empty, shallow -> rows), so bisecting between the requested start
    and now lands on the real boundary in ~12 requests instead of guessing.

    Returns the timestamp of the first candle the venue returns, or None when
    even the newest part of the window is empty.
    """
    try:
        ms_per_bar = max(1, int(exchange.parse_timeframe(timeframe)) * 1000)
    except Exception:
        ms_per_bar = MS_PER_DAY // 24

    # Sanity gate: if the venue serves nothing near the present, there is no
    # boundary to find, and bisecting would just burn requests.
    newest_probe = max(since_ms, end_ms - limit * ms_per_bar)
    if not fetch_page(exchange, symbol, timeframe, newest_probe, limit):
        return None

    lo, hi, earliest = int(since_ms), int(end_ms), None
    for _ in range(max(1, probes)):
        if lo > hi:
            break
        mid = lo + (hi - lo) // 2
        batch = fetch_page(exchange, symbol, timeframe, mid, limit)
        time.sleep(pause_sec)
        if batch:
            earliest = int(batch[0][0])       # authoritative: the row the venue returned
            hi = mid - ms_per_bar             # boundary is at or before this
        else:
            lo = mid + ms_per_bar             # too deep; search shallower
    return earliest


def fetch_ohlcv_window(exchange, symbol: str, timeframe: str, since_ms: int, end_ms: int,
                       limit: int, pause_sec: float, max_chunks: int) -> tuple:
    """Paginate [since_ms, end_ms]; if the venue serves nothing that deep, fall
    back to the earliest window it does serve.

    Returns (rows, pages, largest_page, effective_since_ms). `effective_since_ms`
    is None only when the venue returned nothing at all for this grain.
    """
    rows, pages, largest_page = paginate_ohlcv(
        exchange, symbol, timeframe, since_ms, end_ms, limit, pause_sec, max_chunks)
    if rows:
        return rows, pages, largest_page, int(since_ms)

    log.warning("No %s candles returned from a start of %s — that is deeper than the venue "
                "stores, not an outage; locating the earliest available window",
                timeframe, _iso_ms(since_ms))
    earliest = find_earliest_available(exchange, symbol, timeframe, since_ms, end_ms, limit, pause_sec)
    if earliest is None:
        log.warning("Venue serves no %s candles for %s at all (checked up to %s)",
                    timeframe, symbol, _iso_ms(end_ms))
        return [], pages, largest_page, None

    log.warning("Falling back to the earliest available %s candle: %s (requested start %s)",
                timeframe, _iso_ms(earliest), _iso_ms(since_ms))
    rows, extra_pages, extra_largest = paginate_ohlcv(
        exchange, symbol, timeframe, earliest, end_ms, limit, pause_sec, max_chunks)
    return rows, pages + extra_pages, max(largest_page, extra_largest), int(earliest)


def _iso_ms(ms: int) -> str:
    """Millisecond timestamp -> 'YYYY-MM-DD HH:MM UTC' for logs."""
    try:
        return pd.to_datetime(int(ms), unit="ms").strftime("%Y-%m-%d %H:%M UTC")
    except (TypeError, ValueError):
        return str(ms)


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

        # Pre-load markets for precision and order lot formatting
        if self.mode == "live" and cfg.api_key:
            try:
                self.exchange.load_markets()
            except Exception as e:
                log.debug("Could not pre-load markets during exchange init for %s: %s", cfg.exchange_id, e)

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
        trip a rate limit or return short pages. When the requested window is
        deeper than the venue stores, the start is pulled back to the grain's
        known limit and, failing that, to the earliest candle actually served.
        """
        requested_days = max(1, min(int(days), MAX_DAYS))
        timeframe = timeframe or self.cfg.timeframe
        days = clamp_lookback_days(requested_days, timeframe)
        if days != requested_days:
            log.warning("Lookback %s day(s) exceeds what %s stores for %s — requesting %s day(s) "
                        "instead. %s", requested_days, self.cfg.exchange_id, timeframe, days,
                        lookback_limit_note(timeframe))
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
        rows, pages, largest_page, effective_since = fetch_ohlcv_window(
            self.exchange, self.cfg.symbol, timeframe, since_ms, end_ms, limit, pause, max_chunks)
        df = to_frame(rows)
        served_days = (end_ms - effective_since) / MS_PER_DAY if effective_since else 0
        log.info("Fetched %s candles across %s day(s) for %s (%s) in %d page(s) of up to %d",
                 f"{len(df):,}", f"{served_days:.0f}", self.cfg.symbol, timeframe, pages, largest_page)
        if df.empty:
            log.error("No %s candles available for %s from this venue. %s",
                      timeframe, self.cfg.symbol, lookback_limit_note(timeframe))
        elif effective_since and effective_since > since_ms:
            log.warning("Only the last %s day(s) of %s were served (requested %s) — the backtest "
                        "will cover a shorter window than asked", f"{served_days:.0f}",
                        timeframe, days)
        return df

    def current_price(self) -> float:
        ticker = self.exchange.fetch_ticker(self.cfg.symbol)
        price = ticker.get("last") or ticker.get("close") or ticker.get("bid") or 0.0
        return float(price)

    def amount_to_precision(self, size: float, symbol: str = None) -> float:
        """Format an order size according to the exchange's precision and lot sizing rules.
        Uses CCXT amount_to_precision when markets are loaded, with graceful fallback.
        """
        sym = symbol or self.cfg.symbol
        try:
            if hasattr(self.exchange, "amount_to_precision") and getattr(self.exchange, "markets", None) and sym in self.exchange.markets:
                formatted = self.exchange.amount_to_precision(sym, size)
                return float(formatted)
        except Exception as e:
            log.debug("amount_to_precision fallback for %s: %s", sym, e)
        if 0 < size < 0.000001:
            return float(round(size, 8))
        return float(round(size, 6))

    def price_to_precision(self, price: float, symbol: str = None) -> float:
        """Format an order price according to the exchange's price precision rules.
        Uses CCXT price_to_precision when markets are loaded, with graceful fallback.
        """
        sym = symbol or self.cfg.symbol
        try:
            if hasattr(self.exchange, "price_to_precision") and getattr(self.exchange, "markets", None) and sym in self.exchange.markets:
                formatted = self.exchange.price_to_precision(sym, price)
                return float(formatted)
        except Exception as e:
            log.debug("price_to_precision fallback for %s: %s", sym, e)
        if 0 < price < 1.0:
            if price < 0.0001:
                return float(round(price, 8))
            elif price < 0.01:
                return float(round(price, 6))
            else:
                return float(round(price, 4))
        return float(round(price, 2))

    def fetch_wallet_balance(self, symbol: str = None) -> dict:
        """Fetch available and total wallet balances for the trading pair's base and quote assets.
        In paper mode or unauthenticated mode, mirrors the paper ledger.
        """
        sym = symbol or self.cfg.symbol
        parts = sym.split("/") if "/" in sym else [sym, "USDT"]
        base_curr = parts[0]
        quote_curr = parts[1] if len(parts) > 1 else "USDT"

        if self.mode == "live" and getattr(self.cfg, "api_key", None):
            try:
                balance = self.exchange.fetch_balance()
                free_dict = balance.get("free") or {}
                total_dict = balance.get("total") or {}

                free_quote = float(free_dict.get(quote_curr, 0.0) or (balance.get(quote_curr, {}) or {}).get("free", 0.0) or 0.0)
                free_base = float(free_dict.get(base_curr, 0.0) or (balance.get(base_curr, {}) or {}).get("free", 0.0) or 0.0)
                total_quote = float(total_dict.get(quote_curr, 0.0) or (balance.get(quote_curr, {}) or {}).get("total", 0.0) or 0.0)
                total_base = float(total_dict.get(base_curr, 0.0) or (balance.get(base_curr, {}) or {}).get("total", 0.0) or 0.0)

                return {
                    "mode": "live",
                    "base_currency": base_curr,
                    "quote_currency": quote_curr,
                    "free_quote": free_quote,
                    "free_base": free_base,
                    "total_quote": total_quote,
                    "total_base": total_base,
                    "is_live": True,
                }
            except Exception as e:
                log.warning("Could not fetch live wallet balance: %s", e)

        # Fallback / paper trading mode
        return {
            "mode": "paper",
            "base_currency": base_curr,
            "quote_currency": quote_curr,
            "free_quote": float(self.paper_cash),
            "free_base": float(self.paper_position),
            "total_quote": float(self.paper_cash),
            "total_base": float(self.paper_position),
            "is_live": False,
        }

    def fetch_order_book_spread(self, symbol: str = None) -> dict:
        """Fetch top-of-book bid/ask spread and calculate percentage spread.
        Guards against high slippage and illiquid trading pairs.
        """
        sym = symbol or self.cfg.symbol
        try:
            ob = self.exchange.fetch_order_book(sym, limit=5)
            bids = ob.get("bids") or []
            asks = ob.get("asks") or []
            if bids and asks:
                bid = float(bids[0][0])
                ask = float(asks[0][0])
                spread = max(0.0, ask - bid)
                mid = (bid + ask) / 2.0
                spread_pct = (spread / ask * 100.0) if ask > 0 else 0.0
                return {
                    "symbol": sym,
                    "bid": bid,
                    "ask": ask,
                    "spread": spread,
                    "spread_pct": spread_pct,
                    "mid": mid,
                    "valid": True,
                }
        except Exception as e:
            log.debug("fetch_order_book_spread failed for %s: %s", sym, e)

        # Fallback to current ticker price if order book unavailable
        try:
            curr = self.current_price()
        except Exception:
            curr = 0.0
        return {
            "symbol": sym,
            "bid": curr,
            "ask": curr,
            "spread": 0.0,
            "spread_pct": 0.0,
            "mid": curr,
            "valid": False,
        }

    def load_market_limits(self, symbol: str = None) -> dict:
        sym = symbol or self.cfg.symbol
        try:
            self.exchange.load_markets()
            market = self.exchange.market(sym)
            limits = market.get("limits") or {}
            min_cost = float((limits.get("cost") or {}).get("min") or 5.0)
            min_amount = float((limits.get("amount") or {}).get("min") or 0.00001)
            return {
                "symbol": sym,
                "min_notional": min_cost,
                "min_amount": min_amount,
                "precision_amount": (market.get("precision") or {}).get("amount"),
                "precision_price": (market.get("precision") or {}).get("price"),
            }
        except Exception as e:
            log.warning("Could not load market limits for %s: %s", sym, e)
            return {"symbol": sym, "min_notional": 5.0, "min_amount": 0.00001}

    def place_order(self, side: str, size: float, ref_price: float) -> dict:
        if self.mode == "live":
            return self._place_live_order(side, size)
        return self._simulate_fill(side, size, ref_price)

    def _notify_alert(self, event_type: str, title: str, message: str, level: str = "warning", details: dict = None):
        try:
            from alerts import get_alert_dispatcher
            get_alert_dispatcher().send_alert(event_type, title, message, level=level, details=details)
        except Exception:
            pass

    # -- live --------------------------------------------------------------
    def _place_live_order(self, side: str, size: float) -> dict:
        sym = self.cfg.symbol
        norm_size = self.amount_to_precision(size, sym)
        if norm_size <= 0:
            raise ValueError(f"Normalized order size {norm_size} is <= 0 for {sym} (original size: {size})")

        try:
            order = self.exchange.create_order(sym, "market", side.lower(), norm_size)
            fill_price = order.get("average") or order.get("price") or self.current_price()
            fill_price = self.price_to_precision(fill_price, sym)
            fee = 0.0
            if order.get("fees"):
                fee = sum(float(f.get("cost", 0) or 0) for f in order["fees"] if isinstance(f, dict))
            elif isinstance(order.get("fee"), dict):
                fee = float(order["fee"].get("cost", 0) or 0)
            log.info("LIVE %s order placed: id=%s price=%.2f size=%.6f", side, order.get("id"), fill_price, norm_size)
            return {"side": side, "size": norm_size, "price": fill_price, "fee": fee, "mode": "live", "id": order.get("id")}
        except ccxt.InsufficientFunds as e:
            log.error("Live order rejected - Insufficient Funds on %s: %s", self.cfg.exchange_id, e)
            self._notify_alert("INSUFFICIENT_FUNDS", f"Insufficient Funds ({sym})", str(e), level="critical",
                               details={"symbol": sym, "side": side, "size": norm_size, "error": str(e)})
            raise
        except ccxt.InvalidOrder as e:
            log.error("Live order rejected - Invalid Order on %s: %s", self.cfg.exchange_id, e)
            self._notify_alert("INVALID_ORDER", f"Invalid Order Rejected ({sym})", str(e), level="critical",
                               details={"symbol": sym, "side": side, "size": norm_size, "error": str(e)})
            raise
        except ccxt.RateLimitExceeded as e:
            log.error("Live order rejected - Rate Limit Exceeded on %s: %s", self.cfg.exchange_id, e)
            self._notify_alert("RATE_LIMIT", f"Rate Limit Exceeded ({self.cfg.exchange_id})", str(e), level="critical",
                               details={"symbol": sym, "side": side, "size": norm_size, "error": str(e)})
            raise
        except ccxt.AuthenticationError as e:
            log.error("Live order rejected - Authentication Error on %s: %s", self.cfg.exchange_id, e)
            self._notify_alert("AUTH_ERROR", f"Exchange Auth Failure ({self.cfg.exchange_id})", str(e), level="critical",
                               details={"symbol": sym, "side": side, "size": norm_size, "error": str(e)})
            raise
        except Exception as e:
            log.exception("Live order execution failed: %s", e)
            self._notify_alert("EXCHANGE_ERROR", f"Live Order Execution Failed ({sym})",
                               f"Order failed on {self.cfg.exchange_id}: {e}", level="critical",
                               details={"symbol": sym, "side": side, "size": norm_size, "error": str(e)})
            raise

    # -- paper ---------------------------------------------------------------
    def _simulate_fill(self, side: str, size: float, ref_price: float) -> dict:
        """Fill against ref_price with the same slippage direction a real market
        order would suffer (worse price, never better), respecting precision."""
        sym = self.cfg.symbol
        norm_size = self.amount_to_precision(size, sym)
        slip = ref_price * self.slippage_pct * (1 if side == "BUY" else -1)
        raw_fill_price = ref_price + slip
        fill_price = self.price_to_precision(raw_fill_price, sym)
        fee = fill_price * norm_size * self.fee_pct

        if side == "BUY":
            self.paper_cash -= fill_price * norm_size + fee
            self.paper_position += norm_size
        else:
            self.paper_cash += fill_price * norm_size - fee
            self.paper_position -= norm_size

        log.info("PAPER %s %.6f @ %.2f (fee %.4f)", side, norm_size, fill_price, fee)
        return {"side": side, "size": norm_size, "price": fill_price, "fee": fee, "mode": "paper"}
