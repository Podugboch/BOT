"""
Persistent Execution Tracker for Live and Paper Trading.

Maintains an immutable record of all order actions (fills, rejections, cancellations)
and tracks open and historical positions with real-time unrealized P&L and trailing stops.
Operates completely independently from backtesting to catch live/paper execution drift.
"""
import os
import sqlite3
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

log = logging.getLogger("execution")

DB_PATH = (
    "/tmp/execution.db"
    if (os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))
    else os.path.join(os.path.dirname(os.path.abspath(__file__)), "execution.db")
)


class ExecutionTracker:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        try:
            with self._get_conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS orders (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        order_id TEXT,
                        created_at TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        side TEXT NOT NULL,
                        order_type TEXT NOT NULL,
                        size REAL NOT NULL,
                        price REAL NOT NULL,
                        fee REAL NOT NULL DEFAULT 0.0,
                        mode TEXT NOT NULL,
                        status TEXT NOT NULL,
                        rejection_reason TEXT
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS positions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        position_id TEXT UNIQUE,
                        opened_at TEXT NOT NULL,
                        closed_at TEXT,
                        status TEXT NOT NULL DEFAULT 'OPEN',
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        side TEXT NOT NULL DEFAULT 'BUY',
                        size REAL NOT NULL,
                        entry_price REAL NOT NULL,
                        current_price REAL NOT NULL,
                        current_stop REAL NOT NULL,
                        initial_stop REAL NOT NULL,
                        trailing_stop REAL,
                        target REAL NOT NULL,
                        peak_price REAL,
                        unrealized_pnl REAL DEFAULT 0.0,
                        unrealized_pnl_pct REAL DEFAULT 0.0,
                        realized_pnl REAL,
                        realized_pnl_pct REAL,
                        fees_paid REAL DEFAULT 0.0,
                        exit_price REAL,
                        exit_reason TEXT,
                        mode TEXT NOT NULL
                    )
                """)
                conn.commit()
        except Exception as e:
            log.warning("Could not initialize execution database at %s: %s", self.db_path, e)

    def record_order(
        self,
        symbol: str,
        timeframe: str,
        side: str,
        size: float,
        price: float,
        fee: float = 0.0,
        mode: str = "paper",
        status: str = "FILLED",
        rejection_reason: Optional[str] = None,
        order_id: Optional[str] = None,
        order_type: str = "market",
    ) -> int:
        """Record an order event (filled, rejected, cancelled)."""
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._get_conn() as conn:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO orders (
                        order_id, created_at, symbol, timeframe, side,
                        order_type, size, price, fee, mode, status, rejection_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    order_id or f"ord_{int(datetime.now(timezone.utc).timestamp()*1000)}",
                    now, symbol, timeframe, side.upper(), order_type.lower(),
                    float(size), float(price), float(fee), mode.lower(),
                    status.upper(), rejection_reason
                ))
                conn.commit()
                return cur.lastrowid or 0
        except Exception as e:
            log.exception("Failed to record order in execution database: %s", e)
            return 0

    def record_position_opened(
        self,
        symbol: str,
        timeframe: str,
        side: str,
        size: float,
        entry_price: float,
        stop: float,
        target: float,
        fee: float,
        mode: str,
        initial_stop: Optional[float] = None,
    ) -> str:
        """Record a new open position."""
        now = datetime.now(timezone.utc).isoformat()
        pos_id = f"pos_{int(datetime.now(timezone.utc).timestamp()*1000)}"
        init_stop = initial_stop if initial_stop is not None else stop
        try:
            with self._get_conn() as conn:
                conn.execute("""
                    INSERT INTO positions (
                        position_id, opened_at, status, symbol, timeframe, side,
                        size, entry_price, current_price, current_stop, initial_stop,
                        trailing_stop, target, peak_price, unrealized_pnl, unrealized_pnl_pct,
                        fees_paid, mode
                    ) VALUES (?, ?, 'OPEN', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0.0, 0.0, ?, ?)
                """, (
                    pos_id, now, symbol, timeframe, side.upper(), float(size),
                    float(entry_price), float(entry_price), float(stop), float(init_stop),
                    float(stop), float(target), float(entry_price), float(fee), mode.lower()
                ))
                conn.commit()
                return pos_id
        except Exception as e:
            log.exception("Failed to record position opened: %s", e)
            return pos_id

    def update_open_position(
        self,
        current_price: float,
        current_stop: float,
        trailing_stop: Optional[float] = None,
        peak_price: Optional[float] = None,
    ) -> None:
        """Update live metrics for currently open position."""
        try:
            with self._get_conn() as conn:
                cur = conn.cursor()
                cur.execute("SELECT id, entry_price, size, fees_paid, peak_price, side FROM positions WHERE status = 'OPEN' ORDER BY id DESC LIMIT 1")
                row = cur.fetchone()
                if not row:
                    return
                pos_id = row["id"]
                entry = float(row["entry_price"])
                size = float(row["size"])
                row_side = str(row["side"] if "side" in row.keys() else "BUY").upper()
                prev_peak = float(row["peak_price"] or entry)

                if row_side == "SELL":
                    new_peak = min(prev_peak, float(peak_price or current_price))
                    unrealized_pnl = (entry - float(current_price)) * size - float(row["fees_paid"])
                    unrealized_pnl_pct = ((entry - float(current_price)) / entry) * 100.0 if entry > 0 else 0.0
                else:
                    new_peak = max(prev_peak, float(peak_price or current_price))
                    unrealized_pnl = (float(current_price) - entry) * size - float(row["fees_paid"])
                    unrealized_pnl_pct = ((float(current_price) - entry) / entry) * 100.0 if entry > 0 else 0.0

                cur.execute("""
                    UPDATE positions SET
                        current_price = ?,
                        current_stop = ?,
                        trailing_stop = ?,
                        peak_price = ?,
                        unrealized_pnl = ?,
                        unrealized_pnl_pct = ?
                    WHERE id = ?
                """, (
                    float(current_price), float(current_stop),
                    float(trailing_stop) if trailing_stop is not None else float(current_stop),
                    new_peak, round(unrealized_pnl, 2), round(unrealized_pnl_pct, 2), pos_id
                ))
                conn.commit()
        except Exception as e:
            log.warning("Failed to update open position: %s", e)

    def record_position_closed(
        self,
        exit_price: float,
        exit_fee: float,
        realized_pnl: float,
        exit_reason: str,
    ) -> None:
        """Mark the active open position as closed."""
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._get_conn() as conn:
                cur = conn.cursor()
                cur.execute("SELECT id, entry_price, size, fees_paid, side FROM positions WHERE status = 'OPEN' ORDER BY id DESC LIMIT 1")
                row = cur.fetchone()
                if not row:
                    return
                pos_id = row["id"]
                entry = float(row["entry_price"])
                row_side = str(row["side"] if "side" in row.keys() else "BUY").upper()
                total_fees = float(row["fees_paid"]) + float(exit_fee)
                if row_side == "SELL":
                    ret_pct = ((entry - float(exit_price)) / entry) * 100.0 if entry > 0 else 0.0
                else:
                    ret_pct = ((float(exit_price) - entry) / entry) * 100.0 if entry > 0 else 0.0

                cur.execute("""
                    UPDATE positions SET
                        status = 'CLOSED',
                        closed_at = ?,
                        current_price = ?,
                        exit_price = ?,
                        fees_paid = ?,
                        realized_pnl = ?,
                        realized_pnl_pct = ?,
                        exit_reason = ?,
                        unrealized_pnl = 0.0,
                        unrealized_pnl_pct = 0.0
                    WHERE id = ?
                """, (
                    now, float(exit_price), float(exit_price), round(total_fees, 4),
                    round(float(realized_pnl), 2), round(ret_pct, 2), exit_reason, pos_id
                ))
                conn.commit()
        except Exception as e:
            log.exception("Failed to record position closed: %s", e)

    def get_open_position(self) -> Optional[Dict[str, Any]]:
        """Return the current open position, if any."""
        try:
            with self._get_conn() as conn:
                cur = conn.cursor()
                cur.execute("SELECT * FROM positions WHERE status = 'OPEN' ORDER BY id DESC LIMIT 1")
                row = cur.fetchone()
                return dict(row) if row else None
        except Exception as e:
            log.warning("Could not fetch open position: %s", e)
            return None

    def get_orders(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent orders (both fills and rejections)."""
        try:
            with self._get_conn() as conn:
                cur = conn.cursor()
                cur.execute("SELECT * FROM orders ORDER BY id DESC LIMIT ?", (limit,))
                return [dict(r) for r in cur.fetchall()]
        except Exception as e:
            log.warning("Could not fetch orders: %s", e)
            return []

    def get_closed_positions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return historical closed trades."""
        try:
            with self._get_conn() as conn:
                cur = conn.cursor()
                cur.execute("SELECT * FROM positions WHERE status = 'CLOSED' ORDER BY id DESC LIMIT ?", (limit,))
                return [dict(r) for r in cur.fetchall()]
        except Exception as e:
            log.warning("Could not fetch closed positions: %s", e)
            return []

    def get_summary_stats(self) -> Dict[str, Any]:
        """Aggregate summary metrics of live/paper execution."""
        try:
            with self._get_conn() as conn:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) AS total_orders, SUM(CASE WHEN status='FILLED' THEN 1 ELSE 0 END) AS filled, SUM(CASE WHEN status='REJECTED' THEN 1 ELSE 0 END) AS rejected FROM orders")
                ord_stats = dict(cur.fetchone() or {})

                cur.execute("SELECT COUNT(*) AS total_trades, SUM(realized_pnl) AS net_pnl, SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) AS winners, SUM(CASE WHEN realized_pnl <= 0 THEN 1 ELSE 0 END) AS losers, SUM(fees_paid) AS total_fees FROM positions WHERE status = 'CLOSED'")
                trade_stats = dict(cur.fetchone() or {})

                total_trades = trade_stats.get("total_trades") or 0
                winners = trade_stats.get("winners") or 0
                win_rate = round((winners / total_trades) * 100.0, 1) if total_trades > 0 else 0.0

                return {
                    "total_orders": ord_stats.get("total_orders") or 0,
                    "filled_orders": ord_stats.get("filled") or 0,
                    "rejected_orders": ord_stats.get("rejected") or 0,
                    "total_closed_trades": total_trades,
                    "win_rate_pct": win_rate,
                    "net_realized_pnl": round(trade_stats.get("net_pnl") or 0.0, 2),
                    "total_fees_paid": round(trade_stats.get("total_fees") or 0.0, 4),
                }
        except Exception as e:
            log.warning("Could not calculate execution stats: %s", e)
            return {
                "total_orders": 0, "filled_orders": 0, "rejected_orders": 0,
                "total_closed_trades": 0, "win_rate_pct": 0.0, "net_realized_pnl": 0.0, "total_fees_paid": 0.0
            }


_tracker_instance: Optional[ExecutionTracker] = None


def get_execution_tracker() -> ExecutionTracker:
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = ExecutionTracker()
    return _tracker_instance
