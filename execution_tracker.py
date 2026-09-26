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
        self._cached_orders: List[Dict[str, Any]] = []
        self._cached_closed_positions: List[Dict[str, Any]] = []
        self._cached_open_position: Optional[Dict[str, Any]] = None
        self._init_db()
        self._sync_from_storage()

    def _sync_to_storage(self):
        if "PYTEST_CURRENT_TEST" in os.environ:
            return
        try:
            from storage_adapter import remote_save_app_setting
            if self._cached_orders:
                remote_save_app_setting("execution_orders", self._cached_orders[:50])
            if self._cached_closed_positions:
                remote_save_app_setting("execution_closed_trades", self._cached_closed_positions[:50])
        except Exception:
            pass
        try:
            from pathlib import Path
            import json
            base = Path("/tmp") if (os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME")) else Path(os.path.dirname(os.path.abspath(__file__)))
            if self._cached_orders:
                (base / "execution_orders.json").write_text(json.dumps(self._cached_orders[:50]))
            if self._cached_closed_positions:
                (base / "execution_closed_trades.json").write_text(json.dumps(self._cached_closed_positions[:50]))
        except Exception:
            pass

    def _sync_from_storage(self):
        if "PYTEST_CURRENT_TEST" in os.environ:
            return
        try:
            from storage_adapter import remote_get_app_setting
            orders = remote_get_app_setting("execution_orders")
            if orders and isinstance(orders, list) and not self._cached_orders:
                self._cached_orders = orders
            closed = remote_get_app_setting("execution_closed_trades")
            if closed and isinstance(closed, list) and not self._cached_closed_positions:
                self._cached_closed_positions = closed
        except Exception:
            pass
        if not self._cached_orders:
            try:
                from pathlib import Path
                import json
                base = Path("/tmp") if (os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME")) else Path(os.path.dirname(os.path.abspath(__file__)))
                orders_file = base / "execution_orders.json"
                if orders_file.exists():
                    self._cached_orders = json.loads(orders_file.read_text())
                closed_file = base / "execution_closed_trades.json"
                if closed_file.exists():
                    self._cached_closed_positions = json.loads(closed_file.read_text())
            except Exception:
                pass

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
        oid = order_id or f"ord_{int(datetime.now(timezone.utc).timestamp()*1000)}"
        last_id = 0
        try:
            with self._get_conn() as conn:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO orders (
                        order_id, created_at, symbol, timeframe, side,
                        order_type, size, price, fee, mode, status, rejection_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    oid, now, symbol, timeframe, side.upper(), order_type.lower(),
                    float(size), float(price), float(fee), mode.lower(),
                    status.upper(), rejection_reason
                ))
                conn.commit()
                last_id = cur.lastrowid or 0
        except Exception as e:
            log.exception("Failed to record order in execution database: %s", e)

        ord_dict = {
            "id": last_id or len(self._cached_orders) + 1,
            "order_id": oid,
            "created_at": now,
            "timestamp": now,
            "symbol": symbol,
            "timeframe": timeframe,
            "side": side.upper(),
            "order_type": order_type.lower(),
            "size": float(size),
            "amount": float(size),
            "price": float(price),
            "fee": float(fee),
            "mode": mode.lower(),
            "status": status.upper(),
            "rejection_reason": rejection_reason,
        }
        self._cached_orders.insert(0, ord_dict)
        self._sync_to_storage()
        return last_id

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
                # Mark any prior OPEN positions closed to avoid orphans
                conn.execute("UPDATE positions SET status = 'CLOSED' WHERE status = 'OPEN'")
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
        except Exception as e:
            log.exception("Failed to record position opened: %s", e)

        self._cached_open_position = {
            "id": pos_id,
            "position_id": pos_id,
            "opened_at": now,
            "status": "OPEN",
            "symbol": symbol,
            "timeframe": timeframe,
            "side": side.upper(),
            "size": float(size),
            "amount": float(size),
            "entry_price": float(entry_price),
            "entry": float(entry_price),
            "current_price": float(entry_price),
            "current_stop": float(stop),
            "stop": float(stop),
            "stop_loss": float(stop),
            "initial_stop": float(init_stop),
            "trailing_stop": float(stop),
            "target": float(target),
            "peak_price": float(entry_price),
            "unrealized_pnl": 0.0,
            "unrealized_pnl_pct": 0.0,
            "fees_paid": float(fee),
            "fee_paid": float(fee),
            "mode": mode.lower(),
        }
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
                    if self._cached_open_position:
                        entry = float(self._cached_open_position.get("entry_price") or self._cached_open_position.get("entry", current_price))
                        size = float(self._cached_open_position.get("size") or self._cached_open_position.get("amount", 0.0))
                        row_side = str(self._cached_open_position.get("side", "BUY")).upper()
                        fee_paid = float(self._cached_open_position.get("fees_paid") or self._cached_open_position.get("fee_paid", 0.0))
                        if row_side == "SELL":
                            unrealized_pnl = (entry - float(current_price)) * size - fee_paid
                            unrealized_pnl_pct = ((entry - float(current_price)) / entry) * 100.0 if entry > 0 else 0.0
                        else:
                            unrealized_pnl = (float(current_price) - entry) * size - fee_paid
                            unrealized_pnl_pct = ((float(current_price) - entry) / entry) * 100.0 if entry > 0 else 0.0
                        self._cached_open_position["current_price"] = float(current_price)
                        self._cached_open_position["current_stop"] = float(current_stop)
                        self._cached_open_position["trailing_stop"] = float(trailing_stop) if trailing_stop is not None else float(current_stop)
                        self._cached_open_position["unrealized_pnl"] = round(unrealized_pnl, 2)
                        self._cached_open_position["unrealized_pnl_pct"] = round(unrealized_pnl_pct, 2)
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
                if self._cached_open_position:
                    self._cached_open_position["current_price"] = float(current_price)
                    self._cached_open_position["current_stop"] = float(current_stop)
                    self._cached_open_position["trailing_stop"] = float(trailing_stop) if trailing_stop is not None else float(current_stop)
                    self._cached_open_position["unrealized_pnl"] = round(unrealized_pnl, 2)
                    self._cached_open_position["unrealized_pnl_pct"] = round(unrealized_pnl_pct, 2)
        except Exception as e:
            log.warning("Failed to update open position: %s", e)

    def record_position_closed(
        self,
        exit_price: float,
        exit_fee: float,
        realized_pnl: float,
        exit_reason: str,
        symbol: str = "BTC/USDT",
        side: str = "BUY",
        size: float = 0.0,
        entry_price: float = 0.0,
        opened_at: Optional[str] = None,
    ) -> None:
        """Mark the active open position as closed."""
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._get_conn() as conn:
                cur = conn.cursor()
                cur.execute("SELECT id, position_id, entry_price, size, fees_paid, side, symbol, opened_at FROM positions WHERE status = 'OPEN' ORDER BY id DESC LIMIT 1")
                row = cur.fetchone()
                if row:
                    pos_id = row["id"]
                    pos_identifier = row["position_id"] if ("position_id" in row.keys() and row["position_id"]) else f"pos_{pos_id}"
                    entry = float(row["entry_price"])
                    sym = str(row["symbol"] or symbol)
                    op_at = str(row["opened_at"] or opened_at or now)
                    pos_size = float(row["size"])
                    row_side = str(row["side"] if "side" in row.keys() else side).upper()
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
                    # Mark any other remaining OPEN positions as CLOSED
                    cur.execute("UPDATE positions SET status = 'CLOSED' WHERE status = 'OPEN'")
                    conn.commit()
                else:
                    # No OPEN row in SQLite: record new closed position
                    pos_identifier = f"pos_{int(datetime.now(timezone.utc).timestamp()*1000)}"
                    entry = float(entry_price or exit_price)
                    pos_size = float(size)
                    sym = symbol
                    op_at = opened_at or now
                    total_fees = float(exit_fee)
                    row_side = str(side).upper()
                    if row_side == "SELL":
                        ret_pct = ((entry - float(exit_price)) / entry) * 100.0 if entry > 0 else 0.0
                    else:
                        ret_pct = ((float(exit_price) - entry) / entry) * 100.0 if entry > 0 else 0.0
                    cur.execute("""
                        INSERT INTO positions (
                            position_id, opened_at, closed_at, status, symbol, timeframe, side,
                            size, entry_price, current_price, current_stop, initial_stop,
                            trailing_stop, target, peak_price, unrealized_pnl, unrealized_pnl_pct,
                            realized_pnl, realized_pnl_pct, fees_paid, exit_price, exit_reason, mode
                        ) VALUES (?, ?, ?, 'CLOSED', ?, '15m', ?, ?, ?, ?, 0.0, 0.0, 0.0, 0.0, ?, 0.0, 0.0, ?, ?, ?, ?, ?, 'paper')
                    """, (
                        pos_identifier,
                        op_at, now, sym, row_side, pos_size, entry, float(exit_price),
                        entry, round(float(realized_pnl), 2), round(ret_pct, 2),
                        round(total_fees, 4), float(exit_price), exit_reason
                    ))
                    conn.commit()

                closed_dict = {
                    "position_id": pos_identifier,
                    "opened_at": op_at,
                    "closed_at": now,
                    "status": "CLOSED",
                    "symbol": sym,
                    "side": row_side,
                    "size": pos_size,
                    "amount": pos_size,
                    "entry_price": entry,
                    "entry": entry,
                    "exit_price": float(exit_price),
                    "fees_paid": round(total_fees, 4),
                    "realized_pnl": round(float(realized_pnl), 2),
                    "pnl": round(float(realized_pnl), 2),
                    "realized_pnl_pct": round(ret_pct, 2),
                    "exit_reason": exit_reason,
                }
                self._cached_closed_positions.insert(0, closed_dict)
                self._cached_open_position = None
                self._sync_to_storage()
        except Exception as e:
            log.exception("Failed to record position closed: %s", e)

    def get_open_position(self) -> Optional[Dict[str, Any]]:
        """Return the current open position, if any."""
        try:
            with self._get_conn() as conn:
                cur = conn.cursor()
                cur.execute("SELECT * FROM positions WHERE status = 'OPEN' ORDER BY id DESC LIMIT 1")
                row = cur.fetchone()
                if not row:
                    return self._cached_open_position
                d = dict(row)
                d["entry"] = d.get("entry_price")
                d["amount"] = d.get("size")
                d["stop"] = d.get("current_stop")
                d["stop_loss"] = d.get("current_stop")
                d["notional"] = round(float(d.get("size", 0.0)) * float(d.get("current_price") or d.get("entry_price", 0.0)), 2)
                return d
        except Exception as e:
            log.warning("Could not fetch open position: %s", e)
            return self._cached_open_position

    def get_orders(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent orders (both fills and rejections)."""
        results = []
        seen = set()
        try:
            with self._get_conn() as conn:
                cur = conn.cursor()
                cur.execute("SELECT * FROM orders ORDER BY id DESC LIMIT ?", (limit,))
                for r in cur.fetchall():
                    d = dict(r)
                    d["timestamp"] = d.get("created_at")
                    d["amount"] = d.get("size")
                    k1 = str(d.get("order_id") or "")
                    k2 = f"{d.get('created_at')}_{d.get('price')}_{d.get('size')}"
                    if k1:
                        seen.add(k1)
                    seen.add(k2)
                    results.append(d)
        except Exception as e:
            log.warning("Could not fetch orders: %s", e)

        for o in self._cached_orders:
            k1 = str(o.get("order_id") or "")
            k2 = f"{o.get('created_at')}_{o.get('price')}_{o.get('size')}"
            if (k1 and k1 in seen) or (k2 in seen):
                continue
            if k1:
                seen.add(k1)
            seen.add(k2)
            o["timestamp"] = o.get("created_at") or o.get("timestamp")
            o["amount"] = o.get("size") or o.get("amount")
            results.append(o)

        results.sort(key=lambda x: str(x.get("created_at") or x.get("timestamp") or ""), reverse=True)
        return results[:limit]

    def get_closed_positions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return historical closed trades."""
        results = []
        seen = set()
        try:
            with self._get_conn() as conn:
                cur = conn.cursor()
                cur.execute("SELECT * FROM positions WHERE status = 'CLOSED' ORDER BY id DESC LIMIT ?", (limit,))
                for r in cur.fetchall():
                    d = dict(r)
                    d["entry"] = d.get("entry_price")
                    d["amount"] = d.get("size")
                    d["pnl"] = d.get("realized_pnl")
                    pos_id = str(d.get("position_id") or "")
                    time_key = f"{d.get('closed_at')}_{d.get('realized_pnl')}"
                    if pos_id:
                        seen.add(pos_id)
                    seen.add(time_key)
                    results.append(d)
        except Exception as e:
            log.warning("Could not fetch closed positions: %s", e)

        for c in self._cached_closed_positions:
            pos_id = str(c.get("position_id") or "")
            time_key = f"{c.get('closed_at')}_{c.get('realized_pnl')}"
            if (pos_id and pos_id in seen) or (time_key in seen):
                continue
            if pos_id:
                seen.add(pos_id)
            seen.add(time_key)
            c["entry"] = c.get("entry_price") or c.get("entry")
            c["amount"] = c.get("size") or c.get("amount")
            c["pnl"] = c.get("realized_pnl") if c.get("realized_pnl") is not None else c.get("pnl")
            results.append(c)

        results.sort(key=lambda x: str(x.get("closed_at") or ""), reverse=True)
        return results[:limit]

    def get_summary_stats(self) -> Dict[str, Any]:
        """Aggregate summary metrics of live/paper execution."""
        stats = {
            "total_orders": 0, "filled_orders": 0, "rejected_orders": 0,
            "total_closed_trades": 0, "win_rate_pct": 0.0, "net_realized_pnl": 0.0, "total_fees_paid": 0.0
        }
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

                stats["total_orders"] = ord_stats.get("total_orders") or 0
                stats["filled_orders"] = ord_stats.get("filled") or 0
                stats["rejected_orders"] = ord_stats.get("rejected") or 0
                stats["total_closed_trades"] = total_trades
                stats["win_rate_pct"] = win_rate
                stats["net_realized_pnl"] = round(trade_stats.get("net_pnl") or 0.0, 2)
                stats["total_fees_paid"] = round(trade_stats.get("total_fees") or 0.0, 4)
        except Exception as e:
            log.warning("Could not calculate execution stats: %s", e)

        if stats["total_orders"] == 0 and self._cached_orders:
            stats["total_orders"] = len(self._cached_orders)
            stats["filled_orders"] = sum(1 for o in self._cached_orders if str(o.get("status")).upper() == "FILLED")
            stats["rejected_orders"] = sum(1 for o in self._cached_orders if str(o.get("status")).upper() == "REJECTED")

        if stats["total_closed_trades"] == 0 and self._cached_closed_positions:
            trades = self._cached_closed_positions
            stats["total_closed_trades"] = len(trades)
            winners = sum(1 for t in trades if float(t.get("realized_pnl") or t.get("pnl") or 0.0) > 0)
            stats["win_rate_pct"] = round((winners / len(trades)) * 100.0, 1) if trades else 0.0
            stats["net_realized_pnl"] = round(sum(float(t.get("realized_pnl") or t.get("pnl") or 0.0) for t in trades), 2)
            stats["total_fees_paid"] = round(sum(float(t.get("fees_paid") or 0.0) for t in trades), 4)

        return stats


_tracker_instance: Optional[ExecutionTracker] = None


def get_execution_tracker() -> ExecutionTracker:
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = ExecutionTracker()
    return _tracker_instance

