"""
Alerting Engine for Real-time Live Risk & Execution Events.

Dispatches push/webhook/telegram/log alerts for:
- Circuit breaker triggered (daily loss / consecutive loss streak)
- Max drawdown threshold crossed / emergency kill switch
- Exchange API errors or authentication failures
- Live or paper order rejections
"""
import os
import json
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Optional, Any
import urllib.request
import urllib.error

log = logging.getLogger("alerts")


class AlertDispatcher:
    def __init__(self):
        self.webhook_url = os.getenv("ALERT_WEBHOOK_URL", "").strip()
        self.telegram_bot_token = os.getenv("ALERT_TELEGRAM_BOT_TOKEN", "").strip()
        self.telegram_chat_id = os.getenv("ALERT_TELEGRAM_CHAT_ID", "").strip()
        self._last_alert_time: Dict[str, float] = {}
        self.min_interval_sec = 30.0  # deduplication window per event key
        self.alert_history: list = []

    def is_configured(self) -> bool:
        return bool(self.webhook_url or (self.telegram_bot_token and self.telegram_chat_id))

    def get_config_summary(self) -> Dict[str, Any]:
        webhook_masked = (
            self.webhook_url[:12] + "..." + self.webhook_url[-6:]
            if len(self.webhook_url) > 20 else ("Configured" if self.webhook_url else "Not configured")
        )
        return {
            "has_webhook": bool(self.webhook_url),
            "webhook_preview": webhook_masked,
            "has_telegram": bool(self.telegram_bot_token and self.telegram_chat_id),
            "total_alerts_sent": len(self.alert_history),
        }

    def send_alert(
        self,
        event_type: str,
        title: str,
        message: str,
        level: str = "warning",
        details: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> bool:
        """
        Send an alert across all configured channels.
        event_type: 'CIRCUIT_BREAKER', 'MAX_DRAWDOWN', 'EXCHANGE_ERROR', 'ORDER_REJECTED', 'EXECUTION'
        level: 'info', 'warning', 'critical'
        """
        now = time.time()
        dedup_key = f"{event_type}_{title}"

        # Prevent alert flooding unless critical or forced
        if not force and level != "critical":
            last_sent = self._last_alert_time.get(dedup_key, 0.0)
            if now - last_sent < self.min_interval_sec:
                log.debug("Suppressed duplicate alert: %s (%s)", title, event_type)
                return False

        self._last_alert_time[dedup_key] = now
        timestamp_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        log_fn = log.critical if level == "critical" else (log.warning if level == "warning" else log.info)
        log_fn("[ALERT %s] %s: %s (Details: %s)", event_type, title, message, details or {})

        alert_record = {
            "timestamp": timestamp_iso,
            "event_type": event_type,
            "level": level.upper(),
            "title": title,
            "message": message,
            "details": details or {},
        }
        self.alert_history.append(alert_record)
        if len(self.alert_history) > 100:
            self.alert_history.pop(0)

        success = True
        # 1. Dispatch Webhook (Discord / Slack / Generic HTTP POST)
        if self.webhook_url:
            try:
                self._dispatch_webhook(alert_record)
            except Exception as e:
                log.warning("Failed to dispatch alert webhook: %s", e)
                success = False

        # 2. Dispatch Telegram Bot
        if self.telegram_bot_token and self.telegram_chat_id:
            try:
                self._dispatch_telegram(alert_record)
            except Exception as e:
                log.warning("Failed to dispatch telegram alert: %s", e)
                success = False

        return success

    def _dispatch_webhook(self, record: dict):
        payload = {
            "username": "Trading Bot Risk Guardian",
            "content": f"**[{record['level']}] {record['event_type']}**: {record['title']}\n{record['message']}",
            "embeds": [
                {
                    "title": f"{record['title']} ({record['event_type']})",
                    "description": record["message"],
                    "color": 15158332 if record["level"] == "CRITICAL" else (16753920 if record["level"] == "WARNING" else 3066993),
                    "fields": [
                        {"name": "Timestamp", "value": record["timestamp"], "inline": True},
                        {"name": "Severity", "value": record["level"], "inline": True},
                    ] + [{"name": k, "value": str(v), "inline": True} for k, v in record.get("details", {}).items()][:6],
                }
            ]
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.webhook_url,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "CryptoTradingBot/AlertEngine"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            pass

    def _dispatch_telegram(self, record: dict):
        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        text = (
            f"🚨 *[{record['level']}] {record['event_type']}*\n"
            f"*{record['title']}*\n\n"
            f"{record['message']}\n\n"
            f"🕒 _{record['timestamp']}_"
        )
        if record.get("details"):
            details_str = "\n".join([f"• `{k}`: {v}" for k, v in record["details"].items()])
            text += f"\n\n*Details:*\n{details_str}"

        payload = {
            "chat_id": self.telegram_chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            pass


_alert_dispatcher_instance: Optional[AlertDispatcher] = None


def get_alert_dispatcher() -> AlertDispatcher:
    global _alert_dispatcher_instance
    if _alert_dispatcher_instance is None:
        _alert_dispatcher_instance = AlertDispatcher()
    return _alert_dispatcher_instance
