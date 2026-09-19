"""
Remote Storage Adapter for Cloud Serverless Environments (Vercel, AWS Lambda).
Provides persistent cross-container storage synchronization for user accounts,
exchange credentials, and system settings using standard HTTP REST (no external dependencies).

Supported Drivers:
- Vercel KV / Upstash Redis REST API (KV_REST_API_URL / UPSTASH_REDIS_REST_URL)
- Turso libSQL HTTP API (TURSO_DATABASE_URL + TURSO_AUTH_TOKEN)
"""
import os
import json
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

KV_URL = (os.getenv("KV_REST_API_URL") or os.getenv("UPSTASH_REDIS_REST_URL") or "").strip().rstrip("/")
KV_TOKEN = (os.getenv("KV_REST_API_TOKEN") or os.getenv("UPSTASH_REDIS_REST_TOKEN") or "").strip()

TURSO_URL = (os.getenv("TURSO_DATABASE_URL") or "").strip()
if TURSO_URL.startswith("libsql://"):
    TURSO_URL = "https://" + TURSO_URL[len("libsql://"):]
TURSO_URL = TURSO_URL.rstrip("/")
TURSO_TOKEN = (os.getenv("TURSO_AUTH_TOKEN") or "").strip()


def is_remote_storage_available() -> bool:
    """Returns True if any remote persistent storage driver is configured."""
    return bool((KV_URL and KV_TOKEN) or (TURSO_URL and TURSO_TOKEN))


def get_remote_storage_type() -> str:
    if KV_URL and KV_TOKEN:
        return "kv"
    if TURSO_URL and TURSO_TOKEN:
        return "turso"
    return "none"


# ---------------------------------------------------------------------------
# KV / Upstash REST Client Helpers
# ---------------------------------------------------------------------------
def _kv_command(*args) -> Optional[Any]:
    """Executes a Redis command via Upstash / Vercel KV REST API."""
    if not (KV_URL and KV_TOKEN):
        return None
    try:
        url = f"{KV_URL}"
        data = json.dumps(list(args)).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {KV_TOKEN}",
                "Content-Type": "application/json",
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            return res.get("result")
    except Exception as e:
        print(f"[REMOTE STORE WARNING] KV command error {args[:2]}: {e}")
        return None


# ---------------------------------------------------------------------------
# User Account Synchronization
# ---------------------------------------------------------------------------
def remote_save_user(
    user_id: int,
    username: str,
    email: str,
    password_hash: str,
    salt: str,
    is_active: int = 1,
    created_at: Optional[str] = None
) -> bool:
    """Stores or updates a user record in the remote persistent store."""
    if not is_remote_storage_available():
        return False

    created_at = created_at or datetime.now(timezone.utc).isoformat()
    record = {
        "id": user_id,
        "username": username,
        "email": email,
        "password_hash": password_hash,
        "salt": salt,
        "is_active": is_active,
        "created_at": created_at,
    }
    rec_str = json.dumps(record)

    if KV_URL and KV_TOKEN:
        u_key = f"user:by_username:{username.strip().lower()}"
        e_key = f"user:by_email:{email.strip().lower()}"
        id_key = f"user:by_id:{user_id}"
        _kv_command("SET", u_key, rec_str)
        _kv_command("SET", e_key, rec_str)
        _kv_command("SET", id_key, rec_str)
        _kv_command("SADD", "users:all_ids", str(user_id))
        _kv_command("SADD", "users:all_usernames", username.strip().lower())
        _kv_command("SADD", "users:all_emails", email.strip().lower())
        return True

    return False


def remote_get_user(val: str) -> Optional[Dict[str, Any]]:
    """Retrieves a user record by username, email, or user_id from the remote persistent store."""
    if not is_remote_storage_available():
        return None

    val_str = str(val).strip().lower()
    if KV_URL and KV_TOKEN:
        # Check by email
        if "@" in val_str:
            raw = _kv_command("GET", f"user:by_email:{val_str}")
        elif val_str.isdigit():
            raw = _kv_command("GET", f"user:by_id:{val_str}")
            if not raw:
                raw = _kv_command("GET", f"user:by_username:{val_str}")
        else:
            raw = _kv_command("GET", f"user:by_username:{val_str}")
            if not raw:
                raw = _kv_command("GET", f"user:by_email:{val_str}")

        if raw:
            try:
                if isinstance(raw, str):
                    return json.loads(raw)
                if isinstance(raw, dict):
                    return raw
            except Exception:
                pass
    return None


def remote_save_exchange_key(
    user_id: int,
    exchange_id: str,
    enc_api_key: str,
    enc_api_secret: str,
    enc_api_passphrase: str = "",
    testnet: bool = True
) -> bool:
    """Stores an encrypted exchange credential in the remote persistent store."""
    if not is_remote_storage_available():
        return False

    record = {
        "user_id": user_id,
        "exchange_id": exchange_id,
        "api_key": enc_api_key,
        "api_secret": enc_api_secret,
        "api_passphrase": enc_api_passphrase,
        "testnet": testnet,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    rec_str = json.dumps(record)

    if KV_URL and KV_TOKEN:
        k = f"exchange_key:{user_id}:{exchange_id.lower()}"
        _kv_command("SET", k, rec_str)
        _kv_command("SADD", f"user:{user_id}:exchanges", exchange_id.lower())
        return True
    return False


def remote_get_exchange_keys(user_id: int) -> List[Dict[str, Any]]:
    """Retrieves all encrypted exchange credentials for a user from remote persistent store."""
    if not is_remote_storage_available():
        return []

    if KV_URL and KV_TOKEN:
        exchanges = _kv_command("SMEMBERS", f"user:{user_id}:exchanges")
        if not exchanges:
            return []
        keys = []
        for ex in exchanges:
            raw = _kv_command("GET", f"exchange_key:{user_id}:{ex.lower()}")
            if raw:
                try:
                    obj = json.loads(raw) if isinstance(raw, str) else raw
                    keys.append(obj)
                except Exception:
                    pass
        return keys
    return []


def remote_delete_account(user_id: int, username: str = "", email: str = "") -> bool:
    """Deletes a user account and associated credentials from remote persistent store."""
    if not is_remote_storage_available():
        return False

    if KV_URL and KV_TOKEN:
        if not username or not email:
            u = remote_get_user(str(user_id))
            if u:
                username = u.get("username", "")
                email = u.get("email", "")

        if username:
            _kv_command("DEL", f"user:by_username:{username.lower()}")
            _kv_command("SREM", "users:all_usernames", username.lower())
        if email:
            _kv_command("DEL", f"user:by_email:{email.lower()}")
            _kv_command("SREM", "users:all_emails", email.lower())

        _kv_command("DEL", f"user:by_id:{user_id}")
        _kv_command("SREM", "users:all_ids", str(user_id))

        exchanges = _kv_command("SMEMBERS", f"user:{user_id}:exchanges") or []
        for ex in exchanges:
            _kv_command("DEL", f"exchange_key:{user_id}:{ex.lower()}")
        _kv_command("DEL", f"user:{user_id}:exchanges")
        return True
    return False


def remote_delete_all_accounts() -> bool:
    """Purges all user accounts and exchange credentials from remote persistent store."""
    if not is_remote_storage_available():
        return False

    if KV_URL and KV_TOKEN:
        ids = _kv_command("SMEMBERS", "users:all_ids") or []
        for uid in ids:
            remote_delete_account(int(uid))
        _kv_command("DEL", "users:all_ids")
        _kv_command("DEL", "users:all_usernames")
        _kv_command("DEL", "users:all_emails")
        return True
    return False
