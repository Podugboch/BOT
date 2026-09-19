"""Reliable remote storage for serverless deployments.

The adapter supports one selected backend at a time (KV, Turso, or Blob).  It is
intentionally dependency-free and treats failed remote operations as failures
instead of reporting false success.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

KV_URL = (os.getenv("KV_REST_API_URL") or os.getenv("UPSTASH_REDIS_REST_URL") or "").strip().rstrip("/")
KV_TOKEN = (os.getenv("KV_REST_API_TOKEN") or os.getenv("UPSTASH_REDIS_REST_TOKEN") or "").strip()

TURSO_URL = (os.getenv("TURSO_DATABASE_URL") or "").strip()
if TURSO_URL.startswith("libsql://"):
    TURSO_URL = "https://" + TURSO_URL[len("libsql://"):]
TURSO_URL = TURSO_URL.rstrip("/")
TURSO_TOKEN = (os.getenv("TURSO_AUTH_TOKEN") or "").strip()

BLOB_TOKEN = (os.getenv("BLOB_READ_WRITE_TOKEN") or "").strip()
BLOB_STORE_ID = (os.getenv("BLOB_STORE_ID") or "").strip()

_SAFE_COMPONENT = re.compile(r"[^a-z0-9_.@-]+", re.IGNORECASE)


def _component(value: Any) -> str:
    """Return a stable, path-safe identifier; never allow traversal or slashes."""
    value = str(value).strip().lower()
    value = _SAFE_COMPONENT.sub("_", value).strip("._")
    if not value or value in {".", ".."}:
        raise ValueError("Storage identifier cannot be empty")
    return value


def _configured_driver() -> str:
    # A single backend prevents split-brain reads when several services exist.
    if KV_URL and KV_TOKEN:
        return "kv"
    if TURSO_URL and TURSO_TOKEN:
        return "turso"
    if BLOB_TOKEN:
        return "blob"
    return "none"


def is_remote_storage_available() -> bool:
    return _configured_driver() != "none"


def get_remote_storage_type() -> str:
    return _configured_driver()


# ---------------------------------------------------------------------------
# KV / Upstash REST
# ---------------------------------------------------------------------------
def _kv_command(*args: Any) -> Tuple[bool, Optional[Any]]:
    """Return (request_succeeded, Redis result)."""
    if not (KV_URL and KV_TOKEN):
        return False, None
    try:
        request = urllib.request.Request(
            KV_URL,
            data=json.dumps(list(args)).encode("utf-8"),
            headers={"Authorization": f"Bearer {KV_TOKEN}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return True, payload.get("result")
    except Exception as exc:
        print(f"[REMOTE STORE WARNING] KV command {args[:2]} failed: {exc}")
        return False, None


def _kv_ok(*args: Any) -> bool:
    return _kv_command(*args)[0]


# ---------------------------------------------------------------------------
# Turso libSQL HTTP API
# ---------------------------------------------------------------------------
def _turso_execute(sql: str, args: Sequence[Any] = ()) -> Tuple[bool, List[Dict[str, Any]]]:
    """Execute one statement through Turso's HTTP pipeline API."""
    if not (TURSO_URL and TURSO_TOKEN):
        return False, []
    endpoint = TURSO_URL if TURSO_URL.endswith("/v2/pipeline") else f"{TURSO_URL}/v2/pipeline"
    body = {"requests": [{"type": "execute", "stmt": {"sql": sql, "args": list(args)}}, {"type": "close"}]}
    try:
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {TURSO_TOKEN}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=8.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
        result = payload.get("results", [{}])[0]
        if result.get("type") == "error" or result.get("response", {}).get("type") == "error":
            raise RuntimeError(str(result))
        rows = result.get("response", {}).get("result", {}).get("rows", [])
        columns = result.get("response", {}).get("result", {}).get("cols", [])
        names = [c.get("name", str(i)) if isinstance(c, dict) else str(i) for i, c in enumerate(columns)]
        decoded = []
        for row in rows:
            values = [cell.get("value") if isinstance(cell, dict) else cell for cell in row]
            decoded.append(dict(zip(names, values)))
        return True, decoded
    except Exception as exc:
        print(f"[REMOTE STORE WARNING] Turso query failed: {exc}")
        return False, []


def _turso_init() -> bool:
    statements = [
        "CREATE TABLE IF NOT EXISTS remote_users (id INTEGER PRIMARY KEY, username TEXT NOT NULL, email TEXT NOT NULL, password_hash TEXT NOT NULL, salt TEXT NOT NULL, is_active INTEGER NOT NULL, created_at TEXT NOT NULL)",
        "CREATE UNIQUE INDEX IF NOT EXISTS remote_users_username ON remote_users(username)",
        "CREATE UNIQUE INDEX IF NOT EXISTS remote_users_email ON remote_users(email)",
        "CREATE TABLE IF NOT EXISTS remote_exchange_keys (user_id INTEGER NOT NULL, exchange_id TEXT NOT NULL, api_key TEXT NOT NULL, api_secret TEXT NOT NULL, api_passphrase TEXT NOT NULL, testnet INTEGER NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY(user_id, exchange_id))",
    ]
    return all(_turso_execute(statement)[0] for statement in statements)


# ---------------------------------------------------------------------------
# Vercel Blob
# ---------------------------------------------------------------------------
def _get_blob_store_id() -> str:
    if BLOB_STORE_ID:
        return BLOB_STORE_ID.replace("store_", "").lower()
    # BLOB_STORE_ID should be configured explicitly. The fallback is retained
    # for compatibility, but avoids guessing from the secret token.
    return "cb633j7qmvxtvyhr"


def _blob_url(pathname: str) -> str:
    encoded = urllib.parse.quote(pathname.lstrip("/"), safe="/")
    return f"https://{_get_blob_store_id()}.private.blob.vercel-storage.com/{encoded}"


def _blob_put(pathname: str, data: bytes, content_type: str = "application/json") -> bool:
    if not BLOB_TOKEN:
        return False
    try:
        url = "https://blob.vercel-storage.com/?pathname=" + urllib.parse.quote(pathname, safe="/")
        request = urllib.request.Request(url, data=data, headers={
            "authorization": f"Bearer {BLOB_TOKEN}", "x-api-version": "12",
            "x-vercel-blob-access": "private", "x-add-random-suffix": "0",
            "x-allow-overwrite": "1", "content-type": content_type,
        }, method="PUT")
        with urllib.request.urlopen(request, timeout=6.0) as response:
            return response.status in (200, 201)
    except Exception as exc:
        print(f"[REMOTE STORE WARNING] Blob PUT {pathname} failed: {exc}")
        return False


def _blob_get(pathname: str) -> Optional[bytes]:
    if not BLOB_TOKEN:
        return None
    try:
        request = urllib.request.Request(_blob_url(pathname) + "?cache=0", headers={"authorization": f"Bearer {BLOB_TOKEN}"})
        with urllib.request.urlopen(request, timeout=6.0) as response:
            return response.read() if response.status == 200 else None
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            print(f"[REMOTE STORE WARNING] Blob GET {pathname} returned HTTP {exc.code}")
        return None
    except Exception as exc:
        print(f"[REMOTE STORE WARNING] Blob GET {pathname} failed: {exc}")
        return None


def _blob_del(pathname: str) -> bool:
    if not BLOB_TOKEN:
        return False
    try:
        request = urllib.request.Request(
            "https://blob.vercel-storage.com/delete",
            data=json.dumps({"urls": [_blob_url(pathname)]}).encode("utf-8"),
            headers={"authorization": f"Bearer {BLOB_TOKEN}", "x-api-version": "12", "content-type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=6.0) as response:
            return response.status in (200, 204)
    except Exception as exc:
        print(f"[REMOTE STORE WARNING] Blob DELETE {pathname} failed: {exc}")
        return False


def _json_blob(pathname: str) -> Optional[Dict[str, Any]]:
    raw = _blob_get(pathname)
    if not raw:
        return None
    try:
        value = json.loads(raw.decode("utf-8"))
        return value if isinstance(value, dict) else None
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"[REMOTE STORE WARNING] Invalid JSON in {pathname}: {exc}")
        return None


def _encrypted(value: str) -> str:
    """Encrypt at the storage boundary; encrypt_secret is idempotent."""
    try:
        from secret_manager import encrypt_secret
        return encrypt_secret(value)
    except Exception as exc:
        raise RuntimeError(f"Unable to encrypt remote credential: {exc}") from exc


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------
def remote_save_user(user_id: int, username: str, email: str, password_hash: str, salt: str,
                     is_active: int = 1, created_at: Optional[str] = None,
                     previous_username: str = "", previous_email: str = "") -> bool:
    if not is_remote_storage_available():
        return False
    username_key, email_key = _component(username), _component(email)
    created_at = created_at or datetime.now(timezone.utc).isoformat()
    record = {"id": int(user_id), "username": username, "email": email, "password_hash": password_hash,
              "salt": salt, "is_active": int(is_active), "created_at": created_at}
    raw = json.dumps(record).encode("utf-8")
    driver = _configured_driver()

    if driver == "kv":
        writes = [
            _kv_ok("SET", f"user:by_username:{username_key}", raw.decode()),
            _kv_ok("SET", f"user:by_email:{email_key}", raw.decode()),
            _kv_ok("SET", f"user:by_id:{int(user_id)}", raw.decode()),
            _kv_ok("SADD", "users:all_ids", str(int(user_id))),
            _kv_ok("SADD", "users:all_usernames", username_key),
            _kv_ok("SADD", "users:all_emails", email_key),
        ]
        if previous_username:
            writes.append(_kv_ok("DEL", f"user:by_username:{_component(previous_username)}"))
            writes.append(_kv_ok("SREM", "users:all_usernames", _component(previous_username)))
        if previous_email:
            writes.append(_kv_ok("DEL", f"user:by_email:{_component(previous_email)}"))
            writes.append(_kv_ok("SREM", "users:all_emails", _component(previous_email)))
        return all(writes)

    if driver == "turso":
        return (_turso_init() and _turso_execute(
            "INSERT INTO remote_users(id,username,email,password_hash,salt,is_active,created_at) VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET username=excluded.username,email=excluded.email,password_hash=excluded.password_hash,salt=excluded.salt,is_active=excluded.is_active,created_at=excluded.created_at",
            (int(user_id), username, email, password_hash, salt, int(is_active), created_at),
        )[0])

    return all(_blob_put(path, raw) for path in (
        f"users/by_username/{username_key}.json", f"users/by_email/{email_key}.json", f"users/by_id/{int(user_id)}.json"))


def remote_get_user(val: str) -> Optional[Dict[str, Any]]:
    if not is_remote_storage_available():
        return None
    value = str(val).strip()
    try:
        key = _component(value)
    except ValueError:
        return None
    driver = _configured_driver()
    if driver == "kv":
        candidates = [f"user:by_email:{key}"] if "@" in value else ([f"user:by_id:{key}", f"user:by_username:{key}"] if value.isdigit() else [f"user:by_username:{key}", f"user:by_email:{key}"])
        for candidate in candidates:
            ok, result = _kv_command("GET", candidate)
            if ok and result:
                try:
                    return json.loads(result) if isinstance(result, str) else result
                except (TypeError, json.JSONDecodeError):
                    return None
        return None
    if driver == "turso":
        if not _turso_init():
            return None
        column = "email" if "@" in value else ("id" if value.isdigit() else "username")
        ok, rows = _turso_execute(f"SELECT id,username,email,password_hash,salt,is_active,created_at FROM remote_users WHERE {column}=? LIMIT 1", (int(value) if column == "id" else value,))
        return rows[0] if ok and rows else None
    paths = [f"users/by_email/{key}.json"] if "@" in value else ([f"users/by_id/{key}.json", f"users/by_username/{key}.json"] if value.isdigit() else [f"users/by_username/{key}.json", f"users/by_email/{key}.json"])
    for path in paths:
        result = _json_blob(path)
        if result:
            return result
    return None


# ---------------------------------------------------------------------------
# Exchange credentials
# ---------------------------------------------------------------------------
def remote_save_exchange_key(user_id: int, exchange_id: str, enc_api_key: str, enc_api_secret: str,
                             enc_api_passphrase: str = "", testnet: bool = True) -> bool:
    if not is_remote_storage_available():
        return False
    exchange = _component(exchange_id)
    record = {"user_id": int(user_id), "exchange_id": exchange_id, "api_key": _encrypted(enc_api_key),
              "api_secret": _encrypted(enc_api_secret), "api_passphrase": _encrypted(enc_api_passphrase),
              "testnet": bool(testnet), "created_at": datetime.now(timezone.utc).isoformat()}
    raw = json.dumps(record).encode("utf-8")
    driver = _configured_driver()
    if driver == "kv":
        return _kv_ok("SET", f"exchange_key:{int(user_id)}:{exchange}", raw.decode()) and _kv_ok("SADD", f"user:{int(user_id)}:exchanges", exchange)
    if driver == "turso":
        return _turso_init() and _turso_execute(
            "INSERT INTO remote_exchange_keys(user_id,exchange_id,api_key,api_secret,api_passphrase,testnet,created_at) VALUES(?,?,?,?,?,?,?) ON CONFLICT(user_id,exchange_id) DO UPDATE SET api_key=excluded.api_key,api_secret=excluded.api_secret,api_passphrase=excluded.api_passphrase,testnet=excluded.testnet,created_at=excluded.created_at",
            (int(user_id), exchange_id, record["api_key"], record["api_secret"], record["api_passphrase"], int(bool(testnet)), record["created_at"]),
        )[0]
    return _blob_put(f"exchange_keys/{int(user_id)}/{exchange}.json", raw)


def remote_get_exchange_keys(user_id: int) -> List[Dict[str, Any]]:
    if not is_remote_storage_available():
        return []
    driver = _configured_driver()
    if driver == "kv":
        ok, exchanges = _kv_command("SMEMBERS", f"user:{int(user_id)}:exchanges")
        if not ok or not exchanges:
            return []
        result = []
        for exchange in exchanges:
            ok, raw = _kv_command("GET", f"exchange_key:{int(user_id)}:{_component(exchange)}")
            if ok and raw:
                try:
                    result.append(json.loads(raw) if isinstance(raw, str) else raw)
                except (TypeError, json.JSONDecodeError):
                    print(f"[REMOTE STORE WARNING] Invalid exchange record for user {user_id}")
        return result
    if driver == "turso":
        if not _turso_init():
            return []
        ok, rows = _turso_execute("SELECT user_id,exchange_id,api_key,api_secret,api_passphrase,testnet,created_at FROM remote_exchange_keys WHERE user_id=?", (int(user_id),))
        return rows if ok else []
    # Blob has no list index: list objects under a user cannot be enumerated by
    # the simple REST API, so retain a small index but fail if it cannot update.
    raw = _blob_get(f"exchange_keys/{int(user_id)}/list.json")
    if not raw:
        return []
    try:
        names = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return []
    result = []
    for name in names if isinstance(names, list) else []:
        obj = _json_blob(f"exchange_keys/{int(user_id)}/{_component(name)}.json")
        if obj:
            result.append(obj)
    return result


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------
def remote_delete_account(user_id: int, username: str = "", email: str = "") -> bool:
    if not is_remote_storage_available():
        return False
    driver = _configured_driver()
    user = remote_get_user(str(user_id)) if not username or not email else None
    username = username or (user or {}).get("username", "")
    email = email or (user or {}).get("email", "")
    if driver == "kv":
        ok, exchanges = _kv_command("SMEMBERS", f"user:{int(user_id)}:exchanges")
        operations = [_kv_ok("DEL", f"user:by_id:{int(user_id)}"), _kv_ok("DEL", f"user:{int(user_id)}:exchanges"), _kv_ok("SREM", "users:all_ids", str(int(user_id)))]
        if username:
            name = _component(username); operations += [_kv_ok("DEL", f"user:by_username:{name}"), _kv_ok("SREM", "users:all_usernames", name)]
        if email:
            mail = _component(email); operations += [_kv_ok("DEL", f"user:by_email:{mail}"), _kv_ok("SREM", "users:all_emails", mail)]
        for exchange in exchanges or []:
            operations.append(_kv_ok("DEL", f"exchange_key:{int(user_id)}:{_component(exchange)}"))
        return all(operations)
    if driver == "turso":
        if not _turso_init():
            return False
        return _turso_execute("DELETE FROM remote_exchange_keys WHERE user_id=?", (int(user_id),))[0] and _turso_execute("DELETE FROM remote_users WHERE id=?", (int(user_id),))[0]
    paths = [f"users/by_id/{int(user_id)}.json"]
    if username: paths.append(f"users/by_username/{_component(username)}.json")
    if email: paths.append(f"users/by_email/{_component(email)}.json")
    raw = _blob_get(f"exchange_keys/{int(user_id)}/list.json")
    exchanges = []
    if raw:
        try: exchanges = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError): pass
    paths += [f"exchange_keys/{int(user_id)}/{_component(x)}.json" for x in exchanges if isinstance(x, str)]
    paths.append(f"exchange_keys/{int(user_id)}/list.json")
    return all(_blob_del(path) for path in paths)


def remote_delete_all_accounts() -> bool:
    if not is_remote_storage_available():
        return False
    driver = _configured_driver()
    if driver == "turso":
        return _turso_init() and _turso_execute("DELETE FROM remote_exchange_keys")[0] and _turso_execute("DELETE FROM remote_users")[0]
    if driver == "kv":
        ok, ids = _kv_command("SMEMBERS", "users:all_ids")
        if not ok:
            return False
        success = all(remote_delete_account(int(uid)) for uid in ids or [] if str(uid).isdigit())
        return success and _kv_ok("DEL", "users:all_ids") and _kv_ok("DEL", "users:all_usernames") and _kv_ok("DEL", "users:all_emails")
    # Blob's index is authoritative for bulk deletion.
    index = _json_blob("users/index.json") or {}
    success = all(remote_delete_account(int(uid)) for uid in index.get("ids", []) if str(uid).isdigit())
    return success and _blob_del("users/index.json")
