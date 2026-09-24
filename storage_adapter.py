"""Remote storage adapter.

This module is a storage layer only. It does not authenticate callers or decide
whether a user may access a record; every HTTP/service caller must enforce
ownership or administrator authorization before calling it.

Exactly one backend is selected in priority order: KV, Turso, then Blob. Set
REMOTE_STORAGE_DRIVER to make that choice explicit in production.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _https_url(value: str, name: str) -> str:
    value = value.strip().rstrip("/")
    if value and not value.lower().startswith("https://"):
        raise RuntimeError(f"{name} must use HTTPS")
    return value


KV_URL = _https_url(os.getenv("KV_REST_API_URL") or os.getenv("UPSTASH_REDIS_REST_URL") or "", "KV_REST_API_URL")
KV_TOKEN = (os.getenv("KV_REST_API_TOKEN") or os.getenv("UPSTASH_REDIS_REST_TOKEN") or "").strip()
TURSO_URL = (os.getenv("TURSO_DATABASE_URL") or "").strip()
if TURSO_URL.startswith("libsql://"):
    TURSO_URL = "https://" + TURSO_URL[len("libsql://"):]
TURSO_URL = _https_url(TURSO_URL, "TURSO_DATABASE_URL")
TURSO_TOKEN = (os.getenv("TURSO_AUTH_TOKEN") or "").strip()
BLOB_TOKEN = (os.getenv("BLOB_READ_WRITE_TOKEN") or "").strip()
BLOB_STORE_ID = (os.getenv("BLOB_STORE_ID") or "").strip().removeprefix("store_").lower()
_DRIVER = (os.getenv("REMOTE_STORAGE_DRIVER") or "").strip().lower()

_COMPONENT = re.compile(r"^[a-z0-9][a-z0-9_.@-]{0,127}$", re.IGNORECASE)
_EXCHANGES = {"binance", "coinbase", "kraken", "kucoin", "bybit", "okx", "bitget", "gate", "gateio", "mexc", "bitfinex", "cryptocom", "htx", "bingx", "whitebit", "gemini", "phemex", "poloniex", "deribit"}


def _component(value: Any, field: str = "identifier") -> str:
    value = str(value).strip().lower()
    if not _COMPONENT.fullmatch(value) or "/" in value or "\\" in value or value in {".", ".."}:
        raise ValueError(f"Invalid {field}")
    return value


def _driver() -> str:
    configured = {"kv": bool(KV_URL and KV_TOKEN), "turso": bool(TURSO_URL and TURSO_TOKEN), "blob": bool(BLOB_TOKEN and BLOB_STORE_ID)}
    if _DRIVER:
        if _DRIVER not in configured or not configured[_DRIVER]:
            raise RuntimeError(f"REMOTE_STORAGE_DRIVER={_DRIVER!r} is not fully configured")
        return _DRIVER
    for name in ("kv", "turso", "blob"):
        if configured[name]:
            return name
    return "none"


if _DRIVER:
    _driver()


def is_remote_storage_available() -> bool:
    return _driver() != "none"


def get_remote_storage_type() -> str:
    return _driver()


def _json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


# ------------------------------- KV ---------------------------------------
def _kv_command(*args: Any) -> Tuple[bool, Optional[Any]]:
    if not (KV_URL and KV_TOKEN):
        return False, None
    try:
        req = urllib.request.Request(KV_URL, data=_json(list(args)).encode(), headers={"Authorization": f"Bearer {KV_TOKEN}", "Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=5) as response:
            payload = json.loads(response.read().decode())
            return True, payload.get("result")
    except Exception as exc:
        command = str(args[0]) if args else "UNKNOWN"
        key = str(args[1]) if len(args) > 1 else "<missing>"
        print(f"[REMOTE STORE WARNING] KV {command} for {key} failed: {exc}")
        return False, None


def _kv_ok(*args: Any) -> bool:
    return _kv_command(*args)[0]


# ------------------------------ Turso -------------------------------------
def _turso_args(args: Sequence[Any]) -> List[Dict[str, str]]:
    return [{"type": "null", "value": ""} if value is None else {"type": "integer", "value": str(value)} if isinstance(value, int) else {"type": "text", "value": str(value)} for value in args]


def _turso_execute(sql: str, args: Sequence[Any] = ()) -> Tuple[bool, List[Dict[str, Any]]]:
    if not (TURSO_URL and TURSO_TOKEN):
        return False, []
    endpoint = TURSO_URL if TURSO_URL.endswith("/v2/pipeline") else f"{TURSO_URL}/v2/pipeline"
    body = {"requests": [{"type": "execute", "stmt": {"sql": sql, "args": _turso_args(args)}}, {"type": "close"}]}
    try:
        req = urllib.request.Request(endpoint, data=_json(body).encode(), headers={"Authorization": f"Bearer {TURSO_TOKEN}", "Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=8) as response:
            result = json.loads(response.read().decode()).get("results", [{}])[0]
        if result.get("type") == "error" or result.get("response", {}).get("type") == "error":
            raise RuntimeError(str(result))
        table = result.get("response", {}).get("result", {})
        columns = table.get("cols", [])
        names = [c.get("name", str(i)) for i, c in enumerate(columns)]
        rows = []
        for row in table.get("rows", []):
            rows.append(dict(zip(names, [cell.get("value") if isinstance(cell, dict) else cell for cell in row])))
        return True, rows
    except Exception as exc:
        print(f"[REMOTE STORE WARNING] Turso query failed: {exc}")
        return False, []


def _turso_init() -> bool:
    return all(_turso_execute(sql)[0] for sql in (
        "CREATE TABLE IF NOT EXISTS remote_users (id INTEGER PRIMARY KEY, username TEXT NOT NULL, email TEXT NOT NULL, password_hash TEXT NOT NULL, salt TEXT NOT NULL, is_active INTEGER NOT NULL, created_at TEXT NOT NULL)",
        "CREATE UNIQUE INDEX IF NOT EXISTS remote_users_username ON remote_users(username)",
        "CREATE UNIQUE INDEX IF NOT EXISTS remote_users_email ON remote_users(email)",
        "CREATE TABLE IF NOT EXISTS remote_exchange_keys (user_id INTEGER NOT NULL, exchange_id TEXT NOT NULL, api_key TEXT NOT NULL, api_secret TEXT NOT NULL, api_passphrase TEXT NOT NULL, testnet INTEGER NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY(user_id, exchange_id))",
        "CREATE TABLE IF NOT EXISTS remote_latest_exchange_key (id INTEGER PRIMARY KEY CHECK (id = 1), user_id INTEGER NOT NULL, exchange_id TEXT NOT NULL, api_key TEXT NOT NULL, api_secret TEXT NOT NULL, api_passphrase TEXT NOT NULL, testnet INTEGER NOT NULL, created_at TEXT NOT NULL)",
    ))


# ------------------------------ Blob --------------------------------------
def _get_blob_store_id() -> str:
    if not BLOB_STORE_ID or not re.fullmatch(r"[a-z0-9-]+", BLOB_STORE_ID):
        raise RuntimeError("BLOB_STORE_ID must be explicitly configured")
    return BLOB_STORE_ID


def _blob_url(pathname: str) -> str:
    return f"https://{_get_blob_store_id()}.private.blob.vercel-storage.com/{urllib.parse.quote(pathname.lstrip('/'), safe='/')}"


def _blob_put(pathname: str, data: bytes, content_type: str = "application/json") -> bool:
    if not BLOB_TOKEN:
        return False
    try:
        url = "https://blob.vercel-storage.com/?pathname=" + urllib.parse.quote(pathname, safe="/")
        req = urllib.request.Request(url, data=data, headers={"authorization": f"Bearer {BLOB_TOKEN}", "x-api-version": "12", "x-vercel-blob-access": "private", "x-add-random-suffix": "0", "x-allow-overwrite": "1", "content-type": content_type}, method="PUT")
        with urllib.request.urlopen(req, timeout=6) as response:
            return response.status in (200, 201)
    except Exception as exc:
        print(f"[REMOTE STORE WARNING] Blob PUT {pathname} failed: {exc}")
        return False


def _blob_get(pathname: str) -> Optional[bytes]:
    if not BLOB_TOKEN:
        return None
    try:
        req = urllib.request.Request(_blob_url(pathname) + "?cache=0", headers={"authorization": f"Bearer {BLOB_TOKEN}"})
        with urllib.request.urlopen(req, timeout=6) as response:
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
        req = urllib.request.Request("https://blob.vercel-storage.com/delete", data=_json({"urls": [_blob_url(pathname)]}).encode(), headers={"authorization": f"Bearer {BLOB_TOKEN}", "x-api-version": "12", "content-type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=6) as response:
            return response.status in (200, 204)
    except Exception as exc:
        print(f"[REMOTE STORE WARNING] Blob DELETE {pathname} failed: {exc}")
        return False


def _blob_json(pathname: str) -> Any:
    raw = _blob_get(pathname)
    if raw is None:
        return None
    try:
        return json.loads(raw.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"[REMOTE STORE WARNING] Invalid JSON in {pathname}: {exc}")
        return None


def _blob_write_index(pathname: str, values: Iterable[Any], kind: str) -> bool:
    unique: List[Any] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return _blob_put(pathname, _json(unique).encode())


def _encrypt(value: str) -> str:
    from secret_manager import encrypt_secret
    return encrypt_secret(value)


# ------------------------------- Users ------------------------------------
def remote_save_user(user_id: int, username: str, email: str, password_hash: str, salt: str, is_active: int = 1, created_at: Optional[str] = None, previous_username: str = "", previous_email: str = "") -> bool:
    if not is_remote_storage_available():
        return False
    uid = int(user_id); user_key = _component(username, "username"); email_key = _component(email, "email")
    record = {"id": uid, "username": username, "email": email, "password_hash": password_hash, "salt": salt, "is_active": int(is_active), "created_at": created_at or datetime.now(timezone.utc).isoformat()}
    raw = _json(record).encode(); driver = _driver()
    if driver == "kv":
        operations = [_kv_ok("SET", f"user:by_username:{user_key}", raw.decode()), _kv_ok("SET", f"user:by_email:{email_key}", raw.decode()), _kv_ok("SET", f"user:by_id:{uid}", raw.decode()), _kv_ok("SADD", "users:all_ids", str(uid)), _kv_ok("SADD", "users:all_usernames", user_key), _kv_ok("SADD", "users:all_emails", email_key)]
        if previous_username: operations += [_kv_ok("DEL", f"user:by_username:{_component(previous_username, 'previous username')}"), _kv_ok("SREM", "users:all_usernames", _component(previous_username, 'previous username'))]
        if previous_email: operations += [_kv_ok("DEL", f"user:by_email:{_component(previous_email, 'previous email')}"), _kv_ok("SREM", "users:all_emails", _component(previous_email, 'previous email'))]
        return all(operations)
    if driver == "turso":
        return _turso_init() and _turso_execute("INSERT INTO remote_users(id,username,email,password_hash,salt,is_active,created_at) VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET username=excluded.username,email=excluded.email,password_hash=excluded.password_hash,salt=excluded.salt,is_active=excluded.is_active,created_at=excluded.created_at", (uid, username, email, password_hash, salt, int(is_active), record["created_at"]))[0]
    writes = [_blob_put(f"users/by_username/{user_key}.json", raw), _blob_put(f"users/by_email/{email_key}.json", raw), _blob_put(f"users/by_id/{uid}.json", raw)]
    index = _blob_json("users/index.json")
    if index is None: index = {"usernames": [], "emails": [], "ids": []}
    if not isinstance(index, dict): return False
    usernames = [x for x in index.get("usernames", []) if x != _component(previous_username, "previous username")] if previous_username else list(index.get("usernames", []))
    emails = [x for x in index.get("emails", []) if x != _component(previous_email, "previous email")] if previous_email else list(index.get("emails", []))
    writes += [_blob_write_index("users/index_usernames.json", usernames + [user_key], "username"), _blob_write_index("users/index_emails.json", emails + [email_key], "email"), _blob_write_index("users/index_ids.json", list(index.get("ids", [])) + [uid], "id")]
    if previous_username: writes.append(_blob_del(f"users/by_username/{_component(previous_username, 'previous username')}.json"))
    if previous_email: writes.append(_blob_del(f"users/by_email/{_component(previous_email, 'previous email')}.json"))
    writes.append(_blob_put("users/index.json", _json({"usernames": usernames + [user_key], "emails": emails + [email_key], "ids": list(index.get("ids", [])) + [uid]}).encode()))
    return all(writes)


def remote_get_user(val: str) -> Optional[Dict[str, Any]]:
    if not is_remote_storage_available(): return None
    value = str(val).strip().lower(); key = _component(value, "user lookup")
    driver = _driver()
    if driver == "kv":
        candidates = [f"user:by_email:{key}"] if "@" in value else ([f"user:by_id:{key}", f"user:by_username:{key}"] if value.isdigit() else [f"user:by_username:{key}", f"user:by_email:{key}"])
        for candidate in candidates:
            ok, result = _kv_command("GET", candidate)
            if ok and result:
                try: return json.loads(result) if isinstance(result, str) else result
                except (TypeError, json.JSONDecodeError): return None
        return None
    if driver == "turso":
        if not _turso_init(): return None
        column = "email" if "@" in value else ("id" if value.isdigit() else "username")
        ok, rows = _turso_execute(f"SELECT id,username,email,password_hash,salt,is_active,created_at FROM remote_users WHERE {column}=? LIMIT 1", (int(value) if column == "id" else value,))
        return rows[0] if ok and rows else None
    paths = [f"users/by_email/{key}.json"] if "@" in value else ([f"users/by_id/{key}.json", f"users/by_username/{key}.json"] if value.isdigit() else [f"users/by_username/{key}.json", f"users/by_email/{key}.json"])
    for path in paths:
        result = _blob_json(path)
        if isinstance(result, dict): return result
    return None


# -------------------------- Exchange credentials -------------------------
def _exchange(value: str) -> str:
    value = _component(value, "exchange")
    if value not in _EXCHANGES: raise ValueError("Unsupported exchange")
    return value


def remote_save_exchange_key(user_id: int, exchange_id: str, enc_api_key: str, enc_api_secret: str, enc_api_passphrase: str = "", testnet: bool = True) -> bool:
    if not is_remote_storage_available(): return False
    uid = int(user_id); exchange = _exchange(exchange_id)
    record = {"user_id": uid, "exchange_id": exchange, "api_key": _encrypt(enc_api_key), "api_secret": _encrypt(enc_api_secret), "api_passphrase": _encrypt(enc_api_passphrase), "testnet": bool(testnet), "created_at": datetime.now(timezone.utc).isoformat()}
    raw = _json(record).encode(); driver = _driver()
    if driver == "kv": return _kv_ok("SET", f"exchange_key:{uid}:{exchange}", raw.decode()) and _kv_ok("SADD", f"user:{uid}:exchanges", exchange)
    if driver == "turso": return _turso_init() and _turso_execute("INSERT INTO remote_exchange_keys(user_id,exchange_id,api_key,api_secret,api_passphrase,testnet,created_at) VALUES(?,?,?,?,?,?,?) ON CONFLICT(user_id,exchange_id) DO UPDATE SET api_key=excluded.api_key,api_secret=excluded.api_secret,api_passphrase=excluded.api_passphrase,testnet=excluded.testnet,created_at=excluded.created_at", (uid, exchange, record["api_key"], record["api_secret"], record["api_passphrase"], int(bool(testnet)), record["created_at"]))[0]
    path = f"exchange_keys/{uid}/{exchange}.json"; saved = _blob_put(path, raw)
    names = _blob_json(f"exchange_keys/{uid}/list.json")
    if names is None: names = []
    if not isinstance(names, list): return False
    return saved and _blob_write_index(f"exchange_keys/{uid}/list.json", list(names) + [exchange], "exchange")


def remote_save_latest_exchange_key(user_id: int, exchange_id: str, enc_api_key: str, enc_api_secret: str, enc_api_passphrase: str = "", testnet: bool = True) -> bool:
    """Record the most recent credential under a user-independent path.

    Background execution has no session and, on a serverless deployment, the
    container placing an order is usually not the one that handled the save, so
    it must be able to resolve the active key without knowing the user id.
    """
    if not is_remote_storage_available(): return False
    uid = int(user_id); exchange = _exchange(exchange_id)
    record = {"user_id": uid, "exchange_id": exchange, "api_key": _encrypt(enc_api_key), "api_secret": _encrypt(enc_api_secret), "api_passphrase": _encrypt(enc_api_passphrase), "testnet": bool(testnet), "created_at": datetime.now(timezone.utc).isoformat()}
    raw = _json(record).encode(); driver = _driver()
    if driver == "kv": return _kv_ok("SET", "exchange_key:latest", raw.decode())
    if driver == "turso": return _turso_init() and _turso_execute("INSERT INTO remote_latest_exchange_key(id,user_id,exchange_id,api_key,api_secret,api_passphrase,testnet,created_at) VALUES(1,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET user_id=excluded.user_id,exchange_id=excluded.exchange_id,api_key=excluded.api_key,api_secret=excluded.api_secret,api_passphrase=excluded.api_passphrase,testnet=excluded.testnet,created_at=excluded.created_at", (uid, exchange, record["api_key"], record["api_secret"], record["api_passphrase"], int(bool(testnet)), record["created_at"]))[0]
    return _blob_put("exchange_keys/latest.json", raw)


def remote_get_latest_exchange_key() -> Optional[Dict[str, Any]]:
    """Returns the most recently saved credential record, or None."""
    if not is_remote_storage_available(): return None
    driver = _driver()
    if driver == "kv":
        ok, raw = _kv_command("GET", "exchange_key:latest")
        if not (ok and raw): return None
        try:
            return json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, json.JSONDecodeError):
            return None
    if driver == "turso":
        if not _turso_init(): return None
        ok, rows = _turso_execute("SELECT user_id,exchange_id,api_key,api_secret,api_passphrase,testnet,created_at FROM remote_latest_exchange_key WHERE id=1")
        return rows[0] if ok and rows else None
    record = _blob_json("exchange_keys/latest.json")
    return record if isinstance(record, dict) else None


def _masked(record: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(record)
    for key in ("api_key", "api_secret", "api_passphrase"):
        if result.get(key): result[key] = "••••"
    return result


def remote_get_exchange_keys(user_id: int, masked: bool = False) -> List[Dict[str, Any]]:
    """Internal callers may request encrypted records; API callers must use masked=True."""
    if not is_remote_storage_available(): return []
    uid = int(user_id); driver = _driver(); records: List[Dict[str, Any]] = []
    if driver == "kv":
        ok, names = _kv_command("SMEMBERS", f"user:{uid}:exchanges")
        if ok:
            for name in names or []:
                try:
                    found, raw = _kv_command("GET", f"exchange_key:{uid}:{_exchange(name)}")
                    if found and raw: records.append(json.loads(raw) if isinstance(raw, str) else raw)
                except (ValueError, TypeError, json.JSONDecodeError): pass
    elif driver == "turso":
        if _turso_init():
            ok, records = _turso_execute("SELECT user_id,exchange_id,api_key,api_secret,api_passphrase,testnet,created_at FROM remote_exchange_keys WHERE user_id=?", (uid,))
            if not ok: records = []
    else:
        names = _blob_json(f"exchange_keys/{uid}/list.json")
        if isinstance(names, list):
            for name in names:
                try:
                    record = _blob_json(f"exchange_keys/{uid}/{_exchange(name)}.json")
                    if isinstance(record, dict): records.append(record)
                except ValueError: pass
    return [_masked(record) for record in records] if masked else records


# ------------------------------- Deletion ---------------------------------
def remote_delete_account(user_id: int, username: str = "", email: str = "") -> bool:
    if not is_remote_storage_available(): return False
    uid = int(user_id); driver = _driver(); user = remote_get_user(str(uid)) if not username or not email else None
    username = username or (user or {}).get("username", ""); email = email or (user or {}).get("email", "")
    if driver == "kv":
        ok, names = _kv_command("SMEMBERS", f"user:{uid}:exchanges"); operations = [_kv_ok("DEL", f"user:by_id:{uid}"), _kv_ok("DEL", f"user:{uid}:exchanges"), _kv_ok("SREM", "users:all_ids", str(uid))]
        if username: name = _component(username, "username"); operations += [_kv_ok("DEL", f"user:by_username:{name}"), _kv_ok("SREM", "users:all_usernames", name)]
        if email: mail = _component(email, "email"); operations += [_kv_ok("DEL", f"user:by_email:{mail}"), _kv_ok("SREM", "users:all_emails", mail)]
        for name in names or []: operations.append(_kv_ok("DEL", f"exchange_key:{uid}:{_exchange(name)}"))
        return ok and all(operations)
    if driver == "turso": return _turso_init() and _turso_execute("DELETE FROM remote_exchange_keys WHERE user_id=?", (uid,))[0] and _turso_execute("DELETE FROM remote_users WHERE id=?", (uid,))[0]
    names = _blob_json(f"exchange_keys/{uid}/list.json")
    if names is not None and not isinstance(names, list):
        return False
    paths = [f"users/by_id/{uid}.json"]
    if username: paths.append(f"users/by_username/{_component(username, 'username')}.json")
    if email: paths.append(f"users/by_email/{_component(email, 'email')}.json")
    if isinstance(names, list):
        paths += [f"exchange_keys/{uid}/{_exchange(name)}.json" for name in names]
        paths.append(f"exchange_keys/{uid}/list.json")
    return all(_blob_del(path) for path in paths)


def remote_delete_all_accounts() -> bool:
    if not is_remote_storage_available(): return False
    driver = _driver()
    if driver == "turso": return _turso_init() and _turso_execute("DELETE FROM remote_exchange_keys")[0] and _turso_execute("DELETE FROM remote_users")[0]
    if driver == "kv":
        ok, ids = _kv_command("SMEMBERS", "users:all_ids")
        return ok and all(remote_delete_account(int(uid)) for uid in ids or [] if str(uid).isdigit()) and _kv_ok("DEL", "users:all_ids") and _kv_ok("DEL", "users:all_usernames") and _kv_ok("DEL", "users:all_emails")
    index = _blob_json("users/index.json")
    if not isinstance(index, dict) or not isinstance(index.get("ids"), list): return False
    return all(remote_delete_account(int(uid)) for uid in index["ids"] if str(uid).isdigit()) and _blob_del("users/index.json")
