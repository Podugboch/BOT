"""
Remote Storage Adapter for Cloud Serverless Environments (Vercel, AWS Lambda).
Provides persistent cross-container storage synchronization for user accounts,
exchange credentials, and system settings using standard HTTP REST (no external dependencies).

Supported Drivers:
- Vercel Blob Storage REST API (BLOB_READ_WRITE_TOKEN / BLOB_STORE_ID)
- Vercel KV / Upstash Redis REST API (KV_REST_API_URL / UPSTASH_REDIS_REST_URL)
- Turso libSQL HTTP API (TURSO_DATABASE_URL + TURSO_AUTH_TOKEN)
"""
import os
import json
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

KV_URL = (os.getenv("KV_REST_API_URL") or os.getenv("UPSTASH_REDIS_REST_URL") or "").strip().rstrip("/")
KV_TOKEN = (os.getenv("KV_REST_API_TOKEN") or os.getenv("UPSTASH_REDIS_REST_TOKEN") or "").strip()

TURSO_URL = (os.getenv("TURSO_DATABASE_URL") or "").strip()
if TURSO_URL.startswith("libsql://"):
    TURSO_URL = "https://" + TURSO_URL[len("libsql://"):]
TURSO_URL = TURSO_URL.rstrip("/")
TURSO_TOKEN = (os.getenv("TURSO_AUTH_TOKEN") or "").strip()

BLOB_TOKEN = (os.getenv("BLOB_READ_WRITE_TOKEN") or "").strip()
BLOB_STORE_ID = (os.getenv("BLOB_STORE_ID") or "").strip()


def is_remote_storage_available() -> bool:
    """Returns True if any remote persistent storage driver is configured."""
    return bool((KV_URL and KV_TOKEN) or (TURSO_URL and TURSO_TOKEN) or BLOB_TOKEN)


def get_remote_storage_type() -> str:
    if KV_URL and KV_TOKEN:
        return "kv"
    if TURSO_URL and TURSO_TOKEN:
        return "turso"
    if BLOB_TOKEN:
        return "blob"
    return "none"


# ---------------------------------------------------------------------------
# Vercel Blob Client Helpers
# ---------------------------------------------------------------------------
def _get_blob_store_id() -> str:
    """
    Derives the Vercel Blob store id from BLOB_STORE_ID, or parses it out of
    BLOB_READ_WRITE_TOKEN (format: vercel_blob_rw_<storeid>_<random>).

    No hardcoded fallback id here on purpose: guessing a store id wrong means
    either silently failing against a store that isn't yours, or -- worse --
    silently succeeding against one that is, but isn't the one you meant to
    use. Fail loudly instead so a misconfiguration is obvious immediately,
    not discovered later as "why is my data missing".
    """
    global BLOB_STORE_ID
    if BLOB_STORE_ID:
        return BLOB_STORE_ID.replace("store_", "").lower()
    if BLOB_TOKEN:
        parts = BLOB_TOKEN.split("_")
        if len(parts) >= 4:
            return parts[3].lower()
    raise RuntimeError(
        "Could not determine the Vercel Blob store id. Set BLOB_STORE_ID "
        "explicitly, or check that BLOB_READ_WRITE_TOKEN is the token Vercel "
        "gives you (format: vercel_blob_rw_<storeid>_<random>)."
    )


def _blob_put(pathname: str, data_bytes: bytes, content_type: str = "application/json") -> bool:
    if not BLOB_TOKEN:
        return False
    try:
        url = f"https://blob.vercel-storage.com/?pathname={urllib.parse.quote(pathname)}"
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={
                "authorization": f"Bearer {BLOB_TOKEN}",
                "x-api-version": "12",
                "x-vercel-blob-access": "private",
                "x-add-random-suffix": "0",
                "x-allow-overwrite": "1",
                "content-type": content_type
            },
            method="PUT"
        )
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            return resp.status in (200, 201)
    except Exception as e:
        print(f"[BLOB PUT WARNING] Error saving {pathname}: {e}")
        return False


def _blob_get(pathname: str) -> Optional[bytes]:
    if not BLOB_TOKEN:
        return None
    try:
        store_id = _get_blob_store_id()
    except RuntimeError as e:
        print(f"[BLOB GET WARNING] {e}")
        return None
    try:
        url = f"https://{store_id}.private.blob.vercel-storage.com/{pathname}?cache=0"
        req = urllib.request.Request(
            url,
            headers={
                "authorization": f"Bearer {BLOB_TOKEN}",
            },
            method="GET"
        )
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            if resp.status == 200:
                return resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        print(f"[BLOB GET WARNING] HTTP error {e.code} for {pathname}")
        return None
    except Exception as e:
        print(f"[BLOB GET WARNING] Error reading {pathname}: {e}")
        return None
    return None


def _blob_del(pathname: str) -> bool:
    if not BLOB_TOKEN:
        return False
    try:
        store_id = _get_blob_store_id()
    except RuntimeError as e:
        print(f"[BLOB DEL WARNING] {e}")
        return False
    try:
        blob_url = f"https://{store_id}.private.blob.vercel-storage.com/{pathname}"
        del_url = "https://blob.vercel-storage.com/delete"
        del_data = json.dumps({"urls": [blob_url]}).encode("utf-8")
        req = urllib.request.Request(
            del_url,
            data=del_data,
            headers={
                "authorization": f"Bearer {BLOB_TOKEN}",
                "x-api-version": "12",
                "content-type": "application/json"
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            return resp.status in (200, 204)
    except Exception as e:
        print(f"[BLOB DEL WARNING] Error deleting {pathname}: {e}")
        return False


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
    rec_bytes = rec_str.encode("utf-8")
    saved = False

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
        saved = True

    if BLOB_TOKEN:
        u_clean = username.strip().lower()
        e_clean = email.strip().lower()
        _blob_put(f"users/by_username/{u_clean}.json", rec_bytes)
        _blob_put(f"users/by_email/{e_clean}.json", rec_bytes)
        _blob_put(f"users/by_id/{user_id}.json", rec_bytes)

        try:
            index_raw = _blob_get("users/index.json")
            index = json.loads(index_raw.decode("utf-8")) if index_raw else {"usernames": [], "emails": [], "ids": []}
            if u_clean not in index.get("usernames", []):
                index.setdefault("usernames", []).append(u_clean)
            if e_clean not in index.get("emails", []):
                index.setdefault("emails", []).append(e_clean)
            if user_id not in index.get("ids", []):
                index.setdefault("ids", []).append(user_id)
            _blob_put("users/index.json", json.dumps(index).encode("utf-8"))
        except Exception:
            pass
        saved = True

    return saved


def remote_get_user(val: str) -> Optional[Dict[str, Any]]:
    """Retrieves a user record by username, email, or user_id from the remote persistent store."""
    if not is_remote_storage_available():
        return None

    val_str = str(val).strip().lower()

    if KV_URL and KV_TOKEN:
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

    if BLOB_TOKEN:
        raw_bytes = None
        if "@" in val_str:
            raw_bytes = _blob_get(f"users/by_email/{val_str}.json")
        elif val_str.isdigit():
            raw_bytes = _blob_get(f"users/by_id/{val_str}.json")
            if not raw_bytes:
                raw_bytes = _blob_get(f"users/by_username/{val_str}.json")
        else:
            raw_bytes = _blob_get(f"users/by_username/{val_str}.json")
            if not raw_bytes:
                raw_bytes = _blob_get(f"users/by_email/{val_str}.json")

        if raw_bytes:
            try:
                return json.loads(raw_bytes.decode("utf-8"))
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
    rec_bytes = rec_str.encode("utf-8")
    saved = False

    if KV_URL and KV_TOKEN:
        k = f"exchange_key:{user_id}:{exchange_id.lower()}"
        _kv_command("SET", k, rec_str)
        _kv_command("SADD", f"user:{user_id}:exchanges", exchange_id.lower())
        saved = True

    if BLOB_TOKEN:
        ex_clean = exchange_id.strip().lower()
        _blob_put(f"exchange_keys/{user_id}/{ex_clean}.json", rec_bytes)
        try:
            list_raw = _blob_get(f"exchange_keys/{user_id}/list.json")
            ex_list = json.loads(list_raw.decode("utf-8")) if list_raw else []
            if ex_clean not in ex_list:
                ex_list.append(ex_clean)
                _blob_put(f"exchange_keys/{user_id}/list.json", json.dumps(ex_list).encode("utf-8"))
        except Exception:
            pass
        saved = True

    return saved


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

    if BLOB_TOKEN:
        list_raw = _blob_get(f"exchange_keys/{user_id}/list.json")
        if not list_raw:
            return []
        try:
            ex_list = json.loads(list_raw.decode("utf-8"))
            keys = []
            for ex in ex_list:
                key_raw = _blob_get(f"exchange_keys/{user_id}/{ex}.json")
                if key_raw:
                    keys.append(json.loads(key_raw.decode("utf-8")))
            return keys
        except Exception:
            return []

    return []


def remote_delete_account(user_id: int, username: str = "", email: str = "") -> bool:
    """Deletes a user account and associated credentials from remote persistent store."""
    if not is_remote_storage_available():
        return False

    deleted = False

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
        deleted = True

    if BLOB_TOKEN:
        if not username or not email:
            u = remote_get_user(str(user_id))
            if u:
                username = u.get("username", "")
                email = u.get("email", "")

        u_clean = username.strip().lower() if username else ""
        e_clean = email.strip().lower() if email else ""

        if u_clean:
            _blob_del(f"users/by_username/{u_clean}.json")
        if e_clean:
            _blob_del(f"users/by_email/{e_clean}.json")
        _blob_del(f"users/by_id/{user_id}.json")

        list_raw = _blob_get(f"exchange_keys/{user_id}/list.json")
        if list_raw:
            try:
                ex_list = json.loads(list_raw.decode("utf-8"))
                for ex in ex_list:
                    _blob_del(f"exchange_keys/{user_id}/{ex}.json")
            except Exception:
                pass
            _blob_del(f"exchange_keys/{user_id}/list.json")

        try:
            index_raw = _blob_get("users/index.json")
            if index_raw:
                idx = json.loads(index_raw.decode("utf-8"))
                if u_clean and u_clean in idx.get("usernames", []):
                    idx["usernames"].remove(u_clean)
                if e_clean and e_clean in idx.get("emails", []):
                    idx["emails"].remove(e_clean)
                if user_id in idx.get("ids", []):
                    idx["ids"].remove(user_id)
                _blob_put("users/index.json", json.dumps(idx).encode("utf-8"))
        except Exception:
            pass
        deleted = True

    return deleted


def remote_delete_all_accounts() -> bool:
    """Purges all user accounts and exchange credentials from remote persistent store."""
    if not is_remote_storage_available():
        return False

    deleted = False

    if KV_URL and KV_TOKEN:
        ids = _kv_command("SMEMBERS", "users:all_ids") or []
        for uid in ids:
            remote_delete_account(int(uid))
        _kv_command("DEL", "users:all_ids")
        _kv_command("DEL", "users:all_usernames")
        _kv_command("DEL", "users:all_emails")
        deleted = True

    if BLOB_TOKEN:
        try:
            index_raw = _blob_get("users/index.json")
            if index_raw:
                idx = json.loads(index_raw.decode("utf-8"))
                for uid in idx.get("ids", []):
                    remote_delete_account(int(uid))
                for u in idx.get("usernames", []):
                    _blob_del(f"users/by_username/{u}.json")
                for e in idx.get("emails", []):
                    _blob_del(f"users/by_email/{e}.json")
            _blob_del("users/index.json")
        except Exception:
            pass
        deleted = True

    return deleted
