"""
Authentication and User Account Management module for Trader Queen.
Provides:
- SQLite persistence (file or /tmp for serverless Vercel)
- Secure password hashing with PBKDF2-HMAC-SHA256 and unique salt
- Email account activation flow (with SMTP and in-app preview)
- Secure password reset flow
- Per-user exchange API key encryption/masking & management
"""
import os
import re
import json
import sqlite3
import hashlib
import secrets
import smtplib
import ssl
import socket
from datetime import datetime, timezone, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import html
import urllib.request
import urllib.error
import urllib.parse
from secret_manager import (
    safe_load_dotenv,
    encrypt_secret,
    decrypt_secret,
    is_encrypted,
    SENSITIVE_ENV_KEYS,
    get_master_key,
    Fernet,
)
from storage_adapter import (
    is_remote_storage_available,
    remote_save_user,
    remote_get_user,
    remote_save_exchange_key,
    remote_get_exchange_keys,
    remote_delete_account,
    remote_delete_all_accounts,
)

safe_load_dotenv()


def is_valid_username(username: str) -> bool:
    if not username or not isinstance(username, str):
        return False
    u = username.strip()
    return 3 <= len(u) <= 32 and bool(re.match(r"^[a-zA-Z0-9_.-]+$", u))


def is_valid_email(email: str) -> bool:
    if not email or not isinstance(email, str):
        return False
    e = email.strip().lower()
    if len(e) > 254 or "\r" in e or "\n" in e:
        return False
    return bool(re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", e))


def generate_activation_token(user_id: int, username: str, email: str, *args, expires_hours: int = 24, **kwargs) -> str:
    """Generates a tamper-proof cryptographically signed activation token."""
    exp = int((datetime.now(timezone.utc) + timedelta(hours=expires_hours)).timestamp())
    payload = {
        "act": "activate",
        "uid": user_id,
        "u": username,
        "e": email,
        "exp": exp,
        "nonce": secrets.token_hex(16)
    }
    raw = json.dumps(payload).encode("utf-8")
    f = Fernet(get_master_key())
    return f.encrypt(raw).decode("utf-8")


def verify_activation_token(token: str) -> Optional[Dict[str, Any]]:
    """Verifies and decrypts a cryptographically signed activation token."""
    if not token or not isinstance(token, str):
        return None
    token = token.strip()
    if "%" in token:
        try:
            token = urllib.parse.unquote(token)
        except Exception:
            pass
    try:
        f = Fernet(get_master_key())
        raw = f.decrypt(token.encode("utf-8"), ttl=86400)
        data = json.loads(raw.decode("utf-8"))
        if data.get("act") != "activate":
            return None
        exp = data.get("exp", 0)
        now_ts = datetime.now(timezone.utc).timestamp()
        if now_ts > exp:
            return {"expired": True, "data": data}
        return {"expired": False, "data": data}
    except Exception:
        return None


def create_stateless_session_token(user_id: int, username: str, email: str, *args, expires_days: int = 30, **kwargs) -> str:
    """Generates a tamper-proof cryptographically signed session token for cross-container authentication."""
    exp = int((datetime.now(timezone.utc) + timedelta(days=expires_days)).timestamp())
    payload = {
        "act": "session",
        "uid": user_id,
        "u": username,
        "e": email,
        "exp": exp,
        "nonce": secrets.token_hex(16)
    }
    raw = json.dumps(payload).encode("utf-8")
    f = Fernet(get_master_key())
    return f.encrypt(raw).decode("utf-8")


def verify_stateless_session_token(token: str) -> Optional[Dict[str, Any]]:
    """Verifies and decrypts a cryptographically signed session token."""
    if not token or not isinstance(token, str):
        return None
    token = token.strip()
    if "%" in token:
        try:
            token = urllib.parse.unquote(token)
        except Exception:
            pass
    try:
        f = Fernet(get_master_key())
        raw = f.decrypt(token.encode("utf-8"), ttl=86400 * 60)
        data = json.loads(raw.decode("utf-8"))
        if data.get("act") != "session":
            return None
        exp = data.get("exp", 0)
        now_ts = datetime.now(timezone.utc).timestamp()
        if now_ts > exp:
            return None
        return data
    except Exception:
        return None


def generate_magic_login_token(email: str, username: str = "", expires_hours: int = 2) -> str:
    """Generates a tamper-proof cryptographically signed 1-click magic login token."""
    exp = int((datetime.now(timezone.utc) + timedelta(hours=expires_hours)).timestamp())
    payload = {
        "act": "magic_login",
        "e": email.strip().lower(),
        "u": username.strip(),
        "exp": exp,
        "nonce": secrets.token_hex(8)
    }
    raw = json.dumps(payload).encode("utf-8")
    f = Fernet(get_master_key())
    return f.encrypt(raw).decode("utf-8")


def verify_magic_login_token(token: str) -> Optional[Dict[str, Any]]:
    """Verifies and decrypts a cryptographically signed magic login token."""
    if not token or not isinstance(token, str):
        return None
    token = token.strip()
    if "%" in token:
        try:
            token = urllib.parse.unquote(token)
        except Exception:
            pass
    try:
        f = Fernet(get_master_key())
        raw = f.decrypt(token.encode("utf-8"), ttl=86400 * 2)
        data = json.loads(raw.decode("utf-8"))
        if data.get("act") != "magic_login":
            return None
        exp = data.get("exp", 0)
        now_ts = datetime.now(timezone.utc).timestamp()
        if now_ts > exp:
            return {"expired": True, "data": data}
        return {"expired": False, "data": data}
    except Exception:
        return None


def generate_password_reset_token(user_id: int, username: str, email: str, expires_hours: int = 2) -> str:
    """Generates a tamper-proof cryptographically signed password reset token."""
    exp = int((datetime.now(timezone.utc) + timedelta(hours=expires_hours)).timestamp())
    payload = {
        "act": "reset",
        "uid": user_id,
        "u": username,
        "e": email,
        "exp": exp,
        "nonce": secrets.token_hex(8)
    }
    raw = json.dumps(payload).encode("utf-8")
    f = Fernet(get_master_key())
    return f.encrypt(raw).decode("utf-8")


def verify_password_reset_token(token: str) -> Optional[Dict[str, Any]]:
    """Verifies and decrypts a cryptographically signed password reset token."""
    if not token or not isinstance(token, str):
        return None
    token = token.strip()
    if "%" in token:
        try:
            token = urllib.parse.unquote(token)
        except Exception:
            pass
    try:
        f = Fernet(get_master_key())
        raw = f.decrypt(token.encode("utf-8"), ttl=86400 * 7)
        data = json.loads(raw.decode("utf-8"))
        if data.get("act") != "reset":
            return None
        exp = data.get("exp", 0)
        now_ts = datetime.now(timezone.utc).timestamp()
        if now_ts > exp:
            return {"expired": True, "data": data}
        return {"expired": False, "data": data}
    except Exception:
        return None

# Detect serverless environment for DB path
if os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
    DB_PATH = Path("/tmp/crypto_bot_users.db")
else:
    DB_PATH = Path(__file__).resolve().parent / "users.db"


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def migrate_unencrypted_secrets(conn: sqlite3.Connection):
    """Encrypts any existing plaintext exchange keys and system settings in the database."""
    try:
        cur = conn.cursor()
        # 1. Exchange keys table
        cur.execute("SELECT id, api_key, api_secret, api_passphrase FROM exchange_keys")
        rows = cur.fetchall()
        for r in rows:
            pk = r["id"]
            k = r["api_key"]
            s = r["api_secret"]
            p = r["api_passphrase"]
            needs_update = False
            enc_k = k
            enc_s = s
            enc_p = p
            if k and not is_encrypted(k):
                enc_k = encrypt_secret(k)
                needs_update = True
            if s and not is_encrypted(s):
                enc_s = encrypt_secret(s)
                needs_update = True
            if p and not is_encrypted(p):
                enc_p = encrypt_secret(p)
                needs_update = True
            if needs_update:
                cur.execute(
                    "UPDATE exchange_keys SET api_key = ?, api_secret = ?, api_passphrase = ? WHERE id = ?",
                    (enc_k, enc_s, enc_p, pk)
                )

        # 2. System settings table
        cur.execute("SELECT key, value FROM system_settings")
        s_rows = cur.fetchall()
        for r in s_rows:
            key = r["key"]
            val = r["value"]
            if key in SENSITIVE_ENV_KEYS and val and not is_encrypted(val):
                cur.execute(
                    "UPDATE system_settings SET value = ? WHERE key = ?",
                    (encrypt_secret(val), key)
                )

        conn.commit()
    except Exception as e:
        print(f"[AUTH ERROR] Failed during unencrypted secret migration: {e}")


def sync_env_to_system_settings(conn: sqlite3.Connection):
    """
    Persists any platform-level environment variables (e.g. Vercel, Docker, AWS Lambda)
    into the encrypted system_settings table so they are securely available to the application.
    """
    keys = [
        "EMAIL_PROVIDER",
        "SMTP_HOST",
        "SMTP_PORT",
        "SMTP_USER",
        "SMTP_PASS",
        "FROM_EMAIL",
        "RESEND_API_KEY",
        "BREVO_API_KEY",
        "SENDGRID_API_KEY",
    ]
    try:
        cur = conn.cursor()
        for k in keys:
            val = (os.getenv(k) or "").strip()
            if val:
                cur.execute("SELECT value FROM system_settings WHERE key = ?", (k,))
                row = cur.fetchone()
                if not row or not row["value"]:
                    stored_val = encrypt_secret(val) if k in SENSITIVE_ENV_KEYS else val
                    cur.execute("""
                        INSERT INTO system_settings (key, value, updated_at)
                        VALUES (?, ?, ?)
                        ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                    """, (k, stored_val, datetime.now(timezone.utc).isoformat()))
        conn.commit()
    except Exception as e:
        print(f"[AUTH WARNING] Could not sync environment settings to database: {e}")


def init_auth_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        is_active INTEGER DEFAULT 0,
        activation_token TEXT,
        activation_expires_at TEXT,
        reset_token TEXT,
        reset_expires_at TEXT,
        created_at TEXT NOT NULL
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS exchange_keys (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        exchange_id TEXT NOT NULL,
        api_key TEXT NOT NULL,
        api_secret TEXT NOT NULL,
        api_passphrase TEXT DEFAULT '',
        testnet INTEGER DEFAULT 1,
        created_at TEXT NOT NULL,
        UNIQUE(user_id, exchange_id),
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS email_outbox (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        to_email TEXT NOT NULL,
        subject TEXT NOT NULL,
        body TEXT NOT NULL,
        token TEXT,
        email_type TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS system_settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        token TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    """)
    conn.commit()
    migrate_unencrypted_secrets(conn)
    sync_env_to_system_settings(conn)
    conn.close()
    try:
        from secret_manager import migrate_and_sanitize_env_file
        migrate_and_sanitize_env_file()
    except Exception as e:
        print(f"[AUTH ERROR] Failed to run env migration in init_auth_db: {e}")


def get_system_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT value FROM system_settings WHERE key = ?", (key,))
        row = cur.fetchone()
        conn.close()
        if row and row["value"]:
            val = row["value"]
            if key in SENSITIVE_ENV_KEYS or is_encrypted(val):
                return decrypt_secret(val)
            return val
    except Exception:
        pass
    env_val = os.getenv(key, default)
    if env_val and (key in SENSITIVE_ENV_KEYS or is_encrypted(env_val)):
        return decrypt_secret(env_val)
    return env_val


def set_system_setting(key: str, value: str, conn: Optional[sqlite3.Connection] = None):
    should_close = False
    if conn is None:
        conn = get_db_connection()
        should_close = True
    try:
        cur = conn.cursor()
        stored_value = encrypt_secret(value) if key in SENSITIVE_ENV_KEYS and value else value
        cur.execute("""
            INSERT INTO system_settings (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
        """, (key, stored_value, datetime.now(timezone.utc).isoformat()))
        if should_close:
            conn.commit()
        if key not in SENSITIVE_ENV_KEYS:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)
    except Exception as e:
        print(f"[AUTH ERROR] Failed to set system setting {key}: {e}")
        raise RuntimeError(f"Failed to persist system setting '{key}': {e}") from e
    finally:
        if should_close:
            try:
                conn.close()
            except Exception:
                pass


def get_resend_verified_domains(resend_key: str) -> list:
    """Queries Resend API for verified sending domains associated with this API key."""
    if not resend_key:
        return []
    try:
        req = urllib.request.Request(
            "https://api.resend.com/domains",
            headers={
                "Authorization": f"Bearer {resend_key}",
                "User-Agent": "TraderQueen/2.0"
            }
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            domains = data.get("data", [])
            return [d["name"] for d in domains if d.get("status") == "verified"]
    except Exception:
        return []


def get_email_settings() -> Dict[str, Any]:
    provider = (get_system_setting("EMAIL_PROVIDER", "") or os.getenv("EMAIL_PROVIDER", "") or "").strip().lower()
    smtp_host = (get_system_setting("SMTP_HOST", "") or os.getenv("SMTP_HOST", "") or "").strip()
    smtp_port = (get_system_setting("SMTP_PORT", "587") or os.getenv("SMTP_PORT", "587") or "587").strip()
    smtp_user = (get_system_setting("SMTP_USER", "") or os.getenv("SMTP_USER", "") or "").strip()
    smtp_pass = (get_system_setting("SMTP_PASS", "") or os.getenv("SMTP_PASS", "") or "").strip()
    from_email = (get_system_setting("FROM_EMAIL", "") or os.getenv("FROM_EMAIL", "") or "").strip()
    resend_key = (get_system_setting("RESEND_API_KEY", "") or os.getenv("RESEND_API_KEY", "") or "").strip()
    brevo_key = (get_system_setting("BREVO_API_KEY", "") or os.getenv("BREVO_API_KEY", "") or "").strip()
    sendgrid_key = (get_system_setting("SENDGRID_API_KEY", "") or os.getenv("SENDGRID_API_KEY", "") or "").strip()

    has_smtp = bool(smtp_host and smtp_user and smtp_pass)
    has_resend = bool(resend_key)
    has_brevo = bool(brevo_key)
    has_sendgrid = bool(sendgrid_key)
    is_configured = has_smtp or has_resend or has_brevo or has_sendgrid

    # Auto-detect active provider if not explicitly chosen
    if not provider or provider == "none":
        if has_resend:
            provider = "resend"
        elif has_brevo:
            provider = "brevo"
        elif has_smtp:
            provider = "smtp"
        elif has_sendgrid:
            provider = "sendgrid"
        else:
            provider = "none"

    # Check if Resend has verified domains on account
    resend_domains = []
    if has_resend:
        resend_domains = get_resend_verified_domains(resend_key)
        # If user has a verified custom domain on Resend but from_email is unset or still on sandbox, auto-upgrade
        if resend_domains and ("onboarding@resend.dev" in from_email.lower() or not from_email):
            auto_from = f"Trader Queen <noreply@{resend_domains[0]}>"
            from_email = auto_from
            set_system_setting("FROM_EMAIL", auto_from)

    # Check if Resend is operating in sandbox testing mode (onboarding@resend.dev without verified domain)
    is_resend_sandbox = bool(
        has_resend and ("onboarding@resend.dev" in from_email.lower() or not from_email) and not resend_domains
    )

    # Determine active provider
    active_provider = "none"
    if provider in ["gmail", "brevo", "resend", "sendgrid", "smtp"]:
        if provider == "gmail" and has_smtp:
            active_provider = "gmail"
        elif provider == "brevo" and has_brevo:
            active_provider = "brevo"
        elif provider == "resend" and has_resend:
            active_provider = "resend"
        elif provider == "sendgrid" and has_sendgrid:
            active_provider = "sendgrid"
        elif provider == "smtp" and has_smtp:
            active_provider = "smtp"

    if active_provider == "none":
        if has_resend:
            active_provider = "resend"
        elif has_brevo:
            active_provider = "brevo"
        elif has_sendgrid:
            active_provider = "sendgrid"
        elif has_smtp:
            active_provider = "gmail" if "gmail" in smtp_host.lower() else "smtp"

    can_send_to_all_users = bool(
        (active_provider in ["gmail", "smtp"] and has_smtp)
        or (active_provider == "brevo" and has_brevo)
        or (active_provider == "sendgrid" and has_sendgrid)
        or (active_provider == "resend" and has_resend and not is_resend_sandbox)
    )

    return {
        "is_configured": is_configured,
        "provider": active_provider,
        "smtp_host": smtp_host,
        "smtp_port": smtp_port,
        "smtp_user": smtp_user,
        "smtp_pass_masked": f"{smtp_pass[:2]}••••{smtp_pass[-2:]}" if len(smtp_pass) > 4 else ("••••" if smtp_pass else ""),
        "from_email": from_email,
        "resend_key_masked": f"{resend_key[:5]}••••" if len(resend_key) > 5 else ("••••" if resend_key else ""),
        "brevo_key_masked": f"{brevo_key[:5]}••••" if len(brevo_key) > 5 else ("••••" if brevo_key else ""),
        "sendgrid_key_masked": f"{sendgrid_key[:5]}••••" if len(sendgrid_key) > 5 else ("••••" if sendgrid_key else ""),
        "is_resend_sandbox": is_resend_sandbox,
        "can_send_to_all_users": can_send_to_all_users,
        "resend_domains": resend_domains,
        "verified_domain": resend_domains[0] if resend_domains else ""
    }


def save_email_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    allowed_keys = [
        "EMAIL_PROVIDER",
        "SMTP_HOST",
        "SMTP_PORT",
        "SMTP_USER",
        "SMTP_PASS",
        "FROM_EMAIL",
        "RESEND_API_KEY",
        "BREVO_API_KEY",
        "SENDGRID_API_KEY",
    ]
    for k in allowed_keys:
        if k in settings and settings[k] is not None:
            val = str(settings[k]).replace("\r", "").replace("\n", "").strip()
            # Do not overwrite sensitive credentials with blank if already configured
            if k in ["SMTP_PASS", "RESEND_API_KEY", "BREVO_API_KEY", "SENDGRID_API_KEY"] and val == "":
                continue
            set_system_setting(k, val)

    return {"success": True, "message": "Email settings saved successfully", "settings": get_email_settings()}


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    if salt is None:
        salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000)
    return dk.hex(), salt


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    computed_hash, _ = hash_password(password, salt)
    return secrets.compare_digest(computed_hash, password_hash)


def _send_via_resend(to_email: str, subject: str, body: str, resend_key: str, from_email: str, email_type: str) -> Tuple[bool, str]:
    if not resend_key:
        return False, "Resend API key not configured"

    sender = from_email
    # If sender is using the sandbox address or empty, check if account has a verified domain
    if not sender or "onboarding@resend.dev" in sender.lower():
        verified = get_resend_verified_domains(resend_key)
        if verified:
            sender = f"Trader Queen <noreply@{verified[0]}>"
            set_system_setting("FROM_EMAIL", sender)
        else:
            sender = "Trader Queen <onboarding@resend.dev>"

    try:
        req = urllib.request.Request(
            "https://api.resend.com/emails",
            data=json.dumps({
                "from": sender,
                "to": [to_email],
                "subject": subject,
                "html": body,
            }).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {resend_key}",
                "Content-Type": "application/json",
                "User-Agent": "TraderQueen/2.0"
            }
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            if 200 <= resp.status < 300:
                msg = f"Delivered {email_type} email to {to_email} via Resend API ({sender})"
                print(f"[AUTH EMAIL] {msg}")
                return True, msg
            return False, f"Resend API returned status code {resp.status}"
    except urllib.error.HTTPError as e:
        try:
            raw_err = e.read().decode("utf-8")
            err_json = json.loads(raw_err)
            detail = err_json.get("message", raw_err)
        except Exception:
            detail = str(e)

        if e.code == 403 and ("testing emails to your own email address" in detail.lower() or "verify a domain" in detail.lower()):
            err_msg = (
                "Resend Sandbox Restriction (403): onboarding@resend.dev can only deliver to the Resend account owner. "
                "To send verification emails to all registering users, please add & verify your custom domain at https://resend.com/domains "
                "and set FROM_EMAIL to an address on that domain (e.g. Trader Queen <noreply@yourdomain.com>), or switch to Brevo API or Gmail SMTP."
            )
        else:
            err_msg = f"Resend API error ({e.code}): {detail}"
        print(f"[AUTH EMAIL] {err_msg}")
        return False, err_msg
    except Exception as e:
        err_msg = f"Resend delivery failed: {e}"
        print(f"[AUTH EMAIL] {err_msg}")
        return False, err_msg


def _send_via_brevo(to_email: str, subject: str, body: str, brevo_key: str, from_email: str, email_type: str) -> Tuple[bool, str]:
    if not brevo_key:
        return False, "Brevo API key not configured"

    sender_name = "Trader Queen"
    sender_email = from_email or "noreply@traderqueen.name.ng"
    if "<" in sender_email and ">" in sender_email:
        sender_name = sender_email.split("<")[0].strip() or sender_name
        sender_email = sender_email.split("<")[1].split(">")[0].strip()

    try:
        payload = {
            "sender": {"name": sender_name, "email": sender_email},
            "to": [{"email": to_email}],
            "subject": subject,
            "htmlContent": body,
        }
        req = urllib.request.Request(
            "https://api.brevo.com/v3/smtp/email",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "api-key": brevo_key,
                "Content-Type": "application/json",
                "User-Agent": "TraderQueen/2.0"
            }
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            if 200 <= resp.status < 300:
                msg = f"Delivered {email_type} email to {to_email} via Brevo API"
                print(f"[AUTH EMAIL] {msg}")
                return True, msg
            return False, f"Brevo API returned status code {resp.status}"
    except urllib.error.HTTPError as e:
        try:
            raw_err = e.read().decode("utf-8")
            err_json = json.loads(raw_err)
            detail = err_json.get("message", raw_err)
        except Exception:
            detail = str(e)
        err_msg = f"Brevo API error ({e.code}): {detail}"
        print(f"[AUTH EMAIL] {err_msg}")
        return False, err_msg
    except Exception as e:
        err_msg = f"Brevo delivery failed: {e}"
        print(f"[AUTH EMAIL] {err_msg}")
        return False, err_msg


def _send_via_sendgrid(to_email: str, subject: str, body: str, sendgrid_key: str, from_email: str, email_type: str) -> Tuple[bool, str]:
    if not sendgrid_key:
        return False, "SendGrid API key not configured"

    sender_name = "Trader Queen"
    sender_email = from_email or "noreply@traderqueen.name.ng"
    if "<" in sender_email and ">" in sender_email:
        sender_name = sender_email.split("<")[0].strip() or sender_name
        sender_email = sender_email.split("<")[1].split(">")[0].strip()

    try:
        payload = {
            "personalizations": [{"to": [{"email": to_email}]}],
            "from": {"email": sender_email, "name": sender_name},
            "subject": subject,
            "content": [{"type": "text/html", "value": body}]
        }
        req = urllib.request.Request(
            "https://api.sendgrid.com/v3/mail/send",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {sendgrid_key}",
                "Content-Type": "application/json",
                "User-Agent": "TraderQueen/2.0"
            }
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            if 200 <= resp.status < 300:
                msg = f"Delivered {email_type} email to {to_email} via SendGrid API"
                print(f"[AUTH EMAIL] {msg}")
                return True, msg
            return False, f"SendGrid API returned status code {resp.status}"
    except urllib.error.HTTPError as e:
        try:
            raw_err = e.read().decode("utf-8")
            err_json = json.loads(raw_err)
            errors = err_json.get("errors", [])
            detail = errors[0].get("message") if errors else raw_err
        except Exception:
            detail = str(e)
        err_msg = f"SendGrid API error ({e.code}): {detail}"
        print(f"[AUTH EMAIL] {err_msg}")
        return False, err_msg
    except Exception as e:
        err_msg = f"SendGrid delivery failed: {e}"
        print(f"[AUTH EMAIL] {err_msg}")
        return False, err_msg


def _send_via_smtp(to_email: str, subject: str, body: str, smtp_host: str, smtp_port: int, smtp_user: str, smtp_pass: str, smtp_from: str, email_type: str) -> Tuple[bool, str]:
    if not smtp_host:
        return False, "SMTP host not configured"

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = smtp_from
        msg["To"] = to_email
        msg.attach(MIMEText(body, "html"))

        ssl_ctx = ssl.create_default_context()

        if smtp_port == 465:
            with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=12, context=ssl_ctx) as server:
                if smtp_user and smtp_pass:
                    server.login(smtp_user, smtp_pass)
                server.sendmail(smtp_from, [to_email], msg.as_string())
        else:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=12) as server:
                server.ehlo()
                server.starttls(context=ssl_ctx)
                server.ehlo()
                if smtp_user and smtp_pass:
                    server.login(smtp_user, smtp_pass)
                server.sendmail(smtp_from, [to_email], msg.as_string())

        msg_str = f"Delivered {email_type} email to {to_email} via SMTP ({smtp_host}:{smtp_port})"
        print(f"[AUTH EMAIL] {msg_str}")
        return True, msg_str
    except smtplib.SMTPAuthenticationError as e:
        if "gmail" in smtp_host.lower():
            err_msg = (
                f"Gmail SMTP authentication failed: Invalid username or App Password. "
                f"For Gmail, enable 2-Step Verification and use a 16-character App Password "
                f"generated at https://myaccount.google.com/apppasswords (not your normal Google password)."
            )
        else:
            err_msg = f"SMTP authentication failed ({smtp_host}:{smtp_port}): {e}"
        print(f"[AUTH EMAIL] {err_msg}")
        return False, err_msg
    except (socket.timeout, TimeoutError, smtplib.SMTPConnectError, OSError) as e:
        err_msg = (
            f"SMTP connection to {smtp_host}:{smtp_port} failed: {e}. "
            f"Note: Cloud serverless environments (like Vercel or AWS Lambda) block outbound SMTP ports 587/465. "
            f"Please configure Brevo API or Resend API in Email Settings for serverless deployments."
        )
        print(f"[AUTH EMAIL] {err_msg}")
        return False, err_msg
    except Exception as e:
        err_msg = f"SMTP delivery failed to {to_email} ({smtp_host}:{smtp_port}): {e}"
        print(f"[AUTH EMAIL] {err_msg}")
        return False, err_msg


def send_email_message(to_email: str, subject: str, body: str, token: str, email_type: str) -> Tuple[bool, str]:
    """Stores in outbox for audit and delivers via active provider with automatic fallback.
    Returns (success: bool, status_message: str)."""
    # Always record in database outbox for audit log
    try:
        conn = get_db_connection()
        conn.execute(
            "INSERT INTO email_outbox (to_email, subject, body, token, email_type, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (to_email, subject, body, token, email_type, datetime.now(timezone.utc).isoformat())
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[AUTH ERROR] Error recording email in outbox: {e}")

    settings = get_email_settings()
    active_prov = settings.get("provider", "none")

    resend_key = get_system_setting("RESEND_API_KEY", "") or os.getenv("RESEND_API_KEY", "") or ""
    brevo_key = get_system_setting("BREVO_API_KEY", "") or os.getenv("BREVO_API_KEY", "") or ""
    sendgrid_key = get_system_setting("SENDGRID_API_KEY", "") or os.getenv("SENDGRID_API_KEY", "") or ""

    smtp_host = get_system_setting("SMTP_HOST", "") or os.getenv("SMTP_HOST", "") or os.getenv("EMAIL_SERVER", "") or ""
    smtp_port = int(get_system_setting("SMTP_PORT", "587") or os.getenv("SMTP_PORT", "587") or os.getenv("EMAIL_PORT", "587") or "587")
    smtp_user = get_system_setting("SMTP_USER", "") or os.getenv("SMTP_USER", "") or os.getenv("EMAIL_USER", "") or ""
    smtp_pass = get_system_setting("SMTP_PASS", "") or os.getenv("SMTP_PASS", "") or os.getenv("EMAIL_PASSWORD", "") or ""

    from_email = get_system_setting("FROM_EMAIL", "") or os.getenv("FROM_EMAIL", "") or smtp_user or ""

    # Define dispatch candidates
    candidates = []

    def run_brevo():
        sender = from_email or "Trader Queen <noreply@traderqueen.name.ng>"
        return _send_via_brevo(to_email, subject, body, brevo_key, sender, email_type)

    def run_sendgrid():
        sender = from_email or "Trader Queen <noreply@traderqueen.name.ng>"
        return _send_via_sendgrid(to_email, subject, body, sendgrid_key, sender, email_type)

    def run_resend():
        sender = from_email or "Trader Queen <onboarding@resend.dev>"
        return _send_via_resend(to_email, subject, body, resend_key, sender, email_type)

    def run_smtp():
        s_from = from_email or smtp_user or "noreply@traderqueen.name.ng"
        return _send_via_smtp(to_email, subject, body, smtp_host, smtp_port, smtp_user, smtp_pass, s_from, email_type)

    # 1. Prioritize active provider
    if active_prov == "brevo" and brevo_key:
        candidates.append(("Brevo API", run_brevo))
    elif active_prov == "sendgrid" and sendgrid_key:
        candidates.append(("SendGrid API", run_sendgrid))
    elif active_prov == "resend" and resend_key:
        candidates.append(("Resend API", run_resend))
    elif active_prov in ["gmail", "smtp"] and smtp_host and smtp_user and smtp_pass:
        candidates.append(("SMTP", run_smtp))

    # 2. Add remaining available providers as automatic fallback
    if brevo_key and not any(c[0] == "Brevo API" for c in candidates):
        candidates.append(("Brevo API", run_brevo))
    if resend_key and not any(c[0] == "Resend API" for c in candidates):
        candidates.append(("Resend API", run_resend))
    if sendgrid_key and not any(c[0] == "SendGrid API" for c in candidates):
        candidates.append(("SendGrid API", run_sendgrid))
    if smtp_host and smtp_user and smtp_pass and not any(c[0] == "SMTP" for c in candidates):
        candidates.append(("SMTP", run_smtp))

    if not candidates:
        not_configured_msg = "No email delivery service (Gmail SMTP, Brevo API, Resend API, or SendGrid) is configured on this server."
        print(f"[AUTH EMAIL] Notice: {not_configured_msg} Recorded email in system outbox for {to_email}.")
        return False, not_configured_msg

    errors = []
    for prov_name, dispatch_fn in candidates:
        success, status = dispatch_fn()
        if success:
            return True, status
        errors.append(status)

    # All attempted providers failed; return diagnostic summary
    combined_err = " | ".join(errors)
    return False, combined_err


def test_email_connection(to_email: str) -> Dict[str, Any]:
    """Test sending a verification email to the specified address."""
    to_email = (to_email or "").strip().lower()
    if not is_valid_email(to_email):
        return {"success": False, "message": "Invalid test email address format"}

    settings = get_email_settings()
    if not settings["is_configured"]:
        return {
            "success": False,
            "message": "No email delivery service is configured yet. Please enter your Gmail, Brevo, Resend, or SMTP credentials first.",
            "settings": settings
        }

    test_token = secrets.token_urlsafe(16)
    test_body = f"""
    <div style="font-family: sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; background: #0f172a; color: #f8fafc; border-radius: 12px; border: 1px solid #334155;">
      <h2 style="color: #22c55e; margin-top: 0;">✓ Email Dispatch Test Successful!</h2>
      <p>Hello,</p>
      <p>This is a confirmation test email sent from your <strong>Trader Queen</strong> platform.</p>
      <p>Your email service (<strong>{html.escape(settings['provider'].upper())}</strong>) is configured and working properly!</p>
      <div style="background: #1e293b; padding: 12px 16px; border-radius: 8px; margin: 20px 0; font-family: monospace; font-size: 12px;">
        Provider: {html.escape(settings['provider'].upper())}<br>
        Recipient: {html.escape(to_email)}<br>
        Dispatched at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
      </div>
      <p style="font-size: 12px; color: #64748b; border-top: 1px solid #334155; padding-top: 15px;">Trader Queen Verification Engine</p>
    </div>
    """

    success, status = send_email_message(to_email, "✓ Trader Queen - Email Test Confirmation", test_body, test_token, "test")
    return {
        "success": success,
        "message": status,
        "recipient": to_email,
        "provider": settings["provider"],
        "can_send_to_all_users": settings.get("can_send_to_all_users", False),
        "is_resend_sandbox": settings.get("is_resend_sandbox", False)
    }


class AuthService:
    @staticmethod
    def register(username: str, email: str, password: str, base_url: str = "http://localhost:8000", auto_activate: Optional[bool] = None) -> Dict[str, Any]:
        username = (username or "").strip()
        email = (email or "").strip().lower()

        if not is_valid_username(username):
            return {"success": False, "message": "Username must be between 3 and 32 characters long and contain only letters, numbers, underscores, dashes, and periods"}
        if not is_valid_email(email):
            return {"success": False, "message": "Invalid email address format"}
        if len(password) < 10:
            return {"success": False, "message": "Password must be at least 10 characters long"}
        if len(password) > 128:
            return {"success": False, "message": "Password is too long (maximum 128 characters)"}

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE LOWER(username) = ? OR LOWER(email) = ?", (username.lower(), email))
        existing = cur.fetchone()
        if not existing and is_remote_storage_available():
            remote_u = remote_get_user(username) or remote_get_user(email)
            if remote_u:
                existing = remote_u

        if existing:
            conn.close()
            return {"success": False, "message": "A user with this username or email already exists"}

        password_hash, salt = hash_password(password)
        expires_at = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
        created_at = datetime.now(timezone.utc).isoformat()
        temp_token = secrets.token_urlsafe(32)

        if auto_activate is None:
            auto_activate = bool(os.getenv("AUTO_ACTIVATE", "").lower() == "true")

        initial_active = 1 if auto_activate else 0

        cur.execute("""
            INSERT INTO users (username, email, password_hash, salt, is_active, activation_token, activation_expires_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (username, email, password_hash, salt, initial_active, temp_token, expires_at, created_at))
        user_id = cur.lastrowid

        # Generate cryptographically signed self-sovereign token containing user state
        activation_token = generate_activation_token(user_id, username, email, expires_hours=24)
        cur.execute("UPDATE users SET activation_token = ? WHERE id = ?", (activation_token, user_id))
        conn.commit()
        conn.close()

        session_token = create_stateless_session_token(user_id, username, email)

        # Persist user in remote store if available
        remote_save_user(user_id, username, email, password_hash, salt, is_active=initial_active, created_at=created_at)

        # Send activation email to recipient
        activation_link = f"{base_url}/activate?token={urllib.parse.quote(activation_token)}"
        email_body = f"""
        <div style="font-family: sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; background: #0f172a; color: #f8fafc; border-radius: 12px; border: 1px solid #334155;">
          <h2 style="color: #22c55e; margin-top: 0;">{"Welcome to Trader Queen" if auto_activate else "Activate Your Trader Queen Account"}</h2>
          <p>Hello <strong>{html.escape(username)}</strong>,</p>
          <p>{"Thank you for registering on the Trader Queen platform! Your account is active and ready to trade." if auto_activate else "Thank you for registering on the Trader Queen platform. Please click the button below to accept and activate your account:"}</p>
          <div style="text-align: center; margin: 30px 0;">
            <a href="{activation_link}" style="background-color: #16a34a; color: #ffffff; padding: 12px 24px; text-decoration: none; font-weight: bold; border-radius: 8px; display: inline-block;">{"Open Trading Dashboard" if auto_activate else "Activate Account"}</a>
          </div>
          <p style="font-size: 12px; color: #94a3b8;">Or copy and paste this link into your browser:<br><a href="{activation_link}" style="color: #38bdf8;">{html.escape(activation_link)}</a></p>
          <p style="font-size: 12px; color: #64748b; margin-top: 30px; border-top: 1px solid #334155; padding-top: 15px;">{"This direct access link is secured for your account." if auto_activate else "This activation link expires in 24 hours."}</p>
        </div>
        """
        email_sent, email_status = send_email_message(
            email,
            "Welcome to Trader Queen (Account Created)" if auto_activate else "Activate Your Trader Queen Account",
            email_body,
            activation_token,
            "activation"
        )

        res = {
            "success": True,
            "user_id": user_id,
            "email": email,
            "email_sent": email_sent,
            "email_status": email_status,
            "is_active": initial_active
        }
        if auto_activate:
            res["token"] = session_token
            res["user"] = {
                "id": user_id,
                "username": username,
                "email": email
            }
            res["message"] = f"Account created for {username}. You are now logged in! A confirmation email has also been sent to {email}."
        elif email_sent:
            res["message"] = f"Account created for {username}. A verification email has been sent to {email}. Please check your inbox to activate your account."
        else:
            res["message"] = f"Account created for {username}. Email delivery warning: {email_status}"
            res["dev_activation_link"] = activation_link

        return res

    @staticmethod
    def activate_account(token: str) -> Dict[str, Any]:
        token = token.strip()
        conn = get_db_connection()
        cur = conn.cursor()

        # 1. First check local database
        cur.execute("SELECT id, username, email, password_hash, salt, is_active, activation_expires_at FROM users WHERE activation_token = ?", (token,))
        user = cur.fetchone()

        if user:
            session_token = create_stateless_session_token(user["id"], user["username"], user["email"])
            if user["is_active"] == 1:
                conn.close()
                return {
                    "success": True,
                    "already_active": True,
                    "message": "Account is already activated. You can log in now.",
                    "session_token": session_token,
                    "username": user["username"],
                    "email": user["email"]
                }

            # Check expiration
            expires = datetime.fromisoformat(user["activation_expires_at"])
            if datetime.now(timezone.utc) > expires:
                conn.close()
                return {"success": False, "message": "Activation link has expired. Please request a new one."}

            cur.execute("UPDATE users SET is_active = 1, activation_token = NULL WHERE id = ?", (user["id"],))
            conn.commit()
            conn.close()

            # Sync to remote persistent store
            remote_u = remote_get_user(user["username"])
            if remote_u:
                remote_save_user(user["id"], user["username"], user["email"], remote_u["password_hash"], remote_u["salt"], is_active=1)

            return {
                "success": True,
                "message": f"Account {user['username']} successfully activated! You can now log in.",
                "session_token": session_token,
                "username": user["username"],
                "email": user["email"]
            }

        # 2. Cryptographic self-sovereign token verification (handles cross-container serverless cold starts)
        token_info = verify_activation_token(token)
        if token_info:
            if token_info["expired"]:
                conn.close()
                return {"success": False, "message": "Activation link has expired. Please request a new one."}

            data = token_info["data"]
            username = data["u"]
            email = data["e"]
            uid = data.get("uid", 1)

            cur.execute("UPDATE users SET is_active = 1, activation_token = NULL, activation_expires_at = NULL WHERE id = ? OR LOWER(email) = ?",
                        (uid, email.lower()))
            if cur.rowcount == 0:
                remote_u = remote_get_user(username) or remote_get_user(email)
                pw_hash = remote_u.get("password_hash", "") if remote_u else ""
                salt = remote_u.get("salt", "") if remote_u else ""
                created_at = remote_u.get("created_at", datetime.now(timezone.utc).isoformat()) if remote_u else datetime.now(timezone.utc).isoformat()
                cur.execute("""
                    INSERT OR REPLACE INTO users (id, username, email, password_hash, salt, is_active, created_at)
                    VALUES (?, ?, ?, ?, ?, 1, ?)
                """, (uid, username, email, pw_hash, salt, created_at))
            conn.commit()
            conn.close()

            session_token = create_stateless_session_token(uid, username, email)
            remote_save_user(uid, username, email, "", "", is_active=1)

            return {
                "success": True,
                "message": f"Account {username} successfully activated! You can now log in.",
                "session_token": session_token,
                "username": username,
                "email": email
            }

        conn.close()
        return {"success": False, "message": "Invalid or already used activation token"}

    @staticmethod
    def resend_activation_email(email: str, base_url: str = "http://localhost:8000") -> Dict[str, Any]:
        email = (email or "").strip().lower()
        if not is_valid_email(email):
            return {"success": False, "message": "Invalid email address format"}

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, username, email, password_hash, salt, is_active FROM users WHERE LOWER(email) = ?", (email,))
        user = cur.fetchone()

        if not user:
            remote_u = remote_get_user(email)
            if remote_u:
                try:
                    cur.execute("""
                        INSERT INTO users (id, username, email, password_hash, salt, is_active, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET is_active = excluded.is_active
                    """, (
                        remote_u["id"],
                        remote_u["username"],
                        remote_u["email"],
                        remote_u["password_hash"],
                        remote_u["salt"],
                        remote_u.get("is_active", 0),
                        remote_u.get("created_at", datetime.now(timezone.utc).isoformat())
                    ))
                    conn.commit()
                    cur.execute("SELECT id, username, email, password_hash, salt, is_active FROM users WHERE id = ?", (remote_u["id"],))
                    user = cur.fetchone()
                except Exception:
                    pass

        if not user:
            conn.close()
            # Do not disclose if user exists, but confirm message
            return {"success": True, "message": f"If an account is associated with {email} and pending verification, a new verification link has been sent."}

        if user["is_active"] == 1:
            conn.close()
            return {"success": False, "already_active": True, "message": "Account is already activated. Please sign in."}

        activation_token = generate_activation_token(user["id"], user["username"], user["email"], expires_hours=24)
        expires_at = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()

        cur.execute("UPDATE users SET activation_token = ?, activation_expires_at = ? WHERE id = ?",
                    (activation_token, expires_at, user["id"]))
        conn.commit()
        conn.close()

        activation_link = f"{base_url}/activate?token={urllib.parse.quote(activation_token)}"
        email_body = f"""
        <div style="font-family: sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; background: #0f172a; color: #f8fafc; border-radius: 12px; border: 1px solid #334155;">
          <h2 style="color: #22c55e; margin-top: 0;">Your New Verification Link</h2>
          <p>Hello <strong>{html.escape(user['username'])}</strong>,</p>
          <p>You requested a new verification link for your Trader Queen account. Please click the button below to accept and activate your account:</p>
          <div style="text-align: center; margin: 30px 0;">
            <a href="{activation_link}" style="background-color: #16a34a; color: #ffffff; padding: 12px 24px; text-decoration: none; font-weight: bold; border-radius: 8px; display: inline-block;">Activate Account</a>
          </div>
          <p style="font-size: 12px; color: #94a3b8;">Or copy and paste this link into your browser:<br><a href="{activation_link}" style="color: #38bdf8;">{html.escape(activation_link)}</a></p>
          <p style="font-size: 12px; color: #64748b; margin-top: 30px; border-top: 1px solid #334155; padding-top: 15px;">This verification link expires in 24 hours.</p>
        </div>
        """
        email_sent, email_status = send_email_message(email, "Activate Your Trader Queen Account (New Verification Link)", email_body, activation_token, "activation")

        res = {
            "success": True,
            "email": email,
            "email_sent": email_sent,
            "email_status": email_status
        }
        if email_sent:
            res["message"] = f"A fresh verification email has been sent to {email}. Please check your inbox."
        else:
            res["message"] = f"A new verification token was generated, but the email could not be sent: {email_status}."
            res["dev_activation_link"] = activation_link

        return res

    @staticmethod
    def login(username_or_email: str, password: str, client_token: Optional[str] = None) -> Dict[str, Any]:
        val = (username_or_email or "").strip().lower()
        if not val:
            return {"success": False, "message": "Username or email is required"}
        if not password:
            return {"success": False, "message": "Password is required"}

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE LOWER(username) = ? OR LOWER(email) = ?", (val, val))
        user = cur.fetchone()

        if not user:
            remote_u = remote_get_user(val)
            if remote_u:
                try:
                    cur.execute("""
                        INSERT INTO users (id, username, email, password_hash, salt, is_active, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET password_hash = excluded.password_hash, salt = excluded.salt, is_active = excluded.is_active
                    """, (
                        remote_u["id"],
                        remote_u["username"],
                        remote_u["email"],
                        remote_u["password_hash"],
                        remote_u["salt"],
                        remote_u.get("is_active", 1),
                        remote_u.get("created_at", datetime.now(timezone.utc).isoformat())
                    ))
                    conn.commit()
                    cur.execute("SELECT * FROM users WHERE id = ?", (remote_u["id"],))
                    user = cur.fetchone()
                except Exception as e:
                    print(f"[AUTH WARNING] Failed caching remote user into local DB: {e}")

        if not user:
            conn.close()
            return {
                "success": False,
                "not_found": True,
                "email": val if "@" in val else "",
                "message": "Invalid username/email or password"
            }

        if not verify_password(password, user["password_hash"], user["salt"]):
            conn.close()
            return {"success": False, "message": "Invalid username/email or password"}

        if user["is_active"] == 0:
            conn.close()
            return {
                "success": False,
                "needs_activation": True,
                "email": user["email"],
                "message": "Your account is not activated yet. Please check your email for the activation link."
            }

        session_token = create_stateless_session_token(user["id"], user["username"], user["email"])
        created_at = datetime.now(timezone.utc).isoformat()
        expires_at = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()

        try:
            cur.execute("INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                        (session_token, user["id"], created_at, expires_at))
            conn.commit()
        except Exception:
            pass
        conn.close()

        return {
            "success": True,
            "token": session_token,
            "user": {
                "id": user["id"],
                "username": user["username"],
                "email": user["email"],
            }
        }

    @staticmethod
    def send_magic_login_link(email: str, base_url: str = "http://localhost:8000") -> Dict[str, Any]:
        email = (email or "").strip().lower()
        if not is_valid_email(email):
            return {"success": False, "message": "Invalid email address format"}

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, username FROM users WHERE LOWER(email) = ?", (email,))
        user = cur.fetchone()
        conn.close()

        # Do not generate login tokens for non-existent users
        if not user:
            return {"success": True, "message": "If an account with that email exists, a sign-in link has been sent."}

        username = user["username"]

        magic_token = generate_magic_login_token(email, username=username, expires_hours=2)
        magic_link = f"{base_url}/magic-login?token={urllib.parse.quote(magic_token)}"

        email_body = f"""
        <div style="font-family: sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; background: #0f172a; color: #f8fafc; border-radius: 12px; border: 1px solid #334155;">
          <h2 style="color: #22c55e; margin-top: 0;">Instant 1-Click Sign In</h2>
          <p>Hello <strong>{html.escape(username)}</strong>,</p>
          <p>Click the button below to sign in instantly to your Trader Queen account on this device:</p>
          <div style="text-align: center; margin: 30px 0;">
            <a href="{magic_link}" style="background-color: #16a34a; color: #ffffff; padding: 12px 24px; text-decoration: none; font-weight: bold; border-radius: 8px; display: inline-block;">Sign In Instantly</a>
          </div>
          <p style="font-size: 12px; color: #94a3b8;">Or copy and paste this link into your browser:<br><a href="{magic_link}" style="color: #38bdf8;">{html.escape(magic_link)}</a></p>
          <p style="font-size: 12px; color: #64748b; margin-top: 30px; border-top: 1px solid #334155; padding-top: 15px;">This instant sign-in link expires in 2 hours.</p>
        </div>
        """
        email_sent, email_status = send_email_message(email, "Instant 1-Click Sign In - Trader Queen", email_body, magic_token, "magic_login")

        res = {
            "success": True,
            "email": email,
            "email_sent": email_sent,
            "email_status": email_status
        }
        if email_sent:
            res["message"] = f"A 1-click sign-in link has been sent to {email}. Please check your inbox."
        else:
            res["message"] = f"Could not send sign-in link: {email_status}."
            res["dev_magic_link"] = magic_link

        return res

    @staticmethod
    def magic_login(token: str) -> Dict[str, Any]:
        t_info = verify_magic_login_token(token)
        if not t_info:
            return {"success": False, "message": "Invalid or expired sign-in link"}
        if t_info.get("expired"):
            return {"success": False, "message": "Sign-in link has expired. Please request a new one."}

        data = t_info["data"]
        email = data["e"]
        username = data.get("u") or email.split("@")[0]

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, username, email FROM users WHERE LOWER(email) = ?", (email.lower(),))
        user = cur.fetchone()

        if not user:
            conn.close()
            return {"success": False, "message": "No account found for this email address. Please register first."}

        user_id = user["id"]
        username = user["username"]
        cur.execute("UPDATE users SET is_active = 1 WHERE id = ?", (user_id,))
        conn.commit()
        conn.close()

        session_token = create_stateless_session_token(user_id, username, email)
        return {
            "success": True,
            "message": f"Successfully signed in as {username}!",
            "session_token": session_token,
            "username": username,
            "email": email
        }

    @staticmethod
    def request_password_reset(email: str, base_url: str = "http://localhost:8000") -> Dict[str, Any]:
        email = (email or "").strip().lower()
        if not is_valid_email(email):
            return {"success": False, "message": "Invalid email address format"}

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, username FROM users WHERE LOWER(email) = ?", (email,))
        user = cur.fetchone()

        if not user:
            remote_u = remote_get_user(email)
            if remote_u:
                user = remote_u

        if not user:
            conn.close()
            # Security: Don't disclose if email exists
            return {"success": True, "message": "If an account with that email exists, password reset instructions have been sent."}

        reset_token = generate_password_reset_token(user["id"], user["username"], email, expires_hours=2)
        expires_at = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()

        try:
            cur.execute("UPDATE users SET reset_token = ?, reset_expires_at = ? WHERE id = ?",
                        (reset_token, expires_at, user["id"]))
            conn.commit()
        except Exception:
            pass
        conn.close()

        reset_link = f"{base_url}/reset-password?token={urllib.parse.quote(reset_token)}"
        email_body = f"""
        <div style="font-family: sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; background: #0f172a; color: #f8fafc; border-radius: 12px; border: 1px solid #334155;">
          <h2 style="color: #f59e0b; margin-top: 0;">Reset Your Password</h2>
          <p>Hello <strong>{html.escape(user['username'])}</strong>,</p>
          <p>We received a request to reset your password for your Trader Queen account. Click below to set a new password:</p>
          <div style="text-align: center; margin: 30px 0;">
            <a href="{reset_link}" style="background-color: #d97706; color: #ffffff; padding: 12px 24px; text-decoration: none; font-weight: bold; border-radius: 8px; display: inline-block;">Reset Password</a>
          </div>
          <p style="font-size: 12px; color: #94a3b8;">Or copy and paste this link into your browser:<br><a href="{reset_link}" style="color: #38bdf8;">{html.escape(reset_link)}</a></p>
          <p style="font-size: 12px; color: #64748b; margin-top: 30px; border-top: 1px solid #334155; padding-top: 15px;">If you did not request this, please ignore this email. This link expires in 2 hours.</p>
        </div>
        """
        email_sent, email_status = send_email_message(email, "Reset Your Trader Queen Password", email_body, reset_token, "reset")

        res = {
            "success": True,
            "email": email,
            "email_sent": email_sent,
            "email_status": email_status
        }
        if email_sent:
            res["message"] = "Password reset instructions have been sent to your email. Please check your inbox."
        else:
            res["message"] = f"Password reset was requested, but email could not be delivered: {email_status}."
            res["dev_reset_link"] = reset_link

        return res

    @staticmethod
    def reset_password(token: str, new_password: str) -> Dict[str, Any]:
        if len(new_password) < 10:
            return {"success": False, "message": "Password must be at least 10 characters long"}
        if len(new_password) > 128:
            return {"success": False, "message": "Password is too long (maximum 128 characters)"}

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, username, email, reset_expires_at FROM users WHERE reset_token = ?", (token,))
        user = cur.fetchone()

        if user:
            expires = datetime.fromisoformat(user["reset_expires_at"])
            if datetime.now(timezone.utc) > expires:
                conn.close()
                return {"success": False, "message": "Password reset link has expired"}
            user_id = user["id"]
            username = user["username"]
            email = user["email"]
        else:
            t_info = verify_password_reset_token(token)
            if not t_info:
                conn.close()
                return {"success": False, "message": "Invalid or expired password reset link"}
            if t_info["expired"]:
                conn.close()
                return {"success": False, "message": "Password reset link has expired"}
            d = t_info["data"]
            username = d["u"]
            email = d["e"]
            user_id = d.get("uid", 1)

        password_hash, salt = hash_password(new_password)
        cur.execute("""
            INSERT INTO users (id, username, email, password_hash, salt, is_active, reset_token, reset_expires_at, created_at)
            VALUES (?, ?, ?, ?, ?, 1, NULL, NULL, ?)
            ON CONFLICT(id) DO UPDATE SET
                password_hash = excluded.password_hash,
                salt = excluded.salt,
                reset_token = NULL,
                reset_expires_at = NULL,
                is_active = 1
        """, (user_id, username, email, password_hash, salt, datetime.now(timezone.utc).isoformat()))
        # Invalidate any existing sessions for this user upon password reset
        cur.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        conn.commit()
        conn.close()

        remote_save_user(user_id, username, email, password_hash, salt, is_active=1)
        return {"success": True, "message": "Password has been successfully updated! You can now log in."}

    @staticmethod
    def get_user_from_token(session_token: str) -> Optional[Dict[str, Any]]:
        if not session_token:
            return None
        session_token = session_token.strip()

        # 1. First check local sessions table
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("""
                SELECT u.id, u.username, u.email, s.expires_at
                FROM sessions s
                JOIN users u ON s.user_id = u.id
                WHERE s.token = ?
            """, (session_token,))
            row = cur.fetchone()
            if row:
                expires = datetime.fromisoformat(row["expires_at"])
                if datetime.now(timezone.utc) <= expires:
                    conn.close()
                    return {
                        "id": row["id"],
                        "username": row["username"],
                        "email": row["email"]
                    }
            conn.close()
        except Exception:
            pass

        # 2. Cryptographic stateless token verification (cross-container support)
        data = verify_stateless_session_token(session_token)
        if data:
            return {
                "id": data.get("uid"),
                "username": data.get("u"),
                "email": data.get("e")
            }

        return None

    @staticmethod
    def save_exchange_key(user_id: int, exchange_id: str, api_key: str, api_secret: str, api_passphrase: str = "", testnet: bool = True, conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
        should_close = False
        if conn is None:
            conn = get_db_connection()
            should_close = True
        try:
            cur = conn.cursor()
            created_at = datetime.now(timezone.utc).isoformat()
            enc_api_key = encrypt_secret(api_key)
            enc_api_secret = encrypt_secret(api_secret)
            enc_api_passphrase = encrypt_secret(api_passphrase) if api_passphrase else ""
            cur.execute("""
                INSERT INTO exchange_keys (user_id, exchange_id, api_key, api_secret, api_passphrase, testnet, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, exchange_id) DO UPDATE SET
                    api_key = excluded.api_key,
                    api_secret = excluded.api_secret,
                    api_passphrase = excluded.api_passphrase,
                    testnet = excluded.testnet,
                    created_at = excluded.created_at
            """, (user_id, exchange_id, enc_api_key, enc_api_secret, enc_api_passphrase, 1 if testnet else 0, created_at))
            if should_close:
                conn.commit()

            # Sync to remote persistent store
            remote_save_exchange_key(user_id, exchange_id, enc_api_key, enc_api_secret, enc_api_passphrase, testnet)

            return {"success": True, "message": f"Credentials for {exchange_id} saved successfully"}
        except Exception as e:
            print(f"[AUTH ERROR] Failed to save exchange key for user {user_id} and {exchange_id}: {e}")
            raise RuntimeError(f"Failed to save exchange key: {e}") from e
        finally:
            if should_close:
                try:
                    conn.close()
                except Exception:
                    pass

    @staticmethod
    def get_exchange_keys(user_id: int) -> list:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT exchange_id, api_key, api_passphrase, testnet, created_at FROM exchange_keys WHERE user_id = ?", (user_id,))
        rows = cur.fetchall()

        if not rows:
            # Check remote persistent store and sync into local DB
            rem_keys = remote_get_exchange_keys(user_id)
            if rem_keys:
                for rk in rem_keys:
                    try:
                        cur.execute("""
                            INSERT INTO exchange_keys (user_id, exchange_id, api_key, api_secret, api_passphrase, testnet, created_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(user_id, exchange_id) DO UPDATE SET
                                api_key = excluded.api_key,
                                api_secret = excluded.api_secret,
                                api_passphrase = excluded.api_passphrase,
                                testnet = excluded.testnet
                        """, (
                            user_id,
                            rk["exchange_id"],
                            rk["api_key"],
                            rk["api_secret"],
                            rk.get("api_passphrase", ""),
                            1 if rk.get("testnet", True) else 0,
                            rk.get("created_at", datetime.now(timezone.utc).isoformat())
                        ))
                    except Exception:
                        pass
                conn.commit()
                cur.execute("SELECT exchange_id, api_key, api_passphrase, testnet, created_at FROM exchange_keys WHERE user_id = ?", (user_id,))
                rows = cur.fetchall()

        conn.close()

        result = []
        for r in rows:
            raw_key = decrypt_secret(r["api_key"])
            key_preview = f"••••{raw_key[-4:]}" if len(raw_key) >= 4 else "••••"
            has_pass = bool(decrypt_secret(r["api_passphrase"])) if r["api_passphrase"] else False
            result.append({
                "exchange_id": r["exchange_id"],
                "api_key_masked": key_preview,
                "has_passphrase": has_pass,
                "testnet": bool(r["testnet"]),
                "created_at": r["created_at"],
            })
        return result

    @staticmethod
    def get_raw_exchange_key(user_id: int, exchange_id: str) -> Optional[Dict[str, Any]]:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT api_key, api_secret, api_passphrase, testnet FROM exchange_keys WHERE user_id = ? AND exchange_id = ?",
                    (user_id, exchange_id))
        row = cur.fetchone()

        if not row:
            rem_keys = remote_get_exchange_keys(user_id)
            for rk in rem_keys:
                if rk["exchange_id"].lower() == exchange_id.lower():
                    conn.close()
                    return {
                        "api_key": decrypt_secret(rk["api_key"]),
                        "api_secret": decrypt_secret(rk["api_secret"]),
                        "api_passphrase": decrypt_secret(rk["api_passphrase"]) if rk.get("api_passphrase") else "",
                        "testnet": bool(rk.get("testnet", True)),
                    }

        conn.close()
        if row:
            return {
                "api_key": decrypt_secret(row["api_key"]),
                "api_secret": decrypt_secret(row["api_secret"]),
                "api_passphrase": decrypt_secret(row["api_passphrase"]) if row["api_passphrase"] else "",
                "testnet": bool(row["testnet"]),
            }
        return None

    @staticmethod
    def get_latest_raw_exchange_key() -> Optional[Dict[str, Any]]:
        """Returns the most recently saved raw exchange key (decrypted) for background bot execution."""
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT exchange_id, api_key, api_secret, api_passphrase, testnet FROM exchange_keys ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        conn.close()
        if row:
            return {
                "exchange_id": row["exchange_id"],
                "api_key": decrypt_secret(row["api_key"]),
                "api_secret": decrypt_secret(row["api_secret"]),
                "api_passphrase": decrypt_secret(row["api_passphrase"]) if row["api_passphrase"] else "",
                "testnet": bool(row["testnet"]),
            }
        return None

    @staticmethod
    def delete_all_accounts() -> Dict[str, Any]:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM sessions")
        cur.execute("DELETE FROM exchange_keys")
        cur.execute("DELETE FROM email_outbox")
        cur.execute("DELETE FROM users")
        conn.commit()
        conn.close()
        remote_delete_all_accounts()
        return {"success": True, "message": "All registered user accounts and sessions have been permanently deleted"}

    @staticmethod
    def delete_account(user_id: int) -> Dict[str, Any]:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        cur.execute("DELETE FROM exchange_keys WHERE user_id = ?", (user_id,))
        cur.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        conn.close()
        remote_delete_account(user_id)
        return {"success": True, "message": "Account deleted successfully"}

    @staticmethod
    def get_latest_email_preview(email: Optional[str] = None) -> Optional[Dict[str, Any]]:
        conn = get_db_connection()
        cur = conn.cursor()
        if email:
            cur.execute("SELECT * FROM email_outbox WHERE LOWER(to_email) = ? ORDER BY id DESC LIMIT 1", (email.lower(),))
        else:
            cur.execute("SELECT * FROM email_outbox ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        conn.close()
        if row:
            return dict(row)
        return None


# Initialize DB on module import
init_auth_db()

auth_service = AuthService

