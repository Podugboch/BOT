"""
Authentication and User Account Management module for Crypto Trading Bot.
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
import urllib.request
import urllib.error
from dotenv import load_dotenv

load_dotenv()

from secret_manager import encrypt_secret, decrypt_secret, is_encrypted, SENSITIVE_ENV_KEYS

# Detect serverless environment for DB path
if os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
    DB_PATH = Path("/tmp/crypto_bot_users.db")
else:
    DB_PATH = Path(__file__).resolve().parent / "users.db"


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
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
    conn.close()


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


def set_system_setting(key: str, value: str):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        stored_value = encrypt_secret(value) if key in SENSITIVE_ENV_KEYS and value else value
        cur.execute("""
            INSERT INTO system_settings (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
        """, (key, stored_value, datetime.now(timezone.utc).isoformat()))
        conn.commit()
        conn.close()
        if key not in SENSITIVE_ENV_KEYS:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)
    except Exception as e:
        print(f"[AUTH ERROR] Failed to set system setting {key}: {e}")


def get_resend_verified_domains(resend_key: str) -> list:
    """Queries Resend API for verified sending domains associated with this API key."""
    if not resend_key:
        return []
    try:
        req = urllib.request.Request(
            "https://api.resend.com/domains",
            headers={
                "Authorization": f"Bearer {resend_key}",
                "User-Agent": "CryptoTradingBot/2.0"
            }
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            domains = data.get("data", [])
            return [d["name"] for d in domains if d.get("status") == "verified"]
    except Exception:
        return []


def get_email_settings() -> Dict[str, Any]:
    provider = (get_system_setting("EMAIL_PROVIDER", "") or "").strip().lower()
    smtp_host = (get_system_setting("SMTP_HOST", "") or "").strip()
    smtp_port = (get_system_setting("SMTP_PORT", "587") or "587").strip()
    smtp_user = (get_system_setting("SMTP_USER", "") or "").strip()
    smtp_pass = (get_system_setting("SMTP_PASS", "") or "").strip()
    from_email = (get_system_setting("FROM_EMAIL", "") or "").strip()
    resend_key = (get_system_setting("RESEND_API_KEY", "") or "").strip()
    brevo_key = (get_system_setting("BREVO_API_KEY", "") or "").strip()
    sendgrid_key = (get_system_setting("SENDGRID_API_KEY", "") or "").strip()

    has_smtp = bool(smtp_host and smtp_user and smtp_pass)
    has_resend = bool(resend_key)
    has_brevo = bool(brevo_key)
    has_sendgrid = bool(sendgrid_key)
    is_configured = has_smtp or has_resend or has_brevo or has_sendgrid

    # Check if Resend has verified domains on account
    resend_domains = []
    if has_resend:
        resend_domains = get_resend_verified_domains(resend_key)
        # If user has a verified custom domain on Resend but from_email is unset or still on sandbox, auto-upgrade
        if resend_domains and ("onboarding@resend.dev" in from_email.lower() or not from_email):
            auto_from = f"Crypto Trading Bot <noreply@{resend_domains[0]}>"
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
            val = str(settings[k]).strip()
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
            sender = f"Crypto Trading Bot <noreply@{verified[0]}>"
            set_system_setting("FROM_EMAIL", sender)
        else:
            sender = "Crypto Trading Bot <onboarding@resend.dev>"

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
                "User-Agent": "CryptoTradingBot/2.0"
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
                "and set FROM_EMAIL to an address on that domain (e.g. Crypto Bot <noreply@yourdomain.com>), or switch to Brevo API or Gmail SMTP."
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

    sender_name = "Crypto Trading Bot"
    sender_email = from_email or "noreply@cryptobot.local"
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
                "User-Agent": "CryptoTradingBot/2.0"
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

    sender_name = "Crypto Trading Bot"
    sender_email = from_email or "noreply@cryptobot.local"
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
                "User-Agent": "CryptoTradingBot/2.0"
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

    resend_key = get_system_setting("RESEND_API_KEY", "") or ""
    brevo_key = get_system_setting("BREVO_API_KEY", "") or ""
    sendgrid_key = get_system_setting("SENDGRID_API_KEY", "") or ""

    smtp_host = get_system_setting("SMTP_HOST", "") or get_system_setting("EMAIL_SERVER", "") or ""
    smtp_port = int(get_system_setting("SMTP_PORT", "587") or get_system_setting("EMAIL_PORT", "587") or "587")
    smtp_user = get_system_setting("SMTP_USER", "") or get_system_setting("EMAIL_USER", "") or ""
    smtp_pass = get_system_setting("SMTP_PASS", "") or get_system_setting("EMAIL_PASSWORD", "") or ""

    from_email = get_system_setting("FROM_EMAIL", "") or smtp_user or ""

    # Define dispatch candidates
    candidates = []

    def run_brevo():
        sender = from_email or "Crypto Trading Bot <noreply@cryptobot.local>"
        return _send_via_brevo(to_email, subject, body, brevo_key, sender, email_type)

    def run_sendgrid():
        sender = from_email or "Crypto Trading Bot <noreply@cryptobot.local>"
        return _send_via_sendgrid(to_email, subject, body, sendgrid_key, sender, email_type)

    def run_resend():
        sender = from_email or "Crypto Trading Bot <onboarding@resend.dev>"
        return _send_via_resend(to_email, subject, body, resend_key, sender, email_type)

    def run_smtp():
        s_from = from_email or smtp_user or "noreply@cryptobot.local"
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
    to_email = to_email.strip().lower()
    if not re.match(r"[^@]+@[^@]+\.[^@]+", to_email):
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
      <p>This is a confirmation test email sent from your <strong>Crypto Trading Bot</strong> platform.</p>
      <p>Your email service (<strong>{settings['provider'].upper()}</strong>) is configured and working properly!</p>
      <div style="background: #1e293b; padding: 12px 16px; border-radius: 8px; margin: 20px 0; font-family: monospace; font-size: 12px;">
        Provider: {settings['provider'].upper()}<br>
        Recipient: {to_email}<br>
        Dispatched at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
      </div>
      <p style="font-size: 12px; color: #64748b; border-top: 1px solid #334155; padding-top: 15px;">Crypto Trading Bot Verification Engine</p>
    </div>
    """

    success, status = send_email_message(to_email, "✓ Crypto Trading Bot - Email Test Confirmation", test_body, test_token, "test")
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
    def register(username: str, email: str, password: str, base_url: str = "http://localhost:8000") -> Dict[str, Any]:
        username = username.strip()
        email = email.strip().lower()

        if len(username) < 3:
            return {"success": False, "message": "Username must be at least 3 characters long"}
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            return {"success": False, "message": "Invalid email address format"}
        if len(password) < 6:
            return {"success": False, "message": "Password must be at least 6 characters long"}

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
        existing = cur.fetchone()
        if existing:
            conn.close()
            return {"success": False, "message": "A user with this username or email already exists"}

        password_hash, salt = hash_password(password)
        activation_token = secrets.token_urlsafe(32)
        expires_at = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
        created_at = datetime.now(timezone.utc).isoformat()

        cur.execute("""
            INSERT INTO users (username, email, password_hash, salt, is_active, activation_token, activation_expires_at, created_at)
            VALUES (?, ?, ?, ?, 0, ?, ?, ?)
        """, (username, email, password_hash, salt, activation_token, expires_at, created_at))
        user_id = cur.lastrowid
        conn.commit()
        conn.close()

        # Send activation email to recipient
        activation_link = f"{base_url}/activate?token={activation_token}"
        email_body = f"""
        <div style="font-family: sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; background: #0f172a; color: #f8fafc; border-radius: 12px; border: 1px solid #334155;">
          <h2 style="color: #22c55e; margin-top: 0;">Activate Your Crypto Trading Bot Account</h2>
          <p>Hello <strong>{username}</strong>,</p>
          <p>Thank you for registering on the Crypto Trading Bot platform. Please click the button below to accept and activate your account:</p>
          <div style="text-align: center; margin: 30px 0;">
            <a href="{activation_link}" style="background-color: #16a34a; color: #ffffff; padding: 12px 24px; text-decoration: none; font-weight: bold; border-radius: 8px; display: inline-block;">Activate Account</a>
          </div>
          <p style="font-size: 12px; color: #94a3b8;">Or copy and paste this link into your browser:<br><a href="{activation_link}" style="color: #38bdf8;">{activation_link}</a></p>
          <p style="font-size: 12px; color: #64748b; margin-top: 30px; border-top: 1px solid #334155; padding-top: 15px;">This activation link expires in 24 hours.</p>
        </div>
        """
        email_sent, email_status = send_email_message(email, "Activate Your Crypto Trading Bot Account", email_body, activation_token, "activation")

        res = {
            "success": True,
            "user_id": user_id,
            "email": email,
            "email_sent": email_sent,
            "email_status": email_status
        }
        if email_sent:
            res["message"] = f"Account created for {username}. A verification email has been sent to {email}. Please check your inbox to activate your account."
        else:
            res["message"] = f"Account created for {username}. Email delivery warning: {email_status}"
            res["dev_activation_link"] = activation_link

        return res

    @staticmethod
    def activate_account(token: str) -> Dict[str, Any]:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, username, email, is_active, activation_expires_at FROM users WHERE activation_token = ?", (token,))
        user = cur.fetchone()

        if not user:
            conn.close()
            return {"success": False, "message": "Invalid or already used activation token"}

        if user["is_active"] == 1:
            conn.close()
            return {"success": True, "message": "Account is already activated. You can log in now."}

        # Check expiration
        expires = datetime.fromisoformat(user["activation_expires_at"])
        if datetime.now(timezone.utc) > expires:
            conn.close()
            return {"success": False, "message": "Activation link has expired. Please request a new one."}

        cur.execute("UPDATE users SET is_active = 1, activation_token = NULL WHERE id = ?", (user["id"],))
        conn.commit()
        conn.close()
        return {"success": True, "message": f"Account {user['username']} successfully activated! You can now log in."}

    @staticmethod
    def resend_activation_email(email: str, base_url: str = "http://localhost:8000") -> Dict[str, Any]:
        email = email.strip().lower()
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            return {"success": False, "message": "Invalid email address format"}

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, username, email, is_active FROM users WHERE LOWER(email) = ?", (email,))
        user = cur.fetchone()

        if not user:
            conn.close()
            # Do not disclose if user exists, but confirm message
            return {"success": True, "message": f"If an account is associated with {email} and pending verification, a new verification link has been sent."}

        if user["is_active"] == 1:
            conn.close()
            return {"success": False, "already_active": True, "message": "Account is already activated. Please sign in."}

        activation_token = secrets.token_urlsafe(32)
        expires_at = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()

        cur.execute("UPDATE users SET activation_token = ?, activation_expires_at = ? WHERE id = ?",
                    (activation_token, expires_at, user["id"]))
        conn.commit()
        conn.close()

        activation_link = f"{base_url}/activate?token={activation_token}"
        email_body = f"""
        <div style="font-family: sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; background: #0f172a; color: #f8fafc; border-radius: 12px; border: 1px solid #334155;">
          <h2 style="color: #22c55e; margin-top: 0;">Your New Verification Link</h2>
          <p>Hello <strong>{user['username']}</strong>,</p>
          <p>You requested a new verification link for your Crypto Trading Bot account. Please click the button below to accept and activate your account:</p>
          <div style="text-align: center; margin: 30px 0;">
            <a href="{activation_link}" style="background-color: #16a34a; color: #ffffff; padding: 12px 24px; text-decoration: none; font-weight: bold; border-radius: 8px; display: inline-block;">Activate Account</a>
          </div>
          <p style="font-size: 12px; color: #94a3b8;">Or copy and paste this link into your browser:<br><a href="{activation_link}" style="color: #38bdf8;">{activation_link}</a></p>
          <p style="font-size: 12px; color: #64748b; margin-top: 30px; border-top: 1px solid #334155; padding-top: 15px;">This verification link expires in 24 hours.</p>
        </div>
        """
        email_sent, email_status = send_email_message(email, "Activate Your Crypto Trading Bot Account (New Verification Link)", email_body, activation_token, "activation")

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
    def login(username_or_email: str, password: str) -> Dict[str, Any]:
        val = username_or_email.strip().lower()
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE LOWER(username) = ? OR LOWER(email) = ?", (val, val))
        user = cur.fetchone()

        if not user:
            conn.close()
            return {"success": False, "message": "Invalid username/email or password"}

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

        session_token = secrets.token_urlsafe(32)
        created_at = datetime.now(timezone.utc).isoformat()
        expires_at = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()

        cur.execute("INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                    (session_token, user["id"], created_at, expires_at))
        conn.commit()
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
    def request_password_reset(email: str, base_url: str = "http://localhost:8000") -> Dict[str, Any]:
        email = email.strip().lower()
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, username FROM users WHERE LOWER(email) = ?", (email,))
        user = cur.fetchone()

        if not user:
            conn.close()
            # Security: Don't disclose if email exists
            return {"success": True, "message": "If an account with that email exists, password reset instructions have been sent."}

        reset_token = secrets.token_urlsafe(32)
        expires_at = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()

        cur.execute("UPDATE users SET reset_token = ?, reset_expires_at = ? WHERE id = ?",
                    (reset_token, expires_at, user["id"]))
        conn.commit()
        conn.close()

        reset_link = f"{base_url}/reset-password?token={reset_token}"
        email_body = f"""
        <div style="font-family: sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; background: #0f172a; color: #f8fafc; border-radius: 12px; border: 1px solid #334155;">
          <h2 style="color: #f59e0b; margin-top: 0;">Reset Your Password</h2>
          <p>Hello <strong>{user['username']}</strong>,</p>
          <p>We received a request to reset your password for your Crypto Trading Bot account. Click below to set a new password:</p>
          <div style="text-align: center; margin: 30px 0;">
            <a href="{reset_link}" style="background-color: #d97706; color: #ffffff; padding: 12px 24px; text-decoration: none; font-weight: bold; border-radius: 8px; display: inline-block;">Reset Password</a>
          </div>
          <p style="font-size: 12px; color: #94a3b8;">Or copy and paste this link into your browser:<br><a href="{reset_link}" style="color: #38bdf8;">{reset_link}</a></p>
          <p style="font-size: 12px; color: #64748b; margin-top: 30px; border-top: 1px solid #334155; padding-top: 15px;">If you did not request this, please ignore this email. This link expires in 2 hours.</p>
        </div>
        """
        email_sent, email_status = send_email_message(email, "Reset Your Crypto Trading Bot Password", email_body, reset_token, "reset")

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
        if len(new_password) < 6:
            return {"success": False, "message": "Password must be at least 6 characters long"}

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, reset_expires_at FROM users WHERE reset_token = ?", (token,))
        user = cur.fetchone()

        if not user:
            conn.close()
            return {"success": False, "message": "Invalid or expired password reset link"}

        expires = datetime.fromisoformat(user["reset_expires_at"])
        if datetime.now(timezone.utc) > expires:
            conn.close()
            return {"success": False, "message": "Password reset link has expired"}

        password_hash, salt = hash_password(new_password)
        cur.execute("UPDATE users SET password_hash = ?, salt = ?, reset_token = NULL, reset_expires_at = NULL WHERE id = ?",
                    (password_hash, salt, user["id"]))
        conn.commit()
        conn.close()
        return {"success": True, "message": "Password has been successfully updated! You can now log in."}

    @staticmethod
    def get_user_from_token(session_token: str) -> Optional[Dict[str, Any]]:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT u.id, u.username, u.email, s.expires_at
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.token = ?
        """, (session_token,))
        row = cur.fetchone()
        conn.close()

        if not row:
            return None

        expires = datetime.fromisoformat(row["expires_at"])
        if datetime.now(timezone.utc) > expires:
            return None

        return {
            "id": row["id"],
            "username": row["username"],
            "email": row["email"]
        }

    @staticmethod
    def save_exchange_key(user_id: int, exchange_id: str, api_key: str, api_secret: str, api_passphrase: str = "", testnet: bool = True) -> Dict[str, Any]:
        conn = get_db_connection()
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
        conn.commit()
        conn.close()
        return {"success": True, "message": f"Credentials for {exchange_id} saved successfully"}

    @staticmethod
    def get_exchange_keys(user_id: int) -> list:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT exchange_id, api_key, api_passphrase, testnet, created_at FROM exchange_keys WHERE user_id = ?", (user_id,))
        rows = cur.fetchall()
        conn.close()

        result = []
        for r in rows:
            raw_key = decrypt_secret(r["api_key"])
            key_preview = raw_key[:4] + "***" + raw_key[-4:] if len(raw_key) > 8 else "***"
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
