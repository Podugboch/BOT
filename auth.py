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
from datetime import datetime, timezone, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import urllib.request
import urllib.error
from dotenv import load_dotenv

load_dotenv()

# Detect serverless environment for DB path
if os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
    DB_PATH = Path("/tmp/crypto_bot_users.db")
else:
    DB_PATH = Path(__file__).resolve().parent / "users.db"


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


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
    conn.close()


def get_system_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT value FROM system_settings WHERE key = ?", (key,))
        row = cur.fetchone()
        conn.close()
        if row and row["value"]:
            return row["value"]
    except Exception:
        pass
    return os.getenv(key, default)


def set_system_setting(key: str, value: str):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO system_settings (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
        """, (key, value, datetime.now(timezone.utc).isoformat()))
        conn.commit()
        conn.close()
        os.environ[key] = value
    except Exception as e:
        print(f"[AUTH ERROR] Failed to set system setting {key}: {e}")


def get_email_settings() -> Dict[str, Any]:
    smtp_host = get_system_setting("SMTP_HOST", "") or ""
    smtp_port = get_system_setting("SMTP_PORT", "587") or "587"
    smtp_user = get_system_setting("SMTP_USER", "") or ""
    smtp_pass = get_system_setting("SMTP_PASS", "") or ""
    from_email = get_system_setting("FROM_EMAIL", "") or ""
    resend_key = get_system_setting("RESEND_API_KEY", "") or ""

    has_smtp = bool(smtp_host and smtp_user and smtp_pass)
    has_resend = bool(resend_key)
    is_configured = has_smtp or has_resend

    active_provider = "none"
    if has_resend:
        active_provider = "resend"
    elif has_smtp:
        if "gmail" in smtp_host.lower():
            active_provider = "gmail"
        else:
            active_provider = "smtp"

    return {
        "is_configured": is_configured,
        "provider": active_provider,
        "smtp_host": smtp_host,
        "smtp_port": smtp_port,
        "smtp_user": smtp_user,
        "smtp_pass_masked": f"{smtp_pass[:2]}••••{smtp_pass[-2:]}" if len(smtp_pass) > 4 else ("••••" if smtp_pass else ""),
        "from_email": from_email,
        "resend_key_masked": f"{resend_key[:5]}••••" if len(resend_key) > 5 else ("••••" if resend_key else "")
    }


def save_email_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    for k in ["SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASS", "FROM_EMAIL", "RESEND_API_KEY"]:
        if k in settings:
            val = str(settings[k]).strip()
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


def send_email_message(to_email: str, subject: str, body: str, token: str, email_type: str) -> Tuple[bool, str]:
    """Stores in outbox for audit and delivers via Resend API or SMTP.
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

    # Method 1: Check for Resend API Key (Best for serverless / Vercel cloud delivery)
    resend_key = get_system_setting("RESEND_API_KEY")
    if resend_key:
        try:
            import urllib.request
            from_email = get_system_setting("FROM_EMAIL") or get_system_setting("SMTP_USER") or "Crypto Trading Bot <onboarding@resend.dev>"
            req = urllib.request.Request(
                "https://api.resend.com/emails",
                data=json.dumps({
                    "from": from_email,
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
                    msg = f"Delivered {email_type} email to {to_email} via Resend API"
                    print(f"[AUTH EMAIL] {msg}")
                    return True, msg
                else:
                    err_msg = f"Resend API returned status code {resp.status}"
                    print(f"[AUTH EMAIL] {err_msg}")
                    if not get_system_setting("SMTP_HOST"):
                        return False, err_msg
        except urllib.error.HTTPError as e:
            try:
                raw_err = e.read().decode("utf-8")
                err_json = json.loads(raw_err)
                detail = err_json.get("message", raw_err)
            except Exception:
                detail = str(e)
            err_msg = f"Resend API error ({e.code}): {detail}"
            print(f"[AUTH EMAIL] {err_msg}")
            if not get_system_setting("SMTP_HOST"):
                return False, err_msg
        except Exception as e:
            err_msg = f"Resend delivery failed: {e}"
            print(f"[AUTH EMAIL] {err_msg}")
            if not get_system_setting("SMTP_HOST"):
                return False, err_msg

    # Method 2: Check for standard SMTP configuration (port 465 SSL or 587 STARTTLS)
    smtp_host = get_system_setting("SMTP_HOST") or get_system_setting("EMAIL_SERVER")
    smtp_port = int(get_system_setting("SMTP_PORT") or get_system_setting("EMAIL_PORT") or "587")
    smtp_user = get_system_setting("SMTP_USER") or get_system_setting("EMAIL_USER")
    smtp_pass = get_system_setting("SMTP_PASS") or get_system_setting("EMAIL_PASSWORD")
    smtp_from = get_system_setting("FROM_EMAIL") or smtp_user or "noreply@cryptobot.local"

    if smtp_host:
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = smtp_from
            msg["To"] = to_email
            msg.attach(MIMEText(body, "html"))

            if smtp_port == 465:
                with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=12) as server:
                    if smtp_user and smtp_pass:
                        server.login(smtp_user, smtp_pass)
                    server.sendmail(smtp_from, [to_email], msg.as_string())
            else:
                with smtplib.SMTP(smtp_host, smtp_port, timeout=12) as server:
                    server.starttls()
                    if smtp_user and smtp_pass:
                        server.login(smtp_user, smtp_pass)
                    server.sendmail(smtp_from, [to_email], msg.as_string())

            msg = f"Delivered {email_type} email to {to_email} via SMTP ({smtp_host}:{smtp_port})"
            print(f"[AUTH EMAIL] {msg}")
            return True, msg
        except Exception as e:
            err_msg = f"SMTP delivery failed to {to_email} ({smtp_host}:{smtp_port}): {e}"
            print(f"[AUTH EMAIL] {err_msg}")
            return False, err_msg

    not_configured_msg = "No SMTP_HOST or RESEND_API_KEY configured on this server."
    print(f"[AUTH EMAIL] Notice: {not_configured_msg} Recorded email in system outbox for {to_email}.")
    return False, not_configured_msg


def test_email_connection(to_email: str) -> Dict[str, Any]:
    """Test sending a verification email to the specified address."""
    to_email = to_email.strip().lower()
    if not re.match(r"[^@]+@[^@]+\.[^@]+", to_email):
        return {"success": False, "message": "Invalid test email address format"}

    settings = get_email_settings()
    if not settings["is_configured"]:
        return {
            "success": False,
            "message": "Neither SMTP credentials nor a Resend API Key is configured yet. Please enter your email credentials first.",
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
        Provider: {settings['provider']}<br>
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
        "provider": settings["provider"]
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
            res["message"] = f"Account created for {username}. A verification email has been sent to {email}. Please check your email to activate your account."
        else:
            res["message"] = f"Account created for {username}. Notice: {email_status} To receive verification emails, please configure SMTP or Resend credentials in Email Settings."
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
        cur.execute("""
            INSERT INTO exchange_keys (user_id, exchange_id, api_key, api_secret, api_passphrase, testnet, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, exchange_id) DO UPDATE SET
                api_key = excluded.api_key,
                api_secret = excluded.api_secret,
                api_passphrase = excluded.api_passphrase,
                testnet = excluded.testnet,
                created_at = excluded.created_at
        """, (user_id, exchange_id, api_key, api_secret, api_passphrase, 1 if testnet else 0, created_at))
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
            key_preview = r["api_key"][:4] + "***" + r["api_key"][-4:] if len(r["api_key"]) > 8 else "***"
            result.append({
                "exchange_id": r["exchange_id"],
                "api_key_masked": key_preview,
                "has_passphrase": bool(r["api_passphrase"]),
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
            return dict(row)
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
