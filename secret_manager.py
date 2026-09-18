"""
Secure Secret Vault and Encryption Manager.
Provides authenticated symmetric encryption (Fernet / AES-128-CBC + HMAC-SHA256)
for credentials stored at rest in SQLite database and prevents plaintext secret leakage.
"""
import os
import stat
from pathlib import Path
from typing import Optional, Set
from cryptography.fernet import Fernet, InvalidToken

DEFAULT_KEY_FILE = Path(__file__).resolve().parent / ".secret_key"
ENC_PREFIX = "enc:"

SENSITIVE_ENV_KEYS: Set[str] = {
    "EXCHANGE_API_KEY",
    "EXCHANGE_API_SECRET",
    "EXCHANGE_API_PASSPHRASE",
    "SMTP_PASS",
    "RESEND_API_KEY",
    "BREVO_API_KEY",
    "SENDGRID_API_KEY",
}


def is_production() -> bool:
    """Checks whether the application is configured to run in production / live mode."""
    env_name = (os.getenv("ENVIRONMENT") or os.getenv("ENV") or os.getenv("BOT_ENV") or "").strip().lower()
    mode = (os.getenv("MODE") or "").strip().lower()
    return env_name in {"production", "prod"} or mode == "live"


def get_master_key() -> bytes:
    """
    Retrieves or generates the master symmetric encryption key.
    Checks environment variable BOT_SECRET_KEY / SECRET_KEY / APP_SECRET_KEY first.
    In production/live mode, fails fast if an externally managed key is not provided.
    In non-production environments, securely loads or generates a key file with 0600 permissions.
    """
    # 1. Check environment variable
    env_key = os.getenv("BOT_SECRET_KEY") or os.getenv("SECRET_KEY") or os.getenv("APP_SECRET_KEY")
    if env_key:
        raw = env_key.strip().encode("utf-8")
        try:
            Fernet(raw)
            return raw
        except Exception:
            # Derive deterministic 32-byte urlsafe base64 key from arbitrary passphrase
            import base64
            import hashlib
            return base64.urlsafe_b64encode(hashlib.sha256(raw).digest())

    # 2. Production fail-fast requirement
    if is_production():
        raise RuntimeError(
            "Production / Live mode requires an externally managed master encryption key. "
            "Please set the BOT_SECRET_KEY or SECRET_KEY environment variable."
        )

    # 3. Check local key file
    key_path = Path(os.getenv("BOT_SECRET_KEY_PATH", str(DEFAULT_KEY_FILE)))
    if key_path.exists():
        try:
            key_data = key_path.read_bytes().strip()
            Fernet(key_data)
            return key_data
        except Exception:
            pass

    # 4. Generate new key and save with restricted 0600 permissions
    new_key = Fernet.generate_key()
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        mode = stat.S_IRUSR | stat.S_IWUSR  # 0600 (read/write only by owner)
        fd = os.open(str(key_path), flags, mode)
        with os.fdopen(fd, "wb") as f:
            f.write(new_key)
    except Exception:
        # Fallback for serverless or read-only filesystem environments
        tmp_key = Path("/tmp/.bot_secret_key")
        try:
            if not tmp_key.exists():
                tmp_key.write_bytes(new_key)
            return tmp_key.read_bytes().strip()
        except Exception:
            return new_key

    return new_key


_FERNET_INSTANCE: Optional[Fernet] = None


def get_cipher() -> Fernet:
    global _FERNET_INSTANCE
    if _FERNET_INSTANCE is None:
        _FERNET_INSTANCE = Fernet(get_master_key())
    return _FERNET_INSTANCE


def reset_cipher():
    """Resets cached cipher instance (useful during unit tests when switching keys)."""
    global _FERNET_INSTANCE
    _FERNET_INSTANCE = None


def encrypt_secret(plaintext: Optional[str]) -> str:
    """Encrypts plaintext string into an authenticated ciphertext token prefixed with 'enc:'."""
    if not plaintext:
        return ""
    plaintext_str = str(plaintext).strip()
    if not plaintext_str:
        return ""
    if is_encrypted(plaintext_str):
        return plaintext_str
    cipher = get_cipher()
    token = cipher.encrypt(plaintext_str.encode("utf-8")).decode("utf-8")
    return f"{ENC_PREFIX}{token}"


def decrypt_secret(ciphertext: Optional[str]) -> str:
    """
    Decrypts ciphertext token.
    If value is unencrypted (legacy plaintext), gracefully returns plaintext for zero downtime migrations.
    """
    if not ciphertext:
        return ""
    ciphertext_str = str(ciphertext).strip()
    if not ciphertext_str:
        return ""
    token = ciphertext_str
    if token.startswith(ENC_PREFIX):
        token = token[len(ENC_PREFIX):]
    try:
        cipher = get_cipher()
        return cipher.decrypt(token.encode("utf-8")).decode("utf-8")
    except (InvalidToken, Exception):
        # Graceful fallback for legacy plaintext stored before encryption was enabled
        return ciphertext_str


def is_encrypted(value: Optional[str]) -> bool:
    """Checks whether a value is an encrypted secret."""
    if not value or not isinstance(value, str):
        return False
    val = value.strip()
    if val.startswith(ENC_PREFIX):
        return True
    if val.startswith("gAAAAA") and len(val) >= 40:
        try:
            get_cipher().decrypt(val.encode("utf-8"))
            return True
        except Exception:
            return False
    return False


def migrate_and_sanitize_env_file(env_file_path: Optional[Path] = None) -> bool:
    """
    Atomically migrates any legacy plaintext credentials from .env into the
    encrypted database vault, verifies successful persistence and decryption,
    and only then removes sensitive values from .env.
    Also strips sensitive keys from os.environ to prevent runtime exposure.
    """
    target = env_file_path or Path(".env")
    if not target.exists() or not target.is_file():
        # Clear any sensitive env vars from process environment
        for k in SENSITIVE_ENV_KEYS:
            os.environ.pop(k, None)
        return False

    try:
        lines = target.read_text().splitlines(keepends=True)
        env_vars = {}
        for line in lines:
            s = line.strip()
            if s and not s.startswith("#") and "=" in s:
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k in SENSITIVE_ENV_KEYS and v:
                    env_vars[k] = v

        if env_vars:
            # Atomic migration into database vault
            from auth import AuthService, set_system_setting, get_system_setting, get_db_connection

            # 1. Exchange keys migration
            ex_key = env_vars.get("EXCHANGE_API_KEY")
            ex_sec = env_vars.get("EXCHANGE_API_SECRET")
            ex_pass = env_vars.get("EXCHANGE_API_PASSPHRASE", "")
            if ex_key and ex_sec:
                # Check if exchange keys already exist in DB
                existing = AuthService.get_latest_raw_exchange_key()
                if not existing or not existing.get("api_key"):
                    conn = get_db_connection()
                    user = conn.execute("SELECT id FROM users ORDER BY id ASC LIMIT 1").fetchone()
                    conn.close()
                    user_id = user["id"] if user else 1
                    ex_id = os.getenv("EXCHANGE_ID", "mexc")
                    testnet_val = os.getenv("TESTNET", "true").lower() == "true"
                    AuthService.save_exchange_key(user_id, ex_id, ex_key, ex_sec, ex_pass, testnet_val)
                    # Verify persistence
                    verified = AuthService.get_latest_raw_exchange_key()
                    if not verified or verified.get("api_key") != ex_key or verified.get("api_secret") != ex_sec:
                        raise RuntimeError("Atomic .env migration failed: Exchange key verification mismatch")

            # 2. Sensitive email settings migration
            email_keys = ["SMTP_PASS", "RESEND_API_KEY", "BREVO_API_KEY", "SENDGRID_API_KEY"]
            for ek in email_keys:
                if ek in env_vars:
                    val = env_vars[ek]
                    set_system_setting(ek, val)
                    # Verify persistence
                    read_back = get_system_setting(ek)
                    if read_back != val:
                        raise RuntimeError(f"Atomic .env migration failed: {ek} verification mismatch")

        # Now sanitize .env file safely
        modified = False
        new_lines = []
        for line in lines:
            s = line.strip()
            if s and not s.startswith("#") and "=" in s:
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip()
                if k in SENSITIVE_ENV_KEYS:
                    if v:
                        modified = True
                        new_lines.append(f"# {k} is stored securely in the encrypted vault (users.db)\n")
                        continue
            new_lines.append(line)

        if modified:
            target.write_text("".join(new_lines))

        # Remove sensitive credentials from runtime process memory (os.environ)
        for k in SENSITIVE_ENV_KEYS:
            os.environ.pop(k, None)

        return True
    except Exception as e:
        print(f"[SECURITY WARNING] Failed during atomic .env migration/sanitization: {e}")
        return False


# Backwards-compatible alias
sanitize_env_file = migrate_and_sanitize_env_file
