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

if os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
    DEFAULT_KEY_FILE = Path("/tmp/.secret_key")
else:
    DEFAULT_KEY_FILE = Path(__file__).resolve().parent / ".secret_key"
ENC_PREFIX = "enc:"

SENSITIVE_ENV_KEYS: Set[str] = {
    "BOT_SECRET_KEY",
    "SECRET_KEY",
    "APP_SECRET_KEY",
    "EXCHANGE_API_KEY",
    "EXCHANGE_API_SECRET",
    "EXCHANGE_API_PASSPHRASE",
    "SMTP_PASS",
    "RESEND_API_KEY",
    "BREVO_API_KEY",
    "SENDGRID_API_KEY",
    "BOT_AUTH_TOKEN",
    "JWT_SECRET",
    "COOKIE_SECRET",
}


def _is_serverless() -> bool:
    return bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))


def get_master_key() -> bytes:
    """
    Retrieves the master symmetric encryption key. There is NO safe default for
    this value — every fallback here either reads an explicit secret the
    operator configured, or raises. A silent, guessable, or code-embedded
    fallback would let anyone who can read this source code (e.g. on a public
    repo) decrypt every stored exchange API credential.

    Resolution order:
      1. BOT_SECRET_KEY / SECRET_KEY / APP_SECRET_KEY env var (any of these —
         a real Fernet key, or any passphrase, which gets deterministically
         stretched into one).
      2. A local key file with 0600 permissions (non-serverless only — the
         serverless filesystem doesn't persist, so a generated-and-forgotten
         file there would silently rotate the key on every cold start and
         make old ciphertext permanently undecryptable).
      3. Nothing configured -> raise. This is correct in every environment,
         not just a "production" flag someone forgot to set (Vercel doesn't
         set ENVIRONMENT/ENV/BOT_ENV/MODE by default, so a flag-gated check
         here would silently never fire on a real Vercel deploy).
    """
    # 1. Explicit operator-supplied key/passphrase.
    env_key = os.getenv("BOT_SECRET_KEY") or os.getenv("SECRET_KEY") or os.getenv("APP_SECRET_KEY")
    if env_key:
        raw = env_key.strip().encode("utf-8")
        try:
            Fernet(raw)  # already a valid Fernet key
            return raw
        except Exception:
            # Not a valid Fernet key as-is — deterministically stretch whatever
            # passphrase was given into one, so any reasonably long random
            # value works, not just output from Fernet.generate_key().
            import base64
            import hashlib
            return base64.urlsafe_b64encode(hashlib.sha256(raw).digest())

    is_prod = (
        os.getenv("ENVIRONMENT", "").lower() in ("production", "prod", "live") or
        os.getenv("ENV", "").lower() in ("production", "prod", "live") or
        os.getenv("BOT_ENV", "").lower() in ("production", "prod", "live")
    )
    if is_prod:
        raise RuntimeError("Production / Live mode requires an externally managed master encryption key (BOT_SECRET_KEY, SECRET_KEY, or APP_SECRET_KEY) for persistent zero-loss security.")

    # 2. Local key file — never on serverless, where it can't persist anyway.
    if not _is_serverless():
        key_path = Path(os.getenv("BOT_SECRET_KEY_PATH", str(DEFAULT_KEY_FILE)))
        if key_path.exists():
            try:
                key_data = key_path.read_bytes().strip()
                Fernet(key_data)
                return key_data
            except Exception:
                pass  # corrupt/invalid file — fall through to generation below

        new_key = Fernet.generate_key()
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            mode = stat.S_IRUSR | stat.S_IWUSR  # 0600 — owner read/write only
            fd = os.open(str(key_path), flags, mode)
            with os.fdopen(fd, "wb") as f:
                f.write(new_key)
            return new_key
        except Exception as e:
            raise RuntimeError(
                f"Could not create or read a local encryption key at {key_path}: {e}. "
                "Set BOT_SECRET_KEY explicitly instead."
            ) from e

    # 3. Serverless with nothing configured: no safe default exists. Fail loudly.
    raise RuntimeError(
        "No master encryption key configured. Set the BOT_SECRET_KEY environment "
        "variable (generate one with `python generate_keys.py`) before starting "
        "the app. There is no safe fallback for this value — silently deriving "
        "one from another secret, or from a constant, would let anyone who can "
        "read this source decrypt every stored exchange API credential."
    )


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


def safe_load_dotenv(env_file_path: Optional[Path] = None, override: bool = False) -> None:
    """
    Safely loads environment variables from .env file:
    Loads ONLY non-sensitive configuration keys into os.environ.
    Strictly blocks sensitive credentials (API keys, secrets, email tokens) from
    ever being injected into os.environ from the .env file.
    """
    target = env_file_path or Path(".env")
    if not target.exists() or not target.is_file():
        return

    try:
        from dotenv import dotenv_values
        values = dotenv_values(str(target))
        for k, v in values.items():
            if not k or k in SENSITIVE_ENV_KEYS:
                continue
            if v is not None:
                if override or k not in os.environ:
                    os.environ[k] = str(v)
    except Exception:
        try:
            for line in target.read_text().splitlines():
                s = line.strip()
                if s and not s.startswith("#") and "=" in s:
                    k, v = s.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and k not in SENSITIVE_ENV_KEYS:
                        if override or k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass

    # Sensitive values must never survive in the process environment, regardless of deployment host.
    for k in SENSITIVE_ENV_KEYS:
        os.environ.pop(k, None)


def migrate_and_sanitize_env_file(env_file_path: Optional[Path] = None) -> bool:
    """
    Atomically migrates any legacy plaintext credentials from .env into the
    encrypted database vault within a single transactional boundary.
    Directly queries the database in the transaction to verify persistence and decryption.
    If database persistence or verification fails:
      - Rolls back all database changes.
      - Leaves .env COMPLETELY INTACT.
      - Never sanitizes .env.
    If verification succeeds:
      - Commits the database transaction.
      - Writes the sanitized .env through an atomic temporary file and rename.
    """
    target = env_file_path or Path(".env")
    if _is_serverless():
        return False

    if not target.exists() or not target.is_file():
        return False

    try:
        lines = target.read_text().splitlines(keepends=True)
    except Exception as e:
        print(f"[SECURITY WARNING] Could not read .env file: {e}")
        return False

    env_vars = {}
    for line in lines:
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            k, v = s.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            if k in SENSITIVE_ENV_KEYS and v:
                env_vars[k] = v

    if not env_vars:
        for k in SENSITIVE_ENV_KEYS:
            os.environ.pop(k, None)
        return True

    from auth import AuthService, set_system_setting, get_db_connection

    conn = get_db_connection()
    try:
        conn.execute("BEGIN TRANSACTION")
        cur = conn.cursor()

        ex_key = env_vars.get("EXCHANGE_API_KEY")
        ex_sec = env_vars.get("EXCHANGE_API_SECRET")
        ex_pass = env_vars.get("EXCHANGE_API_PASSPHRASE", "")
        if ex_key and ex_sec:
            cur.execute("SELECT id FROM exchange_keys LIMIT 1")
            existing_row = cur.fetchone()
            if not existing_row:
                cur.execute("SELECT id FROM users ORDER BY id ASC LIMIT 1")
                user = cur.fetchone()
                if not user:
                    from datetime import datetime, timezone
                    now_iso = datetime.now(timezone.utc).isoformat()
                    cur.execute("""
                        INSERT OR IGNORE INTO users (id, username, email, password_hash, salt, is_active, created_at)
                        VALUES (1, 'system_admin', 'admin@system.local', 'n/a', 'n/a', 1, ?)
                    """, (now_iso,))
                    user_id = 1
                else:
                    user_id = user["id"]
                ex_id = os.getenv("EXCHANGE_ID", "mexc")
                testnet_val = os.getenv("TESTNET", "true").lower() == "true"
                AuthService.save_exchange_key(user_id, ex_id, ex_key, ex_sec, ex_pass, testnet_val, conn=conn)

        email_keys = ["SMTP_PASS", "RESEND_API_KEY", "BREVO_API_KEY", "SENDGRID_API_KEY"]
        for ek in email_keys:
            if ek in env_vars:
                set_system_setting(ek, env_vars[ek], conn=conn)

        if ex_key and ex_sec:
            cur.execute("SELECT api_key, api_secret, api_passphrase FROM exchange_keys ORDER BY id DESC LIMIT 1")
            ex_row = cur.fetchone()
            if not ex_row:
                raise RuntimeError("Verification failed: exchange credentials row not found in database")
            dec_key = decrypt_secret(ex_row["api_key"])
            dec_sec = decrypt_secret(ex_row["api_secret"])
            if dec_key != ex_key or dec_sec != ex_sec:
                raise RuntimeError("Verification failed: decrypted exchange credentials do not match source values")

        for ek in email_keys:
            if ek in env_vars:
                cur.execute("SELECT value FROM system_settings WHERE key = ?", (ek,))
                s_row = cur.fetchone()
                if not s_row:
                    raise RuntimeError(f"Verification failed: setting {ek} not found in database")
                dec_val = decrypt_secret(s_row["value"])
                if dec_val != env_vars[ek]:
                    raise RuntimeError(f"Verification failed: decrypted setting {ek} does not match source value")

        conn.commit()
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        print(f"[SECURITY ERROR] Atomic migration failed, database rolled back, .env left intact: {e}")
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass

    try:
        new_lines = []
        for line in lines:
            s = line.strip()
            if s and not s.startswith("#") and "=" in s:
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip()
                if k in SENSITIVE_ENV_KEYS and v:
                    new_lines.append(f"# {k} is stored securely in the encrypted vault (users.db)\n")
                    continue
            new_lines.append(line)

        temp_file = target.with_name(f".{target.name}.tmp.{os.getpid()}")
        try:
            temp_file.write_text("".join(new_lines))
            temp_file.replace(target)
        except Exception as write_err:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass
            raise RuntimeError(f"Atomic file replace failed: {write_err}") from write_err
    except Exception as e:
        print(f"[SECURITY ERROR] Failed to write sanitized .env atomically: {e}")
        return False

    for k in SENSITIVE_ENV_KEYS:
        os.environ.pop(k, None)

    return True


sanitize_env_file = migrate_and_sanitize_env_file
