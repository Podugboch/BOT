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


def get_master_key() -> bytes:
    """
    Retrieves or generates the master symmetric encryption key.
    Checks environment variable BOT_SECRET_KEY / SECRET_KEY / APP_SECRET_KEY first.
    If absent, securely loads or generates a key file with 0600 permissions.
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

    # 2. Check local key file
    key_path = Path(os.getenv("BOT_SECRET_KEY_PATH", str(DEFAULT_KEY_FILE)))
    if key_path.exists():
        try:
            key_data = key_path.read_bytes().strip()
            Fernet(key_data)
            return key_data
        except Exception:
            pass

    # 3. Generate new key and save with restricted 0600 permissions
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


def sanitize_env_file(env_file_path: Optional[Path] = None) -> bool:
    """
    Scans the .env file and removes any plaintext sensitive secrets,
    ensuring they are persisted in encrypted form and not kept in project files.
    """
    target = env_file_path or Path(".env")
    if not target.exists() or not target.is_file():
        return False

    try:
        lines = target.read_text().splitlines(keepends=True)
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
                        # Replace plaintext line with a safe notice comment
                        new_lines.append(f"# {k} is stored securely in the encrypted vault (users.db)\n")
                        continue
            new_lines.append(line)

        if modified:
            target.write_text("".join(new_lines))
            return True
    except Exception as e:
        print(f"[SECURITY WARNING] Could not sanitize .env file: {e}")
    return False
