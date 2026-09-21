#!/usr/bin/env python3
"""
Master Encryption Key Generator for Trader Queen.

Generates an authenticated symmetric encryption key (Fernet / AES-128-CBC + HMAC-SHA256)
used to protect exchange API keys and authentication tokens stored at rest.
"""
import os
import sys
from pathlib import Path
from cryptography.fernet import Fernet


def generate():
    key = Fernet.generate_key().decode()
    print("\n" + "=" * 64)
    print("  TRADER QUEEN MASTER ENCRYPTION KEY GENERATOR")
    print("=" * 64)
    print(f"\nGenerated Key:\n\n  BOT_SECRET_KEY={key}\n")
    print("=" * 64)
    print("\nHow to configure this key:\n")
    print("1. For Vercel Deployment:")
    print("   - Go to your Vercel Project Dashboard: https://vercel.com")
    print("   - Navigate to: Project Settings -> Environment Variables")
    print("   - Add Variable:")
    print("       Key:   BOT_SECRET_KEY")
    print(f"       Value: {key}")
    print("       Target: Production, Preview, Development")
    print("   - Save and redeploy the project.\n")
    print("2. For Local Development:")
    print("   - Add the key to your .env file:")
    print(f"       BOT_SECRET_KEY={key}\n")
    print("=" * 64 + "\n")

    # Update local .env if present
    env_file = Path(__file__).resolve().parent / ".env"
    if env_file.exists():
        content = env_file.read_text(encoding="utf-8")
        if "BOT_SECRET_KEY=" not in content:
            with open(env_file, "a", encoding="utf-8") as f:
                f.write(f"\n# Master encryption key for stored exchange secrets & auth tokens\nBOT_SECRET_KEY={key}\n")
            print(f"[INFO] Automatically added BOT_SECRET_KEY to local {env_file}")
        else:
            print(f"[INFO] BOT_SECRET_KEY is already present in {env_file}")

    return key


if __name__ == "__main__":
    generate()
