#!/usr/bin/env python3
"""
Pre-Commit & CI Security Linter for BOT Repository.
Detects hardcoded secrets, sensitive credentials, blacklisted files,
and dangerous code anti-patterns before commits can be finalized or pushed.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Graceful rich loading
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

__version__ = "1.0.0"

# Files that should NEVER be committed to Git (allows templates like .env.example)
SENSITIVE_FILENAME_PATTERNS = [
    re.compile(r"^\.env(?:\.(?!example|sample|template).+)?$", re.IGNORECASE),
    re.compile(r"^\.secret_key$", re.IGNORECASE),
    re.compile(r"^seed_users\.json$", re.IGNORECASE),
    re.compile(r"^users\.db$", re.IGNORECASE),
    re.compile(r"^execution\.db$", re.IGNORECASE),
    re.compile(r"^.*\.(?:pem|key|pfx|p12|pkcs12)$", re.IGNORECASE),
    re.compile(r"^id_rsa(?:\.pub)?$", re.IGNORECASE),
    re.compile(r"^.*\.(?:sqlite|sqlite3|db)$", re.IGNORECASE),
    re.compile(r"^.*\.log$", re.IGNORECASE),
]

# High-risk secret patterns (RegEx)
SECRET_PATTERNS = [
    (
        "AWS Access Key ID",
        re.compile(r"\b(AKIA[0-9A-Z]{16})\b"),
        "HIGH"
    ),
    (
        "GitHub Personal Access Token",
        re.compile(r"\b(gh[pous]_[A-Za-z0-9_]{36,255}|github_pat_[A-Za-z0-9_]{82})\b"),
        "HIGH"
    ),
    (
        "Stripe Live API Key",
        re.compile(r"\b(?:sk|rk)_live_[0-9a-zA-Z]{24,}\b"),
        "HIGH"
    ),
    (
        "Slack Bot/User Token",
        re.compile(r"\bxox[baprs]-[0-9a-zA-Z]{10,48}\b"),
        "HIGH"
    ),
    (
        "Resend API Key",
        re.compile(r"\bre_[0-9a-zA-Z]{24,}\b"),
        "HIGH"
    ),
    (
        "Brevo API Key",
        re.compile(r"\bxkeysib-[0-9a-zA-Z]{64}\b"),
        "HIGH"
    ),
    (
        "RSA / EC Private Key Header",
        re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
        "HIGH"
    ),
    (
        "Generic High-Entropy API Secret / Token Assignment",
        re.compile(
            r"""(?i)['"]?(?:api[_-]?key|secret[_-]?key|auth[_-]?token|jwt[_-]?secret|private[_-]?key)['"]?\s*[:=]\s*['"]([a-zA-Z0-9_\-\.]{24,})['"]"""
        ),
        "HIGH"
    ),
    (
        "Hardcoded Password Hash",
        re.compile(r"""(?i)['"]?(?:password_hash|pwd_hash|pass_hash|hash_pw)['"]?\s*[:=]\s*['"]([a-f0-9]{32,64})['"]"""),
        "HIGH"
    ),
    (
        "Hardcoded Cryptographic Salt",
        re.compile(r"""(?i)['"]?(?:salt|password_salt)['"]?\s*[:=]\s*['"]([a-f0-9]{16,64})['"]"""),
        "HIGH"
    ),
    (
        "Hardcoded Password Assignment",
        re.compile(r"""(?i)['"]?(?:password|passwd|pwd|passphrase)['"]?\s*[:=]\s*['"]([^\s'"]{8,})['"]"""),
        "HIGH"
    ),
]

# Dangerous code anti-patterns
DANGEROUS_PATTERNS = [
    (
        "Unsafe Dynamic Code Execution (Eval/Exec)",
        re.compile(r"\b(?:eval|exec)\s*\("),
        "HIGH",
        "Dynamic code evaluation can lead to remote code execution. Use ast.literal_eval() instead."
    ),
    (
        "Command Injection Risk (Shell=True)",
        re.compile(r"os\.system\s*\(|subprocess\.(?:call|Popen|run)\s*\([^)]*shell\s*=\s*True"),
        "HIGH",
        "Executing commands via shell=True enables command injection. Use parameter arrays."
    ),
    (
        "Unparameterized Dynamic SQL Execution",
        re.compile(
            r"execute\s*\(\s*(?:f['\"]|['\"][^'\"]*['\"]\s*[%+]|['\"][^'\"]*\{\}[^'\"]*['\"]\s*\.format)",
            re.IGNORECASE
        ),
        "HIGH",
        "Dynamic SQL formatting risks SQL injection. Use parameterized queries (? or %s)."
    ),
]

IGNORED_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
    ".vercel", "node_modules", "dist", "build", ".agents", ".codex"
}


def get_staged_files(repo_root: Path) -> List[Path]:
    """Return list of staged files in git index."""
    try:
        proc = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return [repo_root / line.strip() for line in proc.stdout.splitlines() if line.strip()]
        return []
    except Exception:
        return []


def is_git_ignored(file_path: Path, repo_root: Path) -> bool:
    """Check if a file is ignored by .gitignore."""
    try:
        proc = subprocess.run(
            ["git", "check-ignore", "-q", str(file_path)],
            cwd=str(repo_root),
            capture_output=True,
            check=False
        )
        return proc.returncode == 0
    except Exception:
        return False


def is_test_file(path: Path) -> bool:
    """Return True if path is a unit/integration test file."""
    parts = [p.lower() for p in path.parts]
    name = path.name.lower()
    return "test" in name or "tests" in parts or name.startswith("test_")


def scan_file_for_issues(file_path: Path) -> List[Dict[str, Any]]:
    """Scan an individual file for sensitive names, secrets, or anti-patterns."""
    issues = []
    fname = file_path.name

    # 1. Check sensitive filename blacklist
    for pattern in SENSITIVE_FILENAME_PATTERNS:
        if pattern.match(fname):
            issues.append({
                "file": str(file_path),
                "line": 0,
                "type": "Blacklisted Sensitive File",
                "severity": "HIGH",
                "message": f"File '{fname}' matches sensitive file pattern and must never be committed to Git.",
                "snippet": fname
            })
            return issues

    # Skip non-text or binary files
    if file_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".ico", ".gif", ".zip", ".tar", ".gz", ".db"]:
        return issues

    if not file_path.is_file():
        return issues

    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return issues

    lines = content.splitlines()
    test_file = is_test_file(file_path)

    for idx, line in enumerate(lines, start=1):
        # Allow intentional ignores via comment
        if "nosec" in line.lower() or "pragma: allowlist secret" in line.lower():
            continue

        # Check secret patterns
        for name, pattern, severity in SECRET_PATTERNS:
            # Skip mock test secrets for unit test files unless it's a real RSA key
            if test_file and name in [
                "Generic High-Entropy API Secret / Token Assignment",
                "Resend API Key",
                "Hardcoded Password Assignment",
                "Hardcoded Password Hash",
                "Hardcoded Cryptographic Salt"
            ]:
                continue

            match = pattern.search(line)
            if match:
                snippet = match.group(0)
                # Mask secret snippet for safe output
                if len(snippet) > 8:
                    masked = snippet[:4] + "*" * (len(snippet) - 8) + snippet[-4:]
                else:
                    masked = "****"

                issues.append({
                    "file": str(file_path),
                    "line": idx,
                    "type": f"Hardcoded Secret: {name}",
                    "severity": severity,
                    "message": f"Detected potential hardcoded secret ({name}).",
                    "snippet": line.strip()
                })

        # Check dangerous code anti-patterns (only for python code in non-test files)
        if file_path.suffix.lower() == ".py" and not test_file:
            for name, pattern, severity, recommendation in DANGEROUS_PATTERNS:
                if pattern.search(line):
                    issues.append({
                        "file": str(file_path),
                        "line": idx,
                        "type": f"Code Risk: {name}",
                        "severity": severity,
                        "message": recommendation,
                        "snippet": line.strip()
                    })

    return issues


def render_report(issues: List[Dict[str, Any]], repo_root: Path) -> None:
    """Print clean terminal report."""
    if not issues:
        if HAS_RICH:
            console = Console()
            console.print("[bold green]✔ Pre-Commit Security Scan: 0 issues found. Code is clean![/bold green]")
        else:
            print("[SUCCESS] Pre-Commit Security Scan: 0 issues found. Code is clean!")
        return

    if HAS_RICH:
        console = Console()
        console.print()
        console.print(Panel(
            "[bold red]SECURITY SCAN FAILED: Sensitive Data or Vulnerabilities Detected[/bold red]\n"
            "[yellow]Please resolve the issues below before committing or pushing to GitHub.[/yellow]",
            border_style="red"
        ))

        table = Table(title="Security Findings", header_style="bold red")
        table.add_column("Severity", style="bold red", width=10)
        table.add_column("Location", style="cyan", width=35)
        table.add_column("Issue Type", style="white", width=30)
        table.add_column("Details", style="dim")

        for issue in issues:
            rel_path = os.path.relpath(issue["file"], repo_root)
            loc = f"{rel_path}:{issue['line']}" if issue["line"] > 0 else rel_path
            table.add_row(issue["severity"], loc, issue["type"], issue["message"])

        console.print(table)
        console.print()
    else:
        print("=" * 70)
        print("[FAILED] SECURITY SCAN: Sensitive Data or Vulnerabilities Detected")
        print("=" * 70)
        for issue in issues:
            rel_path = os.path.relpath(issue["file"], repo_root)
            loc = f"{rel_path}:{issue['line']}" if issue["line"] > 0 else rel_path
            print(f"[{issue['severity']}] {loc} - {issue['type']}")
            print(f"  Details: {issue['message']}")
            print(f"  Snippet: {issue.get('snippet', '')}")
            print("-" * 70)


def main():
    parser = argparse.ArgumentParser(description="BOT Repository Pre-Commit Security Scanner")
    parser.add_argument("--staged", action="store_true", help="Scan only git staged files")
    parser.add_argument("--all", action="store_true", help="Scan all tracked project files")
    parser.add_argument("--target", type=str, default=None, help="Scan a specific file or directory")
    parser.add_argument("--repo", type=str, default=".", help="Repository root path")

    args = parser.parse_args()
    repo_root = Path(args.repo).resolve()

    files_to_scan: List[Path] = []

    if args.target:
        tpath = Path(args.target).resolve()
        if tpath.is_file():
            files_to_scan = [tpath]
        elif tpath.is_dir():
            for root, dirs, files in os.walk(tpath):
                dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
                for f in files:
                    files_to_scan.append(Path(root) / f)
    elif args.staged:
        files_to_scan = get_staged_files(repo_root)
        if not files_to_scan:
            if HAS_RICH:
                Console().print("[dim]No staged files detected for pre-commit scan.[/dim]")
            else:
                print("No staged files detected for pre-commit scan.")
            sys.exit(0)
    elif args.all:
        for root, dirs, files in os.walk(repo_root):
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
            for f in files:
                fpath = Path(root) / f
                if not is_git_ignored(fpath, repo_root):
                    files_to_scan.append(fpath)
    else:
        # Default behavior: if files are staged, scan staged files; otherwise scan tracked non-ignored files
        staged = get_staged_files(repo_root)
        if staged:
            files_to_scan = staged
        else:
            for root, dirs, files in os.walk(repo_root):
                dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
                for f in files:
                    fpath = Path(root) / f
                    if not is_git_ignored(fpath, repo_root):
                        files_to_scan.append(fpath)

    all_issues: List[Dict[str, Any]] = []
    for f in files_to_scan:
        all_issues.extend(scan_file_for_issues(f))

    render_report(all_issues, repo_root)

    if all_issues:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
