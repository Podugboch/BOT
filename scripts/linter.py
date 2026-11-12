#!/usr/bin/env python3
"""
Sec Code Ref Linter - Automated Security Linter & Code Refactoring Engine
Designed for Python and TypeScript/JavaScript codebases.
Scans for security vulnerabilities, dangerous anti-patterns, and code smells,
and provides automated refactoring to production standards.
"""

import argparse
import difflib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Attempt to load rich for styled terminal output, fallback gracefully to plain/ANSI
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# Version metadata
__version__ = "1.0.0"
__author__ = "Patrick Sean Odugboche"

# Ignored directories during recursive directory scans
IGNORED_DIRS = {
    ".git", ".svn", ".hg", "node_modules", "__pycache__",
    ".venv", "venv", "env", ".tox", "build", "dist",
    ".next", ".nuxt", ".cache", ".idea", ".vscode"
}

# Supported file extensions by language
LANGUAGE_EXTENSIONS = {
    "python": {".py"},
    "typescript": {".ts", ".tsx"},
    "javascript": {".js", ".jsx", ".mjs", ".cjs"},
}
ALL_EXTENSIONS = {ext for exts in LANGUAGE_EXTENSIONS.values() for ext in exts}

# Severity definitions
SEVERITY_HIGH = "HIGH"
SEVERITY_MEDIUM = "MEDIUM"
SEVERITY_LOW = "LOW"

# Comprehensive Security & Anti-Pattern Rules
SECURITY_RULES = [
    {
        "id": "SEC-001",
        "name": "Hardcoded Secret / API Key",
        "severity": SEVERITY_HIGH,
        "languages": ["python", "typescript", "javascript"],
        "pattern": re.compile(
            r"""['"]?(?:api[_-]?key|secret|password|auth[_-]?token|private[_-]?key|jwt[_-]?secret)['"]?\s*[:=]\s*['"][a-zA-Z0-9_\-\.]{16,}['"]""",
            re.IGNORECASE
        ),
        "fixable": False,
        "description": "Sensitive credential or token hardcoded in source code.",
        "recommendation": "Move secrets into environment variables or a secure secret vault."
    },
    {
        "id": "SEC-002",
        "name": "Unsafe Dynamic Code Execution (Eval)",
        "severity": SEVERITY_HIGH,
        "languages": ["python", "typescript", "javascript"],
        "pattern": re.compile(r"\b(?:eval|exec)\s*\(|new\s+Function\s*\("),
        "fixable": False,
        "description": "Dynamic code evaluation can lead to arbitrary code execution.",
        "recommendation": "Use safe parsing alternatives such as ast.literal_eval() or JSON.parse()."
    },
    {
        "id": "SEC-003",
        "name": "Command Injection Risk",
        "severity": SEVERITY_HIGH,
        "languages": ["python", "typescript", "javascript"],
        "pattern": re.compile(
            r"os\.system\s*\(|subprocess\.(?:call|Popen|run)\s*\([^)]*shell\s*=\s*True|child_process\.(?:exec|execSync)\s*\("
        ),
        "fixable": False,
        "description": "Executing system shell commands with untrusted input risks shell injection.",
        "recommendation": "Use subprocess.run with argument lists and shell=False, or child_process.execFile."
    },
    {
        "id": "SEC-004",
        "name": "SQL Injection Risk (Unparameterized)",
        "severity": SEVERITY_HIGH,
        "languages": ["python", "typescript", "javascript"],
        "pattern": re.compile(
            r"execute\s*\(\s*(?:f['\"]|['\"][^'\"]*['\"]\s*[%+]|['\"][^'\"]*\{\}[^'\"]*['\"]\s*\.format)",
            re.IGNORECASE
        ),
        "fixable": False,
        "description": "Dynamic string interpolation inside database query execution.",
        "recommendation": "Use parameterized queries, e.g. cursor.execute('SELECT ... WHERE id = %s', (id,))."
    },
    {
        "id": "SEC-005",
        "name": "Insecure Deserialization",
        "severity": SEVERITY_HIGH,
        "languages": ["python"],
        "pattern": re.compile(
            r"pickle\.(?:loads?|dump)|yaml\.load\s*\([^)]*Loader\s*=\s*(?:yaml\.)?Loader|yaml\.unsafe_load"
        ),
        "fixable": False,
        "description": "Insecure deserialization formats can execute malicious arbitrary payloads.",
        "recommendation": "Use safe serializers like json or yaml.safe_load()."
    },
    {
        "id": "SEC-006",
        "name": "DOM-Based Cross-Site Scripting (XSS)",
        "severity": SEVERITY_MEDIUM,
        "languages": ["typescript", "javascript"],
        "pattern": re.compile(
            r"dangerouslySetInnerHTML|\.innerHTML\s*=|\bdocument\.write\s*\("
        ),
        "fixable": False,
        "description": "Directly writing unescaped HTML to DOM risks Cross-Site Scripting (XSS).",
        "recommendation": "Sanitize markup with DOMPurify or use textContent/safe React properties."
    },
    {
        "id": "SEC-007",
        "name": "Bare Except Clause",
        "severity": SEVERITY_MEDIUM,
        "languages": ["python"],
        "pattern": re.compile(r"^\s*except\s*:"),
        "fixable": True,
        "description": "Catching all exceptions catches system exit, interrupts, and hides fatal bugs.",
        "recommendation": "Refactor to 'except Exception as err:' to maintain system interrupt handling."
    },
    {
        "id": "SEC-008",
        "name": "Silent Exception Swallowing (Empty Catch)",
        "severity": SEVERITY_MEDIUM,
        "languages": ["typescript", "javascript"],
        "pattern": re.compile(r"catch\s*(?:\([^)]*\))?\s*\{\s*\}"),
        "fixable": True,
        "description": "Empty catch block silently swallows exceptions, hiding runtime failures.",
        "recommendation": "Log or handle caught errors explicitly."
    },
    {
        "id": "SEC-009",
        "name": "Insecure Debug Mode Enabled",
        "severity": SEVERITY_MEDIUM,
        "languages": ["python"],
        "pattern": re.compile(r"app\.run\s*\([^)]*debug\s*=\s*True"),
        "fixable": True,
        "description": "Enabling debug mode in web frameworks exposes interactive remote debuggers.",
        "recommendation": "Set debug=False or configure through an environment variable."
    },
    {
        "id": "SEC-010",
        "name": "Insecure Public Host Binding",
        "severity": SEVERITY_LOW,
        "languages": ["python", "typescript", "javascript"],
        "pattern": re.compile(r"host\s*=\s*['\"]0\.0\.0\.0['\"]|listen\s*\([^)]*['\"]0\.0\.0\.0['\"]"),
        "fixable": False,
        "description": "Binding to 0.0.0.0 exposes the service to all external network interfaces.",
        "recommendation": "Bind explicitly to 127.0.0.1 for local services or enforce firewall isolation."
    }
]


def detect_file_language(filepath: Path) -> Optional[str]:
    """Detect language of a file based on extension."""
    suffix = filepath.suffix.lower()
    for lang, exts in LANGUAGE_EXTENSIONS.items():
        if suffix in exts:
            return lang
    return None


def get_target_files(target_path: Path, language: str) -> List[Path]:
    """Resolve target files from a single file or directory matching language filter."""
    if not target_path.exists():
        return []

    # Target is a single file
    if target_path.is_file():
        file_lang = detect_file_language(target_path)
        if language == "all" or file_lang == language or (language in ["py", "python"] and file_lang == "python") or (language in ["ts", "typescript"] and file_lang == "typescript") or (language in ["js", "javascript"] and file_lang == "javascript"):
            return [target_path]
        return [target_path]  # Always scan explicitly targeted single file

    # Target is a directory
    selected_exts = ALL_EXTENSIONS
    if language in ["python", "py"]:
        selected_exts = LANGUAGE_EXTENSIONS["python"]
    elif language in ["typescript", "ts"]:
        selected_exts = LANGUAGE_EXTENSIONS["typescript"]
    elif language in ["javascript", "js"]:
        selected_exts = LANGUAGE_EXTENSIONS["javascript"]

    files = []
    for root, dirs, filenames in os.walk(target_path):
        # Prune ignored directories in-place
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".")]
        for filename in filenames:
            file_path = Path(root) / filename
            if file_path.suffix.lower() in selected_exts:
                files.append(file_path)

    return sorted(files)


def scan_file(filepath: Path, exclude_tests: bool = False) -> List[Dict[str, Any]]:
    """Scan a single file for security issues and anti-patterns."""
    issues = []
    if exclude_tests and (filepath.name.startswith("test_") or "test" in filepath.parts):
        return issues

    file_lang = detect_file_language(filepath)
    if not file_lang:
        return issues

    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
        lines = content.splitlines()

        for idx, line in enumerate(lines, 1):
            if "nosec" in line.lower() or "pragma: allowlist secret" in line.lower():
                continue

            for rule in SECURITY_RULES:
                # Check if rule applies to file language
                if file_lang not in rule["languages"]:
                    continue

                if rule["pattern"].search(line):
                    issues.append({
                        "id": rule["id"],
                        "name": rule["name"],
                        "severity": rule["severity"],
                        "line_number": idx,
                        "snippet": line.strip(),
                        "fixable": rule["fixable"],
                        "description": rule["description"],
                        "recommendation": rule["recommendation"],
                        "file": str(filepath)
                    })
    except Exception as e:
        print(f"[Error] Failed to read {filepath}: {e}", file=sys.stderr)

    return issues


def refactor_content(content: str, file_lang: str) -> Tuple[str, int]:
    """
    Apply automated refactorings for fixable rules.
    Returns (refactored_content, fixes_applied_count).
    """
    lines = content.splitlines(keepends=True)
    refactored_lines = []
    fixes_count = 0

    for line in lines:
        new_line = line

        # 1. Python Bare Except -> except Exception as err:
        if file_lang == "python":
            match_bare = re.match(r"^(\s*)except\s*:", new_line)
            if match_bare:
                indent = match_bare.group(1)
                trailing = new_line[match_bare.end():]
                new_line = f"{indent}except Exception as err:{trailing}"
                fixes_count += 1

            # 2. Python debug=True in app.run -> debug=False
            match_debug = re.search(r"(app\.run\s*\([^)]*)debug\s*=\s*True", new_line)
            if match_debug:
                new_line = re.sub(r"debug\s*=\s*True", "debug=False", new_line)
                fixes_count += 1

        # 3. JS/TS empty catch block
        if file_lang in ["typescript", "javascript"]:
            match_empty_catch = re.search(r"catch\s*(?:\([^)]*\))?\s*\{\s*\}", new_line)
            if match_empty_catch:
                new_line = re.sub(
                    r"catch\s*(?:\([^)]*\))?\s*\{\s*\}",
                    r'catch (err) { console.error("Unhandled error:", err); }',
                    new_line
                )
                fixes_count += 1

        refactored_lines.append(new_line)

    return "".join(refactored_lines), fixes_count


def generate_diff(original: str, refactored: str, filepath: str) -> str:
    """Generate a unified diff representation of proposed refactoring."""
    orig_lines = original.splitlines(keepends=True)
    refact_lines = refactored.splitlines(keepends=True)
    diff = difflib.unified_diff(
        orig_lines,
        refact_lines,
        fromfile=f"a/{filepath}",
        tofile=f"b/{filepath}",
        lineterm=""
    )
    return "\n".join(diff)


def print_rich_report(
    files_scanned: int,
    all_issues: List[Dict[str, Any]],
    fixes_applied: int,
    duration: float,
    dry_run: bool = False
):
    """Format and print an elegant report using Rich."""
    console = Console()

    console.print()
    console.print(Panel(
        f"[bold cyan]Sec Code Ref Linter & Refactoring Engine[/bold cyan]\n"
        f"[dim]Automated Security Scanning & Production Refactoring v{__version__}[/dim]",
        border_style="cyan"
    ))

    if all_issues:
        table = Table(title="[bold red]Security Vulnerabilities & Anti-Patterns Flagged[/bold red]", show_header=True, header_style="bold magenta")
        table.add_column("Rule ID", style="cyan", width=10)
        table.add_column("Severity", width=10)
        table.add_column("File:Line", style="dim", width=28)
        table.add_column("Issue Name", style="bold", width=30)
        table.add_column("Snippet / Recommendation", style="white")

        for issue in all_issues:
            sev = issue["severity"]
            if sev == SEVERITY_HIGH:
                sev_styled = "[bold red]HIGH[/bold red]"
            elif sev == SEVERITY_MEDIUM:
                sev_styled = "[bold yellow]MED[/bold yellow]"
            else:
                sev_styled = "[bold blue]LOW[/bold blue]"

            file_loc = f"{Path(issue['file']).name}:{issue['line_number']}"
            snippet_desc = f"[italic]{issue['snippet']}[/italic]\n[dim]Fix: {issue['recommendation']}[/dim]"
            table.add_row(issue["id"], sev_styled, file_loc, issue["name"], snippet_desc)

        console.print(table)
    else:
        console.print("[bold green]✔ Clean scan! No security vulnerabilities or anti-patterns detected.[/bold green]")

    # Summary Panel
    high_count = sum(1 for i in all_issues if i["severity"] == SEVERITY_HIGH)
    med_count = sum(1 for i in all_issues if i["severity"] == SEVERITY_MEDIUM)
    low_count = sum(1 for i in all_issues if i["severity"] == SEVERITY_LOW)

    summary_text = (
        f"[bold]Files Scanned:[/bold] {files_scanned}  |  "
        f"[bold]Total Issues:[/bold] {len(all_issues)}  ("
        f"[red]High: {high_count}[/red], [yellow]Med: {med_count}[/yellow], [blue]Low: {low_count}[/blue])\n"
        f"[bold]Refactor Fixes Applied:[/bold] {fixes_applied}"
        f"{' [yellow](Dry Run - No Changes Written)[/yellow]' if dry_run else ''}  |  "
        f"[bold]Duration:[/bold] {duration:.3f}s"
    )
    console.print(Panel(summary_text, title="[bold]Scan & Refactor Summary[/bold]", border_style="green" if not all_issues else "yellow"))


def print_plain_report(
    files_scanned: int,
    all_issues: List[Dict[str, Any]],
    fixes_applied: int,
    duration: float,
    dry_run: bool = False
):
    """Fallback plain text report when Rich is not available."""
    print("=" * 70)
    print(f"Sec Code Ref Linter & Refactoring Engine (v{__version__})")
    print("=" * 70)

    if all_issues:
        print("\n[!] Security Anti-Patterns Flagged:")
        for issue in all_issues:
            print(f"  * [{issue['id']}] [{issue['severity']}] {issue['file']}:{issue['line_number']}")
            print(f"    Issue: {issue['name']}")
            print(f"    Code:  {issue['snippet']}")
            print(f"    Fix:   {issue['recommendation']}")
            print()
    else:
        print("\n✔ Clean scan! No security vulnerabilities detected.")

    high_count = sum(1 for i in all_issues if i["severity"] == SEVERITY_HIGH)
    med_count = sum(1 for i in all_issues if i["severity"] == SEVERITY_MEDIUM)
    low_count = sum(1 for i in all_issues if i["severity"] == SEVERITY_LOW)

    print("-" * 70)
    print(f"Files Scanned: {files_scanned} | Total Issues: {len(all_issues)} (High: {high_count}, Med: {med_count}, Low: {low_count})")
    print(f"Refactor Fixes Applied: {fixes_applied}{' (Dry Run)' if dry_run else ''} | Duration: {duration:.3f}s")
    print("=" * 70)


def run_linter(
    target: str = ".",
    language: str = "all",
    fix: bool = False,
    diff: bool = False,
    dry_run: bool = False,
    output_format: str = "text",
    output_file: Optional[str] = None,
    strict: bool = True,
    exclude_tests: bool = False
) -> int:
    """Execute the linter and refactoring engine."""
    start_time = time.time()
    target_path = Path(target)

    if not target_path.exists():
        print(f"[Error] Target path '{target}' does not exist.", file=sys.stderr)
        return 2

    files = get_target_files(target_path, language.lower())
    if not files:
        if target_path.is_file():
            print(f"[Warning] Target file '{target}' does not match supported extensions (.py, .ts, .js).")
        else:
            print(f"[Notice] No matching files found in '{target}' for language '{language}'.")
        return 0

    all_issues = []
    total_fixes = 0
    diff_outputs = []

    for file_path in files:
        issues = scan_file(file_path, exclude_tests=exclude_tests)
        all_issues.extend(issues)

        # Refactoring logic
        if fix or diff:
            file_lang = detect_file_language(file_path)
            if file_lang:
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    refactored, fixes = refactor_content(content, file_lang)
                    if fixes > 0:
                        total_fixes += fixes
                        if diff:
                            file_diff = generate_diff(content, refactored, str(file_path))
                            if file_diff:
                                diff_outputs.append(file_diff)
                        if fix and not dry_run:
                            file_path.write_text(refactored, encoding="utf-8")
                except Exception as e:
                    print(f"[Error] Failed to refactor {file_path}: {e}", file=sys.stderr)

    duration = time.time() - start_time

    # Output Diffs if requested
    if diff and diff_outputs:
        print("\n--- Proposed Refactoring Diffs ---")
        for d in diff_outputs:
            print(d)
        print("----------------------------------\n")

    # Output Results
    if output_format == "json":
        report_data = {
            "version": __version__,
            "summary": {
                "files_scanned": len(files),
                "total_issues": len(all_issues),
                "fixes_applied": total_fixes,
                "dry_run": dry_run,
                "duration_seconds": round(duration, 4)
            },
            "issues": all_issues
        }
        json_str = json.dumps(report_data, indent=2)
        if output_file:
            Path(output_file).write_text(json_str, encoding="utf-8")
            print(f"JSON report saved to {output_file}")
        else:
            print(json_str)
    else:
        if HAS_RICH:
            print_rich_report(len(files), all_issues, total_fixes, duration, dry_run=dry_run)
        else:
            print_plain_report(len(files), all_issues, total_fixes, duration, dry_run=dry_run)

        if output_file:
            # Save plain text report to file
            import io
            buffer = io.StringIO()
            sys.stdout = buffer
            print_plain_report(len(files), all_issues, total_fixes, duration, dry_run=dry_run)
            sys.stdout = sys.__stdout__
            Path(output_file).write_text(buffer.getvalue(), encoding="utf-8")
            print(f"Report written to {output_file}")

    # Return exit code
    # If strict mode is enabled, exit 1 if there are any unfixed issues
    unfixed_issues = len(all_issues) - (total_fixes if (fix and not dry_run) else 0)
    if strict and unfixed_issues > 0:
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Sec Code Ref Linter - Automated Security Linter & Production Refactoring Skill",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--target", "-t",
        type=str,
        default=".",
        help="Target directory or single source file to scan"
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        default="all",
        choices=["all", "python", "py", "typescript", "ts", "javascript", "js"],
        help="Language filter for files to scan"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Automatically apply safe code hygiene and refactoring fixes"
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        help="Show unified diff of proposed refactoring changes"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate refactoring changes without writing to disk"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        default="text",
        choices=["text", "json"],
        help="Output report format"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Optional file path to write results"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Exit with non-zero exit code if security issues are flagged"
    )
    parser.add_argument(
        "--exclude-tests",
        action="store_true",
        default=False,
        help="Exclude unit/integration test files from scan"
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"Sec Code Ref Linter v{__version__}"
    )

    args = parser.parse_args()

    exit_code = run_linter(
        target=args.target,
        language=args.language,
        fix=args.fix,
        diff=args.diff,
        dry_run=args.dry_run,
        output_format=args.format,
        output_file=args.output,
        strict=args.strict,
        exclude_tests=args.exclude_tests
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
