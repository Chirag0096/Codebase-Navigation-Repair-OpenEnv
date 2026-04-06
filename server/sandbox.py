# server/sandbox.py
"""
Secure subprocess execution for running agent-submitted code.
NEVER run agent code as root. ALWAYS use timeouts.
"""
import subprocess
import os
import sys
import tempfile
from typing import Tuple
from pathlib import Path
import re


EXECUTION_TIMEOUT = 10       # seconds — hard limit per test run
MAX_OUTPUT_BYTES = 50_000    # truncate large outputs
MAX_MEMORY_MB = 256          # memory limit for subprocess


def run_pytest_sandboxed(repo_path: str, test_file: str = None) -> Tuple[float, str, bool]:
    """
    Run pytest in a sandboxed subprocess.

    Returns:
        (pass_rate: float, output: str, timed_out: bool)
    """
    cmd = [sys.executable, "-m", "pytest", "--tb=short", "-q", "--no-header"]

    if test_file:
        test_path = os.path.join(repo_path, test_file)
        if not os.path.exists(test_path):
            return 0.0, f"Test file not found: {test_file}", False
        cmd.append(test_path)
    else:
        tests_dir = os.path.join(repo_path, "tests")
        if os.path.exists(tests_dir):
            cmd.append(tests_dir)
        else:
            cmd.append(repo_path)

    def set_limits():
        """Set resource limits for subprocess — runs in child process."""
        try:
            import resource
            mem_bytes = MAX_MEMORY_MB * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except Exception:
            pass  # Best effort — don't fail if setrlimit unavailable

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=EXECUTION_TIMEOUT,
            cwd=repo_path,
            env={
                **os.environ,
                "PYTHONPATH": repo_path,
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            preexec_fn=set_limits if sys.platform != "win32" else None,
        )

        output = (result.stdout + result.stderr)[:MAX_OUTPUT_BYTES]
        pass_rate = _parse_pass_rate(output, result.returncode)
        return pass_rate, output, False

    except subprocess.TimeoutExpired:
        return 0.0, f"TIMEOUT: Tests exceeded {EXECUTION_TIMEOUT}s limit", True
    except Exception as e:
        return 0.0, f"EXECUTION_ERROR: {str(e)}", False


def _parse_pass_rate(output: str, returncode: int) -> float:
    """Parse pytest output to get pass rate 0.0–1.0."""
    # Look for "X passed, Y failed" or "X passed" or "X failed"
    passed_match = re.search(r'(\d+) passed', output)
    failed_match = re.search(r'(\d+) failed', output)
    error_match = re.search(r'(\d+) error', output)

    passed = int(passed_match.group(1)) if passed_match else 0
    failed = int(failed_match.group(1)) if failed_match else 0
    errors = int(error_match.group(1)) if error_match else 0

    total = passed + failed + errors
    if total == 0:
        # If returncode is 0, all passed; otherwise failure
        return 1.0 if returncode == 0 else 0.0

    return round(passed / total, 3)


def validate_file_path(path: str, repo_root: str) -> bool:
    """
    Ensure agent cannot read/write files outside the repo.
    Prevents path traversal attacks.
    """
    try:
        repo_abs = os.path.abspath(repo_root)
        file_abs = os.path.abspath(os.path.join(repo_root, path))
        return file_abs.startswith(repo_abs + os.sep) or file_abs == repo_abs
    except Exception:
        return False


def search_in_repo(query: str, repo_path: str) -> str:
    """Grep-style search across all Python files in the repo."""
    results = []
    for root, dirs, files in os.walk(repo_path):
        # Skip __pycache__ and hidden dirs
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for fname in files:
            if fname.endswith('.py') or fname.endswith('.md') or fname.endswith('.json'):
                fpath = os.path.join(root, fname)
                rel_path = os.path.relpath(fpath, repo_path)
                try:
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                        for lineno, line in enumerate(f, 1):
                            if query.lower() in line.lower():
                                results.append(f"{rel_path}:{lineno}: {line.rstrip()}")
                except Exception:
                    continue
    if not results:
        return f"No matches found for: {query}"
    return '\n'.join(results[:50])  # Limit to 50 matches
