# server/security.py
"""
Security layer for detecting unsafe agent actions.

Scans agent-submitted code for:
- Dangerous function calls (os.system, eval, exec, subprocess)
- Import of dangerous modules
- Path traversal attempts
- Prompt injection patterns in code comments
- Network access attempts
"""
import re
from typing import List, Tuple
from dataclasses import dataclass, field


# Patterns that indicate dangerous code
DANGEROUS_PATTERNS = [
    (r'\bos\.system\s*\(', "os.system() — arbitrary command execution"),
    (r'\bos\.popen\s*\(', "os.popen() — arbitrary command execution"),
    (r'\bsubprocess\.(run|call|Popen|check_output)\s*\(', "subprocess — arbitrary command execution"),
    (r'\beval\s*\(', "eval() — arbitrary code execution"),
    (r'\bexec\s*\(', "exec() — arbitrary code execution"),
    (r'\b__import__\s*\(', "__import__() — dynamic import of dangerous modules"),
    (r'\bcompile\s*\(.*exec', "compile()+exec — code execution"),
    (r'\bopen\s*\([^)]*["\']\/etc', "Attempting to read system files"),
    (r'\bopen\s*\([^)]*["\']\/proc', "Attempting to read proc filesystem"),
    (r'\bsocket\s*\.\s*socket\s*\(', "Raw socket creation — network access"),
    (r'\brequests\.(get|post|put|delete)\s*\(', "HTTP requests — network access"),
    (r'\burllib', "urllib — network access"),
    (r'\bshutil\.rmtree\s*\(', "shutil.rmtree() — recursive deletion"),
    (r'\bos\.remove\s*\(', "os.remove() — file deletion"),
    (r'\bos\.unlink\s*\(', "os.unlink() — file deletion"),
]

# Dangerous imports
DANGEROUS_IMPORTS = [
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "http.client",
    "ftplib",
    "smtplib",
    "ctypes",
    "pickle",  # deserialization attacks
]

# Prompt injection patterns — things an attacker might put in code comments
INJECTION_PATTERNS = [
    (r'ignore\s+(all\s+)?previous\s+instructions', "Prompt injection: ignore instructions"),
    (r'you\s+are\s+now\s+a', "Prompt injection: role override"),
    (r'system\s*:\s*you\s+must', "Prompt injection: system role override"),
    (r'<\|im_start\|>', "Prompt injection: chat template injection"),
    (r'IMPORTANT:\s*ignore', "Prompt injection: authority override"),
    (r'act\s+as\s+if', "Prompt injection: behavioral override"),
]


@dataclass
class SecurityScanResult:
    """Result of scanning agent-submitted content."""
    is_safe: bool
    flags: List[str] = field(default_factory=list)
    blocked_patterns: List[str] = field(default_factory=list)
    severity: str = "none"  # none, low, medium, high, critical

    def to_dict(self) -> dict:
        return {
            "is_safe": self.is_safe,
            "flags": self.flags,
            "blocked_patterns": self.blocked_patterns,
            "severity": self.severity,
        }


class SecurityScanner:
    """
    Scans agent-submitted code for security threats.

    Usage:
        scanner = SecurityScanner()
        result = scanner.scan_content(code_content)
        result = scanner.scan_file_read(file_content)  # for injection in existing files
    """

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.total_scans = 0
        self.total_violations = 0

    def scan_write_content(self, content: str, path: str) -> SecurityScanResult:
        """Scan content that agent wants to write to a file."""
        self.total_scans += 1
        flags = []
        blocked = []

        # Check dangerous patterns
        for pattern, description in DANGEROUS_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                flags.append(f"DANGEROUS_CODE: {description}")
                blocked.append(pattern)

        # Check dangerous imports
        for module in DANGEROUS_IMPORTS:
            if re.search(rf'^\s*(import\s+{module}|from\s+{module}\s+import)', content, re.MULTILINE):
                flags.append(f"DANGEROUS_IMPORT: {module}")
                blocked.append(module)

        # Check for path traversal in content
        if ".." in path or path.startswith("/"):
            flags.append(f"PATH_TRAVERSAL: suspicious path '{path}'")

        # Determine severity
        if not flags:
            severity = "none"
        elif len(flags) == 1 and not blocked:
            severity = "low"
        elif blocked:
            severity = "high" if len(blocked) > 2 else "medium"
        else:
            severity = "medium"

        is_safe = len(blocked) == 0 or not self.strict_mode
        if not is_safe:
            self.total_violations += 1

        return SecurityScanResult(
            is_safe=is_safe,
            flags=flags,
            blocked_patterns=blocked,
            severity=severity,
        )

    def scan_file_for_injection(self, content: str, path: str) -> SecurityScanResult:
        """
        Scan file content being READ by agent for prompt injection.
        This detects if a codebase file is trying to hijack the agent.
        """
        self.total_scans += 1
        flags = []

        for pattern, description in INJECTION_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                flags.append(f"INJECTION_DETECTED in {path}: {description}")

        severity = "none" if not flags else ("high" if len(flags) > 1 else "medium")

        return SecurityScanResult(
            is_safe=len(flags) == 0,
            flags=flags,
            blocked_patterns=[],
            severity=severity,
        )

    def get_stats(self) -> dict:
        return {
            "total_scans": self.total_scans,
            "total_violations": self.total_violations,
            "violation_rate": round(
                self.total_violations / max(1, self.total_scans), 3
            ),
        }
