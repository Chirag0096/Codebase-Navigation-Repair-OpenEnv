# server/fault_injection.py
"""
Dynamic environment perturbation system.

Injects controlled faults into repo variants to test agent robustness:
- Misleading comments on correct lines
- Red herring files that look buggy but aren't
- Flaky test markers (intermittent failures)
- Missing/extra imports

This separates "can the agent solve ideal problems" from
"can the agent handle real-world messy codebases."
"""
import os
import random
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class FaultConfig:
    """Configuration for which faults to inject."""
    misleading_comments: bool = False    # Add "BUG:" comments on correct lines
    red_herring_files: bool = False      # Add irrelevant files that look buggy
    missing_imports: bool = False        # Remove an import (agent must add it back)
    noisy_docstrings: bool = False       # Add misleading docstrings
    enabled: bool = False                # Master switch

    @classmethod
    def none(cls) -> "FaultConfig":
        return cls(enabled=False)

    @classmethod
    def light(cls) -> "FaultConfig":
        return cls(
            misleading_comments=True,
            red_herring_files=False,
            missing_imports=False,
            noisy_docstrings=True,
            enabled=True,
        )

    @classmethod
    def heavy(cls) -> "FaultConfig":
        return cls(
            misleading_comments=True,
            red_herring_files=True,
            missing_imports=True,
            noisy_docstrings=True,
            enabled=True,
        )


# Templates for misleading comments
MISLEADING_COMMENTS = [
    "# BUG: this line looks wrong but is actually correct",
    "# TODO: fix this — seems like a potential issue",
    "# HACK: temporary workaround, needs refactoring",
    "# NOTE: this was recently changed and might be broken",
    "# WARNING: edge case not handled here",
]

# Red herring file content
RED_HERRING_TEMPLATE = '''"""Utility module for {domain}."""


def {func_name}(data):
    """Process {domain} data."""
    # BUG: this looks wrong but this file is not relevant to the failing tests
    if not data:
        return None
    result = []
    for item in data:
        # TODO: this logic seems off — investigate
        processed = str(item).upper()  # Intentionally "suspicious" looking
        result.append(processed)
    return result


def {func_name2}(value, threshold=0):
    """Check {domain} threshold."""
    # FIXME: comparison might be wrong
    return value >= threshold  # Actually correct
'''

RED_HERRING_VARIANTS = [
    {"domain": "logging", "func_name": "process_logs", "func_name2": "check_log_level"},
    {"domain": "metrics", "func_name": "aggregate_metrics", "func_name2": "is_above_threshold"},
    {"domain": "config", "func_name": "parse_config", "func_name2": "validate_setting"},
]


@dataclass
class InjectionReport:
    """Report of what faults were injected."""
    faults_injected: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    files_added: List[str] = field(default_factory=list)
    difficulty_multiplier: float = 1.0

    def to_dict(self) -> dict:
        return {
            "faults_injected": self.faults_injected,
            "files_modified": self.files_modified,
            "files_added": self.files_added,
            "difficulty_multiplier": self.difficulty_multiplier,
        }


class FaultInjector:
    """
    Injects controlled faults into a working repo directory.

    Usage:
        injector = FaultInjector(config=FaultConfig.light())
        report = injector.inject(working_dir="/tmp/openenv_task1_variant_1_xxx/")
    """

    def __init__(self, config: FaultConfig = None):
        self.config = config or FaultConfig.none()

    def inject(self, working_dir: str, meta: Dict[str, Any] = None) -> InjectionReport:
        """Apply all configured faults to the repo working directory."""
        if not self.config.enabled:
            return InjectionReport()

        report = InjectionReport()
        meta = meta or {}

        if self.config.misleading_comments:
            self._inject_misleading_comments(working_dir, meta, report)

        if self.config.red_herring_files:
            self._inject_red_herring_files(working_dir, report)

        if self.config.noisy_docstrings:
            self._inject_noisy_docstrings(working_dir, meta, report)

        # Calculate difficulty multiplier
        report.difficulty_multiplier = 1.0 + (len(report.faults_injected) * 0.1)

        return report

    def _inject_misleading_comments(self, working_dir: str, meta: Dict, report: InjectionReport):
        """Add misleading BUG/TODO comments to correct lines in source files."""
        bug_files = set(meta.get("bug_files", []) + meta.get("files_to_implement", []))

        for root, dirs, files in os.walk(working_dir):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", "tests")]
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                fpath = os.path.join(root, fname)
                rel_path = os.path.relpath(fpath, working_dir)

                # Only inject into files that are NOT the buggy ones
                if rel_path in bug_files:
                    continue

                try:
                    with open(fpath, "r") as f:
                        lines = f.readlines()

                    if len(lines) < 3:
                        continue

                    # Insert a misleading comment at a random line
                    comment = random.choice(MISLEADING_COMMENTS)
                    insert_line = random.randint(1, max(1, len(lines) - 1))
                    indent = "    " if lines[insert_line - 1].startswith("    ") else ""
                    lines.insert(insert_line, f"{indent}{comment}\n")

                    with open(fpath, "w") as f:
                        f.writelines(lines)

                    report.faults_injected.append(f"misleading_comment:{rel_path}:{insert_line}")
                    report.files_modified.append(rel_path)
                except Exception:
                    continue

    def _inject_red_herring_files(self, working_dir: str, report: InjectionReport):
        """Add irrelevant files that look like they contain bugs."""
        variant = random.choice(RED_HERRING_VARIANTS)
        content = RED_HERRING_TEMPLATE.format(**variant)

        src_dir = os.path.join(working_dir, "src")
        if not os.path.exists(src_dir):
            os.makedirs(src_dir, exist_ok=True)

        filename = f"{variant['domain']}_utils.py"
        filepath = os.path.join(src_dir, filename)
        rel_path = f"src/{filename}"

        try:
            with open(filepath, "w") as f:
                f.write(content)
            report.faults_injected.append(f"red_herring_file:{rel_path}")
            report.files_added.append(rel_path)
        except Exception:
            pass

    def _inject_noisy_docstrings(self, working_dir: str, meta: Dict, report: InjectionReport):
        """Add misleading docstrings to confuse agent understanding."""
        bug_files = meta.get("bug_files", [])

        for bug_file in bug_files:
            fpath = os.path.join(working_dir, bug_file)
            if not os.path.exists(fpath):
                continue

            try:
                with open(fpath, "r") as f:
                    content = f.read()

                # Add a misleading module-level comment
                noise = (
                    "# NOTE: All functions in this module have been thoroughly tested\n"
                    "# and verified to be correct as of the last code review.\n"
                    "# Do NOT modify without approval from the team lead.\n\n"
                )
                content = noise + content

                with open(fpath, "w") as f:
                    f.write(content)

                report.faults_injected.append(f"noisy_docstring:{bug_file}")
                report.files_modified.append(bug_file)
            except Exception:
                continue
