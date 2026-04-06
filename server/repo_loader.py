# server/repo_loader.py
"""
Loads repo template variants and copies them into a working temp directory
so the agent can modify files without corrupting the originals.
"""
import os
import json
import shutil
import random
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "repo_templates")


class RepoVariant:
    """Represents one loaded repo variant with metadata."""

    def __init__(self, task: str, variant_id: str, working_dir: str, meta: Dict[str, Any]):
        self.task = task
        self.variant_id = variant_id
        self.working_dir = working_dir  # temp copy agent can modify
        self.meta = meta

    def get_tree(self) -> list:
        """Return all file paths relative to working_dir."""
        tree = []
        for root, dirs, files in os.walk(self.working_dir):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            for f in sorted(files):
                abs_path = os.path.join(root, f)
                rel_path = os.path.relpath(abs_path, self.working_dir)
                tree.append(rel_path)
        return sorted(tree)

    def get_failing_tests(self) -> list:
        """Return failing test names from meta.json."""
        return self.meta.get("failing_tests", [])

    def cleanup(self):
        """Remove the working temp directory."""
        if self.working_dir and os.path.exists(self.working_dir):
            shutil.rmtree(self.working_dir, ignore_errors=True)


def load_random_variant(task: str) -> RepoVariant:
    """Load a random variant for the given task."""
    task_dir = os.path.join(TEMPLATES_DIR, task)
    if not os.path.exists(task_dir):
        raise ValueError(f"Task directory not found: {task_dir}")

    variants = [d for d in os.listdir(task_dir)
                if os.path.isdir(os.path.join(task_dir, d)) and d.startswith("variant_")]

    if not variants:
        raise ValueError(f"No variants found for task: {task}")

    chosen = random.choice(variants)
    variant_path = os.path.join(task_dir, chosen)

    # Load meta.json
    meta_path = os.path.join(variant_path, "meta.json")
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    # Create a temp working copy
    working_dir = tempfile.mkdtemp(prefix=f"openenv_{task}_{chosen}_")
    shutil.copytree(variant_path, working_dir, dirs_exist_ok=True)

    # Remove meta.json from working dir so agent cannot read the answers
    meta_in_work = os.path.join(working_dir, "meta.json")
    if os.path.exists(meta_in_work):
        os.remove(meta_in_work)

    return RepoVariant(task=task, variant_id=chosen, working_dir=working_dir, meta=meta)


def get_task_description(task: str, meta: Dict[str, Any]) -> str:
    """Generate the task description shown to the agent."""
    descriptions = {
        "task1": (
            f"This Python repository has {meta.get('total_files', 'several')} files. "
            f"Some tests are currently failing due to bugs in the source code. "
            f"Your goal is to find and fix the bugs so that all tests pass. "
            f"You have {meta.get('optimal_steps', 15)} optimal steps but can use up to your step budget. "
            f"Read relevant source files, identify the bugs, fix them with write_file, then submit."
        ),
        "task2": (
            f"This Python repository has a bug that spans two modules — one module is calling "
            f"another with the wrong argument type or method signature. "
            f"You must read both modules to understand the interface contract, then fix the caller. "
            f"You also need to write one regression test that would have caught this bug. "
            f"Fix the bug and add the regression test, then submit."
        ),
        "task3": (
            f"This Python repository needs a new feature implemented. "
            f"Read FEATURE_SPEC.md first for requirements. "
            f"Then read the existing source files to understand the architecture. "
            f"Implement the feature so all tests in the tests/ directory pass. "
            f"Do not modify any test files. Only modify source files."
        ),
    }
    return descriptions.get(task, "Fix the failing tests in this repository.")
