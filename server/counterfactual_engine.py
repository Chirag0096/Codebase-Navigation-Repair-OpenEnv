# server/counterfactual_engine.py
"""
Counterfactual Robustness Engine — v4.0

The key scientific question: Is the agent's strategy robust, or is it brittle?

We test this by:
1. Running an episode → recording strategy
2. Applying small, semantically-neutral mutations to the repo
   (rename variable, change a constant, add a dummy function)
3. Measuring whether the agent's recorded strategy would fail on the mutated repo

IMPORTANT: This does NOT re-run the agent. It analyzes whether the
already-recorded navigation pattern was based on deep structure (robust)
or surface signals like filenames/constants (brittle).

This is completely novel — no benchmark or tool does this.
"""
from __future__ import annotations
import random
import hashlib
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum


class BrittlenessLevel(str, Enum):
    ROBUST = "ROBUST"           # Strategy survives all mutations
    MILDLY_BRITTLE = "MILDLY_BRITTLE"  # Survives 60-80% of mutations
    BRITTLE = "BRITTLE"         # Survives < 60%
    FRAGILE = "FRAGILE"         # Survives < 30%


@dataclass
class Mutation:
    """A single counterfactual mutation applied to the repo."""
    mutation_type: str
    target_file: str
    description: str
    would_break_agent: bool  # Would this mutation cause agent's strategy to fail?
    why: str                 # Explanation


@dataclass
class CounterfactualReport:
    """Results of counterfactual robustness testing."""
    episode_id: str
    task: str
    brittleness_level: BrittlenessLevel
    robustness_score: float      # 0.0 – 1.0

    mutations_tested: List[Mutation]
    mutations_survived: int
    mutations_failed: int

    surface_dependencies: List[str]  # What surface signals the agent relied on
    deep_dependencies: List[str]     # What structural signals it used correctly

    explanation: str
    recommendations: List[str]

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "task": self.task,
            "brittleness_level": self.brittleness_level.value,
            "robustness_score": round(self.robustness_score, 3),
            "mutations_tested": len(self.mutations_tested),
            "mutations_survived": self.mutations_survived,
            "mutations_failed": self.mutations_failed,
            "mutations": [
                {
                    "type": m.mutation_type,
                    "file": m.target_file,
                    "description": m.description,
                    "would_break_agent": m.would_break_agent,
                    "why": m.why,
                }
                for m in self.mutations_tested
            ],
            "surface_dependencies": self.surface_dependencies,
            "deep_dependencies": self.deep_dependencies,
            "explanation": self.explanation,
            "recommendations": self.recommendations,
        }


class CounterfactualEngine:
    """
    Analyzes brittleness by reasoning about what mutations would break the agent.

    We don't need to actually re-run the agent — we analyze the recorded
    trajectory and ask: "If file X was named differently / had a different
    constant, would this agent's navigation pattern still work?"

    Brittle signals:
    - Agent found bug file by pattern-matching on filename (not content search)
    - Agent submitted after reading the same file every run
    - Agent ignored test content and relied on positional heuristics

    Robust signals:
    - Agent used search_code to find function by name
    - Agent read test → traced import → found source
    - Agent ran tests and verified result before submitting
    """

    MUTATION_TEMPLATES = [
        {
            "type": "FILENAME_RENAME",
            "description": "Rename src/X.py to src/X_v2.py (same content)",
            "breaks_if": "agent found file by name pattern, not by search or import tracing",
            "surface_signal": "filename",
            "robust_signal": "import tracing or search_code",
        },
        {
            "type": "CONSTANT_CHANGE",
            "description": "Change a numeric constant by ±1 (semantically neutral for navigation)",
            "breaks_if": "agent hardcoded expected value rather than reading actual code",
            "surface_signal": "constant value pattern matching",
            "robust_signal": "dynamic code reading",
        },
        {
            "type": "DUMMY_FUNCTION",
            "description": "Add a dummy function with a similar name near the bug",
            "breaks_if": "agent used first-match navigation without reading full context",
            "surface_signal": "first result of search or first match in file",
            "robust_signal": "reading complete function signatures before deciding",
        },
        {
            "type": "DIRECTORY_SHUFFLE",
            "description": "Move test file from tests/ to test/ (same content)",
            "breaks_if": "agent hardcoded path prefix tests/ instead of searching",
            "surface_signal": "hardcoded directory prefix",
            "robust_signal": "search or dynamic discovery",
        },
        {
            "type": "DOCSTRING_NOISE",
            "description": "Add misleading docstring claiming a different function causes the bug",
            "breaks_if": "agent read docs instead of tests to understand expected behavior",
            "surface_signal": "docstring content",
            "robust_signal": "test assertions as ground truth",
        },
        {
            "type": "IMPORT_REORDER",
            "description": "Reorder imports in the source file",
            "breaks_if": "agent relied on line numbers instead of function names",
            "surface_signal": "absolute line numbers",
            "robust_signal": "function name search",
        },
    ]

    def analyze(
        self,
        episode_id: str,
        task: str,
        trajectory_steps: List[dict],
        variant_meta: dict,
        files_read: List[str],
        files_written: List[str],
        final_score: float,
    ) -> CounterfactualReport:
        """
        Analyze robustness by simulating mutations and reasoning about
        whether the agent's recorded pattern would survive them.
        """
        action_types = [s.get("action_type", "") for s in trajectory_steps]
        action_paths = [s.get("action_path") for s in trajectory_steps]

        bug_files = set(variant_meta.get("bug_files", []) or
                        variant_meta.get("files_to_implement", []) or [])
        test_files_meta = set(variant_meta.get("test_files", []) or [])

        # Infer what signals agent used
        used_search = "search_code" in action_types
        used_tests_first = self._tests_read_before_src(trajectory_steps, test_files_meta, bug_files)
        used_run_tests = "run_tests" in action_types
        blind_navigation = not used_search and not used_tests_first
        read_count = action_types.count("read_file")
        write_count = action_types.count("write_file")
        immediate_write = write_count > 0 and action_types.index("write_file") <= 2
        verified_before_submit = self._verified_before_submit(trajectory_steps)

        # ── Evaluate each mutation ────────────────────────────────────────────
        mutations: List[Mutation] = []

        for tmpl in self.MUTATION_TEMPLATES:
            target_file = self._pick_target_file(tmpl["type"], files_read, bug_files)
            would_break, why = self._would_break_agent(
                mutation_type=tmpl["type"],
                used_search=used_search,
                used_tests_first=used_tests_first,
                verified_before_submit=verified_before_submit,
                blind_navigation=blind_navigation,
                immediate_write=immediate_write,
                read_count=read_count,
                tmpl=tmpl,
            )
            mutations.append(Mutation(
                mutation_type=tmpl["type"],
                target_file=target_file or "unknown",
                description=tmpl["description"],
                would_break_agent=would_break,
                why=why,
            ))

        survived = sum(1 for m in mutations if not m.would_break_agent)
        failed = len(mutations) - survived

        robustness_score = survived / len(mutations) if mutations else 0.0

        # ── Surface vs deep dependency analysis ──────────────────────────────
        surface_deps = []
        deep_deps = []

        if not used_search:
            surface_deps.append("Filename-based navigation (no search_code used)")
        if not used_tests_first:
            surface_deps.append("Skipped test-informed navigation")
        if immediate_write:
            surface_deps.append("Immediate write after minimal reading (blind fix)")
        if not verified_before_submit:
            surface_deps.append("Submitted without running tests (no verification)")

        if used_search:
            deep_deps.append("Used search_code to find functions by name (content-based)")
        if used_tests_first:
            deep_deps.append("Read tests first — used expected behavior as compass")
        if read_count >= 3:
            deep_deps.append(f"Read {read_count} files — explored structure before committing")
        if verified_before_submit:
            deep_deps.append("Verified fix with run_tests before submitting")

        # ── Brittleness classification ────────────────────────────────────────
        if robustness_score >= 0.80:
            level = BrittlenessLevel.ROBUST
        elif robustness_score >= 0.60:
            level = BrittlenessLevel.MILDLY_BRITTLE
        elif robustness_score >= 0.30:
            level = BrittlenessLevel.BRITTLE
        else:
            level = BrittlenessLevel.FRAGILE

        explanations = {
            BrittlenessLevel.ROBUST: (
                "Agent strategy is robust. It relies on deep structural signals (function names, "
                "test assertions, causal chain traversal) rather than surface patterns. "
                "Minor repo mutations would not break its navigation."
            ),
            BrittlenessLevel.MILDLY_BRITTLE: (
                "Agent strategy is mildly brittle. Some mutations would break its navigation, "
                "particularly those that change surface signals it relied on. "
                "Using search_code and test-first navigation consistently would improve robustness."
            ),
            BrittlenessLevel.BRITTLE: (
                "Agent strategy is brittle. Most mutations would break its navigation. "
                "The agent appears to rely on stable surface patterns (filenames, positions) "
                "rather than understanding the semantic structure of the codebase."
            ),
            BrittlenessLevel.FRAGILE: (
                "Agent strategy is fragile. Almost any perturbation to the repo structure "
                "would cause this agent to fail. This indicates pure pattern-matching on "
                "the specific repo layout rather than generalizable code understanding."
            ),
        }

        recs = []
        if not used_search:
            recs.append("Use search_code to find functions by name — survives filename renames.")
        if not used_tests_first:
            recs.append("Read tests first to anchor your navigation in expected behavior, not filenames.")
        if immediate_write:
            recs.append("Read source files before writing to them — avoid blind writes.")
        if not verified_before_submit:
            recs.append("Run tests after writing — verify your fix holds on the actual behavior.")

        return CounterfactualReport(
            episode_id=episode_id,
            task=task,
            brittleness_level=level,
            robustness_score=robustness_score,
            mutations_tested=mutations,
            mutations_survived=survived,
            mutations_failed=failed,
            surface_dependencies=surface_deps,
            deep_dependencies=deep_deps,
            explanation=explanations[level],
            recommendations=recs,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _tests_read_before_src(
        self, steps: List[dict], test_files: set, bug_files: set
    ) -> bool:
        test_steps = [
            s.get("step_number", 99) for s in steps
            if s.get("action_type") == "read_file"
            and any(tf in (s.get("action_path") or "") for tf in test_files)
        ]
        src_steps = [
            s.get("step_number", 99) for s in steps
            if s.get("action_type") == "read_file"
            and any(bf in (s.get("action_path") or "") for bf in bug_files)
        ]
        if test_steps and src_steps:
            return min(test_steps) < min(src_steps)
        return False

    def _verified_before_submit(self, steps: List[dict]) -> bool:
        submit_step = next(
            (s.get("step_number", 9999) for s in steps if s.get("action_type") == "submit"),
            None,
        )
        if submit_step is None:
            return False
        return any(
            s.get("action_type") == "run_tests"
            and s.get("step_number", 0) < submit_step
            for s in steps
        )

    def _pick_target_file(
        self, mutation_type: str, files_read: List[str], bug_files: set
    ) -> str:
        if mutation_type in ("FILENAME_RENAME", "DUMMY_FUNCTION", "IMPORT_REORDER"):
            for f in bug_files:
                return f
            return files_read[0] if files_read else "src/main.py"
        if mutation_type == "DIRECTORY_SHUFFLE":
            for f in files_read:
                if "test" in f.lower():
                    return f
        return files_read[0] if files_read else "unknown"

    def _would_break_agent(
        self,
        mutation_type: str,
        used_search: bool,
        used_tests_first: bool,
        verified_before_submit: bool,
        blind_navigation: bool,
        immediate_write: bool,
        read_count: int,
        tmpl: dict,
    ) -> Tuple[bool, str]:
        """
        Return (would_break, explanation) by reasoning about the agent's signals.
        """
        if mutation_type == "FILENAME_RENAME":
            if used_search:
                return False, "Agent used search_code — finds function by name, not filename"
            if blind_navigation:
                return True, "Agent navigated by filename without search — rename breaks it"
            return True, "Agent likely relied on filename pattern without search fallback"

        if mutation_type == "CONSTANT_CHANGE":
            # Almost never breaks well-behaved agents
            if read_count >= 2:
                return False, "Agent read files dynamically — adapts to any constant value"
            return True, "Agent may have hardcoded expected value in navigation heuristic"

        if mutation_type == "DUMMY_FUNCTION":
            if used_search and read_count >= 3:
                return False, "Agent searched and read thoroughly — would disambiguate"
            return True, "Agent took first match without thorough reading"

        if mutation_type == "DIRECTORY_SHUFFLE":
            if used_search:
                return False, "search_code finds tests regardless of directory"
            return True, "Agent used hardcoded path prefix — directory change breaks it"

        if mutation_type == "DOCSTRING_NOISE":
            if used_tests_first:
                return False, "Agent used test assertions as ground truth, not docstrings"
            return True, "Agent may have read misleading docstring instead of test"

        if mutation_type == "IMPORT_REORDER":
            # Only brittle if agent relied on line numbers
            if read_count <= 1:
                return True, "Agent skimmed — likely used line numbers for navigation"
            return False, "Agent read full files — import reorder doesn't change function content"

        return False, "Neutral mutation"
