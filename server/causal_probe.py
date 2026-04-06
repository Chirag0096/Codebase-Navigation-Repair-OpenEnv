# server/causal_probe.py
"""
Causal Reasoning Probe — v4.0

The key scientific question: Did the agent understand WHY the bug exists,
or did it accidentally fix it by pattern matching?

We measure causal understanding by checking if the agent traversed the
COMPLETE causal chain: Failing test → tested function → return path → root cause.

An agent that reads only the test and immediately rewrites the function
is guessing. An agent that reads test → traces the call stack → finds the
actual cause first is reasoning causally.

This is NOT in any current benchmark. SWE-bench only checks if the test passes.
We check HOW the agent got there.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class CausalUnderstandingLevel(str, Enum):
    DEEP = "DEEP"             # Full causal chain traversal
    PARTIAL = "PARTIAL"       # Partial chain (some steps missing)
    SUPERFICIAL = "SUPERFICIAL"  # Direct test→rewrite with no chain
    RANDOM = "RANDOM"         # No discernible causal pattern


@dataclass
class CausalChainNode:
    """One node in the reconstructed causal chain."""
    file: str
    role: str   # "test", "caller", "called", "root_cause", "missed"
    was_read: bool
    read_order: Optional[int]  # Which step did agent read this?


@dataclass
class CausalProbeReport:
    """
    Full causal reasoning analysis for one episode.
    This is the primary output of the CausalProbe.
    """
    episode_id: str
    task: str

    # Core verdict
    understanding_level: CausalUnderstandingLevel
    causal_score: float            # 0.0 – 1.0

    # Chain analysis
    expected_chain: List[CausalChainNode]  # What SHOULD have been read
    actual_chain_coverage: float           # Fraction of chain actually traversed
    chain_order_score: float               # Was chain traversed in correct order?

    # Behavioral signals
    read_before_write: bool        # Did agent read all relevant files before writing?
    test_informed_navigation: bool # Did reading tests change which files agent read next?
    search_before_navigate: bool   # Did agent search for function names before reading?
    submit_after_test: bool        # Did agent verify fix before submitting?

    # Signal: understanding vs guessing
    guessing_indicators: List[str]   # Signs agent was guessing
    understanding_indicators: List[str]  # Signs agent understood

    # Calibration
    false_confidence_detected: bool  # Submitted without reading root cause file
    shortcut_learning_detected: bool # Read test file → immediately wrote → submitted

    explanation: str
    recommendations: List[str]

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "task": self.task,
            "understanding_level": self.understanding_level.value,
            "causal_score": round(self.causal_score, 3),
            "chain_coverage": round(self.actual_chain_coverage, 3),
            "chain_order_score": round(self.chain_order_score, 3),
            "behavioral_signals": {
                "read_before_write": self.read_before_write,
                "test_informed_navigation": self.test_informed_navigation,
                "search_before_navigate": self.search_before_navigate,
                "submit_after_test": self.submit_after_test,
            },
            "guessing_indicators": self.guessing_indicators,
            "understanding_indicators": self.understanding_indicators,
            "diagnostics": {
                "false_confidence_detected": self.false_confidence_detected,
                "shortcut_learning_detected": self.shortcut_learning_detected,
            },
            "expected_chain": [
                {"file": n.file, "role": n.role, "read": n.was_read, "order": n.read_order}
                for n in self.expected_chain
            ],
            "explanation": self.explanation,
            "recommendations": self.recommendations,
        }


class CausalProbe:
    """
    Analyzes whether an agent engaged in true causal reasoning.

    The core insight: for a bug in src/X.py called from tests/test_X.py,
    the causal chain is:
        tests/test_X.py → (calls) → src/X.py → (calls) → src/utils.py (maybe)

    A causally-aware agent reads in this order.
    A shortcut agent reads the test, guesses the bug, rewrites without reading source.

    We score order, coverage, and behavioral signals.
    """

    def probe(
        self,
        episode_id: str,
        task: str,
        trajectory_steps: List[dict],
        variant_meta: dict,
        files_read: List[str],
        files_written: List[str],
        final_score: float,
    ) -> CausalProbeReport:
        """Run the causal probe on an episode's trajectory."""

        # ── Build expected causal chain from variant metadata ─────────────────
        test_files = variant_meta.get("test_files", []) or [
            f for f in variant_meta.get("read_first_files", []) if "test" in f
        ]
        bug_files = variant_meta.get("bug_files", []) or variant_meta.get("files_to_implement", [])
        dep_files = variant_meta.get("dependencies", []) or []

        # If metadata sparse, infer from trajectory
        all_files_in_traj = list({
            s.get("action_path") for s in trajectory_steps
            if s.get("action_path") and s.get("action_type") in ("read_file", "write_file")
        })

        if not test_files:
            test_files = [f for f in all_files_in_traj if "test" in f.lower()]
        if not bug_files:
            bug_files = [f for f in all_files_in_traj
                        if "test" not in f.lower() and f.endswith(".py")]

        # Build expected chain
        expected_chain: List[CausalChainNode] = []
        read_set = set(files_read)
        read_order: Dict[str, int] = {}
        for step in trajectory_steps:
            if step.get("action_type") == "read_file" and step.get("action_path"):
                path = step["action_path"]
                if path not in read_order:
                    read_order[path] = step.get("step_number", len(read_order) + 1)

        for tf in test_files:
            expected_chain.append(CausalChainNode(
                file=tf, role="test",
                was_read=tf in read_set,
                read_order=read_order.get(tf),
            ))
        for bf in bug_files:
            expected_chain.append(CausalChainNode(
                file=bf, role="root_cause",
                was_read=bf in read_set,
                read_order=read_order.get(bf),
            ))
        for df in dep_files:
            expected_chain.append(CausalChainNode(
                file=df, role="caller",
                was_read=df in read_set,
                read_order=read_order.get(df),
            ))

        if not expected_chain:
            # Fallback: any file is better than none
            for f in all_files_in_traj[:3]:
                expected_chain.append(CausalChainNode(
                    file=f, role="unknown",
                    was_read=True,
                    read_order=read_order.get(f),
                ))

        # ── Chain coverage ────────────────────────────────────────────────────
        chain_files_read = [n for n in expected_chain if n.was_read and n.role != "missed"]
        actual_chain_coverage = (
            len(chain_files_read) / len(expected_chain) if expected_chain else 0.0
        )

        # ── Chain order score (tests before src = good causal order) ──────────
        chain_order_score = 0.0
        test_orders = [n.read_order for n in expected_chain if n.role == "test" and n.read_order]
        src_orders = [n.read_order for n in expected_chain
                      if n.role in ("root_cause", "caller") and n.read_order]

        if test_orders and src_orders:
            # Good: all tests read before source files
            correct_order_pairs = sum(
                1 for to in test_orders for so in src_orders if to < so
            )
            total_pairs = len(test_orders) * len(src_orders)
            chain_order_score = correct_order_pairs / total_pairs if total_pairs > 0 else 0.0
        elif test_orders and not src_orders:
            chain_order_score = 0.3  # Partial — read tests but not source
        elif src_orders and not test_orders:
            chain_order_score = 0.2  # Read source without reading tests = weaker

        # ── Behavioral signals ────────────────────────────────────────────────
        action_types = [s.get("action_type", "") for s in trajectory_steps]
        action_paths = [s.get("action_path") for s in trajectory_steps]

        # read_before_write: all written files were read at least once before write
        read_before_write = True
        for step in trajectory_steps:
            if step.get("action_type") == "write_file" and step.get("action_path"):
                p = step["action_path"]
                step_n = step.get("step_number", 0)
                was_read_before = any(
                    s2.get("action_type") == "read_file"
                    and s2.get("action_path") == p
                    and s2.get("step_number", 99) < step_n
                    for s2 in trajectory_steps
                )
                if not was_read_before:
                    read_before_write = False
                    break

        # test_informed_navigation: did agent read source files AFTER reading tests?
        test_read_step = min(
            (s.get("step_number", 99) for s in trajectory_steps
             if s.get("action_type") == "read_file"
             and any(tf in (s.get("action_path") or "") for tf in test_files)),
            default=None
        )
        src_read_after_test = test_read_step is not None and any(
            s.get("action_type") == "read_file"
            and s.get("step_number", 0) > test_read_step
            and any(bf in (s.get("action_path") or "") for bf in bug_files)
            for s in trajectory_steps
        )
        test_informed_navigation = src_read_after_test

        # search_before_navigate: used search_code before reading source files
        search_steps = [s for s in trajectory_steps if s.get("action_type") == "search_code"]
        first_src_read = min(
            (s.get("step_number", 99) for s in trajectory_steps
             if s.get("action_type") == "read_file"
             and any(bf in (s.get("action_path") or "") for bf in bug_files)),
            default=None
        )
        search_before_navigate = bool(search_steps) and (
            first_src_read is None or
            any(s.get("step_number", 99) < first_src_read for s in search_steps)
        )

        # submit_after_test: ran tests before submitting
        test_runs = [s for s in trajectory_steps if s.get("action_type") == "run_tests"]
        submit_step = next(
            (s.get("step_number", 99) for s in trajectory_steps
             if s.get("action_type") == "submit"), None
        )
        submit_after_test = bool(test_runs) and submit_step is not None and any(
            s.get("step_number", 0) < submit_step for s in test_runs
        )

        # ── Guessing vs understanding indicators ──────────────────────────────
        guessing_indicators = []
        understanding_indicators = []

        total = len(trajectory_steps)

        # Guessing: short episode with low score
        if total <= 3 and final_score < 0.5:
            guessing_indicators.append(f"Submitted in only {total} steps with score {final_score:.2f}")

        # Guessing: wrote without reading
        if not read_before_write:
            guessing_indicators.append("Wrote to file(s) without first reading them")

        # Guessing: skipped test files
        if not any(n.was_read for n in expected_chain if n.role == "test"):
            guessing_indicators.append("Never read any test files")

        # Guessing: skipped source files
        if not any(n.was_read for n in expected_chain if n.role == "root_cause"):
            guessing_indicators.append("Never read the bug/source file")

        # Understanding: search used
        if search_steps:
            understanding_indicators.append(
                f"Used search_code {len(search_steps)}× to locate bug"
            )

        # Understanding: read tests first
        if chain_order_score > 0.7:
            understanding_indicators.append("Read tests before source files (correct causal order)")

        # Understanding: tested before submitting
        if submit_after_test:
            understanding_indicators.append("Verified fix with run_tests before submitting")

        # Understanding: explored full chain
        if actual_chain_coverage > 0.7:
            understanding_indicators.append(
                f"Covered {actual_chain_coverage:.0%} of expected causal chain"
            )

        # ── Diagnostics ───────────────────────────────────────────────────────
        # False confidence: submitted very early without testing
        false_confidence_detected = (
            submit_step is not None and submit_step <= 3 and not test_runs
        )

        # Shortcut learning: read test → immediate write → submit (skipped source)
        has_write = "write_file" in action_types
        has_src_read = any(
            bf in (s.get("action_path") or "")
            for s in trajectory_steps
            if s.get("action_type") == "read_file"
            for bf in bug_files
        )
        shortcut_sequence = has_write and not has_src_read
        shortcut_learning_detected = shortcut_sequence

        # ── Composite causal score ─────────────────────────────────────────────
        scores = {
            "chain_coverage": actual_chain_coverage * 0.30,
            "chain_order": chain_order_score * 0.25,
            "read_before_write": (0.15 if read_before_write else 0.0),
            "test_informed": (0.15 if test_informed_navigation else 0.0),
            "verified": (0.10 if submit_after_test else 0.0),
            "searched": (0.05 if search_before_navigate else 0.0),
        }
        causal_score = sum(scores.values())
        causal_score = max(0.0, min(1.0, causal_score))

        # ── Understanding level classification ────────────────────────────────
        if causal_score >= 0.75:
            level = CausalUnderstandingLevel.DEEP
        elif causal_score >= 0.45:
            level = CausalUnderstandingLevel.PARTIAL
        elif causal_score >= 0.20:
            level = CausalUnderstandingLevel.SUPERFICIAL
        else:
            level = CausalUnderstandingLevel.RANDOM

        # ── Explanation ───────────────────────────────────────────────────────
        level_explanations = {
            CausalUnderstandingLevel.DEEP: (
                "Agent demonstrated genuine causal reasoning: read tests to understand expected "
                "behavior, traced the call chain to the root cause, made a targeted fix, and "
                "verified with tests before submitting."
            ),
            CausalUnderstandingLevel.PARTIAL: (
                "Agent showed partial causal understanding. Some chain links were traversed "
                "but the reasoning was incomplete — likely missed tracing deeper dependencies "
                "or skipped test verification."
            ),
            CausalUnderstandingLevel.SUPERFICIAL: (
                "Agent showed superficial reasoning. Actions did not follow a clear causal "
                "path from test → failure → root cause. Likely pattern-matched on filenames "
                "or guessed the fix location."
            ),
            CausalUnderstandingLevel.RANDOM: (
                "Agent showed no discernible causal reasoning. Actions appear random relative "
                "to the causal structure of the bug. This is the profile of pure trial-and-error."
            ),
        }
        explanation = level_explanations[level]

        # ── Recommendations ───────────────────────────────────────────────────
        recs = []
        if not any(n.was_read for n in expected_chain if n.role == "test"):
            recs.append("Always read the failing test first — it defines the expected behavior.")
        if not read_before_write:
            recs.append("Never write to a file before reading it — blind writes cause more bugs.")
        if not submit_after_test:
            recs.append("Run tests after every write to verify your fix is correct.")
        if not search_steps:
            recs.append("Use search_code to find function definitions before navigating blindly.")
        if actual_chain_coverage < 0.5:
            recs.append(
                "Explore more of the causal chain. The bug's root cause may be deeper than the first file."
            )
        if not recs:
            recs.append("Excellent reasoning! Maintain this systematic approach.")

        return CausalProbeReport(
            episode_id=episode_id,
            task=task,
            understanding_level=level,
            causal_score=causal_score,
            expected_chain=expected_chain,
            actual_chain_coverage=actual_chain_coverage,
            chain_order_score=chain_order_score,
            read_before_write=read_before_write,
            test_informed_navigation=test_informed_navigation,
            search_before_navigate=search_before_navigate,
            submit_after_test=submit_after_test,
            guessing_indicators=guessing_indicators,
            understanding_indicators=understanding_indicators,
            false_confidence_detected=false_confidence_detected,
            shortcut_learning_detected=shortcut_learning_detected,
            explanation=explanation,
            recommendations=recs,
        )
