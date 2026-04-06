# server/failure_classifier.py
"""
Typed Failure Classification Engine.

Classifies agent failures into precise, actionable categories rather than
vague scores. Each failure type has a root cause, evidence, and remediation.

Failure taxonomy:
  WRONG_FILE_NAVIGATION  — agent read irrelevant files, missed key files
  BLIND_WRITE            — agent wrote code without reading first
  HALLUCINATED_CODE      — agent wrote syntactically/logically wrong code
  NEVER_TESTED           — agent submitted without running any tests
  LOOPING_BEHAVIOR       — agent repeated same action 3+ times
  CONTEXT_OVERFLOW       — agent read enormous amounts of irrelevant data
  SECURITY_VIOLATION     — agent wrote dangerous code
  CORRECT                — no failure detected
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class FailureInstance:
    """One classified failure event."""
    failure_type: str        # e.g. "WRONG_FILE_NAVIGATION"
    severity: str            # "critical" | "major" | "minor"
    step_number: int         # Which step triggered it
    evidence: str            # Specific observation
    root_cause: str          # Why this happens
    remediation: str         # How to fix in next run


@dataclass
class FailureReport:
    """Full failure analysis for one episode."""
    episode_id: str
    task: str
    primary_failure: str        # Most severe failure type
    failures: List[FailureInstance] = field(default_factory=list)
    success: bool = False
    failure_summary: str = ""
    retry_hint: str = ""        # Actionable hint for the next attempt

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "task": self.task,
            "success": self.success,
            "primary_failure": self.primary_failure,
            "failure_count": len(self.failures),
            "failures": [
                {
                    "type": f.failure_type,
                    "severity": f.severity,
                    "step": f.step_number,
                    "evidence": f.evidence,
                    "root_cause": f.root_cause,
                    "remediation": f.remediation,
                }
                for f in self.failures
            ],
            "failure_summary": self.failure_summary,
            "retry_hint": self.retry_hint,
        }


# ── Severity ordering for picking primary failure ─────────────────────────────
SEVERITY_RANK = {"critical": 3, "major": 2, "minor": 1}

FAILURE_REMEDIATION = {
    "WRONG_FILE_NAVIGATION": (
        "Read the failing test file first to understand the module under test, "
        "then navigate directly to the imported source files."
    ),
    "BLIND_WRITE": (
        "Always read the target file before writing. Use read_file → write_file → run_tests."
    ),
    "HALLUCINATED_CODE": (
        "Re-read the source file, understand the function signature, "
        "then write a minimal targeted fix. Run tests to verify."
    ),
    "NEVER_TESTED": (
        "Always call run_tests after writing a fix. "
        "Submit only when test pass rate has demonstrably improved."
    ),
    "LOOPING_BEHAVIOR": (
        "Stop repeating the same action. Use search_code to find the bug location, "
        "then navigate directly to it."
    ),
    "CONTEXT_OVERFLOW": (
        "Focus on files explicitly referenced in the failing test's imports. "
        "Avoid reading utility files unless the test error specifically mentions them."
    ),
    "SECURITY_VIOLATION": (
        "Do not use os.system, eval, exec, or subprocess in fixes. "
        "Write pure Python logic without shell calls."
    ),
    "CORRECT": "No remediation needed.",
}


class FailureClassifier:
    """
    Classifies agent failures from trajectory data.

    Usage:
        clf = FailureClassifier()
        report = clf.classify(
            episode_id="abc123",
            task="task1",
            trajectory_steps=[...],
            variant_meta={...},
            files_read=[...],
            files_written=[...],
            final_score=0.0,
        )
    """

    def classify(
        self,
        episode_id: str,
        task: str,
        trajectory_steps: List[dict],
        variant_meta: Dict[str, Any],
        files_read: List[str],
        files_written: List[str],
        final_score: float,
        security_violations: int = 0,
    ) -> FailureReport:
        """Run all classifiers and build a structured failure report."""
        failures: List[FailureInstance] = []
        success = final_score >= 0.5

        if success and security_violations == 0:
            return FailureReport(
                episode_id=episode_id,
                task=task,
                primary_failure="CORRECT",
                failures=[],
                success=True,
                failure_summary="Agent succeeded without errors.",
                retry_hint="",
            )

        action_sequence = [s.get("action_type", "") for s in trajectory_steps]

        # ── Classifier 1: Wrong File Navigation ───────────────────────────────
        relevant = set(
            variant_meta.get("bug_files", []) +
            variant_meta.get("interface_files", []) +
            variant_meta.get("read_first_files", []) +
            variant_meta.get("files_to_implement", [])
        )
        if relevant and files_read:
            irrelevant_reads = [f for f in files_read if f not in relevant
                                and not f.startswith("tests/")]
            if len(irrelevant_reads) > 1 and not any(f in files_read for f in relevant):
                failures.append(FailureInstance(
                    failure_type="WRONG_FILE_NAVIGATION",
                    severity="critical",
                    step_number=1,
                    evidence=f"Read {len(irrelevant_reads)} irrelevant files: {irrelevant_reads[:3]}. "
                             f"Never read key files: {list(relevant)[:3]}",
                    root_cause="Agent navigated to wrong part of the codebase entirely.",
                    remediation=FAILURE_REMEDIATION["WRONG_FILE_NAVIGATION"],
                ))

        # ── Classifier 2: Blind Write ─────────────────────────────────────────
        write_indices = [i for i, a in enumerate(action_sequence) if a == "write_file"]
        for wi in write_indices:
            reads_before = [a for a in action_sequence[:wi] if a == "read_file"]
            if not reads_before:
                step = trajectory_steps[wi]
                failures.append(FailureInstance(
                    failure_type="BLIND_WRITE",
                    severity="critical",
                    step_number=wi + 1,
                    evidence=f"write_file at step {wi+1} with zero prior read_file actions.",
                    root_cause="Agent attempted to fix code without reading it first — likely hallucinating.",
                    remediation=FAILURE_REMEDIATION["BLIND_WRITE"],
                ))

        # ── Classifier 3: Hallucinated Code ───────────────────────────────────
        # Detect write followed by immediate test failure
        for i, step in enumerate(trajectory_steps):
            if step.get("action_type") == "run_tests":
                prev_write = None
                for j in range(i - 1, -1, -1):
                    if trajectory_steps[j].get("action_type") == "write_file":
                        prev_write = j
                        break
                if prev_write is not None:
                    pass_rate = step.get("test_pass_rate", None)
                    if pass_rate is not None and pass_rate < 0.3:
                        failures.append(FailureInstance(
                            failure_type="HALLUCINATED_CODE",
                            severity="major",
                            step_number=i + 1,
                            evidence=f"Test pass rate {pass_rate:.2f} after write at step {prev_write+1}. "
                                     f"Code change made things worse.",
                            root_cause="Agent wrote syntactically correct but semantically wrong code.",
                            remediation=FAILURE_REMEDIATION["HALLUCINATED_CODE"],
                        ))

        # ── Classifier 4: Never Tested ────────────────────────────────────────
        has_tests = "run_tests" in action_sequence
        has_writes = "write_file" in action_sequence
        has_submit = "submit" in action_sequence
        if has_submit and has_writes and not has_tests:
            failures.append(FailureInstance(
                failure_type="NEVER_TESTED",
                severity="major",
                step_number=len(action_sequence),
                evidence="Agent wrote code changes but submitted without running any tests.",
                root_cause="No feedback loop — agent cannot know if its fix worked.",
                remediation=FAILURE_REMEDIATION["NEVER_TESTED"],
            ))

        # ── Classifier 5: Looping Behavior ────────────────────────────────────
        read_paths = [
            (i, s.get("action_path"))
            for i, s in enumerate(trajectory_steps)
            if s.get("action_type") == "read_file" and s.get("action_path")
        ]
        path_counts: Dict[str, List[int]] = {}
        for idx, path in read_paths:
            path_counts.setdefault(path, []).append(idx)

        for path, indices in path_counts.items():
            if len(indices) >= 3:
                failures.append(FailureInstance(
                    failure_type="LOOPING_BEHAVIOR",
                    severity="major",
                    step_number=indices[2] + 1,
                    evidence=f"Read '{path}' {len(indices)} times (steps {[i+1 for i in indices]}). "
                             f"Agent is stuck in a read loop.",
                    root_cause="Agent cannot extract the needed information and keeps retrying.",
                    remediation=FAILURE_REMEDIATION["LOOPING_BEHAVIOR"],
                ))

        # ── Classifier 6: Context Overflow ────────────────────────────────────
        total_content = sum(
            s.get("action_content_length") or 0
            for s in trajectory_steps
            if s.get("action_type") == "read_file"
        )
        if total_content > 50_000 and final_score < 0.5:
            failures.append(FailureInstance(
                failure_type="CONTEXT_OVERFLOW",
                severity="minor",
                step_number=len(trajectory_steps),
                evidence=f"Agent read {total_content:,} chars total. "
                         f"Most of this was likely irrelevant context.",
                root_cause="Agent wasted token budget reading unnecessary files.",
                remediation=FAILURE_REMEDIATION["CONTEXT_OVERFLOW"],
            ))

        # ── Classifier 7: Security Violation ─────────────────────────────────
        if security_violations > 0:
            sec_steps = [
                s for s in trajectory_steps if s.get("security_flags")
            ]
            for ss in sec_steps:
                failures.append(FailureInstance(
                    failure_type="SECURITY_VIOLATION",
                    severity="critical",
                    step_number=ss.get("step_number", 0),
                    evidence=f"Flags: {ss.get('security_flags', [])}",
                    root_cause="Agent wrote unsafe code patterns that would be dangerous in production.",
                    remediation=FAILURE_REMEDIATION["SECURITY_VIOLATION"],
                ))

        # ── Build report ──────────────────────────────────────────────────────
        if not failures:
            # Failed but no specific classifier triggered — generic low score
            primary = "HALLUCINATED_CODE"
            summary = f"Score {final_score:.2f} — fix was written but insufficient. Re-read the source files more carefully."
            hint = "Read test file → read all src files → write targeted fix → run tests → submit."
        else:
            # Pick most severe failure as primary
            failures.sort(key=lambda f: SEVERITY_RANK.get(f.severity, 0), reverse=True)
            primary = failures[0].failure_type
            summary = "; ".join(f"{f.failure_type} (step {f.step_number})" for f in failures[:3])
            hint = failures[0].remediation

        return FailureReport(
            episode_id=episode_id,
            task=task,
            primary_failure=primary,
            failures=failures,
            success=success,
            failure_summary=summary,
            retry_hint=hint,
        )
