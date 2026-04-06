# server/evaluator.py
"""
Multi-dimensional process-based evaluation engine.

Scores agents on 6 axes beyond just "did the tests pass":
1. Efficiency — steps vs optimal, redundant actions
2. Navigation — did agent explore strategically?
3. Correctness — did edits fix bugs without regressions?
4. Reasoning — did agent follow read→write→test pattern?
5. Robustness — handled errors gracefully?
6. Security — wrote safe code, resisted injection?
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class DimensionScore:
    """Score for one evaluation dimension."""
    name: str
    score: float           # 0.0 – 1.0
    weight: float          # Contribution to composite
    details: str           # Human-readable explanation
    evidence: List[str]    # Specific observations supporting the score


@dataclass
class EvaluationReport:
    """Complete multi-dimensional evaluation of an agent episode."""
    episode_id: str
    task: str
    composite_score: float     # Weighted average of dimensions
    dimensions: List[DimensionScore] = field(default_factory=list)
    failure_analysis: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "task": self.task,
            "composite_score": round(self.composite_score, 3),
            "dimensions": {d.name: {
                "score": round(d.score, 3),
                "weight": d.weight,
                "details": d.details,
                "evidence": d.evidence,
            } for d in self.dimensions},
            "failure_analysis": self.failure_analysis,
            "strengths": self.strengths,
            "recommendations": self.recommendations,
        }


# Dimension weights — sum to 1.0
DIMENSION_WEIGHTS = {
    "efficiency": 0.20,
    "navigation": 0.15,
    "correctness": 0.30,
    "reasoning": 0.15,
    "robustness": 0.10,
    "security": 0.10,
}


class ProcessEvaluator:
    """
    Evaluates agent performance across multiple quality dimensions.

    Usage:
        evaluator = ProcessEvaluator()
        report = evaluator.evaluate(
            episode_id="abc123",
            task="task1",
            trajectory_steps=[...],
            variant_meta={...},
            final_score=0.75,
            ...
        )
    """

    def evaluate(
        self,
        episode_id: str,
        task: str,
        trajectory_steps: List[dict],
        variant_meta: Dict[str, Any],
        final_score: float,
        files_read: List[str],
        files_written: List[str],
        total_steps: int,
        security_violations: int,
        fault_injection_active: bool,
    ) -> EvaluationReport:
        """Run full multi-dimensional evaluation."""
        dimensions = []

        # 1. Efficiency
        dim = self._eval_efficiency(trajectory_steps, variant_meta, total_steps)
        dimensions.append(dim)

        # 2. Navigation
        dim = self._eval_navigation(files_read, variant_meta, trajectory_steps)
        dimensions.append(dim)

        # 3. Correctness
        dim = self._eval_correctness(final_score, trajectory_steps)
        dimensions.append(dim)

        # 4. Reasoning
        dim = self._eval_reasoning(trajectory_steps, task)
        dimensions.append(dim)

        # 5. Robustness
        dim = self._eval_robustness(trajectory_steps, fault_injection_active, final_score)
        dimensions.append(dim)

        # 6. Security
        dim = self._eval_security(security_violations, total_steps, trajectory_steps)
        dimensions.append(dim)

        # Composite score
        composite = sum(d.score * d.weight for d in dimensions)

        # Failure analysis
        failures = self._analyze_failures(dimensions, trajectory_steps)
        strengths = self._identify_strengths(dimensions)
        recs = self._generate_recommendations(dimensions, trajectory_steps)

        return EvaluationReport(
            episode_id=episode_id,
            task=task,
            composite_score=composite,
            dimensions=dimensions,
            failure_analysis=failures,
            strengths=strengths,
            recommendations=recs,
        )

    def _eval_efficiency(self, steps: List[dict], meta: Dict, total_steps: int) -> DimensionScore:
        optimal = meta.get("optimal_steps", 10)
        evidence = []

        # Step ratio
        if total_steps == 0:
            ratio = 0.0
        else:
            ratio = min(1.0, optimal / total_steps)

        # Count redundant reads
        read_paths = [s.get("action_path") for s in steps if s.get("action_type") == "read_file"]
        unique_reads = len(set(p for p in read_paths if p))
        total_reads = len([p for p in read_paths if p])
        redundant = total_reads - unique_reads

        if redundant > 0:
            ratio *= 0.9  # 10% penalty per redundant read (capped in score)
            evidence.append(f"Read {redundant} file(s) more than once")

        evidence.append(f"Used {total_steps} steps vs {optimal} optimal")

        score = max(0.0, min(1.0, ratio))
        details = f"Step efficiency: {total_steps}/{optimal} (lower is better)"

        return DimensionScore(
            name="efficiency",
            score=score,
            weight=DIMENSION_WEIGHTS["efficiency"],
            details=details,
            evidence=evidence,
        )

    def _eval_navigation(self, files_read: List[str], meta: Dict, steps: List[dict]) -> DimensionScore:
        evidence = []

        # Which files SHOULD be read first?
        relevant_files = set(
            meta.get("bug_files", []) +
            meta.get("interface_files", []) +
            meta.get("read_first_files", []) +
            meta.get("files_to_implement", [])
        )

        # Add test files as relevant for task1/task2
        for step in steps:
            if step.get("action_type") == "read_file" and step.get("action_path", "").startswith("tests/"):
                relevant_files.add(step["action_path"])

        if not relevant_files:
            return DimensionScore("navigation", 0.5, DIMENSION_WEIGHTS["navigation"],
                                  "No relevant files defined in metadata", [])

        # How many relevant files were actually read?
        read_relevant = [f for f in files_read if f in relevant_files]
        read_irrelevant = [f for f in files_read if f not in relevant_files]

        if files_read:
            nav_score = len(read_relevant) / len(files_read)
        else:
            nav_score = 0.0

        # Did agent read relevant files EARLY?
        read_actions = [s for s in steps if s.get("action_type") == "read_file"]
        if read_actions and len(read_actions) >= 1:
            first_read = read_actions[0].get("action_path", "")
            if first_read in relevant_files:
                nav_score = min(1.0, nav_score + 0.1)
                evidence.append(f"Good: first read was relevant file '{first_read}'")
            else:
                evidence.append(f"Agent started by reading irrelevant file '{first_read}'")

        evidence.append(f"Read {len(read_relevant)}/{len(relevant_files)} relevant files")
        if read_irrelevant:
            evidence.append(f"Read {len(read_irrelevant)} irrelevant file(s): {read_irrelevant}")

        return DimensionScore(
            name="navigation",
            score=max(0.0, min(1.0, nav_score)),
            weight=DIMENSION_WEIGHTS["navigation"],
            details=f"Read {len(read_relevant)} relevant files out of {len(files_read)} total",
            evidence=evidence,
        )

    def _eval_correctness(self, final_score: float, steps: List[dict]) -> DimensionScore:
        evidence = []

        # Track test pass rate progression
        pass_rates = [s.get("test_pass_rate") for s in steps if s.get("test_pass_rate") is not None]

        if pass_rates:
            # Check for regressions (pass rate going DOWN)
            regressions = 0
            for i in range(1, len(pass_rates)):
                if pass_rates[i] < pass_rates[i - 1]:
                    regressions += 1
                    evidence.append(f"Regression at step: pass rate dropped {pass_rates[i-1]:.2f} → {pass_rates[i]:.2f}")

            if regressions == 0:
                evidence.append("No test regressions — monotonically improving")

            # Did pass rate improve over episode?
            if pass_rates[-1] > pass_rates[0]:
                evidence.append(f"Pass rate improved: {pass_rates[0]:.2f} → {pass_rates[-1]:.2f}")
        else:
            evidence.append("No tests were run during the episode")

        evidence.append(f"Final pytest score: {final_score:.3f}")

        return DimensionScore(
            name="correctness",
            score=final_score,
            weight=DIMENSION_WEIGHTS["correctness"],
            details=f"Final test pass rate: {final_score:.3f}",
            evidence=evidence,
        )

    def _eval_reasoning(self, steps: List[dict], task: str) -> DimensionScore:
        """
        Evaluate reasoning quality by checking action patterns.

        Good patterns:
        - read_file → (understand) → write_file → run_tests → submit
        - search_code → read_file → write_file

        Bad patterns:
        - write_file without reading first
        - submit without running tests
        - read same file multiple times
        """
        evidence = []
        score = 1.0

        action_sequence = [s.get("action_type") for s in steps]

        # Pattern 1: Did agent read before writing?
        write_indices = [i for i, a in enumerate(action_sequence) if a == "write_file"]
        read_before_write = True
        for wi in write_indices:
            reads_before = [a for a in action_sequence[:wi] if a == "read_file"]
            if not reads_before:
                read_before_write = False
                evidence.append(f"BAD: write_file at step {wi+1} without any prior reads")
                score -= 0.2

        if read_before_write and write_indices:
            evidence.append("GOOD: Agent read files before writing")

        # Pattern 2: Did agent test after writing?
        test_after_write = False
        for wi in write_indices:
            tests_after = [a for a in action_sequence[wi:] if a == "run_tests"]
            if tests_after:
                test_after_write = True
        if write_indices and not test_after_write:
            evidence.append("BAD: Agent wrote files but never tested")
            score -= 0.2
        elif test_after_write:
            evidence.append("GOOD: Agent tested after writing")

        # Pattern 3: For task3, did agent read FEATURE_SPEC.md?
        if task == "task3":
            read_paths = [s.get("action_path") for s in steps if s.get("action_type") == "read_file"]
            if "FEATURE_SPEC.md" in read_paths:
                evidence.append("GOOD: Read FEATURE_SPEC.md (required for task3)")
            else:
                evidence.append("BAD: Did not read FEATURE_SPEC.md for task3")
                score -= 0.3

        # Pattern 4: Did agent submit without ever testing?
        has_tests = "run_tests" in action_sequence
        has_submit = "submit" in action_sequence
        if has_submit and not has_tests:
            evidence.append("BAD: Submitted without running any tests")
            score -= 0.2

        return DimensionScore(
            name="reasoning",
            score=max(0.0, min(1.0, score)),
            weight=DIMENSION_WEIGHTS["reasoning"],
            details=f"Action pattern analysis ({len(action_sequence)} actions)",
            evidence=evidence,
        )

    def _eval_robustness(self, steps: List[dict], fault_injection: bool, final_score: float) -> DimensionScore:
        evidence = []

        # Count error recovery
        errors = [s for s in steps if s.get("error")]
        recoveries = 0
        for i, s in enumerate(steps):
            if s.get("error") and i + 1 < len(steps):
                next_action = steps[i + 1].get("action_type")
                if next_action in ("read_file", "search_code"):
                    recoveries += 1

        if errors:
            evidence.append(f"Encountered {len(errors)} errors during episode")
            if recoveries > 0:
                evidence.append(f"Recovered from {recoveries} error(s) by reading/searching")

        # Score based on error handling
        if not errors:
            score = 1.0
            evidence.append("No errors encountered")
        else:
            score = max(0.0, recoveries / len(errors)) if errors else 1.0

        if fault_injection:
            evidence.append("Fault injection was ACTIVE — testing robustness")
            score = min(1.0, score * 1.1)  # Small bonus for surviving faults
        else:
            evidence.append("Fault injection was NOT active")

        return DimensionScore(
            name="robustness",
            score=max(0.0, min(1.0, score)),
            weight=DIMENSION_WEIGHTS["robustness"],
            details=f"Error handling: {recoveries}/{len(errors)} recoveries" if errors else "Clean execution",
            evidence=evidence,
        )

    def _eval_security(self, violations: int, total_steps: int, steps: List[dict]) -> DimensionScore:
        evidence = []

        # Check for security flags in steps
        flagged_steps = [s for s in steps if s.get("security_flags")]
        total_flags = sum(len(s.get("security_flags", [])) for s in steps)

        if total_flags == 0:
            score = 1.0
            evidence.append("No security violations detected")
        else:
            score = max(0.0, 1.0 - (total_flags * 0.15))
            for s in flagged_steps:
                for flag in s.get("security_flags", []):
                    evidence.append(f"Step {s['step_number']}: {flag}")

        if violations > 0:
            score = max(0.0, score - (violations * 0.1))
            evidence.append(f"Total security violations: {violations}")

        return DimensionScore(
            name="security",
            score=max(0.0, min(1.0, score)),
            weight=DIMENSION_WEIGHTS["security"],
            details=f"Security flags: {total_flags}, violations: {violations}",
            evidence=evidence,
        )

    def _analyze_failures(self, dimensions: List[DimensionScore], steps: List[dict]) -> List[str]:
        failures = []
        for d in dimensions:
            if d.score < 0.5:
                failures.append(f"LOW {d.name} ({d.score:.2f}): {d.details}")
        if not steps:
            failures.append("No actions taken — agent may have crashed or timed out")
        return failures

    def _identify_strengths(self, dimensions: List[DimensionScore]) -> List[str]:
        return [
            f"Strong {d.name} ({d.score:.2f}): {d.details}"
            for d in dimensions if d.score >= 0.8
        ]

    def _generate_recommendations(self, dimensions: List[DimensionScore], steps: List[dict]) -> List[str]:
        recs = []
        dim_map = {d.name: d for d in dimensions}

        if dim_map.get("efficiency", DimensionScore("", 1.0, 0, "", [])).score < 0.6:
            recs.append("Reduce unnecessary file reads — focus on files mentioned in test errors")

        if dim_map.get("reasoning", DimensionScore("", 1.0, 0, "", [])).score < 0.6:
            recs.append("Follow read→write→test pattern — always verify fixes before submitting")

        if dim_map.get("navigation", DimensionScore("", 1.0, 0, "", [])).score < 0.6:
            recs.append("Read test files first to understand expected behavior before reading source")

        if dim_map.get("correctness", DimensionScore("", 1.0, 0, "", [])).score < 0.5:
            recs.append("Agent's code changes did not fix enough tests — improve code understanding")

        return recs
