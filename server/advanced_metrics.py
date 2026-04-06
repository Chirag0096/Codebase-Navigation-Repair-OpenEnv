# server/advanced_metrics.py
"""
Advanced Metrics Engine.

Computes metrics that existing benchmarks (SWE-bench, etc.) completely ignore:
- Exploration vs Exploitation ratio across episode
- Consistency score across multiple runs of same task
- Reliability index (weighted aggregate)
- Reasoning efficiency (useful actions / total actions)
- Decision entropy (how predictable/focused the agent is)
"""
import math
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class AdvancedMetricsReport:
    """All advanced metrics for one episode or cross-episode comparison."""

    # Per-episode
    reasoning_efficiency: float    # Useful steps / total steps
    exploration_ratio: float       # Read+search vs write+test ratio
    decision_entropy: float        # Shannon entropy of action distribution
    reliability_index: float       # Composite reliability score
    pivot_rate: float              # Strategy changes per 10 steps
    wasteful_ratio: float          # Redundant actions / total actions

    # Cross-episode (populated when history provided)
    consistency_score: float = 0.0   # Variance across runs (lower variance = higher consistency)
    runs_analyzed: int = 0

    # Breakdowns
    action_distribution: Dict[str, int] = field(default_factory=dict)
    useful_actions: List[str] = field(default_factory=list)
    wasteful_actions: List[str] = field(default_factory=list)
    reliability_breakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "reasoning_efficiency": round(self.reasoning_efficiency, 3),
            "exploration_ratio": round(self.exploration_ratio, 3),
            "decision_entropy": round(self.decision_entropy, 3),
            "reliability_index": round(self.reliability_index, 3),
            "pivot_rate": round(self.pivot_rate, 3),
            "wasteful_ratio": round(self.wasteful_ratio, 3),
            "consistency_score": round(self.consistency_score, 3),
            "runs_analyzed": self.runs_analyzed,
            "action_distribution": self.action_distribution,
            "useful_actions": self.useful_actions,
            "wasteful_actions": self.wasteful_actions,
            "reliability_breakdown": {
                k: round(v, 3) for k, v in self.reliability_breakdown.items()
            },
        }


class AdvancedMetricsEngine:
    """
    Computes advanced behavioral and reliability metrics from trajectory data.

    Usage:
        engine = AdvancedMetricsEngine()
        report = engine.compute(
            trajectory_steps=[...],
            variant_meta={...},
            final_score=0.7,
            files_read=[...],
            files_written=[...],
            history=[],  # Pass previous episode scores for consistency
        )
    """

    def __init__(self):
        self._score_history: List[float] = []  # Tracks scores across episodes

    def compute(
        self,
        trajectory_steps: List[dict],
        variant_meta: Dict[str, Any],
        final_score: float,
        files_read: List[str],
        files_written: List[str],
        history: Optional[List[float]] = None,
    ) -> AdvancedMetricsReport:
        """Compute all advanced metrics for one episode."""
        # Record this score in history
        self._score_history.append(final_score)

        if not trajectory_steps:
            return AdvancedMetricsReport(
                reasoning_efficiency=0.0,
                exploration_ratio=0.5,
                decision_entropy=0.0,
                reliability_index=0.0,
                pivot_rate=0.0,
                wasteful_ratio=1.0,
            )

        action_seq = [s.get("action_type", "unknown") for s in trajectory_steps]
        total = len(action_seq)

        # ── Action distribution ───────────────────────────────────────────────
        from collections import Counter
        dist = Counter(action_seq)
        action_distribution = dict(dist)

        # ── Decision entropy (Shannon entropy of action types) ────────────────
        entropy = 0.0
        for count in dist.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        # Normalize by max possible entropy (log2 of unique action types)
        max_entropy = math.log2(len(dist)) if len(dist) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # ── Exploration vs exploitation ratio ─────────────────────────────────
        explore = dist.get("read_file", 0) + dist.get("search_code", 0)
        exploit = dist.get("write_file", 0) + dist.get("run_tests", 0)
        exploration_ratio = explore / (explore + exploit) if (explore + exploit) > 0 else 0.5

        # ── Redundancy / wasteful actions ─────────────────────────────────────
        read_paths = [
            s.get("action_path")
            for s in trajectory_steps
            if s.get("action_type") == "read_file" and s.get("action_path")
        ]
        seen = set()
        redundant_reads = 0
        for p in read_paths:
            if p in seen:
                redundant_reads += 1
            seen.add(p)

        error_actions = sum(1 for s in trajectory_steps if s.get("error"))
        total_wasteful = redundant_reads + error_actions
        wasteful_ratio = total_wasteful / total if total > 0 else 0.0

        wasteful_actions = []
        if redundant_reads > 0:
            wasteful_actions.append(f"{redundant_reads}x redundant file reads")
        if error_actions > 0:
            wasteful_actions.append(f"{error_actions}x actions that produced errors")

        # ── Useful action detection ───────────────────────────────────────────
        useful_actions = []
        relevant = set(
            variant_meta.get("bug_files", []) +
            variant_meta.get("interface_files", []) +
            variant_meta.get("read_first_files", []) +
            variant_meta.get("files_to_implement", [])
        )
        relevant_reads = [f for f in files_read if f in relevant]
        if relevant_reads:
            useful_actions.append(f"Read {len(relevant_reads)} key files: {relevant_reads[:3]}")

        test_rates = [
            s.get("test_pass_rate")
            for s in trajectory_steps
            if s.get("test_pass_rate") is not None
        ]
        if len(test_rates) >= 2 and test_rates[-1] > test_rates[0]:
            useful_actions.append(
                f"Test pass rate improved from {test_rates[0]:.2f} to {test_rates[-1]:.2f}"
            )

        if files_written:
            useful_actions.append(f"Wrote {len(files_written)} file(s): {files_written[:3]}")

        # ── Reasoning efficiency ──────────────────────────────────────────────
        useful_count = len(relevant_reads) + (1 if files_written else 0) + (1 if test_rates else 0)
        reasoning_efficiency = min(1.0, useful_count / max(total, 1))

        # ── Pivot rate (strategy switches per 10 steps) ───────────────────────
        pivots = 0
        for i in range(1, len(action_seq)):
            prev_explore = action_seq[i-1] in ("read_file", "search_code")
            curr_exploit = action_seq[i] in ("write_file", "run_tests")
            prev_exploit = action_seq[i-1] in ("write_file", "run_tests")
            curr_explore = action_seq[i] in ("read_file", "search_code")
            if (prev_explore and curr_exploit) or (prev_exploit and curr_explore):
                pivots += 1
        pivot_rate = (pivots / total) * 10 if total > 0 else 0.0  # per 10 steps

        # ── Reliability index ─────────────────────────────────────────────────
        # Weighted aggregate: correctness matters most
        reliability_breakdown = {
            "correctness": final_score,
            "efficiency": max(0.0, 1.0 - wasteful_ratio),
            "focus": 1.0 - normalized_entropy,  # Low entropy = focused behavior
            "verification": 1.0 if test_rates else 0.0,
            "safety": 1.0,  # Will be reduced by security violations
        }

        # Check for security flags
        sec_flags = sum(len(s.get("security_flags", [])) for s in trajectory_steps)
        if sec_flags > 0:
            reliability_breakdown["safety"] = max(0.0, 1.0 - sec_flags * 0.2)

        # Weighted reliability index
        weights = {
            "correctness": 0.40,
            "efficiency": 0.20,
            "focus": 0.15,
            "verification": 0.15,
            "safety": 0.10,
        }
        reliability_index = sum(
            reliability_breakdown[k] * weights[k]
            for k in weights
        )

        # ── Consistency score (cross-episode) ────────────────────────────────
        scores_to_use = list(history) if history else self._score_history
        consistency_score = 0.0
        runs_analyzed = len(scores_to_use)

        if runs_analyzed >= 2:
            mean = sum(scores_to_use) / runs_analyzed
            variance = sum((s - mean) ** 2 for s in scores_to_use) / runs_analyzed
            std_dev = math.sqrt(variance)
            # Consistency = 1 - normalized_std_dev (higher = more consistent)
            consistency_score = max(0.0, 1.0 - (std_dev / max(mean, 0.01)))

        return AdvancedMetricsReport(
            reasoning_efficiency=reasoning_efficiency,
            exploration_ratio=exploration_ratio,
            decision_entropy=normalized_entropy,
            reliability_index=reliability_index,
            pivot_rate=pivot_rate,
            wasteful_ratio=wasteful_ratio,
            consistency_score=consistency_score,
            runs_analyzed=runs_analyzed,
            action_distribution=action_distribution,
            useful_actions=useful_actions,
            wasteful_actions=wasteful_actions,
            reliability_breakdown=reliability_breakdown,
        )

    def get_score_history(self) -> List[float]:
        return list(self._score_history)

    def reset_history(self):
        self._score_history = []
