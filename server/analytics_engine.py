# server/analytics_engine.py
"""
Unified Analytics Engine — v4.0

Aggregates ALL scoring dimensions into a single research-grade report.
Produces:
- Reasoning graph (structured DAG of the agent's decision process)
- Root cause analysis (why the agent failed at every level)
- Decision efficiency score
- Overall AI reliability profile (radar chart data)
- Paper-ready JSON suitable for arXiv submission

This module is the "top of the stack" — it calls all other engines
and synthesizes their outputs into one authoritative report.
"""
from __future__ import annotations
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ReasoningNode:
    """One node in the agent's reconstructed reasoning graph."""
    node_id: str
    step_number: int
    action_type: str
    target: Optional[str]     # file path or search query
    reward: float
    was_useful: bool
    connected_to: List[str]   # IDs of subsequent nodes that built on this
    label: str                # Human-readable description


@dataclass
class ReasoningGraph:
    """
    A directed graph reconstruction of the agent's thought process.
    
    Nodes = actions taken.
    Edges = "built on" relationships (e.g., write followed a read = used info from read).
    Clusters = logical reasoning phases (Exploration, Hypothesis, Verification, Commit)
    """
    nodes: List[ReasoningNode]
    phases: Dict[str, List[str]]  # phase_name → [node_ids]
    critical_path: List[str]       # node_ids on the most impactful path
    wasted_nodes: List[str]        # node_ids that contributed nothing
    optimal_path_comparison: Optional[str]  # What should the agent have done

    def to_dict(self) -> dict:
        return {
            "nodes": [
                {
                    "id": n.node_id, "step": n.step_number,
                    "action": n.action_type, "target": n.target,
                    "reward": round(n.reward, 3), "useful": n.was_useful,
                    "connects_to": n.connected_to, "label": n.label,
                }
                for n in self.nodes
            ],
            "phases": self.phases,
            "critical_path": self.critical_path,
            "wasted_nodes": self.wasted_nodes,
            "optimal_path": self.optimal_path_comparison,
        }


@dataclass
class AnalyticsReport:
    """
    The master analytics report — synthesizes all evaluation dimensions.
    Paper-ready, structured for research publication or leaderboard submission.
    """
    report_id: str
    episode_id: str
    task: str
    variant_id: str
    generated_at: float

    # Dimension scores (0.0–1.0 each)
    correctness_score: float        # Did it fix the bug?
    causal_score: float             # Did it understand WHY?
    robustness_score: float         # Is the strategy resilient?
    calibration_score: float        # Was it appropriately confident?
    reliability_index: float        # Weighted multi-dim score
    generalization_hint: float      # Based on strategy (robust strategies generalize better)
    decision_efficiency: float      # Score / Steps ratio (normalized)
    process_quality: float          # How structured was the reasoning process?

    # Composite
    composite_score: float          # Weighted aggregate of all dimensions

    # Graph
    reasoning_graph: ReasoningGraph

    # Root cause trees
    failure_root_causes: List[Dict]  # Each: {cause, effect, evidence, depth}

    # Alternative path analysis
    what_agent_did: List[str]
    what_agent_should_have_done: List[str]
    steps_wasted: int
    steps_optimal: int

    # Profile tags
    profile_tags: List[str]  # e.g., ["OVERCONFIDENT", "SHORTCUT_LEARNER", "WELL_CALIBRATED"]

    # Executive summary
    executive_summary: str
    researcher_notes: str    # More technical deep dive

    def to_dict(self) -> dict:
        return {
            "report_id": self.report_id,
            "episode_id": self.episode_id,
            "task": self.task,
            "variant_id": self.variant_id,
            "generated_at": self.generated_at,
            "dimension_scores": {
                "correctness": round(self.correctness_score, 3),
                "causal_reasoning": round(self.causal_score, 3),
                "robustness": round(self.robustness_score, 3),
                "calibration": round(self.calibration_score, 3),
                "reliability_index": round(self.reliability_index, 3),
                "generalization": round(self.generalization_hint, 3),
                "decision_efficiency": round(self.decision_efficiency, 3),
                "process_quality": round(self.process_quality, 3),
                "composite": round(self.composite_score, 3),
            },
            "reasoning_graph": self.reasoning_graph.to_dict(),
            "failure_root_causes": self.failure_root_causes,
            "alternative_paths": {
                "what_agent_did": self.what_agent_did,
                "optimal_path": self.what_agent_should_have_done,
                "steps_wasted": self.steps_wasted,
                "steps_optimal": self.steps_optimal,
            },
            "profile_tags": self.profile_tags,
            "executive_summary": self.executive_summary,
            "researcher_notes": self.researcher_notes,
        }

    def render_text(self) -> str:
        """Render a human-readable analytics report."""
        def bar(v: float, width: int = 20) -> str:
            filled = int(v * width)
            return "█" * filled + "░" * (width - filled)

        lines = [
            f"{'═'*70}",
            f"  📈 ANALYTICS ENGINE REPORT — {self.task} | {self.variant_id}",
            f"  Episode: {self.episode_id}",
            f"{'═'*70}",
            "",
            "┌─ DIMENSION SCORES ─────────────────────────────────────────────────",
            f"│  Correctness      [{bar(self.correctness_score)}] {self.correctness_score:.3f}",
            f"│  Causal Reasoning [{bar(self.causal_score)}] {self.causal_score:.3f}",
            f"│  Robustness       [{bar(self.robustness_score)}] {self.robustness_score:.3f}",
            f"│  Calibration      [{bar(self.calibration_score)}] {self.calibration_score:.3f}",
            f"│  Reliability      [{bar(self.reliability_index)}] {self.reliability_index:.3f}",
            f"│  Decision Effic.  [{bar(self.decision_efficiency)}] {self.decision_efficiency:.3f}",
            f"│  Process Quality  [{bar(self.process_quality)}] {self.process_quality:.3f}",
            f"│  {'─'*60}",
            f"│  COMPOSITE        [{bar(self.composite_score)}] {self.composite_score:.3f}",
            "└────────────────────────────────────────────────────────────────────",
            "",
        ]

        if self.profile_tags:
            lines.append(f"🏷️  Profile: {' | '.join(self.profile_tags)}")
            lines.append("")

        lines += [
            "📝 Executive Summary",
            f"   {self.executive_summary}",
            "",
        ]

        if self.failure_root_causes:
            lines.append("🔥 Failure Root Cause Analysis")
            for rc in self.failure_root_causes[:3]:
                lines.append(f"   Cause:  {rc.get('cause')}")
                lines.append(f"   Effect: {rc.get('effect')}")
                lines.append(f"   Fix:    {rc.get('remediation')}")
                lines.append("")

        lines += [
            "🗺️  What Agent Did vs Optimal",
            f"   Steps taken: {len(self.what_agent_did)} | Steps optimal: {self.steps_optimal} | Wasted: {self.steps_wasted}",
        ]
        for a, o in zip(
            self.what_agent_did[:5],
            self.what_agent_should_have_done[:5],
        ):
            prefix_a = "  ✓" if a == o else "  ✗"
            lines.append(f"   Agent:   {a}")
            lines.append(f"   Optimal: {o}")
            lines.append("")

        if self.researcher_notes:
            lines += ["🔬 Researcher Notes", f"   {self.researcher_notes}", ""]

        lines.append(f"{'═'*70}")
        return "\n".join(lines)


class AnalyticsEngine:
    """
    Master analytics engine — integrates all evaluation modules.

    Call .analyze() after an episode to get the full AnalyticsReport.
    """

    def analyze(
        self,
        env,
        causal_report=None,
        counterfactual_report=None,
        calibration_report=None,
        advanced_metrics=None,
        failure_report=None,
        strategy_report=None,
    ) -> AnalyticsReport:
        """
        Synthesize all evaluation outputs into one AnalyticsReport.
        Each sub-report is optional — we gracefully handle None.
        """
        import uuid

        traj = env.get_trajectory()
        steps = traj.get("steps", []) if traj else []
        meta = env.variant.meta if env.variant else {}
        episode_id = traj.get("episode_id", "unknown") if traj else "unknown"
        variant_id = traj.get("variant_id", "unknown") if traj else "unknown"
        task = env.current_task or "unknown"
        final_score = env.final_score
        files_read = list(env.files_read)
        files_written = list(env.files_written)

        # ── Run sub-engines if reports not provided ────────────────────────────
        if causal_report is None:
            from server.causal_probe import CausalProbe
            causal_report = CausalProbe().probe(
                episode_id, task, steps, meta, files_read, files_written, final_score
            )
        if counterfactual_report is None:
            from server.counterfactual_engine import CounterfactualEngine
            counterfactual_report = CounterfactualEngine().analyze(
                episode_id, task, steps, meta, files_read, files_written, final_score
            )
        if calibration_report is None:
            from server.confidence_calibrator import ConfidenceCalibrator
            calibration_report = ConfidenceCalibrator().calibrate(
                episode_id, task, steps, final_score
            )
        if advanced_metrics is None:
            from server.advanced_metrics import AdvancedMetricsEngine
            advanced_metrics = AdvancedMetricsEngine().compute(
                steps, meta, final_score, files_read, files_written
            )
        if failure_report is None:
            from server.failure_classifier import FailureClassifier
            failure_report = FailureClassifier().classify(
                episode_id, task, steps, meta, files_read, files_written, final_score
            )
        if strategy_report is None:
            from server.strategy_detector import StrategyDetector
            strategy_report = StrategyDetector().detect(
                steps, task, meta, files_read, final_score
            )

        # ── Compute derived scores ─────────────────────────────────────────────
        causal_score = causal_report.causal_score
        robustness_score = counterfactual_report.robustness_score
        calibration_score = calibration_report.calibration_score
        reliability_index = advanced_metrics.reliability_index
        correctness_score = final_score

        # Decision efficiency: correctness per step, normalized
        total_steps = max(len(steps), 1)
        max_steps_possible = meta.get("max_steps", 20)
        decision_efficiency = (
            final_score /
            max(1.0, total_steps / max(1, max_steps_possible / 3))
        )
        decision_efficiency = min(1.0, decision_efficiency)

        # Process quality: measures structural quality of reasoning process
        read_before_write = causal_report.read_before_write
        tested_before_submit = causal_report.submit_after_test
        used_search = causal_report.search_before_navigate
        full_chain = causal_report.actual_chain_coverage
        process_quality = (
            (0.25 if read_before_write else 0.0) +
            (0.25 if tested_before_submit else 0.0) +
            (0.20 if used_search else 0.0) +
            full_chain * 0.30
        )

        # Generalization hint from strategy robustness
        strategy_generalization_map = {
            "TARGETED_DEBUGGING": 0.75,
            "SYSTEMATIC_SEARCH": 0.70,
            "SPEC_DRIVEN": 0.80,
            "BRUTE_FORCE": 0.40,
            "RANDOM_EXPLORATION": 0.30,
            "MINIMAL_EFFORT": 0.20,
        }
        generalization_hint = strategy_generalization_map.get(strategy_report.strategy, 0.5)
        generalization_hint = (generalization_hint + robustness_score) / 2

        # Composite (research-grade weighted aggregate)
        composite_score = (
            correctness_score * 0.30 +
            causal_score * 0.20 +
            robustness_score * 0.15 +
            calibration_score * 0.12 +
            reliability_index * 0.10 +
            process_quality * 0.08 +
            decision_efficiency * 0.05
        )

        # ── Build reasoning graph ──────────────────────────────────────────────
        reasoning_graph = self._build_reasoning_graph(steps, meta, files_read, files_written)

        # ── Root cause analysis ────────────────────────────────────────────────
        root_causes = self._build_root_cause_tree(
            failure_report, causal_report, calibration_report, final_score
        )

        # ── Alternative path analysis ─────────────────────────────────────────
        what_did = [
            f"{s.get('action_type')} {s.get('action_path') or s.get('action_query') or ''}".strip()
            for s in steps
        ]
        optimal = self._compute_optimal_path(meta, files_read, files_written, final_score)
        steps_wasted = max(0, total_steps - len(optimal))

        # ── Profile tags ───────────────────────────────────────────────────────
        tags = []
        if calibration_report.profile.value != "WELL_CALIBRATED":
            tags.append(calibration_report.profile.value)
        if causal_report.shortcut_learning_detected:
            tags.append("SHORTCUT_LEARNER")
        if causal_report.false_confidence_detected:
            tags.append("FALSE_CONFIDENCE")
        if counterfactual_report.brittleness_level.value in ("BRITTLE", "FRAGILE"):
            tags.append(f"BRITTLE_STRATEGY_{counterfactual_report.brittleness_level.value}")
        if causal_report.understanding_level.value == "DEEP":
            tags.append("DEEP_REASONER")
        if strategy_report.strategy == "TARGETED_DEBUGGING":
            tags.append("TARGETED_DEBUGGER")
        if not tags:
            tags.append("TYPICAL")

        # ── Executive summary ──────────────────────────────────────────────────
        summary_parts = [
            f"Agent scored {final_score:.2f} on {task}.",
            f"Causal understanding: {causal_report.understanding_level.value} ({causal_score:.2f}).",
            f"Strategy: {strategy_report.strategy} (robustness: {robustness_score:.2f}).",
            f"Confidence calibration: {calibration_report.profile.value} (error: {calibration_report.expected_calibration_error:.2f}).",
            f"Composite reliability: {composite_score:.2f}.",
        ]
        executive_summary = " ".join(summary_parts)

        # ── Researcher notes ───────────────────────────────────────────────────
        researcher_notes = (
            f"Observed {total_steps} steps ({steps_wasted} wasted vs estimated {len(optimal)} optimal). "
            f"Chain coverage: {causal_report.actual_chain_coverage:.0%}. "
            f"Chain order score: {causal_report.chain_order_score:.2f}. "
            f"Counterfactual mutations survived: {counterfactual_report.mutations_survived}/{len(counterfactual_report.mutations_tested)}. "
            f"Expected calibration error: {calibration_report.expected_calibration_error:.3f}. "
            f"Decision efficiency: {decision_efficiency:.3f}. "
            f"Process quality: {process_quality:.3f}."
        )

        return AnalyticsReport(
            report_id=f"ar_{uuid.uuid4().hex[:10]}",
            episode_id=episode_id,
            task=task,
            variant_id=variant_id,
            generated_at=time.time(),
            correctness_score=correctness_score,
            causal_score=causal_score,
            robustness_score=robustness_score,
            calibration_score=calibration_score,
            reliability_index=reliability_index,
            generalization_hint=generalization_hint,
            decision_efficiency=decision_efficiency,
            process_quality=process_quality,
            composite_score=composite_score,
            reasoning_graph=reasoning_graph,
            failure_root_causes=root_causes,
            what_agent_did=what_did,
            what_agent_should_have_done=optimal,
            steps_wasted=steps_wasted,
            steps_optimal=len(optimal),
            profile_tags=tags,
            executive_summary=executive_summary,
            researcher_notes=researcher_notes,
        )

    def _build_reasoning_graph(
        self,
        steps: List[dict],
        meta: dict,
        files_read: List[str],
        files_written: List[str],
    ) -> ReasoningGraph:
        """Build a DAG from the trajectory steps."""
        bug_files = set(meta.get("bug_files", []) + meta.get("files_to_implement", []))

        nodes: List[ReasoningNode] = []
        phases: Dict[str, List[str]] = {
            "Exploration": [], "Hypothesis": [], "Verification": [], "Commit": []
        }
        files_read_set = set()
        last_useful_node_id: Optional[str] = None
        all_node_ids: List[str] = []

        for s in steps:
            node_id = f"n{s.get('step_number', len(nodes)+1)}"
            atype = s.get("action_type", "unknown")
            target = s.get("action_path") or s.get("action_query")
            reward = s.get("reward", 0.0)

            # Determine usefulness
            was_useful = (
                reward > 0 or
                (atype == "read_file" and target in bug_files) or
                (atype == "search_code") or
                (atype == "run_tests") or
                (atype == "submit" and reward > 0)
            )

            # Determine phase
            if atype in ("read_file", "search_code"):
                phase = "Exploration"
            elif atype == "write_file":
                phase = "Hypothesis"
            elif atype == "run_tests":
                phase = "Verification"
            else:
                phase = "Commit"

            # Build label
            short_target = (target.split("/")[-1] if target else "")[:20] if target else ""
            label = f"{atype}({short_target})" if short_target else atype

            # Connections: link to previous useful node
            connects_to = [last_useful_node_id] if last_useful_node_id and was_useful else []
            connects_to = [c for c in connects_to if c]

            node = ReasoningNode(
                node_id=node_id,
                step_number=s.get("step_number", len(nodes) + 1),
                action_type=atype,
                target=target,
                reward=reward,
                was_useful=was_useful,
                connected_to=connects_to,
                label=label,
            )
            nodes.append(node)
            phases[phase].append(node_id)
            all_node_ids.append(node_id)
            if was_useful:
                last_useful_node_id = node_id

        # Critical path: nodes with positive reward or that led to the final submit
        critical_path = [n.node_id for n in nodes if n.reward > 0 or n.action_type == "submit"]
        wasted_nodes = [n.node_id for n in nodes if not n.was_useful and n.action_type != "submit"]

        # Optimal path comparison
        optimal_actions = []
        test_files = [f for f in (list(files_read) + list(bug_files)) if "test" in f.lower()]
        src_files = [f for f in (list(files_read) + list(bug_files)) if f not in test_files]
        for tf in test_files[:1]:
            optimal_actions.append(f"read_file({tf.split('/')[-1]})")
        for sf in src_files[:2]:
            optimal_actions.append(f"read_file({sf.split('/')[-1]})")
        optimal_actions += ["write_file(src)", "run_tests", "submit"]
        optimal_path = " → ".join(optimal_actions)

        return ReasoningGraph(
            nodes=nodes,
            phases={k: v for k, v in phases.items() if v},
            critical_path=critical_path,
            wasted_nodes=wasted_nodes,
            optimal_path_comparison=optimal_path,
        )

    def _build_root_cause_tree(
        self, failure_report, causal_report, calibration_report, final_score: float
    ) -> List[Dict]:
        """Build a structured root cause tree."""
        causes = []

        if failure_report and failure_report.failures:
            for f in failure_report.failures[:3]:
                causes.append({
                    "depth": "primary",
                    "cause": f.failure_type if hasattr(f, "failure_type") else str(f),
                    "effect": f.evidence if hasattr(f, "evidence") else "unknown",
                    "remediation": f.remediation if hasattr(f, "remediation") else "See improvement plan",
                })
        elif final_score < 0.5:
            causes.append({
                "depth": "primary",
                "cause": failure_report.primary_failure if failure_report else "LOW_SCORE",
                "effect": f"Final score only {final_score:.2f} — bug not adequately fixed",
                "remediation": "Use test-first navigation and verify with run_tests",
            })

        if causal_report and causal_report.guessing_indicators:
            for ind in causal_report.guessing_indicators[:2]:
                causes.append({
                    "depth": "secondary",
                    "cause": "CAUSAL_GAP",
                    "effect": ind,
                    "remediation": causal_report.recommendations[0] if causal_report.recommendations else "",
                })

        if calibration_report and calibration_report.profile.value == "OVERCONFIDENT":
            causes.append({
                "depth": "secondary",
                "cause": "OVERCONFIDENCE",
                "effect": f"Inferred confidence {calibration_report.inferred_confidence:.2f} vs actual {calibration_report.actual_performance:.2f}",
                "remediation": "Read more before committing. Verify with tests.",
            })

        return causes

    def _compute_optimal_path(
        self, meta: dict, files_read: List[str], files_written: List[str], score: float
    ) -> List[str]:
        """Suggest what the optimal action sequence would have been."""
        test_files = [f for f in files_read if "test" in f.lower()]
        bug_files = meta.get("bug_files", []) or meta.get("files_to_implement", [])

        path = []
        for tf in (test_files or ["tests/test_main.py"])[:1]:
            path.append(f"read_file {tf}")
        for bf in (bug_files or ["src/main.py"])[:2]:
            path.append(f"read_file {bf}")
        path.append("search_code <function_name>")
        path.append("write_file <targeted_fix>")
        path.append("run_tests")
        path.append("submit")
        return path
