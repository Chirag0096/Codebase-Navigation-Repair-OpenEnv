# server/benchmark_runner.py
"""
Benchmark Runner + Leaderboard — v4.0

Automatically runs ALL tasks × selected agent configurations and generates
a research-grade leaderboard output with per-task, per-strategy breakdowns.

Unlike existing benchmarks (SWE-bench, HumanEval) which require manual setup,
this runs end-to-end in-process with deterministic strategies.

Output format:
- Leaderboard table (ranked by composite score)
- Per-task breakdown
- Per-failure-type breakdown
- Generalization score (variance across tasks)
- Robustness score (from counterfactual engine)
- A "benchmark JSON" suitable for publishing or comparing systems
"""
from __future__ import annotations
import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class BenchmarkResult:
    """Result of running one agent on one task variant."""
    agent_name: str
    task: str
    variant_id: str
    final_score: float
    total_steps: int
    cumulative_reward: float
    duration_seconds: float
    strategy: str
    failure_type: str
    reliability_index: float
    causal_score: float
    robustness_score: float
    calibration_score: float
    action_sequence: List[str]


@dataclass
class AgentBenchmarkSummary:
    """Aggregated results for one agent across all tasks."""
    agent_name: str
    tasks_run: int
    mean_score: float
    std_score: float
    generalization_score: float  # 1 - std (lower variance = more generalizable)
    mean_steps: float
    best_task: str
    worst_task: str
    mean_reliability: float
    mean_causal_score: float
    mean_robustness_score: float
    mean_calibration_score: float
    dominant_strategy: str
    dominant_failure: str
    composite_rank_score: float   # Weighted final score for leaderboard
    per_task_scores: Dict[str, float]

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "tasks_run": self.tasks_run,
            "scores": {
                "mean": round(self.mean_score, 3),
                "std": round(self.std_score, 3),
                "generalization": round(self.generalization_score, 3),
                "reliability": round(self.mean_reliability, 3),
                "causal_reasoning": round(self.mean_causal_score, 3),
                "robustness": round(self.mean_robustness_score, 3),
                "calibration": round(self.mean_calibration_score, 3),
                "composite": round(self.composite_rank_score, 3),
            },
            "efficiency": {
                "mean_steps": round(self.mean_steps, 1),
            },
            "behavior": {
                "dominant_strategy": self.dominant_strategy,
                "dominant_failure": self.dominant_failure,
            },
            "per_task_scores": {k: round(v, 3) for k, v in self.per_task_scores.items()},
            "best_task": self.best_task,
            "worst_task": self.worst_task,
        }


@dataclass
class LeaderboardReport:
    """Full benchmark leaderboard."""
    benchmark_id: str
    tasks_evaluated: List[str]
    agents_evaluated: List[str]
    total_episodes: int
    run_duration_seconds: float
    rankings: List[AgentBenchmarkSummary]
    raw_results: List[BenchmarkResult]

    def to_dict(self) -> dict:
        return {
            "benchmark_id": self.benchmark_id,
            "tasks_evaluated": self.tasks_evaluated,
            "agents_evaluated": self.agents_evaluated,
            "total_episodes": self.total_episodes,
            "run_duration_seconds": round(self.run_duration_seconds, 2),
            "leaderboard": [r.to_dict() for r in self.rankings],
            "winner": self.rankings[0].agent_name if self.rankings else "none",
            "insights": self._generate_insights(),
        }

    def _generate_insights(self) -> List[str]:
        if not self.rankings:
            return []
        insights = []
        top = self.rankings[0]
        bottom = self.rankings[-1]

        if top.composite_rank_score - bottom.composite_rank_score > 0.2:
            insights.append(
                f"Large performance gap: '{top.agent_name}' ({top.composite_rank_score:.2f}) "
                f"vs '{bottom.agent_name}' ({bottom.composite_rank_score:.2f})"
            )
        if top.generalization_score > 0.7:
            insights.append(
                f"'{top.agent_name}' shows strong generalization "
                f"(std={top.std_score:.3f} across {top.tasks_run} tasks)"
            )
        for r in self.rankings:
            if r.mean_causal_score > 0.6:
                insights.append(
                    f"'{r.agent_name}' demonstrated genuine causal reasoning "
                    f"(causal_score={r.mean_causal_score:.2f})"
                )
        strategies = [r.dominant_strategy for r in self.rankings]
        if len(set(strategies)) > 1:
            best_strategy = self.rankings[0].dominant_strategy
            insights.append(
                f"Strategy '{best_strategy}' produced the highest composite score."
            )
        return insights

    def render_table(self) -> str:
        """Render ASCII leaderboard table."""
        if not self.rankings:
            return "No results."

        lines = [
            f"{'═'*90}",
            f"  🏆 BENCHMARK LEADERBOARD — {self.benchmark_id}",
            f"  Tasks: {', '.join(self.tasks_evaluated)} | Agents: {len(self.agents_evaluated)} | Episodes: {self.total_episodes}",
            f"{'═'*90}",
            f"{'Rank':<5} {'Agent':<16} {'Score':<8} {'Causal':<8} {'Robust':<8} {'Calibr':<8} {'Genrz':<8} {'Steps':<7} {'Strategy'}",
            f"{'─'*90}",
        ]
        for i, r in enumerate(self.rankings):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"  #{i+1}"
            lines.append(
                f"{medal:<5} {r.agent_name:<16} {r.mean_score:<8.3f} "
                f"{r.mean_causal_score:<8.3f} {r.mean_robustness_score:<8.3f} "
                f"{r.mean_calibration_score:<8.3f} {r.generalization_score:<8.3f} "
                f"{r.mean_steps:<7.1f} {r.dominant_strategy}"
            )
        lines.append(f"{'═'*90}")

        lines.append("\n📊 Per-Task Breakdown:")
        for r in self.rankings:
            task_str = " | ".join(f"{t}: {s:.2f}" for t, s in sorted(r.per_task_scores.items()))
            lines.append(f"  {r.agent_name:<16} {task_str}")

        if self._generate_insights():
            lines.append("\n💡 Insights:")
            lines.extend(f"  → {i}" for i in self._generate_insights())

        return "\n".join(lines)


class BenchmarkRunner:
    """
    Automated benchmark runner.

    Runs each agent in AGENT_CONFIGS across each task, collecting:
    - Final score
    - All intelligence metrics (causal, counterfactual, confidence)
    - Strategy and failure classification
    - Reliability index

    Then generates a ranked leaderboard.
    """

    def run(
        self,
        env,
        tasks: Optional[List[str]] = None,
        agents: Optional[List[str]] = None,
        benchmark_id: Optional[str] = None,
    ) -> LeaderboardReport:
        """Run the full benchmark."""
        import uuid
        from server.models import RepoAction
        from server.strategy_detector import StrategyDetector
        from server.failure_classifier import FailureClassifier
        from server.advanced_metrics import AdvancedMetricsEngine
        from server.causal_probe import CausalProbe
        from server.counterfactual_engine import CounterfactualEngine
        from server.confidence_calibrator import ConfidenceCalibrator

        benchmark_id = benchmark_id or f"bench_{uuid.uuid4().hex[:8]}"
        tasks = tasks or ["task1", "task2", "task3"]
        agent_configs = self._get_agent_configs()
        if agents:
            agent_configs = {k: v for k, v in agent_configs.items() if k in agents}

        clf = FailureClassifier()
        det = StrategyDetector()
        adv = AdvancedMetricsEngine()
        causal = CausalProbe()
        counter = CounterfactualEngine()
        calibrator = ConfidenceCalibrator()

        start_time = time.time()
        all_results: List[BenchmarkResult] = []

        for task in tasks:
            for agent_name, agent_fn in agent_configs.items():
                try:
                    result = self._run_episode(
                        env, task, agent_name, agent_fn,
                        clf, det, adv, causal, counter, calibrator
                    )
                    all_results.append(result)
                except Exception as e:
                    # Don't crash the whole benchmark on one failure
                    all_results.append(BenchmarkResult(
                        agent_name=agent_name, task=task, variant_id="error",
                        final_score=0.0, total_steps=0, cumulative_reward=0.0,
                        duration_seconds=0.0, strategy="ERROR", failure_type="BENCHMARK_ERROR",
                        reliability_index=0.0, causal_score=0.0, robustness_score=0.0,
                        calibration_score=0.0, action_sequence=[],
                    ))

        total_duration = time.time() - start_time
        rankings = self._compute_rankings(all_results, tasks)

        return LeaderboardReport(
            benchmark_id=benchmark_id,
            tasks_evaluated=tasks,
            agents_evaluated=list(agent_configs.keys()),
            total_episodes=len(all_results),
            run_duration_seconds=total_duration,
            rankings=rankings,
            raw_results=all_results,
        )

    def _run_episode(
        self, env, task, agent_name, agent_fn,
        clf, det, adv, causal, counter, calibrator
    ) -> BenchmarkResult:
        from server.models import RepoAction

        reset_result = env.reset(task=task)
        obs = reset_result.observation
        variant_id = reset_result.info.get("variant_id", "unknown")
        context = {}

        obs_dict = obs.model_dump()
        start = time.time()
        cumulative_reward = 0.0
        files_read, files_written, action_sequence = [], [], []
        max_steps = 15

        for step_num in range(1, max_steps + 1):
            if env.done:
                break
            action_dict = agent_fn(obs_dict, step_num, context)
            action = RepoAction(
                action_type=action_dict.get("action_type", "submit"),
                path=action_dict.get("path"),
                query=action_dict.get("query"),
            )
            result = env.step(action)
            obs = result.observation
            obs_dict = obs.model_dump()
            cumulative_reward += result.reward
            action_sequence.append(action.action_type)
            if action.path and action.action_type == "read_file":
                files_read.append(action.path)
            if action.path and action.action_type == "write_file":
                files_written.append(action.path)
            if result.done:
                break

        if not env.done:
            r = env.step(RepoAction(action_type="submit"))
            cumulative_reward += r.reward
            action_sequence.append("submit")

        duration = time.time() - start
        final_score = env.final_score
        traj = env.get_trajectory()
        steps = traj.get("steps", []) if traj else []
        meta = env.variant.meta if env.variant else {}

        # Intelligence metrics
        fail_r = clf.classify(
            traj.get("episode_id", "") if traj else "", task,
            steps, meta, files_read, files_written, final_score
        )
        strat_r = det.detect(steps, task, meta, files_read, final_score)
        adv_r = adv.compute(steps, meta, final_score, files_read, files_written)
        causal_r = causal.probe(
            traj.get("episode_id", "") if traj else "", task,
            steps, meta, files_read, files_written, final_score
        )
        counter_r = counter.analyze(
            traj.get("episode_id", "") if traj else "", task,
            steps, meta, files_read, files_written, final_score
        )
        calib_r = calibrator.calibrate(
            traj.get("episode_id", "") if traj else "", task,
            steps, final_score,
        )

        return BenchmarkResult(
            agent_name=agent_name,
            task=task,
            variant_id=variant_id,
            final_score=final_score,
            total_steps=len(action_sequence),
            cumulative_reward=cumulative_reward,
            duration_seconds=duration,
            strategy=strat_r.strategy,
            failure_type=fail_r.primary_failure,
            reliability_index=adv_r.reliability_index,
            causal_score=causal_r.causal_score,
            robustness_score=counter_r.robustness_score,
            calibration_score=calib_r.calibration_score,
            action_sequence=action_sequence,
        )

    def _compute_rankings(
        self, results: List[BenchmarkResult], tasks: List[str]
    ) -> List[AgentBenchmarkSummary]:
        import math
        from collections import Counter

        # Group by agent
        agent_results: Dict[str, List[BenchmarkResult]] = {}
        for r in results:
            agent_results.setdefault(r.agent_name, []).append(r)

        summaries = []
        for agent_name, agent_res in agent_results.items():
            scores = [r.final_score for r in agent_res]
            mean_score = sum(scores) / len(scores)
            if len(scores) > 1:
                variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
                std_score = math.sqrt(variance)
            else:
                std_score = 0.0
            generalization_score = max(0.0, 1.0 - std_score)

            per_task = {r.task: r.final_score for r in agent_res}
            strategies = Counter(r.strategy for r in agent_res)
            failures = Counter(r.failure_type for r in agent_res)

            mean_steps = sum(r.total_steps for r in agent_res) / len(agent_res)
            mean_reliability = sum(r.reliability_index for r in agent_res) / len(agent_res)
            mean_causal = sum(r.causal_score for r in agent_res) / len(agent_res)
            mean_robustness = sum(r.robustness_score for r in agent_res) / len(agent_res)
            mean_calibration = sum(r.calibration_score for r in agent_res) / len(agent_res)

            # Composite leaderboard score — weighted across all dimensions
            composite = (
                mean_score * 0.35 +
                mean_causal * 0.20 +
                mean_robustness * 0.15 +
                mean_calibration * 0.15 +
                generalization_score * 0.15
            )

            best_task = max(per_task, key=per_task.get)
            worst_task = min(per_task, key=per_task.get)

            summaries.append(AgentBenchmarkSummary(
                agent_name=agent_name,
                tasks_run=len(agent_res),
                mean_score=mean_score,
                std_score=std_score,
                generalization_score=generalization_score,
                mean_steps=mean_steps,
                best_task=best_task,
                worst_task=worst_task,
                mean_reliability=mean_reliability,
                mean_causal_score=mean_causal,
                mean_robustness_score=mean_robustness,
                mean_calibration_score=mean_calibration,
                dominant_strategy=strategies.most_common(1)[0][0],
                dominant_failure=failures.most_common(1)[0][0],
                composite_rank_score=composite,
                per_task_scores=per_task,
            ))

        summaries.sort(key=lambda s: -s.composite_rank_score)
        return summaries

    def _get_agent_configs(self) -> Dict:
        """Reuse built-in strategies from multi_agent.py."""
        from server.multi_agent import MultiAgentComparison
        return MultiAgentComparison.AGENT_CONFIGS
