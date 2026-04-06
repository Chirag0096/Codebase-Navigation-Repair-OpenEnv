# server/multi_agent.py
"""
Multi-Agent Comparison Engine.

Runs multiple agent configurations against the SAME task variant
and produces a side-by-side comparison report.

Agent configurations:
  - Deterministic (rule-based, no LLM) — baseline
  - Test-first (forces reading tests before anything)
  - Search-first (forces search_code before reads)
  - LLM-based (if HF_TOKEN provided)

This is the key feature that answers: "Which agent strategy wins?"
"""
import time
import copy
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class AgentRunResult:
    """Result of one agent configuration running one episode."""
    agent_name: str
    task: str
    variant_id: str
    final_score: float
    total_steps: int
    cumulative_reward: float
    duration_seconds: float
    action_sequence: List[str]
    files_read: List[str]
    files_written: List[str]
    strategy: str              # Detected strategy label
    strategy_score: float
    failure_type: str
    reliability_index: float
    step_timeline: List[dict]

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "task": self.task,
            "variant_id": self.variant_id,
            "final_score": round(self.final_score, 3),
            "total_steps": self.total_steps,
            "cumulative_reward": round(self.cumulative_reward, 3),
            "duration_seconds": round(self.duration_seconds, 2),
            "action_sequence": self.action_sequence,
            "files_read": self.files_read,
            "files_written": self.files_written,
            "strategy": self.strategy,
            "strategy_score": round(self.strategy_score, 3),
            "failure_type": self.failure_type,
            "reliability_index": round(self.reliability_index, 3),
            "step_timeline": self.step_timeline,
        }


@dataclass
class ComparisonReport:
    """Side-by-side comparison of multiple agent configurations."""
    task: str
    variant_id: str
    runs: List[AgentRunResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        if not self.runs:
            return {"error": "No runs to compare"}

        # Rank by score then steps
        ranked = sorted(self.runs, key=lambda r: (-r.final_score, r.total_steps))
        winner = ranked[0]

        return {
            "task": self.task,
            "variant_id": self.variant_id,
            "winner": winner.agent_name,
            "winner_score": winner.final_score,
            "summary_table": [
                {
                    "rank": i + 1,
                    "agent": r.agent_name,
                    "score": round(r.final_score, 3),
                    "steps": r.total_steps,
                    "reward": round(r.cumulative_reward, 3),
                    "strategy": r.strategy,
                    "failure": r.failure_type,
                    "reliability": round(r.reliability_index, 3),
                }
                for i, r in enumerate(ranked)
            ],
            "detailed_runs": [r.to_dict() for r in self.runs],
            "insights": self._generate_insights(ranked),
        }

    def _generate_insights(self, ranked: List[AgentRunResult]) -> List[str]:
        insights = []
        if len(ranked) < 2:
            return insights

        best = ranked[0]
        worst = ranked[-1]

        if best.final_score > worst.final_score + 0.2:
            insights.append(
                f"'{best.agent_name}' significantly outperformed '{worst.agent_name}' "
                f"({best.final_score:.2f} vs {worst.final_score:.2f})"
            )

        step_diffs = [(r.agent_name, r.total_steps) for r in ranked]
        most_efficient = min(ranked, key=lambda r: r.total_steps if r.final_score >= 0.5 else float('inf'))
        if most_efficient.final_score >= 0.5:
            insights.append(
                f"Most step-efficient successful agent: '{most_efficient.agent_name}' "
                f"({most_efficient.total_steps} steps)"
            )

        strategies = [r.strategy for r in ranked]
        if len(set(strategies)) > 1:
            insights.append(
                f"Strategy variance observed: {set(strategies)} — "
                f"'{best.agent_name}' used {best.strategy} which proved most effective."
            )

        return insights


class MultiAgentComparison:
    """
    Runs multiple deterministic agent strategies against the same environment.

    Usage (in-process, no LLM required):
        from server.environment import CodebaseNavEnvironment
        from server.models import RepoAction

        env = CodebaseNavEnvironment()
        engine = MultiAgentComparison()
        report = engine.compare(env, task="task1")
    """

    # ── Built-in agent strategies ─────────────────────────────────────────────

    @staticmethod
    def _agent_test_first(obs: dict, step: int, context: dict) -> dict:
        """Strategy: Read tests before any source file."""
        tree = obs.get("repo_tree", [])
        files_read = set(obs.get("files_read", []))

        test_files = sorted([f for f in tree if f.startswith("tests/")])
        src_files = sorted([f for f in tree if f.startswith("src/") and f.endswith(".py")])
        spec_files = sorted([f for f in tree if f.endswith(".md")])

        # Phase 1: Tests first
        for tf in test_files:
            if tf not in files_read:
                return {"action_type": "read_file", "path": tf}
        # Phase 2: Source files
        for sf in src_files:
            if sf not in files_read:
                return {"action_type": "read_file", "path": sf}
        # Phase 3: Run tests
        if test_files and context.get("tests_run", 0) == 0:
            context["tests_run"] = 1
            return {"action_type": "run_tests", "path": test_files[0]}
        return {"action_type": "submit"}

    @staticmethod
    def _agent_search_first(obs: dict, step: int, context: dict) -> dict:
        """Strategy: Use search_code to locate the bug before reading."""
        tree = obs.get("repo_tree", [])
        files_read = set(obs.get("files_read", []))
        failing = obs.get("failing_tests", [])

        # Step 1: search for the failing test function name
        if step == 1 and failing:
            fn_name = failing[0].split(".")[-1] if failing else "bug"
            context["searched"] = True
            return {"action_type": "search_code", "query": fn_name}

        # Step 2: Read files based on search
        test_files = sorted([f for f in tree if f.startswith("tests/")])
        src_files = sorted([f for f in tree if f.startswith("src/") and f.endswith(".py")])

        for tf in test_files:
            if tf not in files_read:
                return {"action_type": "read_file", "path": tf}
        for sf in src_files:
            if sf not in files_read:
                return {"action_type": "read_file", "path": sf}
        if test_files and context.get("tests_run", 0) == 0:
            context["tests_run"] = 1
            return {"action_type": "run_tests", "path": test_files[0]}
        return {"action_type": "submit"}

    @staticmethod
    def _agent_minimal(obs: dict, step: int, context: dict) -> dict:
        """Strategy: Minimal effort — read one file, submit immediately."""
        tree = obs.get("repo_tree", [])
        files_read = set(obs.get("files_read", []))
        src_files = [f for f in tree if f.startswith("src/") and f.endswith(".py")]
        if src_files and not files_read:
            return {"action_type": "read_file", "path": src_files[0]}
        return {"action_type": "submit"}

    @staticmethod
    def _agent_exhaustive(obs: dict, step: int, context: dict) -> dict:
        """Strategy: Read everything, run tests, then submit."""
        tree = obs.get("repo_tree", [])
        files_read = set(obs.get("files_read", []))

        all_readable = [f for f in tree if f.endswith(".py") or f.endswith(".md")]
        for f in all_readable:
            if f not in files_read:
                return {"action_type": "read_file", "path": f}

        test_files = [f for f in tree if f.startswith("tests/")]
        if test_files and context.get("tests_run", 0) == 0:
            context["tests_run"] = 1
            return {"action_type": "run_tests", "path": test_files[0]}
        if test_files and context.get("tests_run2", 0) == 0:
            context["tests_run2"] = 1
            return {"action_type": "run_tests"}
        return {"action_type": "submit"}

    AGENT_CONFIGS = {
        "test-first": _agent_test_first.__func__,
        "search-first": _agent_search_first.__func__,
        "minimal": _agent_minimal.__func__,
        "exhaustive": _agent_exhaustive.__func__,
    }

    def compare(
        self,
        env,  # CodebaseNavEnvironment instance
        task: str = "task1",
        agents: Optional[List[str]] = None,
        shared_variant: Optional[str] = None,
    ) -> ComparisonReport:
        """
        Run all (or selected) agents against the same task and compare.
        The environment is reset to the same variant for each agent.
        """
        from server.models import RepoAction
        from server.strategy_detector import StrategyDetector
        from server.failure_classifier import FailureClassifier
        from server.advanced_metrics import AdvancedMetricsEngine

        agent_names = agents or list(self.AGENT_CONFIGS.keys())
        strategy_detector = StrategyDetector()
        failure_classifier = FailureClassifier()
        metrics_engine = AdvancedMetricsEngine()

        runs: List[AgentRunResult] = []
        variant_id = None

        for agent_name in agent_names:
            agent_fn = self.AGENT_CONFIGS.get(agent_name)
            if not agent_fn:
                continue

            # Reset environment
            reset_result = env.reset(task=task)
            obs = reset_result.observation
            variant_id = reset_result.info.get("variant_id", "unknown")

            context = {}
            start = time.time()
            max_steps = 15
            files_read = []
            files_written = []
            cumulative_reward = 0.0
            action_sequence = []
            step_timeline = []

            obs_dict = obs.model_dump()

            for step_num in range(1, max_steps + 1):
                if env.done:
                    break

                action_dict = agent_fn(obs_dict, step_num, context)
                action = RepoAction(
                    action_type=action_dict.get("action_type", "submit"),
                    path=action_dict.get("path"),
                    query=action_dict.get("query"),
                    content=action_dict.get("content"),
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

                step_timeline.append({
                    "step": step_num,
                    "action": action.action_type,
                    "path": action.path,
                    "reward": round(result.reward, 3),
                })

                if result.done:
                    break

            # Force submit if not done
            if not env.done:
                result = env.step(RepoAction(action_type="submit"))
                cumulative_reward += result.reward
                action_sequence.append("submit")

            duration = time.time() - start
            final_score = env.final_score

            # Get trajectory for analysis
            trajectory = env.get_trajectory()
            traj_steps = trajectory.get("steps", []) if trajectory else []
            variant_meta = {}
            if env.variant:
                variant_meta = env.variant.meta

            # Detect strategy
            strategy_report = strategy_detector.detect(
                traj_steps, task, variant_meta, files_read, final_score
            )

            # Classify failure
            failure_report = failure_classifier.classify(
                episode_id=trajectory.get("episode_id", "") if trajectory else "",
                task=task,
                trajectory_steps=traj_steps,
                variant_meta=variant_meta,
                files_read=files_read,
                files_written=files_written,
                final_score=final_score,
            )

            # Advanced metrics
            adv_metrics = metrics_engine.compute(
                traj_steps, variant_meta, final_score, files_read, files_written
            )

            runs.append(AgentRunResult(
                agent_name=agent_name,
                task=task,
                variant_id=variant_id or "unknown",
                final_score=final_score,
                total_steps=len(action_sequence),
                cumulative_reward=cumulative_reward,
                duration_seconds=duration,
                action_sequence=action_sequence,
                files_read=files_read,
                files_written=files_written,
                strategy=strategy_report.strategy,
                strategy_score=strategy_report.score,
                failure_type=failure_report.primary_failure,
                reliability_index=adv_metrics.reliability_index,
                step_timeline=step_timeline,
            ))

        return ComparisonReport(
            task=task,
            variant_id=variant_id or "unknown",
            runs=runs,
        )
