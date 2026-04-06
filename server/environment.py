# server/environment.py
"""
Core RL environment — extended with reliability and evaluation layers.

Integrates:
- Trajectory logging (full action/state recording)
- Process-based evaluation (multi-dimensional scoring)
- Fault injection (robustness testing)
- Security scanning (unsafe action detection)
- Memory tracking (context efficiency)
"""
import os
import time
from typing import Optional, Tuple, Dict, Any

from .models import RepoAction, RepoObservation, StepResult, ResetResult
from .repo_loader import RepoVariant, load_random_variant, get_task_description
from .grader import compute_final_score
from .sandbox import (
    run_pytest_sandboxed, validate_file_path,
    search_in_repo, EXECUTION_TIMEOUT
)
from .trajectory import TrajectoryLogger, FileDiff
from .evaluator import ProcessEvaluator
from .fault_injection import FaultInjector, FaultConfig
from .security import SecurityScanner
from .memory import MemoryTracker

MAX_STEPS = {
    "task1": 20,
    "task2": 25,
    "task3": 30,
}

# Reward constants
REWARD_USEFUL_READ = 0.05        # Reading a file that contains the bug/solution
REWARD_TEST_IMPROVEMENT = 0.10   # run_tests shows more passing than before
REWARD_WRITE_RELEVANT = 0.08     # Writing a relevant file
PENALTY_WASTED_READ = -0.01      # Reading the same file twice
PENALTY_WRONG_ACTION = -0.02     # Invalid path or action
PENALTY_PER_EXTRA_STEP = -0.02   # Steps beyond optimal
PENALTY_SECURITY_VIOLATION = -0.05  # Unsafe code detected


class CodebaseNavEnvironment:
    """
    The core RL environment class — extended with evaluation & reliability layers.
    One instance per active session.
    """

    def __init__(self):
        # Core state
        self.variant: Optional[RepoVariant] = None
        self.current_task: Optional[str] = None
        self.steps_taken: int = 0
        self.max_steps: int = 20
        self.done: bool = True
        self.files_read: list = []
        self.files_written: list = []
        self.last_action_result: Optional[str] = None
        self.last_action_error: Optional[str] = None
        self.cumulative_reward: float = 0.0
        self.last_test_pass_rate: float = 0.0
        self.final_score: float = 0.0

        # ── Reliability & Evaluation Layers ──────────────────────────────
        self.trajectory = TrajectoryLogger()
        self.evaluator = ProcessEvaluator()
        self.fault_injector = FaultInjector(FaultConfig.none())
        self.security = SecurityScanner(strict_mode=False)  # Log but don't block
        self.memory = MemoryTracker()

        # Fault injection state
        self.fault_config = FaultConfig.none()
        self.fault_report = None
        self.security_violations = 0

    def set_fault_config(self, level: str):
        """Set fault injection level: 'none', 'light', 'heavy'."""
        if level == "light":
            self.fault_config = FaultConfig.light()
        elif level == "heavy":
            self.fault_config = FaultConfig.heavy()
        else:
            self.fault_config = FaultConfig.none()
        self.fault_injector = FaultInjector(self.fault_config)

    def reset(self, task: str = "task1") -> ResetResult:
        """Start a new episode. Load a random repo variant."""
        # Cleanup previous episode
        if self.variant:
            self.variant.cleanup()

        self.current_task = task
        self.variant = load_random_variant(task)
        self.steps_taken = 0
        self.max_steps = MAX_STEPS.get(task, 20)
        self.done = False
        self.files_read = []
        self.files_written = []
        self.last_action_result = None
        self.last_action_error = None
        self.cumulative_reward = 0.0
        self.final_score = 0.0
        self.security_violations = 0

        # ── Start trajectory recording ───────────────────────────────────
        self.trajectory.start_episode(task, self.variant.variant_id)

        # ── Apply fault injection ────────────────────────────────────────
        self.fault_report = self.fault_injector.inject(
            self.variant.working_dir, self.variant.meta
        )

        # ── Initialize memory tracker ────────────────────────────────────
        relevant_files = (
            self.variant.meta.get("bug_files", []) +
            self.variant.meta.get("interface_files", []) +
            self.variant.meta.get("read_first_files", []) +
            self.variant.meta.get("files_to_implement", [])
        )
        self.memory.start_episode(relevant_files)

        # Run initial test to establish baseline
        initial_pass_rate, _, _ = run_pytest_sandboxed(self.variant.working_dir)
        self.last_test_pass_rate = initial_pass_rate

        obs = self._build_observation()

        info = {
            "variant_id": self.variant.variant_id,
            "fault_injection": self.fault_report.to_dict() if self.fault_report else {},
        }

        return ResetResult(observation=obs, info=info)

    def step(self, action: RepoAction) -> StepResult:
        """Process one agent action. Return next observation, reward, done."""
        if self.done:
            raise RuntimeError("Episode is done. Call reset() first.")

        step_start = time.time()
        self.steps_taken += 1
        self.last_action_error = None
        reward = 0.0
        file_diff = None
        test_pass_rate = None
        security_flags = []

        # Route action to handler
        if action.action_type == "read_file":
            reward = self._handle_read_file(action)
        elif action.action_type == "write_file":
            reward, file_diff, security_flags = self._handle_write_file_extended(action)
        elif action.action_type == "run_tests":
            reward, test_pass_rate = self._handle_run_tests_extended(action)
        elif action.action_type == "search_code":
            reward = self._handle_search_code(action)
            self.memory.record_search()
        elif action.action_type == "submit":
            reward, score = self._handle_submit()
            self.final_score = score
            self.done = True

        # Apply efficiency penalty for steps beyond optimal
        optimal = self.variant.meta.get("optimal_steps", 10)
        if self.steps_taken > optimal:
            reward += PENALTY_PER_EXTRA_STEP

        # Check step budget
        if self.steps_taken >= self.max_steps and not self.done:
            self.done = True
            pass_rate, _, _ = run_pytest_sandboxed(self.variant.working_dir)
            self.final_score = pass_rate
            self.last_action_result = (
                f"[STEP BUDGET EXHAUSTED] Auto-grading... final score: {pass_rate:.2f}"
            )

        reward = max(-1.0, min(1.0, reward))
        self.cumulative_reward += reward

        step_duration = (time.time() - step_start) * 1000  # ms

        # ── Record trajectory step ───────────────────────────────────────
        obs_compact = {
            "files_read": list(self.files_read),
            "files_written": list(self.files_written),
            "steps_remaining": self.max_steps - self.steps_taken,
            "has_error": self.last_action_error is not None,
        }
        self.trajectory.record_step(
            step_number=self.steps_taken,
            action_type=action.action_type,
            action_path=action.path,
            action_query=action.query,
            action_content_length=len(action.content) if action.content else None,
            reward=reward,
            cumulative_reward=self.cumulative_reward,
            done=self.done,
            error=self.last_action_error,
            file_diff=file_diff,
            test_pass_rate=test_pass_rate,
            duration_ms=round(step_duration, 1),
            observation_compact=obs_compact,
            security_flags=security_flags,
        )

        # ── Finalize trajectory on episode end ───────────────────────────
        if self.done:
            self.trajectory.end_episode(self.final_score)

        obs = self._build_observation()
        return StepResult(
            observation=obs,
            reward=round(reward, 3),
            done=self.done,
            info={
                "cumulative_reward": round(self.cumulative_reward, 3),
                "final_score": self.final_score,
                "steps_taken": self.steps_taken,
                "security_flags": security_flags,
            }
        )

    def get_state(self) -> RepoObservation:
        """Return current state without advancing the episode."""
        return self._build_observation()

    # ── Extended action handlers ─────────────────────────────────────────────

    def _handle_read_file(self, action: RepoAction) -> float:
        if not action.path:
            self.last_action_error = "read_file requires 'path'"
            return PENALTY_WRONG_ACTION

        if not validate_file_path(action.path, self.variant.working_dir):
            self.last_action_error = f"Invalid path: {action.path}"
            return PENALTY_WRONG_ACTION

        full_path = os.path.join(self.variant.working_dir, action.path)
        if not os.path.exists(full_path):
            self.last_action_error = f"File not found: {action.path}"
            return PENALTY_WRONG_ACTION

        # Penalty for reading same file twice
        if action.path in self.files_read:
            try:
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                self.last_action_result = content[:5000]
                self.memory.record_read(action.path, len(content), self.steps_taken)
            except Exception as e:
                self.last_action_result = f"Error reading file: {e}"
            return PENALTY_WASTED_READ

        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            self.files_read.append(action.path)
            self.last_action_result = content[:5000]

            # Track memory
            self.memory.record_read(action.path, len(content), self.steps_taken)

            # Scan for prompt injection in file content
            injection_scan = self.security.scan_file_for_injection(content, action.path)
            if not injection_scan.is_safe:
                # Don't block — just flag it in the observation
                self.last_action_result += (
                    f"\n\n⚠️ SECURITY NOTE: Potential prompt injection detected in this file. "
                    f"Flags: {injection_scan.flags}"
                )

            # Small reward if this file is relevant
            bug_files = self.variant.meta.get("bug_files", []) + \
                        self.variant.meta.get("interface_files", []) + \
                        self.variant.meta.get("read_first_files", [])
            if action.path in bug_files:
                return REWARD_USEFUL_READ
            return 0.0

        except Exception as e:
            self.last_action_error = f"Could not read file: {e}"
            return PENALTY_WRONG_ACTION

    def _handle_write_file_extended(self, action: RepoAction) -> Tuple[float, Optional[FileDiff], list]:
        """Extended write handler with security scanning and diff tracking."""
        security_flags = []

        if not action.path or action.content is None:
            self.last_action_error = "write_file requires 'path' and 'content'"
            return PENALTY_WRONG_ACTION, None, []

        if not validate_file_path(action.path, self.variant.working_dir):
            self.last_action_error = f"Invalid path (cannot write outside repo): {action.path}"
            return PENALTY_WRONG_ACTION, None, []

        # Prevent modifying test files (especially for task3)
        if "tests/" in action.path and self.current_task == "task3":
            self.last_action_error = "Cannot modify test files in task3"
            return PENALTY_WRONG_ACTION, None, []

        # ── Security scan ────────────────────────────────────────────────
        scan_result = self.security.scan_write_content(action.content, action.path)
        security_flags = scan_result.flags
        reward_modifier = 0.0

        if security_flags:
            self.security_violations += len(scan_result.blocked_patterns)
            reward_modifier = PENALTY_SECURITY_VIOLATION * len(scan_result.blocked_patterns)
            self.last_action_error = (
                f"Security flags: {'; '.join(security_flags[:3])}"
            )
            # Don't block in non-strict mode, but penalize

        full_path = os.path.join(self.variant.working_dir, action.path)

        # Read existing content for diff
        before_content = None
        if os.path.exists(full_path):
            try:
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    before_content = f.read()
            except Exception:
                pass

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(action.content)

            self.files_written.append(action.path)
            self.memory.record_write(len(action.content))

            self.last_action_result = f"Successfully wrote {len(action.content)} chars to {action.path}"
            if security_flags:
                self.last_action_result += f" ⚠️ Security flags: {security_flags}"

            # Create diff record
            diff = FileDiff(
                path=action.path,
                before=before_content,
                after=action.content,
                chars_changed=abs(len(action.content) - len(before_content or "")),
            )

            # Small reward if writing a relevant file
            fix_files = self.variant.meta.get("bug_files", []) + \
                       self.variant.meta.get("files_to_implement", [])
            base_reward = REWARD_WRITE_RELEVANT if action.path in fix_files else 0.0

            return base_reward + reward_modifier, diff, security_flags

        except Exception as e:
            self.last_action_error = f"Could not write file: {e}"
            return PENALTY_WRONG_ACTION, None, security_flags

    def _handle_run_tests_extended(self, action: RepoAction) -> Tuple[float, Optional[float]]:
        """Extended test handler that returns pass rate for trajectory."""
        test_file = action.path

        pass_rate, output, timed_out = run_pytest_sandboxed(
            self.variant.working_dir, test_file
        )

        if timed_out:
            self.last_action_result = output
            self.last_action_error = "Tests timed out"
            return PENALTY_WRONG_ACTION, 0.0

        self.last_action_result = output[:3000]

        # Reward improvement in pass rate
        improvement = pass_rate - self.last_test_pass_rate
        self.last_test_pass_rate = pass_rate

        if improvement > 0:
            reward = REWARD_TEST_IMPROVEMENT + (improvement * 0.3)
        elif improvement < 0:
            reward = improvement * 0.2
        else:
            reward = 0.0

        return reward, pass_rate

    def _handle_search_code(self, action: RepoAction) -> float:
        if not action.query:
            self.last_action_error = "search_code requires 'query'"
            return PENALTY_WRONG_ACTION

        results = search_in_repo(action.query, self.variant.working_dir)
        self.last_action_result = results
        return 0.0

    def _handle_submit(self) -> Tuple[float, float]:
        """Final grader. Run full test suite and compute score."""
        pass_rate, output, timed_out = run_pytest_sandboxed(self.variant.working_dir)

        score = pass_rate

        # Task 2 bonus: check if agent wrote a regression test
        if self.current_task == "task2":
            bonus = self._check_regression_test()
            score = min(1.0, score + bonus)

        self.last_action_result = (
            f"=== FINAL GRADER RESULTS ===\n"
            f"pytest pass rate: {pass_rate:.2f}\n"
            f"final score: {score:.3f}\n\n"
            f"{output[:2000]}"
        )

        reward = score
        return reward, score

    def _check_regression_test(self) -> float:
        new_tests = [f for f in self.files_written if "test_" in f]
        if not new_tests:
            return 0.0
        return 0.15

    # ── Evaluation & Metrics ─────────────────────────────────────────────────

    def get_trajectory(self) -> Optional[dict]:
        """Get full trajectory of current/latest episode."""
        return self.trajectory.get_trajectory()

    def get_evaluation(self) -> dict:
        """Run multi-dimensional evaluation on current/latest episode."""
        trajectory = self.trajectory.get_trajectory()
        if not trajectory:
            return {"error": "No trajectory available"}

        steps_data = []
        for step in trajectory.get("steps", []):
            steps_data.append({
                "step_number": step.get("step_number"),
                "action_type": step.get("action_type"),
                "action_path": step.get("action_path"),
                "reward": step.get("reward"),
                "error": step.get("error"),
                "test_pass_rate": step.get("test_pass_rate"),
                "security_flags": step.get("security_flags", []),
            })

        report = self.evaluator.evaluate(
            episode_id=trajectory.get("episode_id", "unknown"),
            task=self.current_task or "unknown",
            trajectory_steps=steps_data,
            variant_meta=self.variant.meta if self.variant else {},
            final_score=self.final_score,
            files_read=list(self.files_read),
            files_written=list(self.files_written),
            total_steps=self.steps_taken,
            security_violations=self.security_violations,
            fault_injection_active=self.fault_config.enabled,
        )

        return report.to_dict()

    def get_metrics(self) -> dict:
        """Get comprehensive metrics for the current/latest episode."""
        trajectory = self.trajectory.get_trajectory()
        evaluation = self.get_evaluation()
        memory_stats = self.memory.get_stats()
        security_stats = self.security.get_stats()
        wasteful = self.memory.get_wasteful_patterns()
        timeline = self.trajectory.get_step_timeline()

        dimensions = evaluation.get("dimensions", {})

        return {
            "episode_id": trajectory.get("episode_id") if trajectory else None,

            # Core metrics from evaluation dimensions
            "success_rate": self.final_score,
            "step_efficiency": dimensions.get("efficiency", {}).get("score", 0.0),
            "navigation_score": dimensions.get("navigation", {}).get("score", 0.0),
            "context_efficiency": memory_stats.context_efficiency,
            "reasoning_quality": dimensions.get("reasoning", {}).get("score", 0.0),
            "robustness_score": dimensions.get("robustness", {}).get("score", 0.0),
            "security_score": dimensions.get("security", {}).get("score", 0.0),

            # Detailed breakdowns
            "memory": memory_stats.to_dict(),
            "security": security_stats,
            "fault_injection": self.fault_report.to_dict() if self.fault_report else {},
            "wasteful_patterns": wasteful,
            "timeline": timeline,
        }

    def _build_observation(self) -> RepoObservation:
        if not self.variant:
            return RepoObservation(
                repo_tree=[],
                task_description="No active episode. Call reset() first.",
                failing_tests=[],
                files_read=[],
                last_action_result=None,
                steps_remaining=0,
                current_task="none",
            )

        return RepoObservation(
            repo_tree=self.variant.get_tree(),
            task_description=get_task_description(self.current_task, self.variant.meta),
            failing_tests=self.variant.get_failing_tests(),
            files_read=list(self.files_read),
            last_action_result=self.last_action_result,
            steps_remaining=self.max_steps - self.steps_taken,
            current_task=self.current_task,
            last_action_error=self.last_action_error,
        )

    def close(self):
        """Cleanup temp directories."""
        if self.variant:
            self.variant.cleanup()
