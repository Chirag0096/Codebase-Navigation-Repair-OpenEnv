# server/trajectory.py
"""
Full trajectory recording and deterministic replay system.

Records every action, observation, reward, file diff, and timing.
Enables post-hoc analysis and deterministic replay of agent episodes.
"""
import time
import copy
import hashlib
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict


@dataclass
class FileDiff:
    """Represents a file change made by the agent."""
    path: str
    before: Optional[str]  # None if file was created
    after: str
    chars_changed: int


@dataclass
class TrajectoryStep:
    """Complete record of one agent step."""
    step_number: int
    timestamp: float
    action_type: str
    action_path: Optional[str]
    action_query: Optional[str]
    action_content_length: Optional[int]  # Don't store full content — too large
    observation_snapshot: Dict[str, Any]   # Compact snapshot
    reward: float
    cumulative_reward: float
    done: bool
    error: Optional[str]
    file_diff: Optional[Dict[str, Any]]   # If write_file, the diff
    test_pass_rate: Optional[float]        # If run_tests, the pass rate
    duration_ms: float                     # How long this step took server-side
    security_flags: List[str] = field(default_factory=list)


@dataclass
class TrajectoryRecord:
    """Complete episode trajectory — everything needed for replay + analysis."""
    episode_id: str
    task: str
    variant_id: str
    start_time: float
    end_time: Optional[float] = None
    steps: List[TrajectoryStep] = field(default_factory=list)
    final_score: float = 0.0
    total_steps: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "episode_id": self.episode_id,
            "task": self.task,
            "variant_id": self.variant_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": round(self.end_time - self.start_time, 2) if self.end_time else None,
            "steps": [asdict(s) for s in self.steps],
            "final_score": self.final_score,
            "total_steps": self.total_steps,
            "metadata": self.metadata,
        }


class TrajectoryLogger:
    """
    Records full agent trajectories for analysis and replay.

    Usage:
        logger = TrajectoryLogger()
        logger.start_episode("task1", "variant_3")
        logger.record_step(step_number=1, action=..., obs=..., ...)
        logger.end_episode(final_score=0.75)
        trajectory = logger.get_trajectory()
    """

    def __init__(self):
        self._current: Optional[TrajectoryRecord] = None
        self._history: List[TrajectoryRecord] = []  # Last N episodes
        self._max_history = 10

    def start_episode(self, task: str, variant_id: str) -> str:
        """Start recording a new episode. Returns episode_id."""
        # Finalize previous episode if still active
        if self._current and self._current.end_time is None:
            self._current.end_time = time.time()
            self._history.append(self._current)

        episode_id = hashlib.md5(
            f"{task}_{variant_id}_{time.time()}".encode()
        ).hexdigest()[:12]

        self._current = TrajectoryRecord(
            episode_id=episode_id,
            task=task,
            variant_id=variant_id,
            start_time=time.time(),
        )
        return episode_id

    def record_step(
        self,
        step_number: int,
        action_type: str,
        action_path: Optional[str],
        action_query: Optional[str],
        action_content_length: Optional[int],
        reward: float,
        cumulative_reward: float,
        done: bool,
        error: Optional[str],
        file_diff: Optional[FileDiff],
        test_pass_rate: Optional[float],
        duration_ms: float,
        observation_compact: Dict[str, Any],
        security_flags: List[str] = None,
    ):
        """Record one step in the current trajectory."""
        if not self._current:
            return

        step = TrajectoryStep(
            step_number=step_number,
            timestamp=time.time(),
            action_type=action_type,
            action_path=action_path,
            action_query=action_query,
            action_content_length=action_content_length,
            observation_snapshot=observation_compact,
            reward=reward,
            cumulative_reward=cumulative_reward,
            done=done,
            error=error,
            file_diff=asdict(file_diff) if file_diff else None,
            test_pass_rate=test_pass_rate,
            duration_ms=duration_ms,
            security_flags=security_flags or [],
        )
        self._current.steps.append(step)
        self._current.total_steps = step_number

    def end_episode(self, final_score: float):
        """Finalize the current episode."""
        if not self._current:
            return

        self._current.end_time = time.time()
        self._current.final_score = final_score

        # Maintain history buffer
        self._history.append(self._current)
        if len(self._history) > self._max_history:
            self._history.pop(0)

    def get_trajectory(self) -> Optional[dict]:
        """Get the current/latest trajectory as dict."""
        if self._current:
            return self._current.to_dict()
        if self._history:
            return self._history[-1].to_dict()
        return None

    def get_replay_actions(self) -> List[dict]:
        """Extract action sequence for deterministic replay."""
        if not self._current and not self._history:
            return []

        record = self._current or self._history[-1]
        actions = []
        for step in record.steps:
            action = {"action_type": step.action_type}
            if step.action_path:
                action["path"] = step.action_path
            if step.action_query:
                action["query"] = step.action_query
            # Note: content not stored in trajectory — replay requires re-supplying it
            actions.append(action)
        return actions

    def get_step_timeline(self) -> List[dict]:
        """Get compact timeline of actions and outcomes for visualization."""
        if not self._current:
            return []

        timeline = []
        for step in self._current.steps:
            timeline.append({
                "step": step.step_number,
                "action": step.action_type,
                "path": step.action_path,
                "reward": step.reward,
                "error": step.error,
                "duration_ms": step.duration_ms,
                "pass_rate": step.test_pass_rate,
                "security_flags": step.security_flags,
            })
        return timeline

    def get_history_summary(self) -> List[dict]:
        """Get summary of recent episodes."""
        summaries = []
        for record in self._history:
            summaries.append({
                "episode_id": record.episode_id,
                "task": record.task,
                "variant_id": record.variant_id,
                "final_score": record.final_score,
                "total_steps": record.total_steps,
                "duration_seconds": round(
                    record.end_time - record.start_time, 2
                ) if record.end_time else None,
            })
        return summaries
