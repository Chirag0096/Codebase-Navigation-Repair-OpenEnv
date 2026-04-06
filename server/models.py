# server/models.py
"""
Pydantic models for the OpenEnv API — extended with evaluation & reliability layer.
"""
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field


# ── Core Action/Observation Models ──────────────────────────────────────────

class RepoAction(BaseModel):
    """All actions the agent can take in one step."""
    action_type: Literal[
        "read_file",      # Read a file's contents. Costs 1 step.
        "write_file",     # Write/modify a file. Costs 1 step.
        "run_tests",      # Run pytest on a specific test file. Costs 2 steps.
        "search_code",    # Search for a string across all files. Costs 1 step.
        "submit"          # Finalise submission and trigger full grader. Terminal action.
    ]
    path: Optional[str] = None          # For read_file, write_file, run_tests
    content: Optional[str] = None       # For write_file — the new file content
    query: Optional[str] = None         # For search_code


class RepoObservation(BaseModel):
    """What the agent sees after each step."""
    repo_tree: List[str]                     # All file paths in the repo
    task_description: str                    # Natural language description of the task
    failing_tests: List[str]                 # Test names that are currently failing
    files_read: List[str]                    # Files the agent has read so far
    last_action_result: Optional[str]        # Output of the last action
    steps_remaining: int
    current_task: str                        # "task1", "task2", or "task3"
    last_action_error: Optional[str] = None  # If the last action failed, why


class RepoReward(BaseModel):
    """Reward signal after each step."""
    value: float = Field(ge=-1.0, le=1.0)
    reason: str


# ── API Response Models ─────────────────────────────────────────────────────

class StepResult(BaseModel):
    """Complete result returned by /step endpoint."""
    observation: RepoObservation
    reward: float
    done: bool
    info: Dict[str, Any] = {}


class ResetResult(BaseModel):
    """Result returned by /reset endpoint."""
    observation: RepoObservation
    info: Dict[str, Any] = {}


class StateResult(BaseModel):
    """Result returned by /state endpoint."""
    observation: RepoObservation
    current_score: float
    total_steps_taken: int


# ── Evaluation & Reliability Models ─────────────────────────────────────────

class TrajectoryResponse(BaseModel):
    """Full trajectory of the current/latest episode."""
    episode_id: Optional[str] = None
    task: Optional[str] = None
    variant_id: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None
    steps: List[Dict[str, Any]] = []
    final_score: float = 0.0
    total_steps: int = 0
    metadata: Dict[str, Any] = {}


class EvaluationResponse(BaseModel):
    """Multi-dimensional evaluation of agent performance."""
    episode_id: Optional[str] = None
    task: Optional[str] = None
    composite_score: float = 0.0
    dimensions: Dict[str, Any] = {}
    failure_analysis: List[str] = []
    strengths: List[str] = []
    recommendations: List[str] = []


class MetricsResponse(BaseModel):
    """Comprehensive metrics for the current/latest episode."""
    episode_id: Optional[str] = None

    # Core metrics
    success_rate: float = 0.0
    step_efficiency: float = 0.0
    navigation_score: float = 0.0
    context_efficiency: float = 0.0
    reasoning_quality: float = 0.0
    robustness_score: float = 0.0
    security_score: float = 0.0

    # Memory stats
    memory: Dict[str, Any] = {}

    # Security stats
    security: Dict[str, Any] = {}

    # Fault injection report
    fault_injection: Dict[str, Any] = {}

    # Wasteful patterns detected
    wasteful_patterns: List[str] = []

    # Timeline of actions
    timeline: List[Dict[str, Any]] = []


class FaultConfigRequest(BaseModel):
    """Request body for configuring fault injection."""
    level: Literal["none", "light", "heavy"] = "none"


class ReplayRequest(BaseModel):
    """Request body for replaying an episode."""
    task: str
    variant_id: Optional[str] = None  # If None, uses the variant from trajectory
    actions: List[Dict[str, Any]] = []
