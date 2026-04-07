# server/app.py
"""
FastAPI server — v4.0

Core endpoints:        POST /reset, POST /step, GET /state, GET /health
Evaluation endpoints:  GET /trajectory, GET /evaluate, GET /metrics
Control endpoints:     POST /fault-config
Intelligence (v3):     GET /classify, GET /strategy, GET /advanced-metrics,
                       POST /compare-agents, GET /improvement-plan, GET /viz-data
Research (v4 NEW):     GET /causal-probe, GET /counterfactual, GET /confidence,
                       POST /benchmark, GET /analytics
"""
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os

from .environment import CodebaseNavEnvironment
from .models import (
    RepoAction, StepResult, ResetResult, StateResult,
    TrajectoryResponse, EvaluationResponse, MetricsResponse,
    FaultConfigRequest,
)
from .failure_classifier import FailureClassifier
from .strategy_detector import StrategyDetector
from .advanced_metrics import AdvancedMetricsEngine
from .self_improvement import SelfImprovementEngine
from .multi_agent import MultiAgentComparison

# Global instances
env = CodebaseNavEnvironment()
failure_clf = FailureClassifier()
strategy_det = StrategyDetector()
adv_metrics = AdvancedMetricsEngine()
improvement = SelfImprovementEngine()
multi_agent = MultiAgentComparison()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    env.close()


app = FastAPI(
    title="Codebase Navigation & Repair — OpenEnv v3",
    description=(
        "RL environment for AI coding agents — extended with process-based evaluation, "
        "failure classification, strategy detection, self-improvement loops, "
        "multi-agent comparison, 3D visualization, and advanced metrics."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

# Serve static files (3D visualizer HTML)
_static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.exists(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


# ── Core OpenEnv Endpoints ────────────────────────────────────────────────────

@app.post("/reset", response_model=ResetResult)
async def reset(task: str = "task1"):
    valid_tasks = ["task1", "task2", "task3"]
    if task not in valid_tasks:
        raise HTTPException(status_code=400, detail=f"task must be one of {valid_tasks}")
    try:
        return env.reset(task=task)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=StepResult)
async def step(action: RepoAction):
    if env.done:
        raise HTTPException(status_code=400, detail="Episode is done. POST /reset to start.")
    try:
        return env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state", response_model=StateResult)
async def state():
    return StateResult(
        observation=env.get_state(),
        current_score=env.final_score,
        total_steps_taken=env.steps_taken,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "environment": "codebase-nav-env", "version": "3.0.0"}


# ── Evaluation Endpoints ──────────────────────────────────────────────────────

@app.get("/trajectory", response_model=TrajectoryResponse)
async def get_trajectory():
    traj = env.get_trajectory()
    if not traj:
        return TrajectoryResponse()
    return TrajectoryResponse(**traj)


@app.get("/evaluate", response_model=EvaluationResponse)
async def get_evaluation():
    evaluation = env.get_evaluation()
    if "error" in evaluation:
        return EvaluationResponse()
    return EvaluationResponse(**evaluation)


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    return MetricsResponse(**env.get_metrics())


@app.post("/fault-config")
async def set_fault_config(config: FaultConfigRequest):
    env.set_fault_config(config.level)
    return {
        "status": "ok",
        "fault_level": config.level,
        "message": f"Fault injection set to '{config.level}'. Takes effect on next /reset.",
    }


# ── Intelligence Endpoints (NEW in v3) ────────────────────────────────────────

@app.get("/classify")
async def classify_failure():
    """
    Classify the failure type of the current/latest episode.
    Returns typed failure taxonomy with root cause and remediation.
    """
    traj = env.get_trajectory()
    if not traj:
        return {"error": "No trajectory available. Run an episode first."}

    steps = traj.get("steps", [])
    meta = env.variant.meta if env.variant else {}

    report = failure_clf.classify(
        episode_id=traj.get("episode_id", ""),
        task=env.current_task or "unknown",
        trajectory_steps=steps,
        variant_meta=meta,
        files_read=list(env.files_read),
        files_written=list(env.files_written),
        final_score=env.final_score,
        security_violations=env.security_violations,
    )
    return report.to_dict()


@app.get("/strategy")
async def detect_strategy():
    """
    Detect the behavioral strategy pattern used by the agent.
    Returns: TARGETED_DEBUGGING | SYSTEMATIC_SEARCH | BRUTE_FORCE |
             RANDOM_EXPLORATION | SPEC_DRIVEN | MINIMAL_EFFORT
    """
    traj = env.get_trajectory()
    if not traj:
        return {"error": "No trajectory available."}

    steps = traj.get("steps", [])
    meta = env.variant.meta if env.variant else {}

    report = strategy_det.detect(
        trajectory_steps=steps,
        task=env.current_task or "unknown",
        variant_meta=meta,
        files_read=list(env.files_read),
        final_score=env.final_score,
    )
    return report.to_dict()


@app.get("/advanced-metrics")
async def get_advanced_metrics():
    """
    Compute advanced metrics: reasoning efficiency, decision entropy,
    exploration ratio, reliability index, consistency, pivot rate.
    """
    traj = env.get_trajectory()
    if not traj:
        return {"error": "No trajectory available."}

    steps = traj.get("steps", [])
    meta = env.variant.meta if env.variant else {}

    report = adv_metrics.compute(
        trajectory_steps=steps,
        variant_meta=meta,
        final_score=env.final_score,
        files_read=list(env.files_read),
        files_written=list(env.files_written),
    )
    return report.to_dict()


@app.get("/improvement-plan")
async def get_improvement_plan():
    """
    Generate a self-improvement plan based on failure classification.
    Returns: what_went_wrong, improved_strategy, step-by-step plan,
             system_prompt_addon (for injecting into next agent run).
    """
    traj = env.get_trajectory()
    if not traj:
        return {"error": "No trajectory available."}

    steps = traj.get("steps", [])
    meta = env.variant.meta if env.variant else {}

    # Classify first
    fail_report = failure_clf.classify(
        episode_id=traj.get("episode_id", ""),
        task=env.current_task or "unknown",
        trajectory_steps=steps,
        variant_meta=meta,
        files_read=list(env.files_read),
        files_written=list(env.files_written),
        final_score=env.final_score,
        security_violations=env.security_violations,
    )

    plan = improvement.generate_improvement_plan(
        episode_id=traj.get("episode_id", ""),
        task=env.current_task or "unknown",
        failure_type=fail_report.primary_failure,
        failure_evidence=[f.evidence for f in fail_report.failures],
        original_score=env.final_score,
        trajectory_steps=steps,
        files_read=list(env.files_read),
        files_written=list(env.files_written),
    )
    return plan.to_dict()


@app.post("/compare-agents")
async def compare_agents(task: str = "task1", agents: str = "all"):
    """
    Run multiple agent strategies on the same task and compare side-by-side.
    agents: "all" | comma-separated list of: test-first,search-first,minimal,exhaustive
    """
    valid_tasks = ["task1", "task2", "task3"]
    if task not in valid_tasks:
        raise HTTPException(status_code=400, detail=f"task must be one of {valid_tasks}")

    if agents == "all":
        agent_list = None
    else:
        agent_list = [a.strip() for a in agents.split(",")]

    try:
        report = multi_agent.compare(env, task=task, agents=agent_list)
        return report.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/viz-data")
async def get_viz_data():
    """
    Get structured 3D visualization data for the current/latest episode.
    Returns nodes (files), edges (dependencies), and step trajectory
    in the format expected by the Three.js visualizer.
    """
    traj = env.get_trajectory()
    if not traj:
        return {"error": "No trajectory available."}

    # Build file nodes
    files = []
    visited = set(env.files_read)
    modified = set(env.files_written)
    meta = env.variant.meta if env.variant else {}
    bug_files = set(meta.get("bug_files", []))

    if env.variant:
        tree = env.variant.get_tree()
        for f in tree:
            ftype = "test" if f.startswith("tests/") else \
                    "spec" if f.endswith(".md") else "src"
            files.append({
                "name": f,
                "type": ftype,
                "is_bug_file": f in bug_files,
                "visited": f in visited,
                "modified": f in modified,
            })

    # Build dependency edges from known patterns
    deps = []
    test_files = [f["name"] for f in files if f["type"] == "test"]
    src_files = [f["name"] for f in files if f["type"] == "src"]

    # Simple heuristic: connect tests to src files
    for tf in test_files:
        for sf in src_files:
            deps.append({"from": tf, "to": sf})

    # Build step data for trajectory
    steps_data = []
    for step in traj.get("steps", []):
        steps_data.append({
            "step": step.get("step_number", 0),
            "action": step.get("action_type", ""),
            "path": step.get("action_path"),
            "reward": step.get("reward", 0.0),
            "error": step.get("error"),
            "pass_rate": step.get("test_pass_rate"),
        })

    # Get strategy
    strategy_info = strategy_det.detect(
        traj.get("steps", []),
        env.current_task or "unknown",
        meta,
        list(env.files_read),
        env.final_score,
    ) if traj.get("steps") else None

    return {
        "task": env.current_task or "unknown",
        "variant_id": traj.get("variant_id", "unknown"),
        "final_score": env.final_score,
        "strategy": strategy_info.strategy if strategy_info else "UNKNOWN",
        "failure_type": "—",
        "files": files,
        "dependencies": deps,
        "steps": steps_data,
    }


# ── Research Endpoints (NEW in v4) ────────────────────────────────────────────

from .causal_probe import CausalProbe
from .counterfactual_engine import CounterfactualEngine
from .confidence_calibrator import ConfidenceCalibrator
from .benchmark_runner import BenchmarkRunner
from .analytics_engine import AnalyticsEngine

_causal = CausalProbe()
_counter = CounterfactualEngine()
_calibrator = ConfidenceCalibrator()
_benchmark = BenchmarkRunner()
_analytics = AnalyticsEngine()


@app.get("/causal-probe")
async def causal_probe():
    """
    Causal reasoning probe — did the agent understand WHY the bug exists?
    Returns: causal_score, understanding_level, chain_coverage, shortcut_detection.
    """
    traj = env.get_trajectory()
    if not traj:
        return {"error": "No trajectory available."}
    steps = traj.get("steps", [])
    meta = env.variant.meta if env.variant else {}
    report = _causal.probe(
        episode_id=traj.get("episode_id", ""),
        task=env.current_task or "unknown",
        trajectory_steps=steps,
        variant_meta=meta,
        files_read=list(env.files_read),
        files_written=list(env.files_written),
        final_score=env.final_score,
    )
    return report.to_dict()


@app.get("/counterfactual")
async def counterfactual():
    """
    Counterfactual robustness test — is the agent's strategy brittle?
    Simulates 6 mutations and measures how many the strategy survives.
    Returns: robustness_score, brittleness_level, mutations analysis.
    """
    traj = env.get_trajectory()
    if not traj:
        return {"error": "No trajectory available."}
    steps = traj.get("steps", [])
    meta = env.variant.meta if env.variant else {}
    report = _counter.analyze(
        episode_id=traj.get("episode_id", ""),
        task=env.current_task or "unknown",
        trajectory_steps=steps,
        variant_meta=meta,
        files_read=list(env.files_read),
        files_written=list(env.files_written),
        final_score=env.final_score,
    )
    return report.to_dict()


@app.get("/confidence")
async def confidence_calibration():
    """
    Confidence calibration — is the agent appropriately confident?
    Infers confidence from behavioral proxies and compares to actual performance.
    Returns: profile (WELL_CALIBRATED|OVERCONFIDENT|UNDERCONFIDENT), calibration_score.
    """
    traj = env.get_trajectory()
    if not traj:
        return {"error": "No trajectory available."}
    steps = traj.get("steps", [])
    report = _calibrator.calibrate(
        episode_id=traj.get("episode_id", ""),
        task=env.current_task or "unknown",
        trajectory_steps=steps,
        final_score=env.final_score,
    )
    return report.to_dict()


@app.post("/benchmark")
async def run_benchmark(
    tasks: str = "task1,task2",
    agents: str = "all",
    benchmark_id: str = None,
):
    """
    Automated benchmark leaderboard.
    Runs all selected agents × tasks. Returns ranked leaderboard.
    tasks: comma-separated task IDs. agents: "all" or comma-separated strategy names.
    """
    task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    valid_tasks = ["task1", "task2", "task3"]
    task_list = [t for t in task_list if t in valid_tasks]
    if not task_list:
        raise HTTPException(status_code=400, detail=f"tasks must be one of {valid_tasks}")

    agent_list = None if agents == "all" else [a.strip() for a in agents.split(",")]

    try:
        report = _benchmark.run(env, tasks=task_list, agents=agent_list, benchmark_id=benchmark_id)
        return report.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics")
async def get_analytics():
    """
    Unified research-grade analytics report.
    Synthesizes all v3+v4 evaluation dimensions into one report with:
    reasoning graph, root cause tree, alternative paths, profile tags,
    composite score, executive summary, researcher notes.
    """
    traj = env.get_trajectory()
    if not traj:
        return {"error": "No trajectory available."}
    try:
        report = _analytics.analyze(env)
        return report.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_v4():
    return {
        "status": "ok",
        "environment": "codebase-nav-env",
        "version": "4.0.0",
        "endpoints": [
            "/reset", "/step", "/state", "/health",
            "/trajectory", "/evaluate", "/metrics", "/fault-config",
            "/classify", "/strategy", "/advanced-metrics",
            "/improvement-plan", "/compare-agents", "/viz-data",
            "/causal-probe", "/counterfactual", "/confidence",
            "/benchmark", "/analytics",
        ],
    }

def main():
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=True)

if __name__ == "__main__":
    main()
