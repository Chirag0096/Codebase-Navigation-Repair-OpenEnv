# server/app.py
"""
FastAPI server exposing the OpenEnv-compliant API + reliability layer endpoints.

Core endpoints:      POST /reset, POST /step, GET /state, GET /health
Evaluation endpoints: GET /trajectory, GET /evaluate, GET /metrics
Control endpoints:    POST /fault-config
"""
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from .environment import CodebaseNavEnvironment
from .models import (
    RepoAction, StepResult, ResetResult, StateResult,
    TrajectoryResponse, EvaluationResponse, MetricsResponse,
    FaultConfigRequest,
)

# Global environment instance (one session per container)
env = CodebaseNavEnvironment()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    env.close()


app = FastAPI(
    title="Codebase Navigation & Repair — OpenEnv",
    description=(
        "RL environment where agents navigate and repair Python codebases. "
        "Extended with process-based evaluation, trajectory replay, "
        "fault injection, security scanning, and memory tracking."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ── Core OpenEnv Endpoints ───────────────────────────────────────────────────

@app.post("/reset", response_model=ResetResult)
async def reset(task: str = "task1"):
    """
    Start a new episode.
    task: "task1" | "task2" | "task3"
    """
    valid_tasks = ["task1", "task2", "task3"]
    if task not in valid_tasks:
        raise HTTPException(status_code=400, detail=f"task must be one of {valid_tasks}")
    try:
        result = env.reset(task=task)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=StepResult)
async def step(action: RepoAction):
    """
    Take one action in the current episode.
    """
    if env.done:
        raise HTTPException(status_code=400, detail="Episode is done. POST /reset to start a new one.")
    try:
        result = env.step(action)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state", response_model=StateResult)
async def state():
    """
    Get current state without advancing the episode.
    """
    obs = env.get_state()
    return StateResult(
        observation=obs,
        current_score=env.final_score,
        total_steps_taken=env.steps_taken,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "environment": "codebase-nav-env", "version": "2.0.0"}


# ── Evaluation & Reliability Endpoints ───────────────────────────────────────

@app.get("/trajectory", response_model=TrajectoryResponse)
async def get_trajectory():
    """
    Get the full trajectory of the current or most recent episode.
    Returns every action, observation snapshot, reward, timing, and security flags.
    """
    traj = env.get_trajectory()
    if not traj:
        return TrajectoryResponse()
    return TrajectoryResponse(**traj)


@app.get("/evaluate", response_model=EvaluationResponse)
async def get_evaluation():
    """
    Get multi-dimensional evaluation of the current/latest episode.
    Scores across 6 dimensions: efficiency, navigation, correctness,
    reasoning, robustness, security.
    """
    evaluation = env.get_evaluation()
    if "error" in evaluation:
        return EvaluationResponse()
    return EvaluationResponse(**evaluation)


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get comprehensive metrics including memory usage, security stats,
    fault injection report, wasteful patterns, and action timeline.
    """
    metrics = env.get_metrics()
    return MetricsResponse(**metrics)


@app.post("/fault-config")
async def set_fault_config(config: FaultConfigRequest):
    """
    Configure fault injection for the NEXT episode (takes effect on next /reset).
    Levels: "none" (default), "light" (misleading comments), "heavy" (all faults)
    """
    env.set_fault_config(config.level)
    return {
        "status": "ok",
        "fault_level": config.level,
        "message": f"Fault injection set to '{config.level}'. Takes effect on next /reset.",
    }
