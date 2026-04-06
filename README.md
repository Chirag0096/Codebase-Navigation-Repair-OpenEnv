---
title: Codebase Navigation Repair OpenEnv
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
license: mit
tags:
  - openenv
  - reinforcement-learning
  - coding-agent
---

# Codebase Navigation & Repair — OpenEnv Environment v2.0

**An RL environment + evaluation layer that makes AI coding agents reliable, testable, and debuggable.**

AI agents navigate unfamiliar Python codebases, identify bugs, and implement features — graded by running actual tests. Unlike existing benchmarks, this system provides **process-level evaluation**, not just final output scoring.

## Why This Exists

Every coding agent (Devin, Cursor, Copilot, Codex) fails ~25%+ on complex tasks. Current benchmarks tell you the agent scored 0.4 but not **why** it failed. This environment answers:

- Did the agent explore strategically or waste steps?
- Did it verify its fixes before submitting?
- Can it resist misleading comments and prompt injection?
- How efficiently does it use its context window?

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    FastAPI Server                         │
│  /reset  /step  /state  /trajectory  /evaluate  /metrics │
└──────────┬───────────────────────────────────────────────┘
           │
┌──────────▼───────────────────────────────────────────────┐
│              CodebaseNavEnvironment (extended)             │
│                                                           │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ Trajectory   │  │  Evaluator   │  │  Security       │  │
│  │ Logger       │  │  (process)   │  │  Scanner        │  │
│  └─────────────┘  └──────────────┘  └─────────────────┘  │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ Fault       │  │  Memory      │  │  Grader         │  │
│  │ Injector    │  │  Tracker     │  │  (pytest)       │  │
│  └─────────────┘  └──────────────┘  └─────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

## Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| task1 | Easy | Single-file bug repair (5 variants) |
| task2 | Medium | Cross-module interface bug + regression test (5 variants) |
| task3 | Hard | Feature implementation from spec (5 variants) |

## API Endpoints

### Core (OpenEnv-compliant)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset?task=task1` | POST | Start new episode |
| `/step` | POST | Take one action |
| `/state` | GET | Get current state |
| `/health` | GET | Health check |

### Evaluation Layer (v2.0)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/trajectory` | GET | Full action log with timing, diffs, security flags |
| `/evaluate` | GET | Multi-dimensional scores (6 axes) |
| `/metrics` | GET | Comprehensive stats: memory, security, timeline |
| `/fault-config` | POST | Enable fault injection: "none", "light", "heavy" |

## Multi-Dimensional Evaluation

The `/evaluate` endpoint scores agents across **6 quality dimensions**:

| Dimension | Weight | What It Measures |
|-----------|--------|-----------------|
| Efficiency | 20% | Steps used vs optimal path |
| Navigation | 15% | Read relevant files first? Explored strategically? |
| Correctness | 30% | Final test pass rate + regression detection |
| Reasoning | 15% | read→write→test pattern adherence |
| Robustness | 10% | Error recovery + fault injection handling |
| Security | 10% | Unsafe code detection + prompt injection resistance |

## Fault Injection

Test agent robustness by injecting controlled faults:

```bash
# Enable heavy fault injection
curl -X POST http://localhost:7860/fault-config -d '{"level":"heavy"}'

# Next reset will inject:
# - Misleading "BUG:" comments on correct lines
# - Red herring files that look buggy but aren't
# - Noisy docstrings claiming code is correct
```

## Quick Start

### Local
```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Docker
```bash
docker build -t codebase-nav-env .
docker run -p 7860:7860 codebase-nav-env
```

### Run Inference
```bash
export HF_TOKEN=your_token
export ENV_BASE_URL=http://localhost:7860
python inference.py
```

## Example Output: `/evaluate`
```json
{
  "composite_score": 0.874,
  "dimensions": {
    "efficiency": {"score": 0.8, "evidence": ["Used 5 steps vs 4 optimal"]},
    "navigation": {"score": 1.0, "evidence": ["Good: first read was relevant file"]},
    "correctness": {"score": 0.714, "evidence": ["No test regressions"]},
    "reasoning": {"score": 1.0, "evidence": ["Agent tested after writing"]},
    "robustness": {"score": 1.0, "evidence": ["Clean execution"]},
    "security": {"score": 1.0, "evidence": ["No security violations"]}
  }
}
```

## License

MIT
