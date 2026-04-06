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

# 🔍 Codebase Navigation & Repair — OpenEnv

> **The system that makes AI coding agents reliable, testable, and debuggable.**

## The Problem

AI coding agents (Copilot, Devin, Cursor) fail ~25%+ on complex tasks. Current benchmarks tell you the score but not **why** the agent failed. Was it poor navigation? Wasted steps? Hallucinated code? There is no way to know.

## Our Solution

An RL environment where agents navigate unfamiliar Python repos, find bugs, and fix them — graded by **actual pytest execution** with **process-level evaluation**.

Unlike existing benchmarks, we evaluate **how** the agent works, not just the final output:

| What We Measure | Why It Matters |
|----------------|---------------|
| Navigation efficiency | Did it read relevant files first? |
| Reasoning patterns | Did it follow read→write→test? |
| Context usage | How much of what it read was useful? |
| Security | Did it write safe code? |
| Robustness | Can it handle misleading comments? |

## How It Works

```
Agent resets environment → sees repo file tree (NOT contents)
  → reads files one at a time (costs steps)
  → identifies bugs in source code
  → writes fixed code
  → runs tests to verify
  → submits for final grade
```

### Tasks

| Task | Difficulty | Description | Variants |
|------|-----------|-------------|----------|
| task1 | Easy | Single-file bug repair | 5 |
| task2 | Medium | Cross-module interface bug + regression test | 5 |
| task3 | Hard | Feature implementation from spec | 5 |

Each variant has structurally different code, so the agent can't memorize solutions.

## Quick Start

### 1. Run Locally (No Docker)
```bash
pip install -r requirements.txt
python app.py                    # Gradio UI at http://localhost:7860
```

### 2. Run Agent (No LLM needed)
```bash
python run_agent.py              # deterministic agent demo
python run_agent.py --all-tasks  # run all 3 tasks
```

### 3. Run Agent with LLM
```bash
export HF_TOKEN=hf_xxxxx
python run_agent.py --llm --task task1
```

### 4. Docker
```bash
docker build -t codebase-nav-env .
docker run -p 7860:7860 codebase-nav-env
```

### 5. API Usage
```bash
# Reset
curl -X POST "http://localhost:7860/reset?task=task1"

# Take action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"read_file","path":"src/auth.py"}'

# Submit
curl -X POST http://localhost:7860/step \
  -d '{"action_type":"submit"}'

# Get evaluation
curl http://localhost:7860/evaluate
```

## API Endpoints

### Core (OpenEnv-compliant)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset?task=task1` | POST | Start new episode |
| `/step` | POST | Take one action |
| `/state` | GET | Get current state |
| `/health` | GET | Health check |

### Evaluation Layer
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/trajectory` | GET | Full action log with timing and diffs |
| `/evaluate` | GET | Multi-dimensional scores (6 axes) |
| `/metrics` | GET | Memory, security, timeline stats |
| `/fault-config` | POST | Enable fault injection |

## Evaluation Dimensions

```
efficiency   [████████████████░░░░] 0.800  — 5 steps vs 4 optimal
navigation   [████████████████████] 1.000  — read relevant files first
correctness  [██████████████░░░░░░] 0.714  — 71.4% tests passing
reasoning    [████████████████████] 1.000  — correct read→write→test pattern
robustness   [████████████████████] 1.000  — no errors encountered
security     [████████████████████] 1.000  — no unsafe code detected
```

## Project Structure

```
codebase-nav-env/
├── app.py                  # Gradio UI + FastAPI (HF Space entry point)
├── run_agent.py            # Standalone HF agent (deterministic + LLM)
├── inference.py            # OpenEnv inference script ([START]/[STEP]/[END])
├── server/
│   ├── app.py              # FastAPI endpoints
│   ├── environment.py      # Core RL environment
│   ├── models.py           # Pydantic models
│   ├── grader.py           # pytest runner
│   ├── repo_loader.py      # Template loader
│   ├── sandbox.py          # Secure subprocess
│   ├── trajectory.py       # Full trajectory recording
│   ├── evaluator.py        # 6-dimension scoring engine
│   ├── fault_injection.py  # Robustness testing
│   ├── security.py         # Unsafe code detection
│   └── memory.py           # Context efficiency tracking
├── repo_templates/          # 15 task variants
│   ├── task1/               # 5 single-file bug variants
│   ├── task2/               # 5 cross-module bug variants
│   └── task3/               # 5 feature implementation variants
├── openenv.yaml            # Environment metadata
├── Dockerfile              # Docker build
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## Why This Is Real-World

This isn't a toy benchmark. It tests the **exact capabilities** production coding agents need:

- **Navigate unfamiliar code** — agent sees only file names, not contents
- **Budget exploration** — finite steps mean strategic reading matters
- **Verify fixes** — must run tests, not just hope the fix works
- **Handle noise** — real repos have misleading comments and dead code
- **Write safe code** — production agents can't `eval()` or `os.system()`

## License

MIT
