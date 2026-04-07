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

# 🔍 The Antidote to "Vibe Coding" — AI Reliability & Navigation Platform 

> **The system repairing the era of blind AI coding by making agents structural, testable, and deeply debuggable.**

**Play with the live environment:** [Interactive Hugging Face Space](https://huggingface.co/spaces/Chirag0123/codebase-nav-env)

## 🚨 The End of "Vibe Coding"

We are officially in the era of **Vibe Coding**. The amount of AI-generated code is exploding, yet developers and AI Agents (Copilot, Devin, Claude Code) are increasingly writing and submitting code *blindly*. 

Most agents don't actually know **where the issue exists**, what the **code flow** looks like, or how the **function dependencies** cascade. They simply guess edits based on the prompt until a test arbitrarily passes. When an AI agent claims "I fixed the bug," how do you verify *how* it did it? Did it actually navigate to the source of the crash, or did it randomly change syntax until the test turned green? 

Current benchmarks only evaluate the final outcome. **They don't evaluate cognition.**

## 💡 The Solution: 3D Visualization & Deep Analytic Execution

This project is not just an environment benchmark—it is a **diagnostic platform**. It forces autonomous AI agents to explore an unknown Python repository file-by-file, and then exposes their **exact cognitive flow** using our bespoke state-of-the-art **3D Visualizer**. 

By tracking structural behavior instead of just outcomes, our platform gives researchers and operators complete visibility into the AI's actual thought processes and navigation flow.

### 🎬 See It In Action (Demo)

![3D Visualizer Architecture Trace](https://raw.githubusercontent.com/Chirag0096/Codebase-Navigation-Repair-OpenEnv/assets/assets/demo.png)

*(A live recording of the 3D agent visualizer tracking test files, source files, and resolving dependencies)*

## 🧠 Core Intelligence Modules

Unlike existing models, we evaluate **how** the agent works, using several proprietary research-grade engines:

| Module | What It Does (The Antidote to Vibe Coding) |
|--------|--------------------------------------------|
| **Causal Graph Probe** | Detects "Shortcut Learning". Did the agent actually read the test file, trace its imported module, and fix the root cause, or did it guess blindly? |
| **Confidence Calibrator** | Infers agent behavioral confidence based on commit speed, rewrite hesitation, and test verification ratios. |
| **Counterfactual Engine** | Analyzes the precise trace line to determine if the agent's strategy is brittle and heavily reliant on memorization of specific repository layouts. |
| **Episodic Memory Bank** | A cross-episode RAG store that captures mistakes (like failing to run tests before commiting) and injects hard lessons into future iteration system prompts. |
| **3D Trace Visualizer** | A seamless, fully-interpolated 3D environment engine that renders repos as geometric maps (Cubes for Source, Prisms for Tests) and visualizes the exact agent navigation traces with glowing Catmull-Rom tube curves. |

## ⚙️ How It Works (The OpenEnv Standard)

1. **Agent loads unfamiliar environment** → sees repo file tree (NOT contents).
2. Agent reads files one at a time (costs strict exploration steps).
3. Agent identifies structural vulnerabilities via function-flow analysis.
4. Agent writes fixed code.
5. Agent verifies functionality through containerized `pytest` execution. 
6. Environment scores agent completely dynamically across 6 separate research axes.

## 🚀 Quick Start

### 1. Run Locally (No Docker)
```bash
pip install -r requirements.txt
python app.py                    # Gradio UI + FastAPI at http://localhost:7860
```

### 2. Connect Your Custom LLM Agent
```bash
export HF_TOKEN=hf_xxxxx
# Configure your script to hit the local /step FASTApi environment
python inference.py
```

### 3. Deploy via Docker
```bash
docker build -t codebase-nav-env .
docker run -p 7860:7860 codebase-nav-env
```

## 📊 Evaluation API Layers

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/step` | POST | Takes singular OpenEnv navigation action (`read_file`, `write_file`) |
| `/evaluate` | GET | Baseline evaluation metrics |
| `/causal-probe` | GET | Builds directed acyclic graphs resolving true root-cause logic mapping |
| `/confidence` | GET | Returns behavior-time confidence estimation algorithms |
| `/counterfactual` | POST | Subjects agent to 6 robustness ablation tests to detect hallucination |

*Stop trusting the vibe. Force the cognition.*
