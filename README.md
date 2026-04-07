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

<div align="center">
  <a href="https://huggingface.co/spaces/Chirag0123/codebase-nav-env">
    <img src="https://raw.githubusercontent.com/Chirag0096/Codebase-Navigation-Repair-OpenEnv/assets/assets/demo.webp" width="100%" alt="3D Visualizer Architecture Trace">
  </a>
  
  <br/>
  
  <h1>🔍 Codebase Navigation Repair OpenEnv</h1>
  
  <p><strong>The ultimate diagnostic environment to end "Vibe Coding." Making AI coding agents structural, testable, and deeply debuggable.</strong></p>
  
  <p>
    <a href="https://huggingface.co/spaces/Chirag0123/codebase-nav-env"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue" alt="Hugging Face Space"></a>
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/FastAPI-REST_API-009688.svg" alt="FastAPI">
    <img src="https://img.shields.io/badge/Three.js-3D_Visualizer-black.svg" alt="ThreeJs">
    <img src="https://img.shields.io/badge/Docker-Containerized_Scoring-2496ED.svg" alt="Docker">
  </p>
</div>

---

## 🚨 The End of "Vibe Coding"

We are officially in the era of **Vibe Coding**. The volume of AI-generated code is exploding, yet developers and top-tier AI Agents (Copilot, Devin, Claude Code) are increasingly writing and submitting code *blindly*. 

Most agents don't actually know **where the issue exists**, what the **code flow** looks like, or how the **function dependencies** cascade. Current developer benchmarks only evaluate the final outcome. **They do not evaluate cognition.** 

When an AI agent claims "I fixed the bug," how do you verify *how* it did it? Did it actually navigate to the source of the crash, trace the logical data flow, or did it just randomly change syntax until a test arbitrarily turned green?

## 💡 Our Solution: 3D Visualization & Deep Analytic Execution

This project is not just another benchmark—it is a **Full-Stack Diagnostic Platform**. It actively forces autonomous AI agents to explore an unknown Python repository file-by-file through a strictly monitored API, and then exposes their **exact cognitive layout**.

By tracking structural behavior instead of just binary pass/fail outcomes, our platform gives researchers, engineers, and Hackathon judges unprecedented visibility into an AI's actual thought process and navigation footprint.

---

## 🧠 Core Intelligence Modules (v4.0)

Unlike standard environments, we evaluate **how** the agent works using proprietary, research-grade engines built specifically for this platform:

| 🧩 Module | 🎯 What It Does (The Cure to Vibe Coding) |
|-----------|--------------------------------------------|
| **`3D Trace Visualizer`** | A seamless, fully-interpolated 3D engine that renders repos as geometric maps (Cubes for Source, Prisms for Tests). Visualizes agent navigation traces via glowing Catmull-Rom tube paths. |
| **`Causal Graph Probe`** | Detects "Shortcut Learning". Maps a Directed Acyclic Graph to verify if the agent actually read the test file, traced its imported module, and structurally fixed the root cause—or if it guessed blindly. |
| **`Confidence Calibrator`** | Infers the agent's behavioral confidence entirely based on real-time execution speeds, rewrite hesitation frequencies, and test verification ratios. |
| **`Counterfactual Engine`** | Subjects the agent to 6 robustness ablation tests (mutating the environment behind the scenes) to determine if its strategy relies on brittle memorization. |
| **`Episodic Memory Bank`** | A cross-episode Retrieval-Augmented Generation (RAG) store capturing procedural mistakes (e.g., failing to run tests before committing) to dynamically auto-inject hard lessons into future iteration system prompts. |

---

## ⚙️ How It Works (The OpenEnv Standard)

1. **Blind Start:** Agent loads an unfamiliar environment variant -> sees the repository file tree (NOT contents).
2. **Step Budgeting:** Agent explores variables and reads files one at a time (costing strictly penalized exploration steps).
3. **Flow Navigation:** Agent navigates architecture dependencies and identifies structural vulnerabilities.
4. **Execution:** Agent acts and writes the updated architectural fix.
5. **Verification:** Agent verifies functionality through containerized `pytest` execution loops safely within the RL boundary.
6. **Dynamic Scoring:** Environment scores the agent's complete step trajectory across 6 independent research axes.

---

## 🚀 Quick Start

### 1. Run Locally (No Docker)
Spin up the backend and the 3D analytical dashboard.
```bash
pip install -r requirements.txt
python app.py                    # Gradio UI + FastAPI starts at http://localhost:7860
```

### 2. Connect Your Custom LLM Agent
Wire up your own agent configuration.
```bash
export HF_TOKEN=hf_xxxxx
# Execute your script pointing to the local /step FASTApi environment
python inference.py
```

### 3. Deploy via Docker
```bash
docker build -t codebase-nav-env .
docker run -p 7860:7860 codebase-nav-env
```

---

## 📊 Evaluation API Layers

The environment strictly communicates via a standard RESTful architecture. 

| Endpoint | Method | Operational Description |
|----------|--------|-------------------------|
| `/step` | `POST` | Takes singular OpenEnv navigation action (`read_file`, `write_file`) |
| `/evaluate` | `GET` | Fetches deterministic baseline evaluation metrics |
| `/causal-probe` | `GET` | Generates directed acyclic graphs resolving true root-cause logic mapping |
| `/confidence` | `GET` | Emits behavioral-time confidence calibration algorithms |
| `/counterfactual` | `POST` | Triggers the 6 robustness ablation hallucination detection engine |

<br/>

> *Stop trusting the vibe. Force the cognition.*
