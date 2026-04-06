# 🚀 Codebase Navigation & Repair — OpenEnv

## 🧨 The Problem
AI coding agents fail silently and unpredictably. 
Worse, **no one knows WHY they fail.** 
Current benchmarks just give a final Pass/Fail grade. Did the agent read the wrong files? Hallucinate a fix? Ignore the tests entirely? There is no way to know. 

## 💡 The Solution
**We track and evaluate every step of the agent’s reasoning and actions.**
Codebase Navigation & Repair is a system that makes AI coding agents reliable in real-world scenarios. We don't just grade the final output; we grade the *entire journey*.

---

## 🛠️ What It Is
It is a fully OpenEnv-compliant, production-ready testing environment for AI software engineers. 

You drop an AI agent into an unfamiliar Python repository with a hidden bug. The agent cannot cheat by seeing all files at once. It must explore the codebase step-by-step, find the issue, write the fix, and run actual tests to prove it works—exactly like a human engineer.

---

## 🌎 Why It Matters
For developers building autonomous agents (like Devin, Copilot, or Cursor), **reliability** is the biggest unsolved problem. Our system provides a high-fidelity diagnostic layer so researchers can find the exact weak spots in their models and fix them.

---

## 🎬 Demo Walkthrough: A Realistic Bug Scenario

Imagine an e-commerce agent tasked with fixing an order processing failure.

**BEFORE:** ❌ `test_process_valid_order` is failing. 

1. **Step 1:** Agent reads `tests/test_orders.py` to understand the expected behavior.
2. **Step 2:** Agent reads `src/order_processor.py` and spots the bug: a missing `datetime` import causing the script to crash.
3. **Step 3:** Agent writes the fix to `src/order_processor.py`.
4. **Step 4:** Agent runs `pytest`.
5. **Step 5:** Agent submits the fixed codebase.

**AFTER:** ✅ All tests pass. 

Our system records this perfect execution. But if an agent *fails*, our **Process-Based Evaluation** engine flags exactly what went wrong: e.g., *"Agent wasted 14 steps reading irrelevant files and submitted without testing."*

---

## 🏗️ How It Works (Simplified)
1. **The Server:** A FastAPI engine loads a Python repository with a verifiable bug.
2. **The Agent:** An AI model (we provide a Hugging Face Inference agent) requests the current state and explores the repo tree.
3. **The Loop:** The agent interacts via structured actions (`read_file`, `write_file`, `run_tests`).
4. **The Evaluation:** Every action is logged, timed, and scored against our 6-axis Reliability Grader.
5. **The UI:** A beautiful Gradio interface lets you watch the AI operate in real-time or explore its trajectory post-flight.

---

## 🥇 Why It’s Better (Our USP)

We test **Process and Reliability, not just Correctness**. 

- **Flight Data Recorder:** Full trajectory replay. Debug the AI's thought process step-by-step.
- **Dynamic Fault Injection:** Real code is messy. We inject misleading comments and red herring files to see if the AI gets distracted.
- **Proactive Security:** We scan the AI's output for dangerous patterns (like `os.system`) to prevent destructive actions.
- **Context Efficiency:** We penalize agents that waste API tokens by reading identical files over and over.

---

## ⏱️ Why Now?
The rise of autonomous agents is here. But enterprise adoption is stalled because these agents are unpredictable. Moving from "cool toy" to "reliable teammate" requires rigorous, process-level evaluation. Our system directly solves the reliability bottleneck.

---

## 🤝 Hackathon Alignment
We built this explicitly for the Meta OpenEnv hackathon:
- **100% OpenEnv Compliant:** Implements standard `/reset`, `/step`, and `/state` APIs.
- **Live & Deployed:** Running live on Hugging Face Spaces with a Gradio frontend.
- **Inference Ready:** Built-in agent using Hugging Face inference (`run_agent.py`).
- **Sandboxed:** Secure, dockerized test execution.

---

## 🚀 Why This Wins
This is not infrastructure; it is a **diagnostic product for the AI era**. 
It features immense technical depth (sandboxed execution, multi-dimensional scoring, fault injection), massive real-world relevance, and a polished user experience. It doesn't just test AI agents—it shows us how to make them better.
