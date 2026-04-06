# 🚀 Codebase Navigation & Repair

**AI coding agents fail silently and unpredictably. And worse—no one knows *why* they fail.** 

They get lost in large codebases, hallucinate fixes, and deploy broken code. Existing benchmarks only tell you if an agent failed, not *where* or *why* it went wrong. 

Our solution: **The system that makes AI coding agents reliable in real-world scenarios.** We track, evaluate, and score every single step of the agent’s reasoning, navigation, and execution.

---

## 🌟 What is it?
Codebase Navigation & Repair is a specialized process-evaluation engine for AI coding agents (like Devin, Copilot, or Cursor). 

Instead of spoon-feeding the AI the exact files it needs, we drop the agent into an unfamiliar, multi-file Python repository. The agent must independently navigate the codebase, understand the bug, write a fix, and run the test suite to verify its work—just like a human engineer. 

---

## 🛠️ Why it matters
Right now, evaluating AI agents is binary: Pass or Fail. 

We change that by evaluating the **process**:
1. **Efficiency:** Did it read irrelevant files and waste context window?
2. **Reasoning:** Did it follow best practices (e.g., reading tests before modifying source code)?
3. **Security:** Did it try to inject malicious code during the repair?

This transforms agent development from guesswork into targeted, measurable engineering.

---

## 🎬 Demo Walkthrough

**The Scenario:** A backend API has a bug where `order_processor.py` fails to handle negative inventory.

**Step 1: The Reset (Agent enters the workspace)**
* The agent sees a file tree (no contents) and the failing test: `test_process_valid_order`

**Step 2: Investigation (Agent reads files)**
* *Action:* `read_file tests/test_orders.py` *(Smart move: understand expected behavior first)*
* *Action:* `read_file src/order_processor.py` *(Finds the bug location)*

**Step 3: The Repair (Agent writes code)**
* *Action:* `write_file src/order_processor.py` *(Modifies logic to add `if item.qty < 0: raise ValueError`)*

**Step 4: Verification (Agent runs tests)**
* *Action:* `run_tests tests/test_orders.py`
* *Result:* Tests turn green! `[100% passing]`

**Step 5: Submission & Evaluation**
* The agent submits the fix.
* **Our Engine kicks in:** It evaluates the trajectory and gives the agent a top-tier composite score for flawless navigation, strong reasoning, and optimal step efficiency.

---

## 🏗️ How it works (Simplified)

1. **The Server:** A FastAPI engine loads a sandboxed, hidden-bug repository.
2. **The Agent:** Interacts via strict API calls (`read_file`, `write_file`, `run_tests`), simulating real console usage.
3. **The Grader:** A sandboxed Pytest runner securely executes the agent's code.
4. **The UI:** A live Gradio dashboard lets you watch agents work in real-time or explore dynamic evaluation metrics.

---

## 🥇 Why it’s better

We don't just grade the outcome; we stress-test the AI:
- **Dynamic Fault Injection:** We actively inject misleading code comments and red herring files into the codebase to see if the AI gets tricked.
- **Trajectory Replay:** We record every API call, diff, and timestamp so you can "play back" an agent's failure.
- **Proactive Security:** We monitor the agent's output for dangerous patterns (like `os.system("rm -rf /")`) to ensure production safety.

---

## ⏰ Why Now
Autonomous coding agents are the fastest-growing sector in AI. But **reliability is the biggest unsolved problem holding them back from enterprise adoption.** A system that can definitively evaluate *how* an agent reasons and *why* it fails is the missing infrastructure for the next generation of AI product development.

---

## 🤝 Hackathon Alignment
We built this explicitly for the Meta OpenEnv standard:
- **OpenEnv Compliant:** Implements standard `/reset`, `/step`, and `/state` APIs out-of-the-box.
- **Hugging Face Ready:** Fully dockerized, sandboxed, and deployed via Gradio to HF Spaces.
- **HF Inference Agent:** Includes a standalone Python script (`run_agent.py`) using Hugging Face inference endpoints—no OpenAI lock-in required.

---

## 🚀 Why This Wins
This project isn't just a hackathon toy—it is a piece of **core infrastructure** the AI industry actually needs right now. 

It combines **real-world relevance** (fixing broken tests in messy, multi-file repos) with **deep technical rigor** (process-based evaluation, fault injection, secure sandboxing). We've taken the base OpenEnv standard and turned it into a completely observable, visually impressive, state-of-the-art testing layer that is impossible to ignore.
