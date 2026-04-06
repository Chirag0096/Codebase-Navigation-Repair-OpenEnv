# 🚀 Codebase Navigation & Repair — OpenEnv Pitch

Welcome to our Meta OpenEnv Hackathon submission! This document explains our project in simple, clear terms—what it is, why it's better than existing tools, and how it works under the hood.

---

## 🌟 What is it?
**Codebase Navigation & Repair** is a specialized training and testing ground (an "environment") for AI coding agents like Devin, GitHub Copilot, or Cursor. 

Imagine dropping a developer into a massive codebase they have never seen before and telling them, "Fix the bug." They have to look around, read the right files, understand the problem, write the fix, and run the tests to prove it works. 

Our environment forces AI agents to do exactly that. We don't just give the AI all the files at once (which is unrealistic and expensive); instead, the AI must *navigate* the repo step-by-step, just like a human engineer would.

---

## 🛠️ How is it helpful?
Today, if an AI coding agent fails to fix a bug, developers usually don't know *why*. Did it read the wrong files? Did it waste time reading irrelevant things? Did it hallucinate code? Did it test the fix?

Our environment solves this by providing a **Process-Based Evaluation Engine**. We don't just grade the final output (Pass/Fail). We grade the *entire journey*:
1. Did it find the right files quickly?
2. Did it follow good engineering practices (Read → Write → Test)?
3. Did it try to do anything unsafe or malicious?
4. How efficiently did it use its context window?

This helps researchers and developers find the exact weak spots in their AI models and improve them targetedly.

---

## 🥇 Why is it better than other tools? (Our USP)

Our Unique Selling Proposition (USP) is that we test **Process and Reliability, not just Correctness**. 

Unlike standard benchmarks (like SWE-bench) which just check if a test passed at the end, our system features:
- **Full Trajectory Replay:** We record every single action the agent takes, like a flight data recorder, so you can debug the AI's thought process.
- **Dynamic Fault Injection:** Real-world code is messy. We can inject misleading comments, red herring files, and noisy documentation into the environment to see if the AI gets tricked or stays focused.
- **Proactive Security Scanning:** We scan the AI's output for dangerous code (like attempting to run `os.system("rm -rf /")`), ensuring the agent is safe to run in production.
- **Context Memory Tracking:** We penalize agents that waste API tokens by re-reading identical files unnecessarily.

---

## 🏗️ Architecture: How does it work?

The system is built as a complete, self-contained **FastAPI + Gradio** web application packaged in a **Docker Container**, making it perfect for Hugging Face Spaces.

Here is the flow:
1. **The Server (Environment):** Built with FastAPI. It loads a Python repository with a hidden bug.
2. **The Agent (Inference):** The AI model (we provide a Hugging Face Inference agent) requests the current state—it only sees a list of file names, not the contents.
3. **The Loop:** 
   - The Agent asks to `read_file`. It gets the contents.
   - The Agent asks to `write_file` to fix the bug.
   - The Agent asks to `run_tests` to verify if its fix worked via our sandboxed Pytest runner.
   - Every action is logged, scored, and evaluated by our Reliability Grader.
4. **The UI:** A beautiful Gradio interface lets human users interact with the environment manually or watch the built-in AI agent work in real-time. It also provides beautiful evaluation dashboards.

---

## 🚀 Steps to work with it

You have several ways to use this environment:

### 1. In your Browser (Easiest)
Simply visit our Hugging Face Space: [Chirag0123/codebase-nav-env](https://huggingface.co/spaces/Chirag0123/codebase-nav-env)
You can play the environment like a text-based game using the **Interactive** tab, or watch the AI solve it in the **Run Agent** tab.

### 2. Run it Locally with Docker
If you want to run it on your own machine securely:
```bash
docker build -t codebase-nav-env .
docker run -p 7860:7860 codebase-nav-env
```
Then visit `http://localhost:7860` in your browser.

### 3. Test Your Own AI Model
If you are building an AI agent, you can hook it up to our API.
```bash
# Provide your Hugging Face API Token (or OpenAI, etc.)
export HF_TOKEN="hf_your_token_here"
# Run the included agent that talks to our environment
python run_agent.py --llm --task task1
```

---

## 🎯 Hackathon Requirements Satisfied

We have strictly followed all rules and requirements for the Meta OpenEnv Hackathon:
✅ **OpenEnv Compliant:** Implements standard `/reset`, `/step`, and `/state` API endpoints.
✅ **Dockerized & Sandboxed:** Securely runs code in a non-root environment using Docker.
✅ **Hugging Face Space Ready:** Deployed and running live on Hugging Face Spaces with a Gradio UI entry point.
✅ **Inference Script Provided:** Includes `run_agent.py` and `inference.py` which utilize Hugging Face's Inference endpoints (not OpenAI) to solve tasks.
✅ **Realistic Tasks:** Complex, multi-file bug fixing and feature implementations verified by real `pytest` executions.
✅ **Gradio UI:** Features a multi-tab visual dashboard to demonstrate the environment's capabilities intuitively.
