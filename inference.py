#!/usr/bin/env python3
"""
inference.py — Mandatory OpenEnv baseline inference script.
Runs an LLM agent against all 3 tasks and emits required log format.

Environment variables required:
    API_BASE_URL   — LLM API endpoint
    MODEL_NAME     — model identifier
    HF_TOKEN       — Hugging Face API token
"""
import os
import json
import textwrap
from typing import List, Optional

from openai import OpenAI
import httpx

# ── Configuration ─────────────────────────────────────────────────────────────
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

MAX_STEPS_PER_TASK = {"task1": 12, "task2": 18, "task3": 22}
TEMPERATURE = 0.2
MAX_TOKENS = 800
SUCCESS_THRESHOLD = 0.5

TASKS = ["task1", "task2", "task3"]


# ── Logging helpers ────────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Environment client ─────────────────────────────────────────────────────────
class EnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=60.0)

    def reset(self, task: str) -> dict:
        r = self.client.post(f"{self.base_url}/reset", params={"task": task})
        r.raise_for_status()
        return r.json()

    def step(self, action: dict) -> dict:
        r = self.client.post(f"{self.base_url}/step", json=action)
        r.raise_for_status()
        return r.json()

    def state(self) -> dict:
        r = self.client.get(f"{self.base_url}/state")
        r.raise_for_status()
        return r.json()

    def close(self):
        self.client.close()


# ── LLM Agent ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert software engineer working inside a Python code repository.
    You can take the following actions (respond with ONLY a valid JSON object):

    {"action_type": "read_file", "path": "src/some_file.py"}
    {"action_type": "write_file", "path": "src/some_file.py", "content": "...full new content..."}
    {"action_type": "run_tests", "path": "tests/test_something.py"}
    {"action_type": "search_code", "query": "function_name_or_keyword"}
    {"action_type": "submit"}

    Strategy:
    1. ALWAYS read relevant source files before writing any fixes
    2. For task1/task2: read failing test file first to understand what is expected
    3. For task3: read FEATURE_SPEC.md first, then existing source files
    4. Run tests after writing a fix to verify improvement
    5. Submit only when confident tests will pass

    Reply with ONLY the JSON action object. No explanation. No markdown. No extra text.
""").strip()


def build_user_prompt(obs: dict, step: int, history: List[str]) -> str:
    tree_str = "\n".join(obs.get("repo_tree", []))
    files_read_str = ", ".join(obs.get("files_read", [])) or "none yet"
    failing_str = ", ".join(obs.get("failing_tests", [])) or "unknown"
    last_result = obs.get("last_action_result") or "none"
    last_error = obs.get("last_action_error") or "none"
    steps_left = obs.get("steps_remaining", 0)
    history_str = "\n".join(history[-5:]) if history else "none"

    return textwrap.dedent(f"""
        Step: {step}
        Task: {obs.get('current_task')}
        Description: {obs.get('task_description')}
        Steps remaining: {steps_left}

        Repository files:
        {tree_str}

        Files already read: {files_read_str}
        Known failing tests: {failing_str}
        Last action result: {last_result[:1000]}
        Last action error: {last_error}

        Recent history:
        {history_str}

        What is your next action? Reply with ONLY a JSON action object.
    """).strip()


def get_agent_action(client: OpenAI, obs: dict, step: int, history: List[str]) -> dict:
    user_prompt = build_user_prompt(obs, step, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        action = json.loads(text)
        return action
    except json.JSONDecodeError:
        print(f"[DEBUG] Failed to parse action JSON: {text[:200]}", flush=True)
        return {"action_type": "submit"}  # Fallback
    except Exception as e:
        print(f"[DEBUG] LLM call failed: {e}", flush=True)
        return {"action_type": "submit"}


def run_task(env_client: EnvClient, llm_client: OpenAI, task: str) -> tuple:
    """Run one complete episode for a task. Returns (score, steps, rewards)."""
    max_steps = MAX_STEPS_PER_TASK.get(task, 15)
    benchmark = "codebase-nav-env"

    rewards = []
    history = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task, env=benchmark, model=MODEL_NAME)

    try:
        reset_result = env_client.reset(task=task)
        obs = reset_result["observation"]

        for step_num in range(1, max_steps + 1):
            if obs.get("steps_remaining", 0) <= 0:
                break

            action = get_agent_action(llm_client, obs, step_num, history)
            action_str = json.dumps(action)

            try:
                step_result = env_client.step(action)
            except Exception as e:
                log_step(step_num, action_str, 0.0, True, str(e))
                break

            reward = step_result.get("reward", 0.0)
            done = step_result.get("done", False)
            error = step_result["observation"].get("last_action_error")

            rewards.append(reward)
            steps_taken = step_num
            obs = step_result["observation"]

            history.append(f"Step {step_num}: {action.get('action_type')} -> reward {reward:+.2f}")

            log_step(step=step_num, action=action_str[:200], reward=reward, done=done, error=error)

            if done:
                # Get final score from state
                state = env_client.state()
                score = state.get("current_score", 0.0)
                break

        # If not done yet (step budget exhausted), force submit
        if not obs.get("last_action_result", "").startswith("=== FINAL GRADER"):
            try:
                step_result = env_client.step({"action_type": "submit"})
                state = env_client.state()
                score = state.get("current_score", 0.0)
            except Exception:
                pass

        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score, steps_taken, rewards


def main():
    env_client = EnvClient(ENV_BASE_URL)
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_scores = []
    for task in TASKS:
        score, steps, rewards = run_task(env_client, llm_client, task)
        all_scores.append(score)
        print(f"[INFO] {task} complete: score={score:.3f} steps={steps}", flush=True)

    avg_score = sum(all_scores) / len(all_scores)
    print(f"[INFO] Average score across all tasks: {avg_score:.3f}", flush=True)

    env_client.close()


if __name__ == "__main__":
    main()
