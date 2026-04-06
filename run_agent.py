#!/usr/bin/env python3
"""
run_agent.py — Standalone HF Inference agent for OpenEnv.

Uses Hugging Face InferenceClient (NOT OpenAI SDK).
Runs directly against the environment in-process — no server needed.
Solves bug-fixing tasks step-by-step and prints the full execution trace.

Usage:
    python run_agent.py                            # uses built-in env
    HF_TOKEN=hf_xxx python run_agent.py            # with LLM agent
    HF_TOKEN=hf_xxx python run_agent.py --task task2  # specific task
"""
import os
import sys
import json
import argparse
import textwrap
from typing import List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from server.environment import CodebaseNavEnvironment
from server.models import RepoAction


# ── Configuration ────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
MAX_STEPS = {"task1": 12, "task2": 18, "task3": 22}


# ── HF Inference Client (lazy import) ───────────────────────────────────────
def get_hf_client():
    """Create HF InferenceClient. Returns None if no token."""
    if not HF_TOKEN:
        return None
    try:
        from huggingface_hub import InferenceClient
        return InferenceClient(model=MODEL_ID, token=HF_TOKEN)
    except ImportError:
        print("[WARN] huggingface_hub not installed. Using deterministic agent.", flush=True)
        return None
    except Exception as e:
        print(f"[WARN] Could not create HF client: {e}. Using deterministic agent.", flush=True)
        return None


# ── Prompts ──────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert Python developer debugging a code repository.
    You interact with the repo via JSON actions. Reply with ONLY a JSON object.

    Available actions:
    {"action_type": "read_file", "path": "src/file.py"}
    {"action_type": "write_file", "path": "src/file.py", "content": "...full file..."}
    {"action_type": "run_tests", "path": "tests/test_file.py"}
    {"action_type": "search_code", "query": "keyword"}
    {"action_type": "submit"}

    Strategy:
    1. Read the failing test first to understand expected behavior
    2. Read the buggy source file(s) identified by test imports
    3. Fix the bug by writing the corrected file
    4. Run tests to verify your fix
    5. Submit when all tests pass

    RESPOND WITH ONLY A JSON OBJECT. No markdown, no explanation.
""").strip()


def build_prompt(obs: dict, step: int, history: List[str]) -> str:
    tree = "\n".join(obs.get("repo_tree", []))
    read = ", ".join(obs.get("files_read", [])) or "none"
    failing = ", ".join(obs.get("failing_tests", [])) or "unknown"
    result = (obs.get("last_action_result") or "none")[:1500]
    error = obs.get("last_action_error") or "none"
    steps_left = obs.get("steps_remaining", 0)
    hist = "\n".join(history[-5:]) if history else "none"

    return (
        f"Step {step} | Task: {obs.get('current_task')} | Steps left: {steps_left}\n\n"
        f"Description: {obs.get('task_description')}\n\n"
        f"Files:\n{tree}\n\n"
        f"Already read: {read}\nFailing tests: {failing}\n\n"
        f"Last result:\n{result}\n\nLast error: {error}\n\n"
        f"History:\n{hist}\n\n"
        f"Next action? Reply with ONLY a JSON object."
    )


def llm_action(client, obs: dict, step: int, history: List[str]) -> dict:
    """Get action from HF Inference API."""
    prompt = build_prompt(obs, step, history)
    try:
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=800,
            temperature=0.2,
        )
        text = response.choices[0].message.content.strip()

        # Strip code fences
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        return json.loads(text)
    except json.JSONDecodeError:
        print(f"  [PARSE ERROR] Could not parse: {text[:100]}")
        return {"action_type": "submit"}
    except Exception as e:
        print(f"  [LLM ERROR] {e}")
        return {"action_type": "submit"}


# ── Deterministic Agent (no LLM needed) ─────────────────────────────────────
def deterministic_agent(obs: dict, step: int, files_read: set) -> dict:
    """
    A rule-based agent that follows optimal patterns for each task type.
    Works without any LLM — useful for testing and demos.
    """
    tree = obs.get("repo_tree", [])
    task = obs.get("current_task", "task1")
    test_files = sorted([f for f in tree if f.startswith("tests/")])
    src_files = sorted([f for f in tree if f.startswith("src/") and f.endswith(".py")])
    spec_files = sorted([f for f in tree if f.endswith("FEATURE_SPEC.md")])

    # Phase 1: Read spec/test files first
    if task == "task3" and spec_files:
        for sf in spec_files:
            if sf not in files_read:
                return {"action_type": "read_file", "path": sf}

    for tf in test_files:
        if tf not in files_read:
            return {"action_type": "read_file", "path": tf}

    # Phase 2: Read all source files
    for sf in src_files:
        if sf not in files_read:
            return {"action_type": "read_file", "path": sf}

    # Phase 3: Run tests to see current state
    if step <= 2 + len(src_files) + len(test_files):
        if test_files:
            return {"action_type": "run_tests", "path": test_files[0]}
        return {"action_type": "run_tests"}

    # Phase 4: Submit
    return {"action_type": "submit"}


# ── Main Runner ──────────────────────────────────────────────────────────────
def run_episode(env: CodebaseNavEnvironment, task: str, use_llm: bool = False):
    """Run one complete episode."""
    hf_client = get_hf_client() if use_llm else None
    using_llm = hf_client is not None

    max_steps = MAX_STEPS.get(task, 15)
    history = []
    files_read = set()

    print(f"\n{'='*60}")
    print(f"  [START] task={task} agent={'llm' if using_llm else 'deterministic'}")
    print(f"{'='*60}")

    # Reset
    reset_result = env.reset(task=task)
    obs = reset_result.observation
    variant = reset_result.info.get("variant_id", "?")

    print(f"  Variant: {variant}")
    print(f"  Files: {obs.repo_tree}")
    print(f"  Failing: {obs.failing_tests}")
    print(f"  Steps budget: {obs.steps_remaining}")
    print()

    rewards = []
    final_score = 0.0

    for step_num in range(1, max_steps + 1):
        if env.done:
            break

        # Get action from LLM or deterministic agent
        obs_dict = obs.model_dump()
        if using_llm:
            action_dict = llm_action(hf_client, obs_dict, step_num, history)
        else:
            action_dict = deterministic_agent(obs_dict, step_num, files_read)

        action_type = action_dict.get("action_type", "submit")
        action_path = action_dict.get("path")

        # Construct action
        action = RepoAction(
            action_type=action_type,
            path=action_dict.get("path"),
            query=action_dict.get("query"),
            content=action_dict.get("content"),
        )

        # Execute step
        result = env.step(action)
        obs = result.observation
        reward = result.reward

        rewards.append(reward)
        if action_path:
            files_read.add(action_path)

        # Print step log
        detail = action_path or action_dict.get("query") or ""
        err = f" ❌ {obs.last_action_error}" if obs.last_action_error else ""
        print(
            f"  [STEP] step={step_num} action={action_type:12s} "
            f"{detail:30s} reward={reward:+.3f}{err}"
        )

        history.append(f"Step {step_num}: {action_type} → {reward:+.3f}")

        if result.done:
            final_score = result.info.get("final_score", 0.0)
            break

    # Force submit if not done
    if not env.done:
        result = env.step(RepoAction(action_type="submit"))
        final_score = result.info.get("final_score", 0.0)
        rewards.append(result.reward)

    # Summary
    total_reward = sum(rewards)
    total_steps = len(rewards)
    success = final_score >= 0.5

    print()
    print(f"  [END] success={str(success).lower()} steps={total_steps} "
          f"score={final_score:.3f} total_reward={total_reward:.3f}")
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"  [END] rewards={rewards_str}")

    # Evaluation summary
    ev = env.get_evaluation()
    if "composite_score" in ev:
        print(f"\n  📊 Evaluation:")
        print(f"     Composite: {ev['composite_score']:.3f}")
        for name, dim in ev.get("dimensions", {}).items():
            print(f"     {name:15s}: {dim['score']:.3f}")

    return final_score, total_steps, rewards


def main():
    parser = argparse.ArgumentParser(description="Run agent against OpenEnv codebase-nav")
    parser.add_argument("--task", default="task1", choices=["task1", "task2", "task3"])
    parser.add_argument("--all-tasks", action="store_true", help="Run all 3 tasks")
    parser.add_argument("--llm", action="store_true", help="Use HF LLM agent (needs HF_TOKEN)")
    args = parser.parse_args()

    env = CodebaseNavEnvironment()

    if args.all_tasks:
        tasks = ["task1", "task2", "task3"]
    else:
        tasks = [args.task]

    all_scores = []
    for task in tasks:
        score, steps, rewards = run_episode(env, task, use_llm=args.llm)
        all_scores.append(score)

    if len(all_scores) > 1:
        avg = sum(all_scores) / len(all_scores)
        print(f"\n{'='*60}")
        print(f"  OVERALL: avg_score={avg:.3f} tasks={len(all_scores)}")
        print(f"{'='*60}")

    env.close()


if __name__ == "__main__":
    main()
