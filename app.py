#!/usr/bin/env python3
"""
app.py — Gradio UI + FastAPI endpoints for the OpenEnv environment.
This is the HF Space entry point.
"""
import os
import json
import gradio as gr
from server.environment import CodebaseNavEnvironment
from server.models import RepoAction

# ── Global environment instance ──────────────────────────────────────────────
env = CodebaseNavEnvironment()


# ── Gradio callback functions ────────────────────────────────────────────────

def reset_environment(task: str):
    """Reset environment and return initial state."""
    try:
        result = env.reset(task=task)
        obs = result.observation
        tree = "\n".join(f"  📄 {f}" for f in obs.repo_tree)
        failing = ", ".join(obs.failing_tests) if obs.failing_tests else "None listed"
        info_data = result.info

        status_text = (
            f"✅ Episode started\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Task: {task}\n"
            f"Variant: {info_data.get('variant_id', 'unknown')}\n"
            f"Steps remaining: {obs.steps_remaining}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📁 Repository Files:\n{tree}\n\n"
            f"🔴 Failing Tests: {failing}\n\n"
            f"📋 Task: {obs.task_description}"
        )
        return status_text, "", "0", "0.000"
    except Exception as e:
        return f"❌ Error: {e}", "", "0", "0.000"


def take_step(action_type: str, path: str, query: str, content: str):
    """Execute one agent step."""
    if env.done:
        return "❌ Episode is done. Reset first.", "", "", ""

    try:
        action = RepoAction(
            action_type=action_type,
            path=path if path.strip() else None,
            query=query if query.strip() else None,
            content=content if content.strip() else None,
        )
        result = env.step(action)
        obs = result.observation

        action_result = obs.last_action_result or "No output"
        error = obs.last_action_error or ""
        if error:
            error = f"⚠️ {error}"

        status = (
            f"Step {result.info['steps_taken']} | "
            f"Reward: {result.reward:+.3f} | "
            f"Steps left: {obs.steps_remaining}"
        )
        if result.done:
            status += f"\n\n🏁 EPISODE DONE — Final Score: {result.info['final_score']:.3f}"

        flags = result.info.get("security_flags", [])
        if flags:
            status += f"\n🔒 Security: {flags}"

        return (
            status,
            action_result[:3000],
            str(result.info["steps_taken"]),
            f"{result.info.get('cumulative_reward', 0):.3f}",
        )
    except Exception as e:
        return f"❌ Error: {e}", "", "", ""


def get_evaluation():
    """Get multi-dimensional evaluation report."""
    try:
        ev = env.get_evaluation()
        if "error" in ev:
            return "No evaluation available. Run an episode first."

        lines = [
            f"🎯 Composite Score: {ev['composite_score']:.3f}",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ]
        for name, dim in ev.get("dimensions", {}).items():
            bar = "█" * int(dim["score"] * 20) + "░" * (20 - int(dim["score"] * 20))
            lines.append(f"  {name:15s} [{bar}] {dim['score']:.3f}")
            for e in dim.get("evidence", []):
                lines.append(f"    → {e}")

        if ev.get("strengths"):
            lines.append("\n💪 Strengths:")
            for s in ev["strengths"]:
                lines.append(f"  ✅ {s}")

        if ev.get("failure_analysis"):
            lines.append("\n⚠️ Failures:")
            for f in ev["failure_analysis"]:
                lines.append(f"  ❌ {f}")

        if ev.get("recommendations"):
            lines.append("\n💡 Recommendations:")
            for r in ev["recommendations"]:
                lines.append(f"  → {r}")

        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def get_metrics():
    """Get comprehensive metrics."""
    try:
        m = env.get_metrics()
        return json.dumps(m, indent=2, default=str)
    except Exception as e:
        return f"Error: {e}"


def get_trajectory():
    """Get full trajectory."""
    try:
        t = env.get_trajectory()
        if not t:
            return "No trajectory available."

        lines = [
            f"Episode: {t.get('episode_id', 'N/A')}",
            f"Task: {t.get('task', 'N/A')} | Variant: {t.get('variant_id', 'N/A')}",
            f"Duration: {t.get('duration_seconds', 'N/A')}s | Score: {t.get('final_score', 0):.3f}",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ]
        for step in t.get("steps", []):
            emoji = "📖" if step["action_type"] == "read_file" else \
                    "✏️" if step["action_type"] == "write_file" else \
                    "🧪" if step["action_type"] == "run_tests" else \
                    "🔍" if step["action_type"] == "search_code" else "🏁"
            path = step.get("action_path") or step.get("action_query") or ""
            err = f" ❌ {step['error']}" if step.get("error") else ""
            lines.append(
                f"  {emoji} Step {step['step_number']:2d}: "
                f"{step['action_type']:12s} {path:30s} "
                f"reward={step['reward']:+.3f} "
                f"({step['duration_ms']:.0f}ms){err}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def run_builtin_agent(task: str):
    """Run the built-in deterministic agent for a quick demo."""
    try:
        # Reset
        result = env.reset(task=task)
        obs = result.observation
        log_lines = [f"🚀 Starting {task} (variant: {result.info.get('variant_id')})"]
        log_lines.append(f"   Files: {obs.repo_tree}")
        log_lines.append(f"   Failing: {obs.failing_tests}")

        # Strategy: read test file → read source → fix → run tests → submit
        test_files = [f for f in obs.repo_tree if f.startswith("tests/")]
        src_files = [f for f in obs.repo_tree if f.startswith("src/") and f.endswith(".py")]
        spec_files = [f for f in obs.repo_tree if f.endswith(".md")]

        steps_done = 0
        max_demo_steps = 15

        # Step 1: read spec or test
        if task == "task3" and spec_files:
            target = spec_files[0]
        elif test_files:
            target = test_files[0]
        else:
            target = obs.repo_tree[0]

        step_result = env.step(RepoAction(action_type="read_file", path=target))
        steps_done += 1
        log_lines.append(f"   Step {steps_done}: read_file {target} → reward={step_result.reward:+.3f}")

        # Step 2+: read all source files
        for sf in src_files:
            if env.done or steps_done >= max_demo_steps - 2:
                break
            step_result = env.step(RepoAction(action_type="read_file", path=sf))
            steps_done += 1
            log_lines.append(f"   Step {steps_done}: read_file {sf} → reward={step_result.reward:+.3f}")

        # Step N-1: run tests
        if not env.done and steps_done < max_demo_steps - 1:
            step_result = env.step(RepoAction(action_type="run_tests"))
            steps_done += 1
            log_lines.append(f"   Step {steps_done}: run_tests → reward={step_result.reward:+.3f}")

        # Step N: submit
        if not env.done:
            step_result = env.step(RepoAction(action_type="submit"))
            steps_done += 1
            log_lines.append(f"   Step {steps_done}: submit → reward={step_result.reward:+.3f}")

        log_lines.append(f"\n🏁 Final Score: {env.final_score:.3f}")
        log_lines.append(f"   Total Steps: {steps_done}")
        log_lines.append(f"   Cumulative Reward: {env.cumulative_reward:.3f}")

        return "\n".join(log_lines)
    except Exception as e:
        return f"❌ Error: {e}"


# ── Build the Gradio UI ─────────────────────────────────────────────────────

with gr.Blocks(
    title="Codebase Navigation & Repair — OpenEnv",
) as demo:
    gr.Markdown(
        "# 🔍 Codebase Navigation & Repair — OpenEnv\n"
        "**RL environment for testing AI coding agents.** "
        "Agents navigate repos, find bugs, and fix them — graded by actual pytest execution."
    )

    with gr.Tabs():
        # ── Tab 1: Interactive Environment ────────────────────────────────
        with gr.TabItem("🎮 Interactive"):
            with gr.Row():
                with gr.Column(scale=1):
                    task_select = gr.Dropdown(
                        choices=["task1", "task2", "task3"],
                        value="task1",
                        label="Task",
                        info="task1=single-file bugs, task2=cross-module, task3=feature impl"
                    )
                    reset_btn = gr.Button("🔄 Reset Environment", variant="primary")

                    gr.Markdown("### Take an Action")
                    action_type = gr.Dropdown(
                        choices=["read_file", "write_file", "run_tests", "search_code", "submit"],
                        value="read_file",
                        label="Action Type",
                    )
                    action_path = gr.Textbox(label="Path (for read/write/run_tests)", placeholder="src/auth.py")
                    action_query = gr.Textbox(label="Query (for search_code)", placeholder="validate_token")
                    action_content = gr.Textbox(label="Content (for write_file)", lines=5, placeholder="# new file content...")
                    step_btn = gr.Button("▶️ Execute Step", variant="secondary")

                with gr.Column(scale=2):
                    status_box = gr.Textbox(label="Status", lines=15, interactive=False)
                    result_box = gr.Textbox(label="Last Action Result", lines=10, interactive=False)
                    with gr.Row():
                        steps_box = gr.Textbox(label="Steps Taken", value="0", interactive=False)
                        reward_box = gr.Textbox(label="Cumulative Reward", value="0.000", interactive=False)

            reset_btn.click(
                reset_environment, inputs=[task_select],
                outputs=[status_box, result_box, steps_box, reward_box],
            )
            step_btn.click(
                take_step,
                inputs=[action_type, action_path, action_query, action_content],
                outputs=[status_box, result_box, steps_box, reward_box],
            )

        # ── Tab 2: Run Agent ─────────────────────────────────────────────
        with gr.TabItem("🤖 Run Agent"):
            gr.Markdown(
                "### Built-in Demonstration Agent\n"
                "Runs a deterministic read-all-then-submit agent. "
                "For LLM-based agent, use `run_agent.py` or `inference.py`."
            )
            agent_task = gr.Dropdown(
                choices=["task1", "task2", "task3"], value="task1", label="Task"
            )
            run_btn = gr.Button("🚀 Run Agent", variant="primary")
            agent_output = gr.Textbox(label="Agent Log", lines=20, interactive=False)
            run_btn.click(run_builtin_agent, inputs=[agent_task], outputs=[agent_output])

        # ── Tab 3: Evaluation Dashboard ──────────────────────────────────
        with gr.TabItem("📊 Evaluation"):
            with gr.Row():
                eval_btn = gr.Button("🎯 Get Evaluation", variant="primary")
                metrics_btn = gr.Button("📈 Get Metrics", variant="secondary")
                traj_btn = gr.Button("🗺️ Get Trajectory", variant="secondary")
            eval_output = gr.Textbox(label="Evaluation Report", lines=25, interactive=False)
            eval_btn.click(get_evaluation, outputs=[eval_output])
            metrics_btn.click(get_metrics, outputs=[eval_output])
            traj_btn.click(get_trajectory, outputs=[eval_output])

        # ── Tab 4: API Docs ──────────────────────────────────────────────
        with gr.TabItem("📖 API"):
            gr.Markdown("""
### REST API Endpoints

The FastAPI endpoints are mounted alongside this UI at `/api/`.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/reset?task=task1` | POST | Start new episode |
| `/api/step` | POST | Take action (JSON body) |
| `/api/state` | GET | Get current state |
| `/api/health` | GET | Health check |
| `/api/trajectory` | GET | Full action log |
| `/api/evaluate` | GET | Multi-dimensional scores |
| `/api/metrics` | GET | Comprehensive stats |
| `/api/fault-config` | POST | Enable fault injection |

### Example: Reset + Read + Submit
```bash
BASE="https://YOUR-SPACE.hf.space/api"

# Reset
curl -X POST "$BASE/reset?task=task1"

# Read a file
curl -X POST "$BASE/step" -H "Content-Type: application/json" \\
  -d '{"action_type":"read_file","path":"src/auth.py"}'

# Submit
curl -X POST "$BASE/step" -H "Content-Type: application/json" \\
  -d '{"action_type":"submit"}'

# Get evaluation
curl "$BASE/evaluate"
```
""")


# ── Mount FastAPI under /api ─────────────────────────────────────────────────
from server.app import app as fastapi_app

gr_app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(fastapi_app, host="0.0.0.0", port=7860)
