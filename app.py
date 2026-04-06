#!/usr/bin/env python3
"""
app.py — Gradio UI v3.0 — Full Platform Entry Point

Tabs:
  🎮 Interactive          — manual step-by-step control
  🤖 Run Agent            — built-in deterministic agent demo
  📊 Evaluation           — 6-dimension evaluation report
  🧠 Intelligence         — failure classification, strategy, advanced metrics
  🔁 Self-Improve         — improvement plan after failure
  ⚖️ Compare Agents       — side-by-side multi-agent comparison
  🌐 3D Visualizer        — Three.js trajectory visualization
  📖 API                  — REST API reference
"""
import os
import json
import gradio as gr
from server.environment import CodebaseNavEnvironment
from server.models import RepoAction
from server.failure_classifier import FailureClassifier
from server.strategy_detector import StrategyDetector
from server.advanced_metrics import AdvancedMetricsEngine
from server.self_improvement import SelfImprovementEngine
from server.multi_agent import MultiAgentComparison

# ── Global instances ──────────────────────────────────────────────────────────
env = CodebaseNavEnvironment()
failure_clf = FailureClassifier()
strategy_det = StrategyDetector()
adv_metrics_engine = AdvancedMetricsEngine()
improvement_engine = SelfImprovementEngine()
multi_agent_engine = MultiAgentComparison()


# ── Tab 1: Interactive ────────────────────────────────────────────────────────

def reset_environment(task: str):
    try:
        result = env.reset(task=task)
        obs = result.observation
        tree = "\n".join(f"  📄 {f}" for f in obs.repo_tree)
        failing = ", ".join(obs.failing_tests) if obs.failing_tests else "None listed"
        fi = result.info.get("fault_injection", {})
        faults = ""
        if fi.get("faults_injected"):
            faults = f"\n\n⚠️ Fault Injection ({fi.get('difficulty_multiplier', 1.0):.1f}x):\n"
            faults += "\n".join(f"  • {f}" for f in fi["faults_injected"][:5])

        status = (
            f"✅ Episode Started — {task} (variant: {result.info.get('variant_id', '?')})\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Steps: {obs.steps_remaining} remaining\n\n"
            f"📁 Files:\n{tree}\n\n"
            f"🔴 Failing Tests: {failing}\n\n"
            f"📋 Task: {obs.task_description}{faults}"
        )
        return status, "", "0", "0.000"
    except Exception as e:
        return f"❌ Error: {e}", "", "0", "0.000"


def take_step(action_type: str, path: str, query: str, content: str):
    if env.done:
        return "❌ Episode done. Reset first.", "", "", ""
    try:
        action = RepoAction(
            action_type=action_type,
            path=path.strip() or None,
            query=query.strip() or None,
            content=content.strip() or None,
        )
        result = env.step(action)
        obs = result.observation
        result_text = obs.last_action_result or "No output"
        error = f"\n⚠️ {obs.last_action_error}" if obs.last_action_error else ""
        flags = result.info.get("security_flags", [])
        sec = f"\n🔒 Security: {flags}" if flags else ""

        status = (
            f"Step {result.info['steps_taken']} | "
            f"Reward: {result.reward:+.3f} | "
            f"Steps left: {obs.steps_remaining}{error}{sec}"
        )
        if result.done:
            status += f"\n\n🏁 DONE — Score: {result.info['final_score']:.3f}"

        return (
            status,
            result_text[:3000],
            str(result.info["steps_taken"]),
            f"{result.info.get('cumulative_reward', 0):.3f}",
        )
    except Exception as e:
        return f"❌ Error: {e}", "", "", ""


# ── Tab 2: Run Agent ──────────────────────────────────────────────────────────

def run_builtin_agent(task: str):
    try:
        result = env.reset(task=task)
        obs = result.observation
        log = [
            f"🚀 {task} (variant: {result.info.get('variant_id')})",
            f"   Files: {obs.repo_tree}",
            f"   Failing: {obs.failing_tests}",
        ]
        tree = obs.repo_tree
        test_files = sorted([f for f in tree if f.startswith("tests/")])
        src_files = sorted([f for f in tree if f.startswith("src/") and f.endswith(".py")])
        spec_files = sorted([f for f in tree if f.endswith(".md")])
        steps = 0

        if task == "task3" and spec_files:
            for sf in spec_files:
                if env.done: break
                r = env.step(RepoAction(action_type="read_file", path=sf))
                steps += 1
                log.append(f"   Step {steps}: read_file {sf} → {r.reward:+.3f}")

        for tf in test_files:
            if env.done: break
            r = env.step(RepoAction(action_type="read_file", path=tf))
            steps += 1
            log.append(f"   Step {steps}: read_file {tf} → {r.reward:+.3f}")

        for sf in src_files:
            if env.done or steps >= 12: break
            r = env.step(RepoAction(action_type="read_file", path=sf))
            steps += 1
            log.append(f"   Step {steps}: read_file {sf} → {r.reward:+.3f}")

        if not env.done and test_files:
            r = env.step(RepoAction(action_type="run_tests", path=test_files[0]))
            steps += 1
            log.append(f"   Step {steps}: run_tests → {r.reward:+.3f}")

        if not env.done:
            r = env.step(RepoAction(action_type="submit"))
            steps += 1
            log.append(f"   Step {steps}: submit → {r.reward:+.3f}")

        log += [
            f"\n🏁 Score: {env.final_score:.3f}",
            f"   Steps: {steps}",
            f"   Reward: {env.cumulative_reward:.3f}",
        ]
        return "\n".join(log)
    except Exception as e:
        return f"❌ Error: {e}"


# ── Tab 3: Evaluation ─────────────────────────────────────────────────────────

def get_evaluation():
    try:
        ev = env.get_evaluation()
        if "error" in ev:
            return "No evaluation available. Run an episode first."
        lines = [
            f"🎯 Composite Score: {ev['composite_score']:.3f}",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ]
        for name, dim in ev.get("dimensions", {}).items():
            bar = "█" * int(dim["score"] * 20) + "░" * (20 - int(dim["score"] * 20))
            lines.append(f"  {name:15s} [{bar}] {dim['score']:.3f}")
            for e in dim.get("evidence", [])[:2]:
                lines.append(f"    → {e}")
        if ev.get("strengths"):
            lines += ["\n💪 Strengths:"] + [f"  ✅ {s}" for s in ev["strengths"]]
        if ev.get("failure_analysis"):
            lines += ["\n⚠️ Failures:"] + [f"  ❌ {f}" for f in ev["failure_analysis"]]
        if ev.get("recommendations"):
            lines += ["\n💡 Recommendations:"] + [f"  → {r}" for r in ev["recommendations"]]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def get_metrics():
    try:
        return json.dumps(env.get_metrics(), indent=2, default=str)
    except Exception as e:
        return f"Error: {e}"


def get_trajectory():
    try:
        t = env.get_trajectory()
        if not t:
            return "No trajectory. Run an episode first."
        lines = [
            f"Episode: {t.get('episode_id')}",
            f"Task: {t.get('task')} | Variant: {t.get('variant_id')}",
            f"Score: {t.get('final_score', 0):.3f} | Duration: {t.get('duration_seconds', '?')}s",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ]
        emojis = {"read_file": "📖", "write_file": "✏️", "run_tests": "🧪",
                  "search_code": "🔍", "submit": "🏁"}
        for step in t.get("steps", []):
            em = emojis.get(step["action_type"], "•")
            p = step.get("action_path") or step.get("action_query") or ""
            err = " ❌" if step.get("error") else ""
            lines.append(
                f"  {em} {step['step_number']:2d}: {step['action_type']:12s} {p:30s} "
                f"reward={step['reward']:+.3f} ({step['duration_ms']:.0f}ms){err}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


# ── Tab 4: Intelligence ───────────────────────────────────────────────────────

def get_failure_classification():
    try:
        traj = env.get_trajectory()
        if not traj:
            return "No trajectory. Run an episode first."
        meta = env.variant.meta if env.variant else {}
        report = failure_clf.classify(
            episode_id=traj.get("episode_id", ""),
            task=env.current_task or "unknown",
            trajectory_steps=traj.get("steps", []),
            variant_meta=meta,
            files_read=list(env.files_read),
            files_written=list(env.files_written),
            final_score=env.final_score,
            security_violations=env.security_violations,
        )
        d = report.to_dict()
        lines = [
            f"{'✅ SUCCESS' if d['success'] else '❌ FAILURE'}",
            f"Primary Failure Type: {d['primary_failure']}",
            f"Failures Detected: {d['failure_count']}",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ]
        for f in d.get("failures", []):
            lines += [
                f"\n[{f['severity'].upper()}] {f['type']} @ Step {f['step']}",
                f"  Evidence: {f['evidence']}",
                f"  Root Cause: {f['root_cause']}",
                f"  Fix: {f['remediation']}",
            ]
        if d.get("failure_summary"):
            lines += ["\n📋 Summary:", f"  {d['failure_summary']}"]
        if d.get("retry_hint"):
            lines += ["\n🔁 Retry Hint:", f"  {d['retry_hint']}"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def get_strategy_detection():
    try:
        traj = env.get_trajectory()
        if not traj:
            return "No trajectory. Run an episode first."
        meta = env.variant.meta if env.variant else {}
        report = strategy_det.detect(
            trajectory_steps=traj.get("steps", []),
            task=env.current_task or "unknown",
            variant_meta=meta,
            files_read=list(env.files_read),
            final_score=env.final_score,
        )
        d = report.to_dict()
        score_bar = "█" * int(d["score"] * 20) + "░" * (20 - int(d["score"] * 20))
        lines = [
            f"🧭 Strategy: {d['strategy']}",
            f"   Score:  [{score_bar}] {d['score']:.3f}",
            f"   Confidence: {d['confidence']:.0%}",
            f"\n📖 {d['strategy_description']}",
            f"\n📊 Exploration Ratio: {d['exploration_ratio']:.2f} "
            f"({'explore-heavy' if d['exploration_ratio'] > 0.6 else 'exploit-heavy' if d['exploration_ratio'] < 0.4 else 'balanced'})",
            f"   Strategy Pivots: {d['pivot_count']}",
        ]
        if d.get("sub_patterns"):
            lines += ["\n🔖 Sub-patterns:"] + [f"  • {p}" for p in d["sub_patterns"]]
        if d.get("evidence"):
            lines += ["\n🔍 Evidence:"] + [f"  → {e}" for e in d["evidence"]]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def get_advanced_metrics():
    try:
        traj = env.get_trajectory()
        if not traj:
            return "No trajectory. Run an episode first."
        meta = env.variant.meta if env.variant else {}
        report = adv_metrics_engine.compute(
            trajectory_steps=traj.get("steps", []),
            variant_meta=meta,
            final_score=env.final_score,
            files_read=list(env.files_read),
            files_written=list(env.files_written),
        )
        d = report.to_dict()

        def bar(v):
            return "█" * int(v * 20) + "░" * (20 - int(v * 20))

        lines = [
            "⚡ ADVANCED METRICS",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"  Reasoning Efficiency  [{bar(d['reasoning_efficiency'])}] {d['reasoning_efficiency']:.3f}",
            f"  Reliability Index     [{bar(d['reliability_index'])}] {d['reliability_index']:.3f}",
            f"  Exploration Ratio     [{bar(d['exploration_ratio'])}] {d['exploration_ratio']:.3f}",
            f"  Decision Entropy      [{bar(d['decision_entropy'])}] {d['decision_entropy']:.3f}",
            f"  Wasteful Ratio        [{bar(d['wasteful_ratio'])}] {d['wasteful_ratio']:.3f}",
            f"  Pivot Rate            {d['pivot_rate']:.2f} per 10 steps",
            f"  Consistency           [{bar(d['consistency_score'])}] {d['consistency_score']:.3f} ({d['runs_analyzed']} runs)",
            "\n📊 Action Distribution:",
        ]
        for action, count in d.get("action_distribution", {}).items():
            lines.append(f"  {action:15s}: {count}")
        if d.get("useful_actions"):
            lines += ["\n✅ Useful Actions:"] + [f"  • {a}" for a in d["useful_actions"]]
        if d.get("wasteful_actions"):
            lines += ["\n⚠️ Wasteful Actions:"] + [f"  • {a}" for a in d["wasteful_actions"]]
        lines += ["\n🔒 Reliability Breakdown:"]
        for k, v in d.get("reliability_breakdown", {}).items():
            lines.append(f"  {k:15s}: {v:.3f}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


# ── Tab 5: Self-Improve ───────────────────────────────────────────────────────

def get_improvement_plan():
    try:
        traj = env.get_trajectory()
        if not traj:
            return "No trajectory. Run an episode first."
        meta = env.variant.meta if env.variant else {}
        steps = traj.get("steps", [])

        fail_report = failure_clf.classify(
            episode_id=traj.get("episode_id", ""),
            task=env.current_task or "unknown",
            trajectory_steps=steps,
            variant_meta=meta,
            files_read=list(env.files_read),
            files_written=list(env.files_written),
            final_score=env.final_score,
            security_violations=env.security_violations,
        )
        plan = improvement_engine.generate_improvement_plan(
            episode_id=traj.get("episode_id", ""),
            task=env.current_task or "unknown",
            failure_type=fail_report.primary_failure,
            failure_evidence=[f.evidence for f in fail_report.failures],
            original_score=env.final_score,
            trajectory_steps=steps,
            files_read=list(env.files_read),
            files_written=list(env.files_written),
        )
        d = plan.to_dict()
        lines = [
            f"🔁 SELF-IMPROVEMENT PLAN",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"Original Score: {d['original_score']:.3f}",
            f"Failure Type: {d['failure_type']}",
            f"\n❌ What Went Wrong:\n  {d['what_went_wrong']}",
            f"\n🎯 Improved Strategy:\n  {d['improved_strategy']}",
            f"\n📋 Step-by-Step Plan:",
        ]
        for step in d.get("step_by_step_plan", []):
            lines.append(f"  {step}")
        if d.get("specific_errors"):
            lines += ["\n🔎 Specific Errors:"] + [f"  • {e}" for e in d["specific_errors"][:5]]
        lines += [
            "\n💉 System Prompt Injection (for next LLM run):",
            "─────────────────────────────────────",
            d.get("system_prompt_addon", "No injection needed."),
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


# ── Tab 6: Compare Agents ─────────────────────────────────────────────────────

def run_comparison(task: str, selected_agents: list):
    try:
        agents = selected_agents if selected_agents else None
        report = multi_agent_engine.compare(env, task=task, agents=agents)
        d = report.to_dict()

        lines = [
            f"⚖️ MULTI-AGENT COMPARISON — {task} (variant: {d.get('variant_id')})",
            f"🏆 Winner: {d.get('winner')} (score: {d.get('winner_score', 0):.3f})",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"{'Rank':<6} {'Agent':<16} {'Score':<8} {'Steps':<8} {'Strategy':<22} {'Failure':<22} {'Reliability':<12}",
            "─" * 100,
        ]
        for row in d.get("summary_table", []):
            lines.append(
                f"#{row['rank']:<5} {row['agent']:<16} {row['score']:<8.3f} "
                f"{row['steps']:<8} {row['strategy']:<22} {row['failure']:<22} {row['reliability']:<12.3f}"
            )
        lines.append("━" * 100)

        if d.get("insights"):
            lines += ["\n💡 Insights:"] + [f"  → {i}" for i in d["insights"]]

        lines.append("\n📊 Per-Agent Action Sequences:")
        for run in d.get("detailed_runs", []):
            seq = " → ".join(run.get("action_sequence", []))
            lines.append(f"  {run['agent_name']:16s}: {seq}")

        return "\n".join(lines)
    except Exception as e:
        return f"❌ Error: {e}"


# ── Tab 7: 3D Visualizer ──────────────────────────────────────────────────────

def get_viz_html():
    """Generate the 3D visualizer HTML with current trajectory data injected."""
    # Load the static HTML template
    static_path = os.path.join(os.path.dirname(__file__), "static", "viz3d.html")
    if not os.path.exists(static_path):
        return "<p style='color:red'>viz3d.html not found in static/</p>"

    with open(static_path, "r") as f:
        html = f.read()

    # Get viz data from current environment
    traj = env.get_trajectory()
    if traj:
        meta = env.variant.meta if env.variant else {}
        bug_files = set(meta.get("bug_files", []))
        files = []
        if env.variant:
            for fname in env.variant.get_tree():
                ftype = "test" if fname.startswith("tests/") else \
                        "spec" if fname.endswith(".md") else "src"
                files.append({
                    "name": fname,
                    "type": ftype,
                    "is_bug_file": fname in bug_files,
                    "visited": fname in env.files_read,
                    "modified": fname in env.files_written,
                })

        test_files = [f["name"] for f in files if f["type"] == "test"]
        src_files = [f["name"] for f in files if f["type"] == "src"]
        deps = []
        for tf in test_files:
            for sf in src_files:
                deps.append({"from": tf, "to": sf})

        steps_data = []
        for step in traj.get("steps", []):
            steps_data.append({
                "step": step.get("step_number", 0),
                "action": step.get("action_type", ""),
                "path": step.get("action_path"),
                "reward": step.get("reward", 0.0),
                "error": step.get("error"),
                "pass_rate": step.get("test_pass_rate"),
            })

        strategy_report = strategy_det.detect(
            traj.get("steps", []),
            env.current_task or "unknown",
            meta,
            list(env.files_read),
            env.final_score,
        ) if traj.get("steps") else None

        viz_data = {
            "task": env.current_task or "unknown",
            "variant_id": traj.get("variant_id", "unknown"),
            "final_score": env.final_score,
            "strategy": strategy_report.strategy if strategy_report else "UNKNOWN",
            "failure_type": "—",
            "files": files,
            "dependencies": deps,
            "steps": steps_data,
        }
        data_json = json.dumps(viz_data)
    else:
        data_json = ""

    # Inject data into HTML
    html = html.replace(
        '<div id="viz-data" style="display:none"></div>',
        f'<div id="viz-data" style="display:none">{data_json}</div>'
    )
    return html


# ── Build Gradio UI ───────────────────────────────────────────────────────────

with gr.Blocks(title="Codebase Navigation & Repair — OpenEnv v3") as demo:
    gr.Markdown(
        "# 🔍 Codebase Navigation & Repair — OpenEnv v3\n"
        "**The most advanced debugging + evaluation platform for AI coding agents.** "
        "Navigate codebases · Fix bugs · Evaluate process · Visualize in 3D."
    )

    with gr.Tabs():

        # ── Tab 1: Interactive ────────────────────────────────────────────────
        with gr.TabItem("🎮 Interactive"):
            with gr.Row():
                with gr.Column(scale=1):
                    task_select = gr.Dropdown(
                        ["task1", "task2", "task3"], value="task1",
                        label="Task",
                        info="task1=bugs, task2=cross-module, task3=feature impl"
                    )
                    reset_btn = gr.Button("🔄 Reset Environment", variant="primary")
                    gr.Markdown("### Action")
                    act_type = gr.Dropdown(
                        ["read_file", "write_file", "run_tests", "search_code", "submit"],
                        value="read_file", label="Action Type",
                    )
                    act_path = gr.Textbox(label="Path", placeholder="src/auth.py")
                    act_query = gr.Textbox(label="Query (search_code)", placeholder="validate_token")
                    act_content = gr.Textbox(label="Content (write_file)", lines=4)
                    step_btn = gr.Button("▶️ Execute Step", variant="secondary")
                with gr.Column(scale=2):
                    status_box = gr.Textbox(label="Status", lines=14, interactive=False)
                    result_box = gr.Textbox(label="Last Result", lines=8, interactive=False)
                    with gr.Row():
                        steps_box = gr.Textbox(label="Steps", value="0", interactive=False)
                        reward_box = gr.Textbox(label="Cumulative Reward", value="0.000", interactive=False)
            reset_btn.click(reset_environment, [task_select], [status_box, result_box, steps_box, reward_box])
            step_btn.click(take_step, [act_type, act_path, act_query, act_content], [status_box, result_box, steps_box, reward_box])

        # ── Tab 2: Run Agent ──────────────────────────────────────────────────
        with gr.TabItem("🤖 Run Agent"):
            gr.Markdown("### Built-in Demonstration Agent\nRuns deterministic read→submit strategy.")
            agent_task = gr.Dropdown(["task1", "task2", "task3"], value="task1", label="Task")
            run_btn = gr.Button("🚀 Run Agent", variant="primary")
            agent_output = gr.Textbox(label="Agent Log", lines=20, interactive=False)
            run_btn.click(run_builtin_agent, [agent_task], [agent_output])

        # ── Tab 3: Evaluation ─────────────────────────────────────────────────
        with gr.TabItem("📊 Evaluation"):
            with gr.Row():
                eval_btn = gr.Button("🎯 Evaluation Report", variant="primary")
                metrics_btn = gr.Button("📈 Metrics JSON", variant="secondary")
                traj_btn = gr.Button("🗺️ Trajectory", variant="secondary")
            eval_out = gr.Textbox(label="Output", lines=28, interactive=False)
            eval_btn.click(get_evaluation, outputs=[eval_out])
            metrics_btn.click(get_metrics, outputs=[eval_out])
            traj_btn.click(get_trajectory, outputs=[eval_out])

        # ── Tab 4: 🧠 Intelligence ─────────────────────────────────────────────
        with gr.TabItem("🧠 Intelligence"):
            gr.Markdown(
                "### Deep Agent Intelligence Analysis\n"
                "Failure classification, strategy detection, and advanced behavioral metrics."
            )
            with gr.Row():
                classify_btn = gr.Button("🔬 Classify Failure", variant="primary")
                strategy_btn = gr.Button("🧭 Detect Strategy", variant="secondary")
                adv_btn = gr.Button("⚡ Advanced Metrics", variant="secondary")
            intel_out = gr.Textbox(label="Analysis", lines=32, interactive=False)
            classify_btn.click(get_failure_classification, outputs=[intel_out])
            strategy_btn.click(get_strategy_detection, outputs=[intel_out])
            adv_btn.click(get_advanced_metrics, outputs=[intel_out])

        # ── Tab 5: 🔁 Self-Improve ─────────────────────────────────────────────
        with gr.TabItem("🔁 Self-Improve"):
            gr.Markdown(
                "### Self-Improvement Loop\n"
                "After a failure, this generates an actionable improvement plan and a "
                "system prompt injection for the agent's next attempt."
            )
            improve_btn = gr.Button("🔁 Generate Improvement Plan", variant="primary")
            improve_out = gr.Textbox(label="Improvement Plan", lines=32, interactive=False)
            improve_btn.click(get_improvement_plan, outputs=[improve_out])

        # ── Tab 6: ⚖️ Compare ──────────────────────────────────────────────────
        with gr.TabItem("⚖️ Compare Agents"):
            gr.Markdown(
                "### Multi-Agent Strategy Comparison\n"
                "Runs 4 built-in agent strategies on the same task to compare "
                "efficiency, strategy, and reliability side-by-side."
            )
            with gr.Row():
                comp_task = gr.Dropdown(["task1", "task2", "task3"], value="task1", label="Task")
                comp_agents = gr.CheckboxGroup(
                    ["test-first", "search-first", "minimal", "exhaustive"],
                    value=["test-first", "search-first", "minimal", "exhaustive"],
                    label="Agents to Compare",
                )
            comp_btn = gr.Button("⚖️ Run Comparison", variant="primary")
            comp_out = gr.Textbox(label="Comparison Report", lines=30, interactive=False)
            comp_btn.click(run_comparison, [comp_task, comp_agents], [comp_out])

        # ── Tab 7: 🌐 3D Visualizer ────────────────────────────────────────────
        with gr.TabItem("🌐 3D Visualizer"):
            gr.Markdown(
                "### Agent Trajectory 3D Visualization\n"
                "Files = 3D nodes · Dependencies = edges · Agent path = animated beam · "
                "Timeline = scrubbable replay. **Run an episode first, then refresh.**"
            )
            refresh_viz_btn = gr.Button("🔄 Load Trajectory into Visualizer", variant="primary")
            viz_html = gr.HTML(value="<p style='color:#64748b;text-align:center;padding:40px'>Click 'Load Trajectory' after running an episode.</p>")
            refresh_viz_btn.click(get_viz_html, outputs=[viz_html])

        # ── Tab 8: API ────────────────────────────────────────────────────────
        with gr.TabItem("📖 API"):
            gr.Markdown("""
### REST API — v3.0 Endpoints

#### Core (OpenEnv-compliant)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset?task=task1` | POST | Start new episode |
| `/step` | POST | Take action |
| `/state` | GET | Current state |
| `/health` | GET | Health check |

#### Evaluation
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/trajectory` | GET | Full action log |
| `/evaluate` | GET | 6-dimension scores |
| `/metrics` | GET | Memory + security stats |
| `/fault-config` | POST | Enable fault injection |

#### Intelligence (NEW in v3)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/classify` | GET | Typed failure classification |
| `/strategy` | GET | Behavioral strategy detection |
| `/advanced-metrics` | GET | Entropy, reliability, consistency |
| `/improvement-plan` | GET | Self-improvement feedback |
| `/compare-agents` | POST | Multi-agent comparison |
| `/viz-data` | GET | 3D visualization data |

```bash
BASE="http://localhost:7860"
curl -X POST "$BASE/reset?task=task1"
curl -X POST "$BASE/step" -H "Content-Type: application/json" -d '{"action_type":"read_file","path":"src/auth.py"}'
curl -X POST "$BASE/step" -d '{"action_type":"submit"}'
curl "$BASE/classify"
curl "$BASE/strategy"
curl "$BASE/advanced-metrics"
curl "$BASE/improvement-plan"
curl -X POST "$BASE/compare-agents?task=task1"
```
""")


# ── Mount FastAPI under same process ──────────────────────────────────────────
from server.app import app as fastapi_app
gr_app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(fastapi_app, host="0.0.0.0", port=7860)
