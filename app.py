#!/usr/bin/env python3
"""
app.py — Gradio UI v4.0 — Full Research Platform

13 tabs:
  🎮 Interactive        — manual control
  🤖 Run Agent          — deterministic demo agent
  📊 Evaluation         — 6-dimension process evaluation
  🧠 Intelligence       — failure, strategy, advanced metrics
  🔁 Self-Improve       — improvement plan with prompt injection
  ⚖️ Compare Agents     — multi-agent strategy comparison
  🌐 3D Visualizer      — Three.js trajectory viz (FIXED: iframe)
  🧪 Causal Probe       — causal reasoning vs guessing
  🎭 Counterfactual     — brittleness / robustness testing
  📐 Confidence         — calibration: overconfident vs underconfident
  🏆 Benchmark          — automated leaderboard
  📈 Analytics          — unified research-grade report
  📖 API                — REST reference
"""
import os
import json
import gradio as gr
from server.app import (
    app as fastapi_app,
    env,
    failure_clf,
    strategy_det,
    adv_metrics as adv_metrics_engine,
    improvement as improvement_engine,
    multi_agent as multi_agent_engine,
    _causal as causal_probe,
    _counter as counterfactual_engine,
    _calibrator as confidence_calibrator,
    _benchmark as benchmark_runner,
    _analytics as analytics_engine,
)
from server.models import RepoAction
from server.memory_bank import get_global_memory

# ── Global instances ──────────────────────────────────────────────────────────
# All engines and the environment are imported from server.app so that 
# Gradio interactions and direct HTTP REST calls use the exact same state.
memory_bank = get_global_memory()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_traj_and_meta():
    traj = env.get_trajectory()
    if not traj:
        return None, None, None, None
    meta = env.variant.meta if env.variant else {}
    steps = traj.get("steps", [])
    return traj, meta, steps, traj.get("episode_id", "")


def _no_traj():
    return "⚠️ No trajectory. Run an episode first (Interactive or Run Agent tab)."


# ── Tab 1: Interactive ────────────────────────────────────────────────────────

def reset_environment(task):
    try:
        result = env.reset(task=task)
        obs = result.observation
        tree = "\n".join(f"  📄 {f}" for f in obs.repo_tree)
        failing = ", ".join(obs.failing_tests) if obs.failing_tests else "None"
        fi = result.info.get("fault_injection", {})
        faults = ""
        if fi.get("faults_injected"):
            faults = f"\n\n⚠️ Fault Injection ({fi.get('difficulty_multiplier',1):.1f}×):\n"
            faults += "\n".join(f"  • {f}" for f in fi["faults_injected"][:5])
        status = (
            f"✅ Episode started — {task} (variant: {result.info.get('variant_id','?')})\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Steps remaining: {obs.steps_remaining}\n\n"
            f"📁 Files:\n{tree}\n\n"
            f"🔴 Failing Tests: {failing}\n\n"
            f"📋 {obs.task_description}{faults}"
        )
        return status, "", "0", "0.000"
    except Exception as e:
        return f"❌ Error: {e}", "", "0", "0.000"


def take_step(action_type, path, query, content):
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
        result_text = obs.last_action_result or ""
        err = f"\n⚠️ {obs.last_action_error}" if obs.last_action_error else ""
        flags = result.info.get("security_flags", [])
        sec = f"\n🔒 {flags}" if flags else ""
        status = (
            f"Step {result.info['steps_taken']} | Reward: {result.reward:+.3f} | "
            f"Left: {obs.steps_remaining}{err}{sec}"
        )
        if result.done:
            status += f"\n\n🏁 DONE — Score: {result.info['final_score']:.3f}"
        return status, result_text[:3000], str(result.info["steps_taken"]), f"{result.info.get('cumulative_reward',0):.3f}"
    except Exception as e:
        return f"❌ {e}", "", "", ""


# ── Tab 2: Run Agent ──────────────────────────────────────────────────────────

def run_builtin_agent(task):
    try:
        result = env.reset(task=task)
        obs = result.observation
        tree = obs.repo_tree
        log = [f"🚀 {task} (variant: {result.info.get('variant_id')})", f"   Files: {tree}"]
        test_files = sorted([f for f in tree if f.startswith("tests/")])
        src_files = sorted([f for f in tree if f.startswith("src/") and f.endswith(".py")])
        spec_files = sorted([f for f in tree if f.endswith(".md")])
        steps = 0

        if task == "task3" and spec_files:
            for sf in spec_files[:2]:
                if env.done: break
                r = env.step(RepoAction(action_type="read_file", path=sf))
                steps += 1; log.append(f"   Step {steps}: read_file {sf} → {r.reward:+.3f}")

        for tf in test_files:
            if env.done: break
            r = env.step(RepoAction(action_type="read_file", path=tf))
            steps += 1; log.append(f"   Step {steps}: read_file {tf} → {r.reward:+.3f}")

        if not env.done:
            r = env.step(RepoAction(action_type="search_code", query="def "))
            steps += 1; log.append(f"   Step {steps}: search_code → {r.reward:+.3f}")

        for sf in src_files:
            if env.done or steps >= 14: break
            r = env.step(RepoAction(action_type="read_file", path=sf))
            steps += 1; log.append(f"   Step {steps}: read_file {sf} → {r.reward:+.3f}")

        if not env.done and test_files:
            r = env.step(RepoAction(action_type="run_tests", path=test_files[0]))
            steps += 1; log.append(f"   Step {steps}: run_tests → {r.reward:+.3f}")

        if not env.done:
            r = env.step(RepoAction(action_type="submit"))
            steps += 1; log.append(f"   Step {steps}: submit → {r.reward:+.3f}")

        log += ["", f"🏁 Score: {env.final_score:.3f} | Steps: {steps} | Reward: {env.cumulative_reward:.3f}"]

        # Store in memory
        traj = env.get_trajectory()
        if traj:
            meta = env.variant.meta if env.variant else {}
            fail_r = failure_clf.classify(
                traj.get("episode_id",""), task, traj.get("steps",[]), meta,
                list(env.files_read), list(env.files_written), env.final_score
            )
            strat_r = strategy_det.detect(traj.get("steps",[]), task, meta, list(env.files_read), env.final_score)
            imp_plan = improvement_engine.generate_improvement_plan(
                traj.get("episode_id",""), task, fail_r.primary_failure,
                [], env.final_score, traj.get("steps",[]),
                list(env.files_read), list(env.files_written)
            )
            memory_bank.store(
                traj.get("episode_id",""), task, fail_r.primary_failure,
                fail_r.failure_summary or "", env.final_score,
                strat_r.strategy, traj.get("steps",[]), imp_plan.to_dict()
            )
            log.append(f"💾 Stored lesson in memory bank ({memory_bank.get_stats()['total_entries']} total)")

        return "\n".join(log)
    except Exception as e:
        return f"❌ {e}"


# ── Tab 3: Evaluation ─────────────────────────────────────────────────────────

def get_evaluation():
    try:
        ev = env.get_evaluation()
        if "error" in ev:
            return _no_traj()
        lines = [f"🎯 Composite Score: {ev['composite_score']:.3f}", "━"*50]
        for name, dim in ev.get("dimensions", {}).items():
            bar = "█" * int(dim["score"]*20) + "░" * (20-int(dim["score"]*20))
            lines.append(f"  {name:15s} [{bar}] {dim['score']:.3f}")
            for e in dim.get("evidence",[])[:2]:
                lines.append(f"    → {e}")
        if ev.get("strengths"):
            lines += ["\n💪 Strengths:"] + [f"  ✅ {s}" for s in ev["strengths"]]
        if ev.get("failure_analysis"):
            lines += ["\n⚠️ Failures:"] + [f"  ❌ {f}" for f in ev["failure_analysis"]]
        if ev.get("recommendations"):
            lines += ["\n💡 Recs:"] + [f"  → {r}" for r in ev["recommendations"]]
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
        if not t: return _no_traj()
        lines = [
            f"Episode: {t.get('episode_id')}", f"Task: {t.get('task')} | Variant: {t.get('variant_id')}",
            f"Score: {t.get('final_score',0):.3f} | Duration: {t.get('duration_seconds','?')}s", "━"*60,
        ]
        em = {"read_file":"📖","write_file":"✏️","run_tests":"🧪","search_code":"🔍","submit":"🏁"}
        for step in t.get("steps",[]):
            p = step.get("action_path") or step.get("action_query") or ""
            err = " ❌" if step.get("error") else ""
            lines.append(f"  {em.get(step['action_type'],'•')} {step['step_number']:2d}: {step['action_type']:12s} {p:25s} reward={step['reward']:+.3f}{err}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


# ── Tab 4: Intelligence ───────────────────────────────────────────────────────

def get_failure_classification():
    try:
        traj, meta, steps, ep_id = _get_traj_and_meta()
        if not traj: return _no_traj()
        r = failure_clf.classify(ep_id, env.current_task or "?", steps, meta,
                                  list(env.files_read), list(env.files_written), env.final_score)
        d = r.to_dict()
        lines = [
            f"{'✅ SUCCESS' if d['success'] else '❌ FAILURE'}",
            f"Primary: {d['primary_failure']} | Count: {d['failure_count']}", "━"*50,
        ]
        for f in d.get("failures",[]):
            lines += [f"\n[{f['severity'].upper()}] {f['type']} @ step {f['step']}",
                      f"  Evidence: {f['evidence']}", f"  Fix: {f['remediation']}"]
        if d.get("failure_summary"):
            lines += ["\n📋 Summary:", f"  {d['failure_summary']}"]
        if d.get("retry_hint"):
            lines += [f"\n🔁 Retry hint: {d['retry_hint']}"]
        return "\n".join(lines)
    except Exception as e: return f"Error: {e}"


def get_strategy_detection():
    try:
        traj, meta, steps, _ = _get_traj_and_meta()
        if not traj: return _no_traj()
        r = strategy_det.detect(steps, env.current_task or "?", meta, list(env.files_read), env.final_score)
        d = r.to_dict()
        bar = "█"*int(d["score"]*20)+"░"*(20-int(d["score"]*20))
        lines = [
            f"🧭 Strategy: {d['strategy']}", f"   [{bar}] {d['score']:.3f} (confidence: {d['confidence']:.0%})",
            f"\n{d['strategy_description']}",
            f"\nExploration: {d['exploration_ratio']:.2f} | Pivots: {d['pivot_count']}",
        ]
        if d.get("sub_patterns"): lines += ["\nSub-patterns:"] + [f"  • {p}" for p in d["sub_patterns"]]
        if d.get("evidence"): lines += ["\nEvidence:"] + [f"  → {e}" for e in d["evidence"]]
        return "\n".join(lines)
    except Exception as e: return f"Error: {e}"


def get_advanced_metrics():
    try:
        traj, meta, steps, _ = _get_traj_and_meta()
        if not traj: return _no_traj()
        r = adv_metrics_engine.compute(steps, meta, env.final_score, list(env.files_read), list(env.files_written))
        d = r.to_dict()
        def bar(v): return "█"*int(v*20)+"░"*(20-int(v*20))
        lines = ["⚡ ADVANCED METRICS", "━"*50,
            f"  Reasoning Efficiency  [{bar(d['reasoning_efficiency'])}] {d['reasoning_efficiency']:.3f}",
            f"  Reliability Index     [{bar(d['reliability_index'])}] {d['reliability_index']:.3f}",
            f"  Exploration Ratio     [{bar(d['exploration_ratio'])}] {d['exploration_ratio']:.3f}",
            f"  Decision Entropy      [{bar(d['decision_entropy'])}] {d['decision_entropy']:.3f}",
            f"  Wasteful Ratio        [{bar(d['wasteful_ratio'])}] {d['wasteful_ratio']:.3f}",
            f"  Pivot Rate  {d['pivot_rate']:.2f}/10 steps | Consistency {d['consistency_score']:.3f} ({d['runs_analyzed']} runs)",
        ]
        if d.get("action_distribution"):
            lines += ["\nAction Distribution:"] + [f"  {a:14s}: {c}" for a,c in d["action_distribution"].items()]
        return "\n".join(lines)
    except Exception as e: return f"Error: {e}"


# ── Tab 5: Self-Improve ───────────────────────────────────────────────────────

def get_improvement_plan():
    try:
        traj, meta, steps, ep_id = _get_traj_and_meta()
        if not traj: return _no_traj()
        fail_r = failure_clf.classify(ep_id, env.current_task or "?", steps, meta,
                                       list(env.files_read), list(env.files_written), env.final_score)
        plan = improvement_engine.generate_improvement_plan(
            ep_id, env.current_task or "?", fail_r.primary_failure,
            [f.evidence for f in fail_r.failures], env.final_score,
            steps, list(env.files_read), list(env.files_written)
        )
        d = plan.to_dict()
        lines = [
            "🔁 SELF-IMPROVEMENT PLAN", "━"*50,
            f"Original Score: {d['original_score']:.3f} | Failure: {d['failure_type']}",
            f"\n❌ What went wrong:\n  {d['what_went_wrong']}",
            f"\n🎯 Improved strategy:\n  {d['improved_strategy']}",
            "\n📋 Step-by-step plan:",
        ] + [f"  {s}" for s in d.get("step_by_step_plan",[])]
        lines += ["\n💉 System Prompt Injection:", "─"*40, d.get("system_prompt_addon","None")]
        return "\n".join(lines)
    except Exception as e: return f"Error: {e}"


def get_memory_context_for_task(task):
    try:
        ctx = memory_bank.retrieve(task=task, max_lessons=3)
        stats = memory_bank.get_stats()
        lines = [
            f"🧠 MEMORY BANK — {stats['total_entries']} total lessons",
            f"Retrieving for: {task}", "━"*50,
        ]
        if not ctx.relevant_lessons:
            lines.append("No lessons stored yet. Run episodes to build memory.")
        else:
            lines.append(f"\n📚 {ctx.lessons_count} relevant lesson(s):\n")
            for i, e in enumerate(ctx.relevant_lessons, 1):
                lines += [
                    f"[Lesson {i}] Task: {e.task} | Failure: {e.failure_type} | Score: {e.score:.2f}",
                    f"  Title: {e.lesson_title}",
                    f"  Lesson: {e.lesson_body[:120]}",
                    f"  Hint: {e.lesson_hint[:120]}" if e.lesson_hint else "",
                    "",
                ]
            lines += ["\n💉 System Prompt Injection:", "─"*40, ctx.system_prompt_injection]
        return "\n".join(l for l in lines)
    except Exception as e: return f"Error: {e}"


# ── Tab 6: Compare Agents ─────────────────────────────────────────────────────

def run_comparison(task, selected_agents):
    try:
        agents = selected_agents or None
        report = multi_agent_engine.compare(env, task=task, agents=agents)
        d = report.to_dict()
        lines = [
            f"⚖️ MULTI-AGENT COMPARISON — {task} (variant: {d.get('variant_id')})",
            f"🏆 Winner: {d.get('winner')} (score: {d.get('winner_score',0):.3f})", "━"*80,
            f"{'Rank':<5} {'Agent':<16} {'Score':<8} {'Steps':<7} {'Strategy':<22} {'Failure':<20} {'Reliability'}",
            "─"*80,
        ]
        for row in d.get("summary_table",[]):
            lines.append(f"#{row['rank']:<4} {row['agent']:<16} {row['score']:<8.3f} {row['steps']:<7} {row['strategy']:<22} {row['failure']:<20} {row['reliability']:.3f}")
        lines.append("━"*80)
        if d.get("insights"):
            lines += ["\n💡 Insights:"] + [f"  → {i}" for i in d["insights"]]
        lines.append("\n📊 Action Sequences:")
        for run in d.get("detailed_runs",[]):
            seq = " → ".join(run.get("action_sequence",[]))
            lines.append(f"  {run['agent_name']:16s}: {seq}")
        return "\n".join(lines)
    except Exception as e: return f"❌ {e}"


# ── Tab 7: 3D Visualizer ──────────────────────────────────────────────────────

def get_viz_iframe():
    """Return iframe pointing to /static/viz3d.html — fixes Three.js canvas rendering."""
    # Add a cache-busting timestamp so Gradio re-renders on refresh
    import time
    ts = int(time.time())
    return (
        f'<iframe src="/static/viz3d.html?t={ts}" '
        f'width="100%" height="640" frameborder="0" '
        f'style="border-radius:10px;border:1px solid rgba(125,211,252,0.2);'
        f'background:#0a0e1a;" '
        f'allow="accelerometer; autoplay" loading="lazy">'
        f'</iframe>'
    )


# ── Tab 8: Causal Probe ───────────────────────────────────────────────────────

def get_causal_probe():
    try:
        traj, meta, steps, ep_id = _get_traj_and_meta()
        if not traj: return _no_traj()
        r = causal_probe.probe(ep_id, env.current_task or "?", steps, meta,
                                list(env.files_read), list(env.files_written), env.final_score)
        d = r.to_dict()
        bar = lambda v: "█"*int(v*20)+"░"*(20-int(v*20))
        lines = [
            f"🧪 CAUSAL REASONING PROBE",
            f"━"*55,
            f"Understanding Level: {d['understanding_level']}",
            f"Causal Score:        [{bar(d['causal_score'])}] {d['causal_score']:.3f}",
            f"Chain Coverage:      [{bar(d['chain_coverage'])}] {d['chain_coverage']:.3f}",
            f"Chain Order Score:   [{bar(d['chain_order_score'])}] {d['chain_order_score']:.3f}",
            f"\n📡 Behavioral Signals:",
        ]
        sigs = d.get("behavioral_signals",{})
        for k,v in sigs.items():
            lines.append(f"  {'✅' if v else '❌'} {k.replace('_',' ').title()}")
        if d.get("understanding_indicators"):
            lines += ["\n✅ Understanding Indicators:"] + [f"  • {i}" for i in d["understanding_indicators"]]
        if d.get("guessing_indicators"):
            lines += ["\n❌ Guessing Indicators:"] + [f"  • {i}" for i in d["guessing_indicators"]]
        diag = d.get("diagnostics",{})
        if diag.get("false_confidence_detected"):
            lines.append("\n⚠️ FALSE CONFIDENCE DETECTED — submitted without adequate exploration")
        if diag.get("shortcut_learning_detected"):
            lines.append("⚠️ SHORTCUT LEARNING DETECTED — wrote without reading source")
        lines += [f"\n📝 {d['explanation']}"]
        if d.get("recommendations"):
            lines += ["\n💡 Recommendations:"] + [f"  → {r_}" for r_ in d["recommendations"]]
        return "\n".join(lines)
    except Exception as e: return f"Error: {e}"


# ── Tab 9: Counterfactual ─────────────────────────────────────────────────────

def get_counterfactual():
    try:
        traj, meta, steps, ep_id = _get_traj_and_meta()
        if not traj: return _no_traj()
        r = counterfactual_engine.analyze(ep_id, env.current_task or "?", steps, meta,
                                           list(env.files_read), list(env.files_written), env.final_score)
        d = r.to_dict()
        bar = lambda v: "█"*int(v*20)+"░"*(20-int(v*20))
        lines = [
            f"🎭 COUNTERFACTUAL ROBUSTNESS TEST",
            f"━"*55,
            f"Brittleness Level:  {d['brittleness_level']}",
            f"Robustness Score:   [{bar(d['robustness_score'])}] {d['robustness_score']:.3f}",
            f"Mutations Tested:   {d['mutations_tested']}",
            f"Mutations Survived: {d['mutations_survived']} ✅ | Failed: {d['mutations_failed']} ❌",
            f"\n🧬 Mutation Results:",
        ]
        for m in d.get("mutations",[]):
            icon = "✅" if not m["would_break_agent"] else "❌"
            lines.append(f"  {icon} [{m['type']}] {m['description'][:55]}")
            lines.append(f"     {m['why'][:80]}")
        if d.get("surface_dependencies"):
            lines += ["\n⚠️ Surface Dependencies:"] + [f"  • {s}" for s in d["surface_dependencies"]]
        if d.get("deep_dependencies"):
            lines += ["\n✅ Deep Dependencies:"] + [f"  • {s}" for s in d["deep_dependencies"]]
        lines += [f"\n📝 {d['explanation']}"]
        if d.get("recommendations"):
            lines += ["\n💡 Recommendations:"] + [f"  → {r_}" for r_ in d["recommendations"]]
        return "\n".join(lines)
    except Exception as e: return f"Error: {e}"


# ── Tab 10: Confidence Calibration ────────────────────────────────────────────

def get_calibration():
    try:
        traj, meta, steps, ep_id = _get_traj_and_meta()
        if not traj: return _no_traj()
        r = confidence_calibrator.calibrate(ep_id, env.current_task or "?", steps, env.final_score)
        d = r.to_dict()
        bar = lambda v: "█"*int(v*20)+"░"*(20-int(v*20))
        lines = [
            f"📐 CONFIDENCE CALIBRATION REPORT",
            f"━"*55,
            f"Calibration Profile: {d['profile']}",
            f"Calibration Score:   [{bar(d['calibration_score'])}] {d['calibration_score']:.3f}",
            f"Inferred Confidence: [{bar(d['inferred_confidence'])}] {d['inferred_confidence']:.3f}",
            f"Actual Performance:  [{bar(d['actual_performance'])}] {d['actual_performance']:.3f}",
            f"Calibration Error:   {d['expected_calibration_error']:.3f} (lower=better)",
            f"Conf-Acc Correlation: {d['confidence_accuracy_correlation']:.3f}",
            f"\n📊 Behavioral Signals:",
        ]
        sigs = d.get("signals",{})
        lines.append(f"  Commitment Speed:    {sigs.get('commitment_speed',0):.3f} (high=fast commit)")
        lines.append(f"  Re-Exploration Rate: {sigs.get('re_exploration_rate',0):.3f} (high=uncertain)")
        lines.append(f"  Verification Rate:   {sigs.get('verification_rate',0):.3f} tests/write")
        lines.append(f"  Submit Speed:        {sigs.get('submit_speed',0):.3f} (high=early submit)")
        lines += [f"\n📝 {d['diagnosis']}"]
        if d.get("recommendations"):
            lines += ["\n💡 Recommendations:"] + [f"  → {r_}" for r_ in d["recommendations"]]
        if d.get("confidence_trajectory"):
            lines.append("\n📈 Confidence Trajectory:")
            for s in d["confidence_trajectory"][:8]:
                acc_str = f" | acc={s['accuracy']:.2f}" if s['accuracy'] is not None else ""
                lines.append(f"  S{s['step']}: {s['action']:12s} conf={s['confidence']:.2f}{acc_str}")
        return "\n".join(lines)
    except Exception as e: return f"Error: {e}"


# ── Tab 11: Benchmark ─────────────────────────────────────────────────────────

def run_benchmark(tasks_selected, agents_selected):
    try:
        tasks = tasks_selected if tasks_selected else ["task1", "task2", "task3"]
        agents = agents_selected if agents_selected else None
        report = benchmark_runner.run(env, tasks=tasks, agents=agents)
        return report.render_table()
    except Exception as e:
        return f"❌ Benchmark error: {e}"


# ── Tab 12: Analytics ─────────────────────────────────────────────────────────

def get_analytics():
    try:
        if not env.get_trajectory():
            return _no_traj()
        report = analytics_engine.analyze(env)
        return report.render_text()
    except Exception as e:
        return f"Error: {e}"

def get_analytics_json():
    try:
        if not env.get_trajectory():
            return _no_traj()
        report = analytics_engine.analyze(env)
        return json.dumps(report.to_dict(), indent=2, default=str)
    except Exception as e:
        return f"Error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# Gradio UI
# ══════════════════════════════════════════════════════════════════════════════

with gr.Blocks(title="Codebase Navigation & Repair — OpenEnv v4") as demo:
    gr.Markdown(
        "# 🔍 Codebase Navigation & Repair — OpenEnv v4\n"
        "**The first platform that scientifically measures, explains, and improves AI agent reasoning.** "
        "Navigate · Fix · Evaluate Process · Probe Causality · Test Counterfactuals · Calibrate Confidence · Benchmark."
    )

    with gr.Tabs():

        # ── Tab 0: Quick Start Guide ──────────────────────────────────────────
        with gr.TabItem("📖 Quick Start Guide"):
            gr.Markdown("""
### Welcome to Codebase Navigation & Repair — OpenEnv v4

This interactive dashboard allows you to experience the environment infrastructure, run simulations, and analyze advanced agent logic.

#### 🚀 Step-by-Step Evaluation Guide:

1. **Initialize an Episode** 
   - Navigate to the **🤖 Run Agent** tab.
   - Select a task (`task1`, `task2`, or `task3`) and click **"Run Agent"**.
   - *This simulates an AI executing an episode dynamically against the environment and stores the trajectory.*

2. **Trigger Advanced Intelligence Diagnostics (v3/v4 Features)**
   - Go to **🧪 Causal Probe** and click it to evaluate if the agent truly understood the bug, or if it was just pattern-matching.
   - Go to **🎭 Counterfactual** to run mutation tests and analyze the brittleness of the agent's logic.
   - Go to **📐 Confidence** to see if the agent over-explored or submitted too early.
   - Go to **🧠 Intelligence** to execute failure classification and strategy detection.

3. **Visualize the Thought Process**
   - Head over to the **🌐 3D Visualizer** tab.
   - Click **"Load / Refresh Visualizer"**.
   - Using Three.js, this generates a dynamic 3D web of exactly how the agent traversed the repository files (cubes) and tests (prisms).

4. **Experiment Manually**
   - Want to play the game yourself? Go to the **🎮 Interactive** tab.
   - Click **Reset Environment**, then use the dropdowns to `read_file`, `write_file`, and finally `submit` to grade yourself.

5. **REST API / CLI Runner**
   - The entire platform operates out of incredibly fast, natively compliant REST endpoints. Check the **📖 API** tab for standard cURL routing.
            """)

        # ── Tab 1: Interactive ────────────────────────────────────────────────
        with gr.TabItem("🎮 Interactive"):
            with gr.Row():
                with gr.Column(scale=1):
                    task_sel = gr.Dropdown(["task1","task2","task3"], value="task1", label="Task")
                    reset_btn = gr.Button("🔄 Reset Environment", variant="primary")
                    gr.Markdown("### Action")
                    act_type = gr.Dropdown(["read_file","write_file","run_tests","search_code","submit"], value="read_file", label="Action Type")
                    act_path = gr.Textbox(label="Path", placeholder="src/auth.py")
                    act_query = gr.Textbox(label="Query", placeholder="validate_token")
                    act_content = gr.Textbox(label="Content (write_file)", lines=4)
                    step_btn = gr.Button("▶️ Execute Step", variant="secondary")
                with gr.Column(scale=2):
                    status_box = gr.Textbox(label="Status", lines=14, interactive=False)
                    result_box = gr.Textbox(label="Last Result", lines=8, interactive=False)
                    with gr.Row():
                        steps_box = gr.Textbox(label="Steps", value="0", interactive=False)
                        reward_box = gr.Textbox(label="Cumulative Reward", value="0.000", interactive=False)
            reset_btn.click(reset_environment, [task_sel], [status_box, result_box, steps_box, reward_box])
            step_btn.click(take_step, [act_type, act_path, act_query, act_content], [status_box, result_box, steps_box, reward_box])

        # ── Tab 2: Run Agent ──────────────────────────────────────────────────
        with gr.TabItem("🤖 Run Agent"):
            gr.Markdown("### Built-in Demonstration Agent\nRuns test-first deterministic strategy + stores lesson in memory bank.")
            agent_task = gr.Dropdown(["task1","task2","task3"], value="task1", label="Task")
            run_btn = gr.Button("🚀 Run Agent", variant="primary")
            agent_out = gr.Textbox(label="Agent Log", lines=22, interactive=False)
            run_btn.click(run_builtin_agent, [agent_task], [agent_out])

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

        # ── Tab 4: Intelligence ───────────────────────────────────────────────
        with gr.TabItem("🧠 Intelligence"):
            gr.Markdown("### Deep Agent Intelligence Analysis")
            with gr.Row():
                clf_btn = gr.Button("🔬 Classify Failure", variant="primary")
                strat_btn = gr.Button("🧭 Detect Strategy", variant="secondary")
                adv_btn = gr.Button("⚡ Advanced Metrics", variant="secondary")
            intel_out = gr.Textbox(label="Analysis", lines=32, interactive=False)
            clf_btn.click(get_failure_classification, outputs=[intel_out])
            strat_btn.click(get_strategy_detection, outputs=[intel_out])
            adv_btn.click(get_advanced_metrics, outputs=[intel_out])

        # ── Tab 5: Self-Improve ───────────────────────────────────────────────
        with gr.TabItem("🔁 Self-Improve"):
            gr.Markdown("### Self-Improvement Loop + Episodic Memory")
            with gr.Row():
                improve_btn = gr.Button("🔁 Improvement Plan", variant="primary")
                mem_task = gr.Dropdown(["task1","task2","task3"], value="task1", label="Task for Memory")
                mem_btn = gr.Button("🧠 Retrieve Memory", variant="secondary")
            improve_out = gr.Textbox(label="Output", lines=32, interactive=False)
            improve_btn.click(get_improvement_plan, outputs=[improve_out])
            mem_btn.click(get_memory_context_for_task, [mem_task], [improve_out])

        # ── Tab 6: Compare Agents ─────────────────────────────────────────────
        with gr.TabItem("⚖️ Compare Agents"):
            gr.Markdown("### Multi-Agent Strategy Comparison")
            with gr.Row():
                comp_task = gr.Dropdown(["task1","task2","task3"], value="task1", label="Task")
                comp_agents = gr.CheckboxGroup(
                    ["test-first","search-first","minimal","exhaustive"],
                    value=["test-first","search-first","minimal","exhaustive"],
                    label="Agents",
                )
            comp_btn = gr.Button("⚖️ Run Comparison", variant="primary")
            comp_out = gr.Textbox(label="Report", lines=30, interactive=False)
            comp_btn.click(run_comparison, [comp_task, comp_agents], [comp_out])

        # ── Tab 7: 3D Visualizer ──────────────────────────────────────────────
        with gr.TabItem("🌐 3D Visualizer"):
            gr.Markdown(
                "### Agent Trajectory 3D Visualization\n"
                "Files = glowing 3D spheres · Dependencies = edges · Agent = animated beam · **Run an episode first.**"
            )
            refresh_btn = gr.Button("🔄 Load / Refresh Visualizer", variant="primary")
            viz_html = gr.HTML(
                value='<div style="text-align:center;padding:60px;color:#475569;background:#0a0e1a;border-radius:10px">'
                      '<p style="font-size:24px">🌐</p>'
                      '<p style="color:#7dd3fc;font-weight:700">Run an episode then click Load</p></div>'
            )
            refresh_btn.click(get_viz_iframe, outputs=[viz_html])

        # ── Tab 8: Causal Probe ───────────────────────────────────────────────
        with gr.TabItem("🧪 Causal Probe"):
            gr.Markdown(
                "### Causal Reasoning Evaluation\n"
                "Did the agent truly understand WHY the bug exists, "
                "or did it pattern-match and guess? "
                "Measures chain coverage, order, and shortcut learning."
            )
            causal_btn = gr.Button("🧪 Run Causal Probe", variant="primary")
            causal_out = gr.Textbox(label="Causal Reasoning Report", lines=32, interactive=False)
            causal_btn.click(get_causal_probe, outputs=[causal_out])

        # ── Tab 9: Counterfactual ─────────────────────────────────────────────
        with gr.TabItem("🎭 Counterfactual"):
            gr.Markdown(
                "### Counterfactual Robustness Testing\n"
                "Applies 6 semantic-neutral mutations (filename rename, constant change, "
                "dummy function, directory shift, docstring noise, import reorder) "
                "and measures whether the agent's strategy survives."
            )
            cf_btn = gr.Button("🎭 Run Counterfactual Analysis", variant="primary")
            cf_out = gr.Textbox(label="Robustness Report", lines=32, interactive=False)
            cf_btn.click(get_counterfactual, outputs=[cf_out])

        # ── Tab 10: Confidence ────────────────────────────────────────────────
        with gr.TabItem("📐 Confidence"):
            gr.Markdown(
                "### Confidence Calibration Analysis\n"
                "Infers agent confidence from behavioral proxies (commitment speed, "
                "re-exploration rate, verification rate, submit timing) "
                "and compares to actual performance. Detects overconfident and underconfident agents."
            )
            calib_btn = gr.Button("📐 Analyze Calibration", variant="primary")
            calib_out = gr.Textbox(label="Calibration Report", lines=32, interactive=False)
            calib_btn.click(get_calibration, outputs=[calib_out])

        # ── Tab 11: Benchmark ─────────────────────────────────────────────────
        with gr.TabItem("🏆 Benchmark"):
            gr.Markdown(
                "### Automated Benchmark Leaderboard\n"
                "Runs all selected agent strategies × all selected tasks automatically. "
                "Ranks by composite score: correctness + causal reasoning + robustness + calibration + generalization."
            )
            with gr.Row():
                bench_tasks = gr.CheckboxGroup(["task1","task2","task3"], value=["task1","task2"], label="Tasks to Benchmark")
                bench_agents = gr.CheckboxGroup(
                    ["test-first","search-first","minimal","exhaustive"],
                    value=["test-first","minimal"],
                    label="Agent Strategies",
                )
            bench_btn = gr.Button("🏆 Run Benchmark (2–4 min)", variant="primary")
            bench_out = gr.Textbox(label="Leaderboard", lines=35, interactive=False)
            bench_btn.click(run_benchmark, [bench_tasks, bench_agents], [bench_out])

        # ── Tab 12: Analytics ─────────────────────────────────────────────────
        with gr.TabItem("📈 Analytics"):
            gr.Markdown(
                "### Unified Research-Grade Analytics\n"
                "Synthesizes ALL evaluation dimensions into one report: "
                "reasoning graph, root cause tree, alternative paths, profile tags, "
                "decision efficiency, composite score. Paper-ready JSON available."
            )
            with gr.Row():
                analytics_btn = gr.Button("📈 Full Analytics Report", variant="primary")
                analytics_json_btn = gr.Button("📋 Export JSON", variant="secondary")
            analytics_out = gr.Textbox(label="Analytics Report", lines=40, interactive=False)
            analytics_btn.click(get_analytics, outputs=[analytics_out])
            analytics_json_btn.click(get_analytics_json, outputs=[analytics_out])

        # ── Tab 13: API ───────────────────────────────────────────────────────
        with gr.TabItem("📖 API"):
            gr.Markdown("""
### REST API — v4.0 Endpoints

#### Core
| `/reset` POST | `/step` POST | `/state` GET | `/health` GET |

#### Evaluation
| `/trajectory` GET | `/evaluate` GET | `/metrics` GET | `/fault-config` POST |

#### Intelligence (v3)
| `/classify` GET | `/strategy` GET | `/advanced-metrics` GET | `/improvement-plan` GET | `/compare-agents` POST | `/viz-data` GET |

#### Research (v4 NEW)
| `/causal-probe` GET | `/counterfactual` GET | `/confidence` GET | `/benchmark` POST | `/analytics` GET |

```bash
BASE="http://localhost:7860"
# Run a full episode
curl -X POST "$BASE/reset?task=task1"
curl -X POST "$BASE/step" -H "Content-Type: application/json" -d '{"action_type":"read_file","path":"tests/test_formatter.py"}'
curl -X POST "$BASE/step" -d '{"action_type":"submit"}'

# All intelligence endpoints
curl "$BASE/classify"
curl "$BASE/causal-probe"
curl "$BASE/counterfactual"
curl "$BASE/confidence"
curl "$BASE/analytics"

# Benchmark
curl -X POST "$BASE/benchmark?tasks=task1,task2"
```
""")


# ── Mount FastAPI ─────────────────────────────────────────────────────────────
from server.app import app as fastapi_app
gr_app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(fastapi_app, host="0.0.0.0", port=7860)
