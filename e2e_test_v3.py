#!/usr/bin/env python3
"""
e2e_test_v3.py — Full End-to-End test suite for v3.0

Tests every endpoint, all 3 tasks, all new intelligence modules,
multi-agent comparison, and the 3D viz-data endpoint.
"""
import sys
import json
import time
import requests

BASE = "http://localhost:7860"
PASS = 0
FAIL = 0
RESULTS = []


def check(name, condition, detail=""):
    global PASS, FAIL
    status = "✅ PASS" if condition else "❌ FAIL"
    if condition:
        PASS += 1
    else:
        FAIL += 1
    msg = f"  {status}  {name}"
    if detail:
        msg += f"  →  {detail}"
    print(msg)
    RESULTS.append({"name": name, "passed": condition, "detail": detail})


def section(title):
    print(f"\n{'━'*60}")
    print(f"  {title}")
    print(f"{'━'*60}")


# ─────────────────────────────────────────────────────────────────────────────
section("1. HEALTH & BASIC CONNECTIVITY")
# ─────────────────────────────────────────────────────────────────────────────

r = requests.get(f"{BASE}/health")
check("GET /health returns 200", r.status_code == 200)
data = r.json()
check("Health version is 3.0.0", data.get("version") == "3.0.0", data.get("version"))
check("Health status is ok", data.get("status") == "ok")


# ─────────────────────────────────────────────────────────────────────────────
section("2. CORE OPENENV — ALL 3 TASKS")
# ─────────────────────────────────────────────────────────────────────────────

for task in ["task1", "task2", "task3"]:
    r = requests.post(f"{BASE}/reset?task={task}")
    check(f"POST /reset?task={task} → 200", r.status_code == 200, f"status={r.status_code}")
    if r.status_code == 200:
        d = r.json()
        obs = d.get("observation", {})
        check(f"  {task}: has repo_tree", bool(obs.get("repo_tree")), str(obs.get("repo_tree", [])[:2]))
        check(f"  {task}: has variant_id", bool(d.get("info", {}).get("variant_id")))
        check(f"  {task}: steps_remaining > 0", obs.get("steps_remaining", 0) > 0)

# ─────────────────────────────────────────────────────────────────────────────
section("3. STEP ACTIONS — FULL EPISODE (task1)")
# ─────────────────────────────────────────────────────────────────────────────

r = requests.post(f"{BASE}/reset?task=task1")
obs = r.json()["observation"]
tree = obs["repo_tree"]
test_files = [f for f in tree if f.startswith("tests/")]
src_files = [f for f in tree if f.startswith("src/")]

# read_file
r = requests.post(f"{BASE}/step", json={"action_type": "read_file", "path": test_files[0]})
check("POST /step read_file test file → 200", r.status_code == 200)
check("read_file reward >= 0", r.json().get("reward", -1) >= 0, str(r.json().get("reward")))

r = requests.post(f"{BASE}/step", json={"action_type": "read_file", "path": src_files[0]})
check("POST /step read_file src file → 200", r.status_code == 200)

# search_code
r = requests.post(f"{BASE}/step", json={"action_type": "search_code", "query": "def "})
check("POST /step search_code → 200", r.status_code == 200)

# run_tests
r = requests.post(f"{BASE}/step", json={"action_type": "run_tests"})
check("POST /step run_tests → 200", r.status_code == 200, f"reward={r.json().get('reward')}")

# submit
r = requests.post(f"{BASE}/step", json={"action_type": "submit"})
check("POST /step submit → 200", r.status_code == 200)
final_score = r.json()["info"].get("final_score", 0)
check("Episode done after submit", r.json().get("done") == True)

# Try stepping after done → should get 400
r = requests.post(f"{BASE}/step", json={"action_type": "read_file", "path": "x.py"})
check("POST /step after done → 400", r.status_code == 400)

# ─────────────────────────────────────────────────────────────────────────────
section("4. STATE ENDPOINT")
# ─────────────────────────────────────────────────────────────────────────────

requests.post(f"{BASE}/reset?task=task1")
requests.post(f"{BASE}/step", json={"action_type": "read_file", "path": test_files[0]})
r = requests.get(f"{BASE}/state")
check("GET /state → 200", r.status_code == 200)
d = r.json()
check("State has observation", "observation" in d)
check("State total_steps_taken >= 1", d.get("total_steps_taken", 0) >= 1)


# ─────────────────────────────────────────────────────────────────────────────
section("5. TRAJECTORY & EVALUATION")
# ─────────────────────────────────────────────────────────────────────────────

requests.post(f"{BASE}/step", json={"action_type": "submit"})

r = requests.get(f"{BASE}/trajectory")
check("GET /trajectory → 200", r.status_code == 200)
traj = r.json()
check("Trajectory has episode_id", bool(traj.get("episode_id")))
check("Trajectory steps > 0", len(traj.get("steps", [])) > 0, f"steps={len(traj.get('steps',[]))}")

r = requests.get(f"{BASE}/evaluate")
check("GET /evaluate → 200", r.status_code == 200)
ev = r.json()
check("Evaluation has composite_score", "composite_score" in ev, str(ev.get("composite_score")))
check("Evaluation has 6 dimensions", len(ev.get("dimensions", {})) == 6, str(list(ev.get("dimensions", {}).keys())))

r = requests.get(f"{BASE}/metrics")
check("GET /metrics → 200", r.status_code == 200)
m = r.json()
check("Metrics has timeline", "timeline" in m, str(list(m.keys())[:5]))


# ─────────────────────────────────────────────────────────────────────────────
section("6. FAULT INJECTION")
# ─────────────────────────────────────────────────────────────────────────────

r = requests.post(f"{BASE}/fault-config", json={"level": "light"})
check("POST /fault-config light → 200", r.status_code == 200)
r = requests.post(f"{BASE}/reset?task=task1")
check("Reset with fault injection → 200", r.status_code == 200)
fi = r.json().get("info", {}).get("fault_injection", {})
check("Fault injection info present", "difficulty_multiplier" in fi or "faults_injected" in fi, str(fi))

# Reset back
requests.post(f"{BASE}/fault-config", json={"level": "none"})


# ─────────────────────────────────────────────────────────────────────────────
section("7. INTELLIGENCE — FAILURE CLASSIFIER")
# ─────────────────────────────────────────────────────────────────────────────

# Run a fresh episode with minimal effort to get a known failure
requests.post(f"{BASE}/reset?task=task1")
requests.post(f"{BASE}/step", json={"action_type": "submit"})  # Submit without doing anything

r = requests.get(f"{BASE}/classify")
check("GET /classify → 200", r.status_code == 200)
d = r.json()
check("Classify has episode_id", "episode_id" in d, d.get("episode_id"))
check("Classify has primary_failure", "primary_failure" in d, d.get("primary_failure"))
check("Classify has success field", "success" in d)
check("Classify success=False for minimal effort", d.get("success") == False)
check("Classify has retry_hint", bool(d.get("retry_hint")), d.get("retry_hint", "")[:60])


# ─────────────────────────────────────────────────────────────────────────────
section("8. INTELLIGENCE — STRATEGY DETECTOR")
# ─────────────────────────────────────────────────────────────────────────────

r = requests.get(f"{BASE}/strategy")
check("GET /strategy → 200", r.status_code == 200)
d = r.json()
check("Strategy has strategy field", "strategy" in d, d.get("strategy"))
VALID_STRATEGIES = ["TARGETED_DEBUGGING", "SYSTEMATIC_SEARCH", "BRUTE_FORCE",
                    "RANDOM_EXPLORATION", "SPEC_DRIVEN", "MINIMAL_EFFORT"]
check("Strategy is a known label", d.get("strategy") in VALID_STRATEGIES, d.get("strategy"))
check("Strategy has score 0-1", 0 <= d.get("score", -1) <= 1, str(d.get("score")))
check("Strategy has exploration_ratio", "exploration_ratio" in d)
check("Strategy has sub_patterns list", isinstance(d.get("sub_patterns"), list))


# ─────────────────────────────────────────────────────────────────────────────
section("9. INTELLIGENCE — ADVANCED METRICS")
# ─────────────────────────────────────────────────────────────────────────────

r = requests.get(f"{BASE}/advanced-metrics")
check("GET /advanced-metrics → 200", r.status_code == 200)
d = r.json()
expected_keys = ["reasoning_efficiency", "exploration_ratio", "decision_entropy",
                 "reliability_index", "pivot_rate", "wasteful_ratio", "consistency_score"]
for key in expected_keys:
    check(f"  advanced-metrics has '{key}'", key in d, str(d.get(key, "MISSING")))
check("reliability_index in [0,1]", 0 <= d.get("reliability_index", -1) <= 1)
check("action_distribution is dict", isinstance(d.get("action_distribution"), dict))


# ─────────────────────────────────────────────────────────────────────────────
section("10. INTELLIGENCE — IMPROVEMENT PLAN")
# ─────────────────────────────────────────────────────────────────────────────

r = requests.get(f"{BASE}/improvement-plan")
check("GET /improvement-plan → 200", r.status_code == 200)
d = r.json()
check("Plan has failure_type", "failure_type" in d, d.get("failure_type"))
check("Plan has what_went_wrong", bool(d.get("what_went_wrong")))
check("Plan has improved_strategy", bool(d.get("improved_strategy")))
check("Plan has step_by_step_plan list", isinstance(d.get("step_by_step_plan"), list))
check("Plan step_by_step_plan not empty", len(d.get("step_by_step_plan", [])) > 0)
check("Plan has system_prompt_addon", "system_prompt_addon" in d)


# ─────────────────────────────────────────────────────────────────────────────
section("11. MULTI-AGENT COMPARISON")
# ─────────────────────────────────────────────────────────────────────────────

r = requests.post(f"{BASE}/compare-agents?task=task1&agents=test-first,minimal")
check("POST /compare-agents (2 agents) → 200", r.status_code == 200, f"status={r.status_code}")
if r.status_code == 200:
    d = r.json()
    check("Comparison has winner", "winner" in d, d.get("winner"))
    check("Comparison has summary_table", "summary_table" in d)
    check("Summary table has 2 rows", len(d.get("summary_table", [])) == 2,
          str(len(d.get("summary_table", []))))
    check("Each row has score/steps/strategy", all(
        "score" in row and "steps" in row and "strategy" in row
        for row in d.get("summary_table", [])
    ))
    check("Comparison has insights", "insights" in d)
    check("Comparison has detailed_runs", len(d.get("detailed_runs", [])) == 2)

# Test all 4 agents
r = requests.post(f"{BASE}/compare-agents?task=task1")
check("POST /compare-agents (all agents) → 200", r.status_code == 200)
if r.status_code == 200:
    d = r.json()
    check("All 4 agents ran", len(d.get("summary_table", [])) == 4,
          f"rows={len(d.get('summary_table',[]))}")


# ─────────────────────────────────────────────────────────────────────────────
section("12. 3D VISUALIZATION DATA")
# ─────────────────────────────────────────────────────────────────────────────

# Run a full episode first for viz data
requests.post(f"{BASE}/reset?task=task1")
requests.post(f"{BASE}/step", json={"action_type": "read_file", "path": test_files[0]})
requests.post(f"{BASE}/step", json={"action_type": "submit"})

r = requests.get(f"{BASE}/viz-data")
check("GET /viz-data → 200", r.status_code == 200)
d = r.json()
check("Viz-data has files array", isinstance(d.get("files"), list), f"len={len(d.get('files',[]))}")
check("Viz-data files > 0", len(d.get("files", [])) > 0)
check("Viz-data has dependencies", isinstance(d.get("dependencies"), list))
check("Viz-data has steps", isinstance(d.get("steps"), list))
check("Viz-data has strategy", "strategy" in d, d.get("strategy"))
check("Viz-data has final_score", "final_score" in d)
if d.get("files"):
    f = d["files"][0]
    check("File node has name/type/is_bug_file", all(k in f for k in ["name","type","is_bug_file"]))


# ─────────────────────────────────────────────────────────────────────────────
section("13. INVALID ACTION HANDLING")
# ─────────────────────────────────────────────────────────────────────────────

requests.post(f"{BASE}/reset?task=task1")

# Invalid task
r = requests.post(f"{BASE}/reset?task=task99")
check("Invalid task → 400", r.status_code == 400)

# Invalid action type
r = requests.post(f"{BASE}/step", json={"action_type": "hack_system"})
check("Invalid action_type → 400 or 422", r.status_code in (400, 422))

# Non-existent file
r = requests.post(f"{BASE}/step", json={"action_type": "read_file", "path": "non_existent.py"})
check("Read non-existent file → 200 with error", r.status_code == 200)
obs = r.json().get("observation", {})
check("Non-existent file has error in obs", bool(obs.get("last_action_error")), obs.get("last_action_error","")[:60])


# ─────────────────────────────────────────────────────────────────────────────
section("14. SECURITY SCANNING")
# ─────────────────────────────────────────────────────────────────────────────

requests.post(f"{BASE}/reset?task=task1")
# Try to write a file with dangerous code
r = requests.post(f"{BASE}/step", json={
    "action_type": "write_file",
    "path": src_files[0] if src_files else "src/hack.py",
    "content": "import os\nos.system('rm -rf /')\n"
})
check("Write dangerous code → 200", r.status_code == 200)
if r.status_code == 200:
    info = r.json().get("info", {})
    flags = info.get("security_flags", [])
    check("Security flags populated for os.system", len(flags) > 0, str(flags[:2]))


# ─────────────────────────────────────────────────────────────────────────────
section("15. GRADIO UI ENDPOINTS")
# ─────────────────────────────────────────────────────────────────────────────

r = requests.get(f"{BASE}/")
check("GET / (Gradio UI) → 200", r.status_code == 200)
check("Response is HTML", "text/html" in r.headers.get("content-type", ""))

r = requests.get(f"{BASE}/static/viz3d.html")
check("GET /static/viz3d.html → 200", r.status_code == 200)
check("viz3d.html is HTML", "html" in r.text.lower()[:200])
check("viz3d.html has Three.js", "three" in r.text.lower())
check("viz3d.html has timeline-slider", "timeline-slider" in r.text)


# ─────────────────────────────────────────────────────────────────────────────
section("16. TASK2 & TASK3 FULL EPISODE")
# ─────────────────────────────────────────────────────────────────────────────

for task in ["task2", "task3"]:
    r = requests.post(f"{BASE}/reset?task={task}")
    check(f"{task} reset → 200", r.status_code == 200)
    obs = r.json()["observation"]
    tree = obs["repo_tree"]
    tf = [f for f in tree if f.startswith("tests/")]
    sf = [f for f in tree if f.startswith("src/")]
    md = [f for f in tree if f.endswith(".md")]

    if task == "task3" and md:
        requests.post(f"{BASE}/step", json={"action_type": "read_file", "path": md[0]})
    if tf:
        requests.post(f"{BASE}/step", json={"action_type": "read_file", "path": tf[0]})
    if sf:
        requests.post(f"{BASE}/step", json={"action_type": "read_file", "path": sf[0]})

    r = requests.post(f"{BASE}/step", json={"action_type": "submit"})
    check(f"{task} submit → done", r.json().get("done") == True)

    # Verify all intelligence endpoints work post-episode
    r = requests.get(f"{BASE}/classify")
    check(f"{task} /classify works", r.status_code == 200 and "primary_failure" in r.json())
    r = requests.get(f"{BASE}/strategy")
    check(f"{task} /strategy works", r.status_code == 200 and "strategy" in r.json())


# ─────────────────────────────────────────────────────────────────────────────
section("17. CONSISTENCY — 3 RUNS SAME TASK")
# ─────────────────────────────────────────────────────────────────────────────

scores = []
for i in range(3):
    requests.post(f"{BASE}/reset?task=task1")
    r = requests.get(f"{BASE}/state")
    tree = r.json()["observation"]["repo_tree"]
    tf = [f for f in tree if f.startswith("tests/")]
    if tf:
        requests.post(f"{BASE}/step", json={"action_type": "read_file", "path": tf[0]})
    requests.post(f"{BASE}/step", json={"action_type": "submit"})
    metrics = requests.get(f"{BASE}/advanced-metrics").json()
    scores.append(requests.get(f"{BASE}/evaluate").json().get("composite_score", 0))

check("3 runs completed", len(scores) == 3, str(scores))
check("All runs have valid scores", all(0 <= s <= 1 for s in scores), str(scores))

# Consistency metric
r = requests.get(f"{BASE}/advanced-metrics")
d = r.json()
check("Consistency score populated after multiple runs", d.get("runs_analyzed", 0) >= 1,
      f"runs={d.get('runs_analyzed')}, consistency={d.get('consistency_score'):.3f}")


# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'═'*60}")
print(f"  E2E RESULTS: {PASS} passed | {FAIL} failed | {PASS+FAIL} total")
print(f"  Score: {PASS/(PASS+FAIL)*100:.1f}%")
print(f"{'═'*60}")

if FAIL > 0:
    print("\nFailed tests:")
    for r in RESULTS:
        if not r["passed"]:
            print(f"  ❌ {r['name']}: {r['detail']}")

sys.exit(0 if FAIL == 0 else 1)
