#!/usr/bin/env python3
"""Quick E2E test for the deployed HF Space."""
import httpx, json, sys

BASE = "https://Chirag0123-codebase-nav-env.hf.space"
client = httpx.Client(timeout=120.0)
ok = 0

def test(label, fn):
    global ok
    try:
        result = fn()
        ok += 1
        print(f"  ✅ {label}: {json.dumps(result)[:200]}")
    except Exception as e:
        print(f"  ❌ {label}: {e}")

print("Testing deployed Space...")

# 1. Health
test("Health", lambda: client.get(f"{BASE}/health").json())

# 2. Reset
test("Reset task1", lambda: (r := client.post(f"{BASE}/reset", params={"task": "task1"}).json(), r["info"]["variant_id"])[1])

# 3. Read file
test("Read file", lambda: (r := client.post(f"{BASE}/step", json={"action_type": "read_file", "path": client.get(f"{BASE}/state").json()["observation"]["repo_tree"][0]}).json(), f"reward={r['reward']}")[1])

# 4. Run tests
test("Run tests", lambda: (r := client.post(f"{BASE}/step", json={"action_type":"run_tests"}).json(), f"reward={r['reward']}")[1])

# 5. Submit
test("Submit", lambda: (r := client.post(f"{BASE}/step", json={"action_type":"submit"}).json(), f"score={r['info']['final_score']}")[1])

# 6. Trajectory
test("Trajectory", lambda: (r := client.get(f"{BASE}/trajectory").json(), f"steps={r['total_steps']}")[1])

# 7. Evaluate
test("Evaluate", lambda: (r := client.get(f"{BASE}/evaluate").json(), f"composite={r['composite_score']}")[1])

# 8. Metrics
test("Metrics", lambda: (r := client.get(f"{BASE}/metrics").json(), f"efficiency={r['step_efficiency']}")[1])

# 9. Fault config
test("Fault config", lambda: client.post(f"{BASE}/fault-config", json={"level":"light"}).json())

# 10. Reset with faults
test("Reset+faults", lambda: (r := client.post(f"{BASE}/reset", params={"task":"task2"}).json(), f"faults={len(r['info'].get('fault_injection',{}).get('faults_injected',[]))}")[1])

# 11. Disable faults
test("Disable faults", lambda: client.post(f"{BASE}/fault-config", json={"level":"none"}).json())

print(f"\n{'='*50}")
print(f"  Result: {ok}/11 tests passed")
print(f"{'='*50}")
client.close()
