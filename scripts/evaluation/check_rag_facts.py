#!/usr/bin/env python3
"""Check if RAG facts are present in the trajectory.json of the RAG run."""
import json
import glob
import os

# Find the most recent RAG episode
base = "/root/autodl-tmp/QWEN/QWEN-project/output/agent_loops"
episodes = sorted(glob.glob(f"{base}/vt_0000_*"))
if len(episodes) < 2:
    print("Need at least 2 episodes (baseline + RAG)")
    exit(1)

baseline_dir = episodes[-2]
rag_dir = episodes[-1]

print(f"Baseline: {os.path.basename(baseline_dir)}")
print(f"RAG:      {os.path.basename(rag_dir)}")

# Check baseline (should have empty rag_facts)
traj = json.load(open(f"{baseline_dir}/trajectory.json"))
baseline_has_rag = any(step["context"].get("rag_facts") for step in traj["steps"])
print(f"\nBaseline rag_facts present: {baseline_has_rag}")

# Check RAG run
traj = json.load(open(f"{rag_dir}/trajectory.json"))
for step in traj["steps"]:
    facts = step["context"].get("rag_facts", [])
    print(f"  step {step['step_idx']}: rag_facts={len(facts)}")
    for f in facts[:2]:
        print(f"    - {f[:70]}")

rag_has_facts = any(step["context"].get("rag_facts") for step in traj["steps"])
print(f"\nRAG run rag_facts present: {rag_has_facts}")

if rag_has_facts and not baseline_has_rag:
    print("\nE1 VERIFICATION: PASSED - RAG facts injected correctly")
elif rag_has_facts:
    print("\nE1 VERIFICATION: PARTIAL - RAG facts present in both (unexpected)")
else:
    print("\nE1 VERIFICATION: FAILED - No RAG facts found in RAG run")
