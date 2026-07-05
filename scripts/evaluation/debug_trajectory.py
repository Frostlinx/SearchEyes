#!/usr/bin/env python3
"""Debug script: run a few samples and save full trajectories."""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from eval_multiturn import (
    VLLMClient, LocalKBBackend, load_pkc_test, run_agent_episode
)

client = VLLMClient(
    api_base="https://routify.alibaba-inc.com/protocol/openai/v1",
    model_name="gpt-4o",
    max_tokens=8192,
    temperature=0.6,
    top_p=0.95,
    api_key="sk-9eba1adb38fa4cb1af5dca05f58f8472",
)
backend = LocalKBBackend("/tmp/pgkc_full_kb.json")
samples = load_pkc_test("data/vissearch_bench.jsonl", max_samples=3)

trajectories = []
for i, sample in enumerate(samples):
    print(f"\n{'='*60}")
    print(f"[{i+1}/3] {sample.sample_id}")
    print(f"Q: {sample.question[:120]}...")
    print(f"Gold: {sample.golden_answers}")
    
    episode = run_agent_episode(
        client=client,
        search_backend=backend,
        question=sample.question,
        image_path=sample.image_path,
        max_turns=15,
    )
    
    print(f"Pred: {episode['final_answer']}")
    print(f"Termination: {episode['termination']}")
    print(f"Turns: {episode['total_turns']}, Search: {episode['num_search_calls']}, Read: {episode['num_read_calls']}")
    
    # Print each turn
    for t in episode["turns"]:
        ti = t["turn_index"]
        tn = t.get("tool_name", "")
        ta = t.get("tool_args", {})
        obs = t.get("observation", "")[:300]
        at = t.get("assistant_text", "")[:400]
        print(f"\n  --- Turn {ti} ---")
        print(f"  Assistant: {at}")
        if tn:
            print(f"  Tool: {tn}({json.dumps(ta)})")
            print(f"  Observation: {obs}")
    
    trajectories.append({
        "sample_id": sample.sample_id,
        "question": sample.question,
        "gold": sample.golden_answers,
        "prediction": episode["final_answer"],
        "termination": episode["termination"],
        "num_turns": episode["total_turns"],
        "num_search": episode["num_search_calls"],
        "num_read": episode["num_read_calls"],
        "turns": episode["turns"],
    })

with open("/tmp/debug_trajectories.json", "w") as f:
    json.dump(trajectories, f, indent=2, ensure_ascii=False)
print(f"\n\nSaved to /tmp/debug_trajectories.json")
