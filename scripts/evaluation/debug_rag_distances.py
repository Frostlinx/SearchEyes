#!/usr/bin/env python3
"""Debug: check actual RAG distances from a screenshot to WIT images."""
import sys, json, glob
sys.path.insert(0, "/root/autodl-tmp/QWEN/QWEN-project")

from searcheyes.multimodal_rag import MultimodalRAG, RagConfig

# Use a very high threshold to see ALL results
config = RagConfig(
    chroma_db_path="/root/autodl-tmp/QWEN/QWEN-project/data/wit_subset_hf/chroma_db",
    embedding_server_url="http://localhost:8766",
    top_k=5,
    score_threshold=2.0,  # Accept everything
)
rag = MultimodalRAG(config)

# Find a screenshot from the last episode
episodes = sorted(glob.glob("/root/autodl-tmp/QWEN/QWEN-project/output/agent_loops/vt_0000_*"))
if episodes:
    screenshot = f"{episodes[-1]}/step_00/screenshot.png"
    print(f"Querying with: {screenshot}")
    facts = rag.get_rag_facts(screenshot)
    print(f"Results: {len(facts)} facts")
    for f in facts:
        print(f"  {f.wit_id}: score={f.score:.4f} title={f.title[:40]} caption={f.caption[:40]}")
else:
    print("No episodes found")
