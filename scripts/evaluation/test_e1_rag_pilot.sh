#!/bin/bash
# E1: End-to-end RAG pilot test
export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/QWEN/QWEN-project

echo "===== E1: RAG Pilot Test ====="
echo "Embedding server should be running on port 8766"
curl -s http://localhost:8766/health && echo

echo ""
echo "===== Run 1: WITHOUT RAG (baseline) ====="
python3 run_vlm_pilot.py \
    --backend scripted \
    --task-jsonl /root/autodl-tmp/QWEN/QWEN-project/data/tasks/visual_tasks.jsonl \
    --task-index 0

echo ""
echo "===== Run 2: WITH RAG ====="
python3 run_vlm_pilot.py \
    --backend scripted \
    --task-jsonl /root/autodl-tmp/QWEN/QWEN-project/data/tasks/visual_tasks.jsonl \
    --task-index 0 \
    --rag-db /root/autodl-tmp/QWEN/QWEN-project/data/wit_subset_hf/chroma_db \
    --embedding-url http://localhost:8766

echo ""
echo "===== E1 COMPLETE ====="
