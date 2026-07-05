#!/bin/bash
# Download WIT subset (1000 images) via hf-mirror
export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com

echo "[$(date)] Starting WIT 1000 download..." | tee /tmp/wit_download.log

python3 /root/autodl-tmp/QWEN/QWEN-project/searcheyes/wit_downloader_hf.py \
    --output-dir /root/autodl-tmp/QWEN/QWEN-project/data/wit_subset_hf \
    --count 1000 \
    >> /tmp/wit_download.log 2>&1

echo "[$(date)] WIT script finished" >> /tmp/wit_download.log
