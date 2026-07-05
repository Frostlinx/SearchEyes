#!/bin/bash
# Download Qwen3-VL-Embedding-2B via hf-mirror
export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com

echo "[$(date)] Starting Qwen3-VL-Embedding-2B download..." | tee /tmp/embed_download.log

python3 -c "
from huggingface_hub import snapshot_download
import os
print(f'Mirror: {os.environ.get(\"HF_ENDPOINT\")}', flush=True)
snapshot_download(
    'Qwen/Qwen3-VL-Embedding-2B',
    local_dir='/root/autodl-tmp/QWEN/Qwen3-VL-Embedding-2B',
    resume_download=True,
)
print('EMBED DOWNLOAD COMPLETE', flush=True)
" >> /tmp/embed_download.log 2>&1

echo "[$(date)] Script finished" >> /tmp/embed_download.log
