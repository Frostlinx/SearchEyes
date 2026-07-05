"""
config.py — 单一配置源，所有实验脚本从这里读参数
"""
from pathlib import Path

# ── 项目根目录 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── KB ──────────────────────────────────────────────────────
CHROMA_DB_PATH   = PROJECT_ROOT / "data" / "wit_kb_v2" / "chroma_db"
COLLECTION_NAME  = "wit_knowledge_v2_qwen"
IMAGES_DIR       = PROJECT_ROOT / "data" / "wit_kb_v2" / "images"
META_JSONL       = PROJECT_ROOT / "data" / "wit_kb_v2" / "meta.jsonl"

# ── 任务 ────────────────────────────────────────────────────
TASKS_JSONL      = PROJECT_ROOT / "data" / "tasks" / "research_tasks_v2.jsonl"
EVAL_MAX_TASKS   = 50   # 固定eval split，A vs B必须用同一批

# ── Embedding server ────────────────────────────────────────
EMBEDDING_URL    = "http://localhost:8766"

# ── 本地VLM（Option B caption生成）──────────────────────────
# 支持环境变量覆盖（服务器路径与本地不同时使用）
import os as _os
VLM_MODEL_PATH   = Path(_os.environ.get("VLM_MODEL_PATH", str(PROJECT_ROOT / "Qwen3-VL-4B-Instruct")))
VLM_DEVICE       = "cuda"
VLM_MAX_NEW_TOKENS = 128

# ── 检索参数 ────────────────────────────────────────────────
TOP_K_DEFAULT    = 20    # 与主项目保持一致

# ── 结果输出 ────────────────────────────────────────────────
RESULTS_DIR      = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
