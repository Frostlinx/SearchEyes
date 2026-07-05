#!/bin/bash
# run_train.sh — 一键启动 Qwen3-VL-4B SFT 训练
# ================================================
# 用法: bash run_train.sh
# 可选环境变量:
#   EPOCHS=3          训练轮数
#   LR=2e-4           学习率
#   OUTPUT_DIR=checkpoints/sft_v1
#   NO_LORA=0         设为 1 则全参数微调

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-4}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/sft_v1}"
NO_LORA="${NO_LORA:-0}"

echo "========================================================"
echo "  Qwen3-VL-4B SFT 训练启动脚本"
echo "========================================================"
echo "  项目目录: $PROJECT_DIR"
echo "  Epochs:   $EPOCHS"
echo "  LR:       $LR"
echo "  输出目录: $OUTPUT_DIR"
echo "========================================================"

# ── 1. 检查 Python ────────────────────────────────────────
echo ""
echo "[1/5] 检查 Python 环境..."
python3 --version || { echo "❌ 未找到 python3"; exit 1; }

# ── 2. 检查 GPU ───────────────────────────────────────────
echo ""
echo "[2/5] 检查 GPU..."
python3 -c "
import torch
if not torch.cuda.is_available():
    print('❌ CUDA 不可用，请确认 GPU 已开启')
    exit(1)
name = torch.cuda.get_device_name(0)
mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'✅ GPU: {name}  显存: {mem:.1f} GB')
" || exit 1

# ── 3. 安装依赖 ───────────────────────────────────────────
echo ""
echo "[3/5] 检查并安装依赖..."
pip install -q --upgrade \
    "transformers>=4.51.0" \
    "trl>=0.17.0" \
    "peft>=0.14.0" \
    "accelerate>=1.4.0" \
    "datasets>=3.0.0" \
    "Pillow>=10.0.0" \
    "bitsandbytes>=0.45.0" \
    "playwright" \
    "qwen-vl-utils"

# 安装 playwright 浏览器（数据生成需要）
python3 -m playwright install chromium --with-deps 2>/dev/null || true

echo "✅ 依赖检查完成"

# ── 4. 生成/检查训练数据 ──────────────────────────────────
echo ""
echo "[4/5] 准备训练数据..."

TRAIN_DATA="$PROJECT_DIR/data/sft_train.jsonl"
EVAL_DATA="$PROJECT_DIR/data/sft_eval.jsonl"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "  训练数据不存在，开始生成..."

    # 先生成任务集（如果不存在）
    TASK_FILE="$PROJECT_DIR/data/tasks/visual_tasks.jsonl"
    if [ ! -f "$TASK_FILE" ]; then
        echo "  生成任务集..."
        python3 -c "
import sys; sys.path.insert(0, '.')
from searcheyes.task_generator import generate_task_set
generate_task_set(count=100, seed=42, validate=True)
"
    fi

    # 生成 RL rollout 数据（scripted pilot）
    echo "  生成 SFT rollout 数据..."
    python3 -c "
import sys, json
sys.path.insert(0, '.')
from pathlib import Path
from searcheyes.task_schema import VisualTask
from searcheyes.rl_adapter import RLEnvironment
from searcheyes.vlm_agent import ScriptedPilotBackend

task_file = Path('data/tasks/visual_tasks.jsonl')
tasks = []
with open(task_file) as f:
    for line in f:
        line = line.strip()
        if line:
            tasks.append(VisualTask.load_from_dict(json.loads(line)))

print(f'共 {len(tasks)} 个任务，开始 rollout...')
for i, task in enumerate(tasks[:50]):  # 先跑前50个
    try:
        env = RLEnvironment(task)
        obs = env.reset()
        script = [s.__dict__ for s in task.ground_truth_trajectory]
        backend = ScriptedPilotBackend(script)
        done = False
        step = 0
        while not done and step < task.dag_depth * 2:
            from searcheyes.agent_loop import AgentLoop
            break  # 直接用 RLEnvironment scripted 模式
        env.export_episode_jsonl(verl_format=False)
        if (i+1) % 10 == 0:
            print(f'  完成 {i+1}/50')
    except Exception as e:
        print(f'  任务 {task.task_id} 失败: {e}')
print('rollout 完成')
" 2>&1 | tail -20

    # 预处理数据
    python3 prepare_sft_data.py
else
    echo "  训练数据已存在，跳过生成"
    TRAIN_COUNT=$(wc -l < "$TRAIN_DATA")
    EVAL_COUNT=$(wc -l < "$EVAL_DATA" 2>/dev/null || echo 0)
    echo "  训练集: $TRAIN_COUNT 条, 验证集: $EVAL_COUNT 条"
fi

# ── 5. 启动训练 ───────────────────────────────────────────
echo ""
echo "[5/5] 启动训练..."
echo "  输出目录: $OUTPUT_DIR"
echo ""

LORA_FLAG=""
if [ "$NO_LORA" = "1" ]; then
    LORA_FLAG="--no-lora"
fi

python3 train_sft.py \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 1 \
    --grad-accum 8 \
    --max-length 2048 \
    $LORA_FLAG \
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo ""
echo "========================================================"
echo "✅ 训练完成！"
echo "  模型保存至: $PROJECT_DIR/$OUTPUT_DIR/final"
echo "  训练日志:   $PROJECT_DIR/$OUTPUT_DIR/train.log"
echo ""
echo "  验证训练效果:"
echo "  python3 verify_training.py --model-dir $OUTPUT_DIR/final"
echo "========================================================"
