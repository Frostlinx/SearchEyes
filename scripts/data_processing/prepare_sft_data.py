#!/usr/bin/env python3
"""
prepare_sft_data.py — 把 SFT demo JSONL 转换为 HuggingFace 训练格式
=====================================================================
1. 扫描 data/benchmark/batch_rollouts/ 下所有 sft_demos/*.jsonl
2. Fix legacy paths to current project root
3. 过滤掉截图文件不存在的样本
4. 转换为 Qwen3-VL apply_chat_template 兼容的格式
5. 输出到 data/sft_train.jsonl 和 data/sft_eval.jsonl
"""

from __future__ import annotations
import json
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
ROLLOUT_DIR = DATA_DIR / "benchmark" / "batch_rollouts"
OUTPUT_TRAIN = DATA_DIR / "sft_train.jsonl"
OUTPUT_EVAL  = DATA_DIR / "sft_eval.jsonl"

# 旧路径前缀 → 新路径前缀的映射规则
OLD_PATH_PREFIXES = [
    ".",
    ".",
]


def fix_path(old_path: str) -> str:
    """把 Mac 开发机路径替换为当前服务器路径"""
    for prefix in OLD_PATH_PREFIXES:
        if old_path.startswith(prefix):
            rel = old_path[len(prefix):]
            return str(PROJECT_ROOT / rel.lstrip("/"))
    return old_path


def fix_prompt(prompt_list: list) -> list:
    """递归修复 prompt 中所有 image_url / image 路径"""
    fixed = []
    for msg in prompt_list:
        new_msg = dict(msg)
        if isinstance(msg.get("content"), list):
            new_content = []
            for part in msg["content"]:
                new_part = dict(part)
                if part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    new_url = fix_path(url)
                    new_part["image_url"] = {"url": new_url}
                elif part.get("type") == "image":
                    new_part["image"] = fix_path(part["image"])
                new_content.append(new_part)
            new_msg["content"] = new_content
        fixed.append(new_msg)
    return fixed


def convert_to_chat_format(record: dict) -> dict | None:
    """
    把单条 SFT demo 记录转换为 chat 格式：
    messages = [
        {"role": "user",      "content": [image + text]},
        {"role": "assistant", "content": chosen_action_json},
    ]
    """
    prompt = record.get("prompt", [])
    chosen = record.get("chosen", "")
    reward = record.get("reward", 0.0)

    if not prompt or not chosen:
        return None

    # 修复路径
    fixed_prompt = fix_prompt(prompt)

    # 检查截图是否存在
    for msg in fixed_prompt:
        for part in msg.get("content", []):
            if part.get("type") == "image_url":
                img_path = part["image_url"]["url"]
                if img_path.startswith("/") and not Path(img_path).exists():
                    return None  # 截图不存在，跳过
            elif part.get("type") == "image":
                img_path = part["image"]
                if img_path.startswith("/") and not Path(img_path).exists():
                    return None

    # 构建 assistant 回复（结构化 JSON）
    try:
        chosen_obj = json.loads(chosen)
        assistant_text = json.dumps(chosen_obj, ensure_ascii=False)
    except Exception:
        assistant_text = chosen

    messages = fixed_prompt + [
        {"role": "assistant", "content": assistant_text}
    ]

    return {
        "messages": messages,
        "task_id": record.get("task_id", ""),
        "step_idx": record.get("step_idx", 0),
        "reward": reward,
    }


def collect_all_samples() -> list[dict]:
    samples = []
    jsonl_files = sorted(ROLLOUT_DIR.rglob("sft_demos/*.jsonl"))
    print(f"找到 {len(jsonl_files)} 个 JSONL 文件")

    skipped = 0
    for jf in jsonl_files:
        with open(jf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    converted = convert_to_chat_format(record)
                    if converted is not None:
                        samples.append(converted)
                    else:
                        skipped += 1
                except Exception as e:
                    skipped += 1

    print(f"有效样本: {len(samples)}, 跳过: {skipped}")
    return samples


def split_and_save(samples: list[dict], eval_ratio: float = 0.05, seed: int = 42):
    rng = random.Random(seed)
    rng.shuffle(samples)

    n_eval = max(1, int(len(samples) * eval_ratio))
    eval_samples = samples[:n_eval]
    train_samples = samples[n_eval:]

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_TRAIN, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(OUTPUT_EVAL, "w", encoding="utf-8") as f:
        for s in eval_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"训练集: {len(train_samples)} 条 → {OUTPUT_TRAIN}")
    print(f"验证集: {len(eval_samples)} 条 → {OUTPUT_EVAL}")


if __name__ == "__main__":
    print("=" * 60)
    print("SFT 数据预处理")
    print("=" * 60)
    samples = collect_all_samples()
    if not samples:
        print("❌ 没有找到有效样本，请检查路径和截图文件")
        sys.exit(1)
    split_and_save(samples)
    print("✅ 数据预处理完成")
