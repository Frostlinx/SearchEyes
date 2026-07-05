#!/usr/bin/env python3
"""
gen_tasks_v2.py — 从 wit_kb_v2/meta.jsonl 重新生成 research_tasks_v2.jsonl
所有 ground_truth_wit_id 指向新 KB（witb_ 前缀条目）。
"""
import json, random
from pathlib import Path

ROOT      = Path("/root/autodl-tmp/QWEN/QWEN-project")
META      = ROOT / "data/wit_kb_v2/meta.jsonl"
OUT       = ROOT / "data/tasks/research_tasks_v2.jsonl"
BACKUP    = ROOT / "data/tasks/research_tasks_v2_old.jsonl"
N_TASKS   = 200
N_DISTRACTORS = 5   # GT 之外的干扰项数
SEED      = 42
random.seed(SEED)

# ── 读 KB ─────────────────────────────────────────────────
entries = [json.loads(l) for l in META.read_text().splitlines() if l.strip()]
# 过滤：有图片 + caption 够长
valid = [e for e in entries if e.get("image_filename") and len(e.get("caption","")) >= 20]
print(f"KB 有效条目: {len(valid)}")
assert len(valid) >= N_TASKS + N_DISTRACTORS, "KB 条目不足"

# ── 备份旧任务 ────────────────────────────────────────────
if OUT.exists() and not BACKUP.exists():
    BACKUP.write_bytes(OUT.read_bytes())
    print(f"备份旧任务 → {BACKUP}")

# ── 难度分配 ──────────────────────────────────────────────
# easy:medium:hard = 1:2:1
difficulty_pool = (
    ["easy"]   * (N_TASKS // 4) +
    ["medium"] * (N_TASKS // 2) +
    ["hard"]   * (N_TASKS // 4)
)
random.shuffle(difficulty_pool)

DIFF_TAGS = {
    "easy":   ["rag_knowledge"],
    "medium": ["rag_knowledge", "visual_ambiguity"],
    "hard":   ["rag_knowledge", "visual_ambiguity", "zoom_required"],
}
DIFF_STEPS = {"easy": 8, "medium": 10, "hard": 12}

# ── 生成任务 ──────────────────────────────────────────────
# 随机选 N_TASKS 条作为 GT（不重复）
gt_entries = random.sample(valid, N_TASKS)
all_ids    = {e["wit_id"] for e in valid}

tasks = []
for idx, gt in enumerate(gt_entries):
    diff   = difficulty_pool[idx]
    gt_id  = gt["wit_id"]
    cap    = gt["caption"]
    title  = gt["page_title"]

    # 干扰项：从非 GT 的条目随机抽
    distractor_pool = [e for e in valid if e["wit_id"] != gt_id]
    distractors = random.sample(distractor_pool, N_DISTRACTORS)

    # wit_bindings：GT 插入随机位置
    candidates = distractors[:]
    gt_pos = random.randint(0, N_DISTRACTORS)
    candidates.insert(gt_pos, gt)

    wit_bindings = [
        {
            "wit_id":        e["wit_id"],
            "caption":       e["caption"],
            "image_filename":e["image_filename"],
            "result_id":     i + 1,
        }
        for i, e in enumerate(candidates)
    ]

    # goal：基于 caption 和 title 生成研究语义目标
    goal = (
        f"查看各文档图片进行视觉调研，找到关于「{cap[:60]}」"
        f"最相关的文档并引用相关证据"
    )

    task = {
        "task_id":              f"task_{idx:04d}",
        "goal":                 goal,
        "difficulty":           diff,
        "fact_set": {
            "facts": [{
                "wit_id":     gt_id,
                "caption":    cap,
                "page_title": title,
                "is_primary": True,
            }]
        },
        "ground_truth_wit_id":  gt_id,
        "ground_truth_caption": cap,
        "max_steps":            DIFF_STEPS[diff],
        "min_citations_for_submit": 1,
        "wit_bindings":         wit_bindings,
        "difficulty_tags":      DIFF_TAGS[diff],
        "requires_rag":         True,
    }
    tasks.append(task)

# ── 写出 ─────────────────────────────────────────────────
with open(OUT, "w", encoding="utf-8") as f:
    for t in tasks:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")

print(f"写入 {len(tasks)} 条任务 → {OUT}")

# ── 验证 ─────────────────────────────────────────────────
kb_ids = {e["wit_id"] for e in valid}
gt_ids = [t["ground_truth_wit_id"] for t in tasks]
all_in_kb = all(gid in kb_ids for gid in gt_ids)
dup_gt    = len(gt_ids) - len(set(gt_ids))

diff_dist = {d: sum(1 for t in tasks if t["difficulty"]==d) for d in ["easy","medium","hard"]}
print(f"所有GT在KB中: {all_in_kb}")
print(f"重复GT: {dup_gt}")
print(f"难度分布: {diff_dist}")
print(f"\n[{'PASS' if all_in_kb and dup_gt==0 else 'FAIL'}] 任务生成完成")
print("\n第一条任务预览:")
print(json.dumps(tasks[0], indent=2, ensure_ascii=False)[:600])
