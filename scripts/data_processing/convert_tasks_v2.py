"""
convert_tasks_v2.py — 批量转换 rag_tasks.jsonl → research_tasks_v2.jsonl
==========================================================================
学长 philosophy: task goal 定义 agent 学什么行为。
"比较价格后购买" → 购物行为
"分析内容后引用证据" → 研究行为
这不是简单的文本替换，是 env 定义的一部分。
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from searcheyes.research_contracts import (
    ResearchTask, ResearchDifficulty, FactSet,
)

# ═══════════════════════════════════════════════════════════
# Template-level goal cleaning (6 templates → 6 research equivalents)
# ═══════════════════════════════════════════════════════════

_GOAL_TEMPLATES = [
    # (regex pattern, replacement) — ordered most specific first
    (
        r"放大查看每个商品的图片进行视觉调研，找到与(「[^」]+」)最相关的商品并购买",
        r"查看各文档图片进行视觉调研，找到关于\1最相关的文档并引用相关证据",
    ),
    (
        r"搜索商品并放大查看图片，确认哪个商品展示了(「[^」]+」)后购买",
        r"搜索并查看文档图片，确认哪个文档展示了\1后引用相关证据",
    ),
    (
        r"搜索(「[^」]+」)相关商品，比较价格后购买最便宜的",
        r"搜索\1相关文档，分析内容后引用最相关的证据",
    ),
    (
        r"查看各商品图片，找到展示了(「[^」]+」)的商品并购买",
        r"查看各文档图片，找到展示了\1的文档并引用相关证据",
    ),
    (
        r"搜索商品，找到与(「[^」]+」)相关的产品并购买",
        r"搜索文档，找到关于\1的相关文档并引用证据",
    ),
    (
        r"根据图片内容判断哪个商品展示了(「[^」]+」)，将其加入购物车",
        r"根据图片内容判断哪个文档展示了\1，引用相关证据",
    ),
]

# Fallback word-level replacements (in case of edge cases)
_WORD_FALLBACKS = [
    ("比较价格后购买最便宜的", "分析内容后引用最相关的证据"),
    ("将其加入购物车", "引用相关证据"),
    ("并购买", "并引用相关证据"),
    ("后购买", "后引用相关证据"),
    ("购买", "引用"),
    ("产品", "文档"),
    ("商品", "文档"),
    ("找到与", "找到关于"),
]


def clean_goal(goal: str) -> str:
    """Template-first cleaning, then word-level fallback."""
    for pattern, replacement in _GOAL_TEMPLATES:
        new_goal = re.sub(pattern, replacement, goal)
        if new_goal != goal:
            return new_goal
    # Fallback: word-level
    for old, new in _WORD_FALLBACKS:
        goal = goal.replace(old, new)
    return goal


def convert_one(vt_data: dict) -> dict:
    """Convert one v1 task dict to v2 ResearchTask dict."""
    goal = clean_goal(vt_data.get("goal", ""))
    gt_wit_id = vt_data.get("ground_truth_wit_id", "")
    gt_caption = vt_data.get("ground_truth_caption", "")

    # Build fact_set
    fact_set = {
        "facts": [{
            "wit_id": gt_wit_id,
            "caption": gt_caption,
            "is_primary": True,
        }]
    }

    # Convert wit_bindings: product_id → result_id (keep both for compat)
    wit_bindings = []
    for b in vt_data.get("wit_bindings", []):
        nb = dict(b)
        nb["result_id"] = nb.pop("product_id", nb.get("result_id", 0))
        wit_bindings.append(nb)

    return {
        "task_id": vt_data["task_id"],
        "goal": goal,
        "difficulty": vt_data.get("difficulty", "medium"),
        "fact_set": fact_set,
        "ground_truth_wit_id": gt_wit_id,
        "ground_truth_caption": gt_caption,
        "max_steps": vt_data.get("dag_depth", 5) * 2,
        "min_citations_for_submit": 1,
        "wit_bindings": wit_bindings,
        "difficulty_tags": vt_data.get("difficulty_tags", []),
        "requires_rag": vt_data.get("requires_rag", True),
    }


def main():
    src = Path(__file__).parent / "data" / "tasks" / "rag_tasks.jsonl"
    dst = Path(__file__).parent / "data" / "tasks" / "research_tasks_v2.jsonl"

    tasks_v1 = []
    with open(src) as f:
        for line in f:
            tasks_v1.append(json.loads(line))

    print(f"Source: {src}")
    print(f"Tasks to convert: {len(tasks_v1)}")

    # Convert
    tasks_v2 = []
    template_hits = {i: 0 for i in range(len(_GOAL_TEMPLATES))}
    fallback_count = 0

    for vt in tasks_v1:
        original_goal = vt["goal"]
        # Check which template matched
        matched = False
        for i, (pattern, _) in enumerate(_GOAL_TEMPLATES):
            if re.search(pattern, original_goal):
                template_hits[i] += 1
                matched = True
                break
        if not matched:
            fallback_count += 1

        v2 = convert_one(vt)
        tasks_v2.append(v2)

    # Write
    with open(dst, "w", encoding="utf-8") as f:
        for t in tasks_v2:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"\nOutput: {dst}")
    print(f"Converted: {len(tasks_v2)} tasks")

    # Stats
    print("\n=== Template match stats ===")
    template_names = [
        "放大查看...视觉调研...购买",
        "搜索商品并放大...确认...购买",
        "搜索相关商品...比较价格...最便宜",
        "查看各商品图片...展示了...购买",
        "搜索商品...找到与...产品...购买",
        "根据图片内容判断...加入购物车",
    ]
    for i, name in enumerate(template_names):
        print(f"  T{i+1} [{template_hits[i]:3d}x] {name}")
    print(f"  Fallback: {fallback_count}")
    total_matched = sum(template_hits.values())
    print(f"  Coverage: {total_matched}/{len(tasks_v1)} ({100*total_matched/len(tasks_v1):.1f}%)")

    # Validate via ResearchTask.load_from_dict
    print("\n=== Validation ===")
    errors = []
    for t_dict in tasks_v2:
        rt = ResearchTask.load_from_dict(t_dict)
        valid, errs = rt.validate()
        if not valid:
            errors.append((t_dict["task_id"], errs))

    if errors:
        print(f"FAILED: {len(errors)} tasks with validation errors")
        for tid, errs in errors[:5]:
            print(f"  {tid}: {errs}")
    else:
        print(f"ALL {len(tasks_v2)} tasks passed validation")

    # Spot check: show 3 conversions
    print("\n=== Spot check (3 samples) ===")
    samples = [0, 2, 13]  # one from each main template
    for idx in samples:
        if idx < len(tasks_v1):
            print(f"\n  {tasks_v1[idx]['task_id']}:")
            print(f"    v1: {tasks_v1[idx]['goal'][:80]}")
            print(f"    v2: {tasks_v2[idx]['goal'][:80]}")

    # Check for residual shopping terms
    print("\n=== Residual shopping terms check ===")
    shopping_terms = ["商品", "产品", "购买", "购物车", "价格", "便宜"]
    residual = []
    for t in tasks_v2:
        for term in shopping_terms:
            if term in t["goal"]:
                residual.append((t["task_id"], term, t["goal"][:60]))
                break

    if residual:
        print(f"WARNING: {len(residual)} tasks still contain shopping terms:")
        for tid, term, g in residual[:5]:
            print(f"  {tid} [{term}]: {g}")
    else:
        print("CLEAN: No residual shopping terms in any goal")

    # Difficulty distribution
    from collections import Counter
    diffs = Counter(t["difficulty"] for t in tasks_v2)
    print(f"\n=== Difficulty distribution ===")
    for d, c in diffs.most_common():
        print(f"  {d}: {c}")

    print("\nDone.")


if __name__ == "__main__":
    main()
