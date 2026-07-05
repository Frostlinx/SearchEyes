"""
query_rewriter.py — Query Rewrite v1（正则规则版）
===================================================
将脏的任务 goal 清洗为适合 RAG 检索的核心语义短语。

规则（优先级从高到低）：
  1. 有「...」或"..."引号时：提取引号内 caption hint（主要语义载体）
  2. 无引号时：去掉模板词后取剩余核心短语
  3. 有结构化约束（价格比较/cheapest 等）：这类是决策约束，不进入检索 query

设计原则：
  - 只做 text 清洗，不引入额外模型
  - 不截断真正有信息量的语义内容
  - 结果应该接近人类手写的搜索词，而不是完整句子
"""

from __future__ import annotations

import re


# 中文模板词（任务 goal 中的包装语言，无检索信息量）
_ZH_TEMPLATE_TOKENS = [
    r"搜索商品[，,]?",
    r"找到与?",
    r"最?相关的商品并购买",
    r"相关的产品并购买",
    r"相关的产品",
    r"并购买",
    r"放大查看每个商品的图片进行视觉调研[，,]?",
    r"放大查看[^，,。]*?[，,]",
    r"浏览搜索结果[，,]?",
    r"识别展示了?",
    r"的商品并购买",
    r"搜索并购买最便宜的",
    r"比较价格后购买最便宜的",
    r"查看各商品图片[，,]?",
    r"找到展示了?",
    r"进行视觉调研[，,]?",
    r"视觉调研[，,]?",
]

# 英文模板词
_EN_TEMPLATE_TOKENS = [
    r"search for and buy",
    r"search and buy",
    r"find and buy",
    r"browse.*?and buy the",
    r"look for",
    r"purchase the",
    r"buy the",
    r"find the",
    r"related to",
    r"associated with",
]

# 决策约束词（价格/比较类），不应进入检索 query
_DECISION_CONSTRAINT_PATTERNS = [
    r"最便宜",
    r"cheapest",
    r"compare.*price",
    r"比较价格",
    r"最低价",
    r"lowest price",
]


def rewrite(goal: str) -> str:
    """将任务 goal 清洗为检索 query。

    >>> rewrite("搜索商品，找到与「a stone bridge over a river」最相关的产品并购买")
    'a stone bridge over a river'

    >>> rewrite("放大查看每个商品的图片进行视觉调研，找到与「Location within Genesee County」最相关的商品并购买")
    'Location within Genesee County'

    >>> rewrite("搜索并购买最便宜的商品")
    '搜索并购买最便宜的商品'  # 决策约束型，原样保留（后续单独处理）
    """
    goal = goal.strip()

    # 规则 1：提取「...」或"..."或"..."中的内容（最可靠的语义载体）
    for pattern in [r'「([^」]+)」', r'"([^"]+)"', r'"([^"]+)"', r"'([^']+)'"]:
        m = re.search(pattern, goal)
        if m:
            extracted = m.group(1).strip()
            if len(extracted) > 3:  # 太短的引号内容不可信
                return extracted

    # 规则 2a：决策约束型任务，原样保留（不适合做检索 query，但不要破坏）
    for pattern in _DECISION_CONSTRAINT_PATTERNS:
        if re.search(pattern, goal, re.IGNORECASE):
            return goal  # 这类任务 query rewrite 意义不大，保持原样

    # 规则 2b：去模板词后取剩余核心短语
    cleaned = goal
    for pattern in _ZH_TEMPLATE_TOKENS:
        cleaned = re.sub(pattern, "", cleaned)
    for pattern in _EN_TEMPLATE_TOKENS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    # 去掉首尾标点和空白
    cleaned = cleaned.strip("，,。.　 \t\n「」\"\"''")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # 如果清洗后太短或为空，回退到原始 goal
    if len(cleaned) < 4:
        return goal

    return cleaned


def batch_rewrite(goals: list[str]) -> list[str]:
    """批量清洗。"""
    return [rewrite(g) for g in goals]


# ── 简单测试 ────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        "搜索商品，找到与「a stone bridge over a river in a wooded area」最相关的产品并购买",
        "放大查看每个商品的图片进行视觉调研，找到与「Location within Genesee County (red) and the administered...」最相关的商品并购买",
        "搜索并购买最便宜的商品",
        "浏览搜索结果，识别展示了「Ahmedabad Airport domestic terminal」的商品并购买",
        "search for and buy the product related to \"a historic church in a European city\"",
        "放大查看图片，找到与「Jastrow illusion toy railway tracks」相关的产品",
        "搜索商品，找到展示了「Cheerleaders from 1922」的商品并购买",
    ]

    for goal in test_cases:
        result = rewrite(goal)
        changed = "✓" if result != goal else "="
        print(f"{changed} IN:  {goal[:80]}")
        print(f"   OUT: {result}")
        print()
