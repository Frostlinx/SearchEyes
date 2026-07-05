# Vision-DeepResearch 实验计划

> 指导文件 — 基于 Claude × Codex 双方辩论结论（2026-04-14）

## 背景与目标

学长指示：验证"图搜图"两种方案可行性，在进入 GRPO 训练前稳定动作空间。

参考论文：Vision-DeepResearch (arXiv:2601.22060v3, CUHK MMLab, 2026-03-23)

**核心判断**：算法有效性优先，训练其次。GRPO 之前必须完成本实验。

---

## 两种方案定义

### Option A — 直接图搜图（baseline，已实现）
```
image → Qwen3-VL-Embedding-2B → 向量 → ChromaDB query → top-k结果
```
现状：L1=100%（GT in top-20），这就是 Option A 的基线数字。

### Option B — Text Bridge（待实现，本实验核心）
```
image → Qwen3-VL-4B caption → 关键词/描述文本 → text embedding → ChromaDB+BM25 query → top-k结果
```
来自 Vision-DeepResearch 的 Text Bridging 思路。
论文数据：WIS=16% → WIS+TS=29.3%（文本搜索大幅提升）

---

## 文件结构

```
experiments/vision_deepresearch/
  README.md              ← 本文件：计划 + 结论记录
  config.py              ← 路径、端口、eval参数（单一配置源）
  caption_bridge.py      ← Option B核心：image → caption/keywords
  retrieval_adapters.py  ← A/B/A+B 三种检索方式封装
  crop_engine.py         ← 多尺度裁剪（zoom_search预备）
  metrics.py             ← recall@k, hit@1, zoom_recovery
  run_ablation.py        ← 主实验：A vs B on fixed eval split → JSON
  run_zoom_fsm_eval.py   ← FSM eval with zoom_search（第二阶段）
  prompts/
    caption_bridge.txt   ← VLM caption生成的固定prompt
  results/               ← 实验结果JSON输出
```

---

## 执行计划（3天）

### Day 1：实现 Option B + 跑基础对比
1. `caption_bridge.py` — 用 Qwen3-VL-4B 给图片生成文字描述
2. `retrieval_adapters.py` — 封装 A、B、A+B 三种方式
3. `run_ablation.py` — 在 data/tasks/research_tasks_v2.jsonl 前50条上跑 A vs B

### Day 2：分析结果 + zoom_search 设计
1. 分析 hit@1 / hit@5 / hit@20 三档数字
2. 决策：B更好→集成进search_controller；B更差→直接放弃
3. 设计 `zoom_search` action stub（enable_zoom_search=False 默认关）

### Day 3：FSM集成 + 写入论文消融实验
1. 加 `zoom_search` 到 transition_engine（不破坏现有eval）
2. 在 eval_research.py 加 `zoom_recovery@k` 指标
3. 结论写进论文 Section 消融实验

---

## 成功标准

- Option B hit@20 ≥ Option A hit@20 → 集成进主项目
- Option B hit@20 < Option A hit@20 → 放弃，节省变量，专注GRPO
- 无论哪种结果都要写进论文作为消融证据

---

## 关键设计约束（来自Codex审查）

1. **不污染现有L1** — `zoom_search` 用独立指标 `zoom_recovery@k`，不混入原 L1
2. **enable_zoom_search=False 默认关闭** — 不影响当前eval路径
3. **A vs B 用相同eval split** — 保证可比性，用 research_tasks_v2.jsonl 前50条
4. **Caption hallucination风险** — 记录失败case，特别是Wikipedia专有名词匹配

---

## 第一阶段实验结果 (2026-04-15)

| 方案 | hit@1 | hit@5 | hit@20 | 结论 |
|---|---|---|---|---|
| Option A (image→embed) | 100% | 100% | 100% | **Trivial self-match**，不是有效baseline |
| B_lazy (GT caption) | 72% | 96% | 100% | Text Bridge理论上限 |
| **B_vlm (Qwen3-VL caption)** | 50% | **98%** | **100%** | **真实可用性能** |

**核心发现**：
1. Text Bridge 完全可行（hit@20=100%）
2. 反直觉：VLM caption > GT caption（98% vs 96% @top-5）
3. Option A的100%是artifact，query与KB是同一张图

---

## 第二阶段待办（下一步行动）

### 方案1：Crop-as-Query（推zoom_search的关键铺垫）
**目的**：暴露Option A的真实性能，验证Text Bridge在不完整视觉信息下的鲁棒性

**做法**：
1. 对每个 GT 图片生成 50% 面积的 crop（center/top_half/bottom_half 三个位置）
2. 用 crop 作为 query，分别跑 A_crop 和 B_vlm_crop
3. 对比 A_crop vs A_full 和 B_vlm_crop vs B_vlm_full 的 hit@k 下降幅度

**预期**：
- A_crop hit@1 大幅下降（图片embed对遮挡不鲁棒）
- B_vlm_crop hit@1 小幅下降（VLM描述聚焦可见内容）
- 若B_vlm_crop hit@20 仍 ≥ 90%，Text Bridge + crop 可作为 zoom_search 的核心机制

**新增文件**：
- `run_crop_ablation.py` — 调用 `crop_engine.py` 生成 crop，复用 A/B retrievers
- `results/ablation_crop_*.json`

### 方案2：Hold-out KB（论文消融用，最严谨的检验）
**目的**：模拟"用新图去查KB"的真实场景，消除self-match artifact

**做法**：
1. 从2000条KB里随机抽50条做**hold-out set**，从ChromaDB里删除
2. 用这50条的图片作为query，看A和B能否找回**语义最相似**但不是自己的条目
3. 定义新指标：top-k overlap（与semantic judge标注的相关条目重合率）

**预期**：
- A 性能大幅下降（从100%掉到未知）
- B_vlm 下降幅度小（因为文字描述泛化性好）
- 这才是论文能报的**真实 retrieval 能力**

**新增文件**：
- `run_holdout_ablation.py` — 建立hold-out KB副本，跑A/B
- `holdout_labeler.py` — 用 Qwen3-VL-4B 标注 query 与候选的语义相似度（0/1）
- `results/ablation_holdout_*.json`

### 方案1 vs 方案2 优先级
- **先做方案1**（3小时）— 直接推动 zoom_search 的设计依据
- **再做方案2**（1天）— 论文消融章节的严谨性证据
- 两个都要在 GRPO 之前完成

---

## 与主项目的接口

- KB：`data/wit_kb_v2/chroma_db/`，collection `wit_knowledge_v2_qwen`
- 任务：`data/tasks/research_tasks_v2.jsonl`（200条，用前50条）
- Embedding server：`http://localhost:8766`
- VLM model：`Qwen3-VL-4B-Instruct/`（本地，按需加载）
- 主RAG：`searcheyes/multimodal_rag.py`（可直接import）
