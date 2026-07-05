# 技能驱动任务生成报告

## 概览

- **生成时间**: 1773390936.0
- **总任务数**: 100
- **有效任务数**: 100
- **通过率**: 100.0%

## 技能图谱

- **技能总数**: 14
- **技能组合**: 12
- **难度分布**: {1: 3, 2: 3, 3: 3, 4: 2, 5: 1}

## 任务统计

### 难度分布

| 难度 | 数量 | 占比 |
|------|------|------|
| 1 | 25 | 25.0% |
| 2 | 30 | 30.0% |
| 3 | 25 | 25.0% |
| 4 | 15 | 15.0% |
| 5 | 5 | 5.0% |

### 技能覆盖

| 技能 | 使用次数 |
|------|----------|
| click_product | 101 |
| search | 79 |
| buy | 63 |
| zoom | 39 |
| sort | 24 |
| scan | 22 |
| aggregate | 22 |
| filter | 20 |
| back | 14 |
| memory | 14 |

### 特殊要求

- **需要 zoom**: 34 (34.0%)
- **需要 memory**: 14 (14.0%)
- **有干扰项**: 11 (11.0%)
- **平均技能数**: 4.12

## 验证结果

- **可验证**: 100/100 (100.0%)
- **可解**: 100/100 (100.0%)
- **步数合理**: 100/100 (100.0%)
- **技能匹配**: 100/100 (100.0%)
- **平均步数**: 3.9

## 任务示例


### 难度 1

**vt_proposed_2574a02d**: 鞋子然后购买
- 技能: click_product, buy
- 步数: 2-2

**vt_proposed_e479b543**: 搜索商品鞋子
- 技能: search
- 步数: 1-1


### 难度 2

**vt_proposed_8bd62a0d**: 鞋子，放大查看价格和库存然后购买
- 技能: click_product, zoom, buy
- 步数: 3-3

**vt_proposed_c3fb21bd**: 搜索商品鞋子
- 技能: search, sort, click_product
- 步数: 3-3


### 难度 3

**vt_proposed_aec8dd62**: 搜索商品最贵的鞋子然后购买
- 技能: search, scan, aggregate, click_product, buy
- 步数: 4-5

**vt_proposed_7c7bc87a**: 搜索商品最贵的鞋子然后购买
- 技能: search, scan, aggregate, click_product, buy
- 步数: 4-5


### 难度 4

**vt_proposed_e0123d8b**: 搜索商品鞋子，放大查看价格和库存，记住之前看到的商品然后购买
- 技能: search, click_product, zoom, back, memory, click_product, buy
- 步数: 6-8

**vt_proposed_567764cf**: 搜索商品鞋子，放大查看价格和库存，记住之前看到的商品然后购买
- 技能: search, click_product, zoom, back, memory, click_product, buy
- 步数: 6-8


### 难度 5

**vt_proposed_a5ec1f0b**: 搜索商品green色，价格在300-1184元之间的鞋子，放大查看价格和库存，比较多个商品，记住之前看到的商品然后购买
- 技能: search, filter, sort, zoom, click_product, back, memory, click_product, zoom, compare, buy
- 步数: 8-12

**vt_proposed_b80197ef**: 搜索商品价格低于500元的鞋子，放大查看价格和库存，比较多个商品，记住之前看到的商品然后购买
- 技能: search, filter, sort, zoom, click_product, back, memory, click_product, zoom, compare, buy
- 步数: 8-12


## 使用方法

```python
# 加载任务
import json
with open('data/tasks/visual_tasks_skill_driven.jsonl', 'r', encoding='utf-8') as f:
    tasks = [json.loads(line) for line in f]

# 按难度筛选
easy_tasks = [t for t in tasks if t['difficulty'] == 1]
hard_tasks = [t for t in tasks if t['difficulty'] >= 4]

# 按技能筛选
zoom_tasks = [t for t in tasks if t['requires_zoom']]
memory_tasks = [t for t in tasks if t['requires_memory']]
```
