# SearchEyes

**Towards Frontier Multimodal Deep Search Intelligence via Search World Simulation**

[Paper](https://arxiv.org/abs/xxxx.xxxxx) | [Project Page](https://github.com/searcheyes)

SearchEyes uses a typed knowledge graph as the backbone of a simulated search world that unifies training data synthesis, environment simulation, and RL reward signals — eliminating the structural disconnect that limits existing multimodal search agent pipelines.

<p align="center">
  <img src="assets/overview.png" width="90%" alt="SearchEyes Overview"/>
</p>

## Highlights

- **Unified Search World Simulation**: A single typed knowledge graph simultaneously defines data synthesis, retrieval environment, and step-level reward anchors.
- **Perception-Knowledge Chain (PKC)**: Graph-constrained multi-hop path sampling with P–K alternation, anti-shortcut filtering, and information concealment.
- **Hop-Anchored Policy Optimization (HaPO)**: Step-level credit assignment via gold entity anchors — no separately trained process reward model needed.
- **State-of-the-Art Results**: SearchEyes-27B improves over the strongest open-source baseline by **6.2 points** on average across six benchmarks.

## Results

| Model | SimpleVQA | VDR | MMSearch | LiveVQA | BC-VL | FVQA | Avg. |
|-------|-----------|-----|----------|---------|-------|------|------|
| SearchEyes-9B | 75.4 | 28.3 | 69.2 | 66.1 | 42.3 | 74.2 | 59.3 |
| SearchEyes-27B | 80.9 | 39.4 | 82.4 | 77.3 | 49.3 | 79.1 | 68.1 |
| OpenSearch-VL-32B | 76.2 | 33.8 | 72.3 | 70.5 | 43.8 | 74.7 | 61.9 |
| Gemini-3.1-Pro (Agentic) | – | – | 86.1 | 76.6 | 64.1 | 84.0 | – |

## Installation

```bash
pip install -e .

# For training
pip install -e ".[train]"

# For evaluation
pip install -e ".[eval]"

# For data processing
pip install -e ".[data]"
```

## Project Structure

```
searcheyes/
├── searcheyes/              # Core Python package
│   ├── hapo.py              # Hop-Anchored Policy Optimization
│   ├── pgkc_synthesizer.py  # PKC question synthesis
│   ├── pgkc_graph.py        # Knowledge graph operations
│   ├── pgkc_filter.py       # Anti-shortcut filtering
│   ├── pgkc_pipeline.py     # End-to-end PKC pipeline
│   ├── sft_synthesis.py     # SFT trajectory synthesis
│   ├── rag_engine.py        # Hybrid BM25+dense retrieval engine
│   ├── vdr_agent.py         # Multi-turn search agent
│   ├── vdr_tools.py         # Tool implementations
│   ├── reward_fn.py         # Reward function for verl
│   └── ...
├── scripts/
│   ├── training/            # Training scripts (SFT, GRPO, HaPO)
│   ├── evaluation/          # Evaluation & benchmarking
│   └── data_processing/     # Data preparation & indexing
├── configs/                 # DeepSpeed & tool configs
├── data/                    # VisSearch Bench & task data
├── experiments/             # Ablation experiments
└── tests/
```

## Quick Start

### 1. Data Synthesis (PKC Pipeline)

```bash
python -m searcheyes.pgkc_pipeline \
    --kg-path data/wikidata5m/ \
    --output-dir data/pkc_output/ \
    --num-questions 10000
```

### 2. SFT Training

```bash
bash scripts/training/run_train.sh
```

### 3. HaPO (RL Training)

```bash
bash scripts/training/run_rl.sh
```

### 4. Evaluation

```bash
bash scripts/evaluation/run_eval.sh --model-path <checkpoint> --benchmark mmsearch
```

## VisSearch Bench

We introduce **VisSearch Bench**, a 1000-question benchmark for multi-hop visual search with guaranteed P–K alternating structure. Available at `data/vissearch_bench.json`.

```python
import json

with open("data/vissearch_bench.json") as f:
    bench = json.load(f)

print(f"Total questions: {len(bench)}")
print(f"Example: {bench[0]['question']}")
```

## Method Overview

### Perception-Knowledge Chains (PKC)

Starting from a typed knowledge graph (Wikidata5M ∩ Wiki6M ∩ Wikipedia images), PKC samples constrained multi-hop paths with:
- **P–K alternation**: Visual perception hops alternate with knowledge retrieval hops
- **Treewidth ≥ 2**: Disambiguating constraint edges prevent trivial single-chain solutions
- **Domain diversity**: Adjacent hops span different semantic domains (Person, Work, Org, Geo)
- **Anti-shortcut filtering**: Hub exclusion, predicate blacklist, and deduplication

### Hop-Anchored Policy Optimization (HaPO)

HaPO addresses sparse trajectory-level rewards by:
1. Grouping trajectories by shared gold entity anchors at each hop
2. Computing step-level advantages within each anchor group
3. Blending hop-level and trajectory-level signals: `A_final = α·A_ep + (1-α)·A_hop`
4. Fatal-aware masking to suppress degenerate trajectory suffixes

## Citation

```bibtex
@article{jiao2026searcheyes,
  title={SearchEyes: Towards Frontier Multimodal Deep Search Intelligence via Search World Simulation},
  author={Jiao, Zhengbo and Cheng, Yiming and Jiang, Yilei and Feng, Kaituo and Huang, Rui and Jiang, Tianyi and Tian, Juanxi and Wang, Qunzhong and Chen, Tailai and Wei, Qianshan and Xiao, Chuan and Rong, Shanyu and Li, Yangfu and Zhou, Yanhan and Zhang, Yifan and Yue, Xiangyu},
  journal={arXiv preprint},
  year={2026}
}
```

## License

This project is released under the [Apache 2.0 License](LICENSE).

## Acknowledgments

We thank the teams behind Wikidata5M, Wiki6M (OVEN-Wiki), and the open-source RL training frameworks (verl, TRL) that made this work possible.
