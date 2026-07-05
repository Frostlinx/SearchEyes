"""
summarize_results.py — 把 results/*.json 汇总成对比表

用法:
  python experiments/vision_deepresearch/summarize_results.py
"""
from __future__ import annotations
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def main():
    rows = []
    for fp in sorted(RESULTS_DIR.glob("ablation_*.json")):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"skip {fp.name}: {e}")
            continue
        xform = data.get("query_transform", "exact")
        for run_key, run in data.get("runs", {}).items():
            m = run.get("metric", {})
            rows.append({
                "file": fp.name,
                "mode": m.get("mode", run_key),
                "transform": xform,
                "n": m.get("n_tasks", 0),
                "hit@1":  f"{m.get('hit_at_1',0)*100:.0f}%",
                "hit@5":  f"{m.get('hit_at_5',0)*100:.0f}%",
                "hit@20": f"{m.get('hit_at_20',0)*100:.0f}%",
                "median_rank": m.get("median_rank", 0),
                "not_found": m.get("not_found", 0),
            })

    if not rows:
        print("no results found")
        return

    # 表头
    cols = ["mode", "transform", "n", "hit@1", "hit@5", "hit@20", "median_rank", "not_found"]
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    sep = "  "
    print(sep.join(c.ljust(widths[c]) for c in cols))
    print(sep.join("-" * widths[c] for c in cols))
    for r in sorted(rows, key=lambda x: (x["mode"], x["transform"])):
        print(sep.join(str(r[c]).ljust(widths[c]) for c in cols))


if __name__ == "__main__":
    main()
