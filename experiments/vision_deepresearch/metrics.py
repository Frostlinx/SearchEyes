"""
metrics.py — 检索评估指标

recall@k, hit@1/5/20, mean_rank, median_rank
"""
from __future__ import annotations
from dataclasses import dataclass, field
import statistics


@dataclass
class MetricResult:
    mode: str
    n_tasks: int = 0
    hit_at_1: float = 0.0
    hit_at_5: float = 0.0
    hit_at_20: float = 0.0
    mean_rank: float = 0.0
    median_rank: float = 0.0
    not_found: int = 0      # rank > top_k
    ranks: list[int] = field(default_factory=list, repr=False)

    def compute(self):
        if not self.ranks:
            return self
        n = len(self.ranks)
        self.n_tasks = n
        self.hit_at_1 = sum(1 for r in self.ranks if r == 1) / n
        self.hit_at_5 = sum(1 for r in self.ranks if r <= 5) / n
        self.hit_at_20 = sum(1 for r in self.ranks if r <= 20) / n
        self.not_found = sum(1 for r in self.ranks if r > 20)
        self.mean_rank = statistics.mean(self.ranks)
        self.median_rank = statistics.median(self.ranks)
        return self

    def report(self) -> str:
        return (
            f"[{self.mode}] n={self.n_tasks} | "
            f"hit@1={self.hit_at_1:.1%} hit@5={self.hit_at_5:.1%} hit@20={self.hit_at_20:.1%} | "
            f"mean_rank={self.mean_rank:.1f} median={self.median_rank:.1f} not_found={self.not_found}"
        )


def compare(a: MetricResult, b: MetricResult) -> str:
    delta_1  = b.hit_at_1  - a.hit_at_1
    delta_5  = b.hit_at_5  - a.hit_at_5
    delta_20 = b.hit_at_20 - a.hit_at_20
    verdict = "B > A" if delta_20 > 0.02 else ("A > B" if delta_20 < -0.02 else "A ≈ B")
    return (
        f"Δhit@1={delta_1:+.1%} Δhit@5={delta_5:+.1%} Δhit@20={delta_20:+.1%} → {verdict}"
    )
