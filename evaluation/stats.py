"""Statistics for the paper: confidence intervals, paired significance tests, breakdowns.

All functions are dependency-free and deterministic given a seed.
"""
from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(values: Sequence[float], n_resamples: int = 10000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval of the mean (Efron & Tibshirani, 1993)."""
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples))
    lo = means[int((alpha / 2) * n_resamples)]
    hi = means[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    return (lo, hi)


def paired_bootstrap_p(a: Sequence[float], b: Sequence[float], n_resamples: int = 10000, seed: int = 0) -> float:
    """Two-sided paired bootstrap test of mean(a) - mean(b) = 0 (resampling question indices)."""
    if len(a) != len(b) or not a:
        raise ValueError("paired samples must be non-empty and of equal length")
    rng = random.Random(seed)
    n = len(a)
    diffs = [x - y for x, y in zip(a, b)]
    observed = sum(diffs) / n
    centered = [d - observed for d in diffs]
    extreme = 0
    for _ in range(n_resamples):
        m = sum(centered[rng.randrange(n)] for _ in range(n)) / n
        if abs(m) >= abs(observed) - 1e-12:
            extreme += 1
    return (extreme + 1) / (n_resamples + 1)


def mcnemar_exact_p(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> Tuple[int, int, float]:
    """Exact (binomial) McNemar test on paired binary outcomes (McNemar, 1947).
    Returns (b01, b10, two-sided p) where b01 = a wrong & b right, b10 = a right & b wrong."""
    b01 = sum(1 for x, y in zip(a_correct, b_correct) if not x and y)
    b10 = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    k = min(b01, b10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return b01, b10, min(1.0, 2 * tail)


def set_f1(pred: Iterable[int], gold: Iterable[int]) -> float:
    p, g = set(pred), set(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    tp = len(p & g)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(p), tp / len(g)
    return 2 * prec * rec / (prec + rec)


def cohens_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    """Cohen's kappa for two annotators' categorical labels (Cohen, 1960)."""
    if len(a) != len(b) or not a:
        return float("nan")
    n = len(a)
    labels = set(a) | set(b)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def group_by(records: Sequence[Dict], key: str) -> Dict[object, List[Dict]]:
    out: Dict[object, List[Dict]] = {}
    for r in records:
        out.setdefault(r.get(key), []).append(r)
    return dict(sorted(out.items(), key=lambda kv: str(kv[0])))


def summarize(records: Sequence[Dict], seed: int = 0) -> Dict[str, float]:
    """Headline numbers for one system's per-question records."""
    em = [float(r["em"]) for r in records]
    f1 = [float(r["f1"]) for r in records]
    sup = [set_f1(r.get("support_pred", []), r.get("support_gold", [])) for r in records if r.get("answerable", True)]
    em_lo, em_hi = bootstrap_ci(em, seed=seed)
    f1_lo, f1_hi = bootstrap_ci(f1, seed=seed)
    tokens = [r.get("usage", {}).get("total_tokens", 0) for r in records]
    seconds = [r.get("usage", {}).get("seconds", 0.0) for r in records]
    refusals = [1.0 if r.get("refused") else 0.0 for r in records]
    return {
        "n": len(records),
        "em": 100 * mean(em), "em_lo": 100 * em_lo, "em_hi": 100 * em_hi,
        "f1": 100 * mean(f1), "f1_lo": 100 * f1_lo, "f1_hi": 100 * f1_hi,
        "support_f1": 100 * mean(sup) if sup else float("nan"),
        "refusal_rate": 100 * mean(refusals),
        "tokens_per_q": mean(tokens),
        "seconds_per_q": mean(seconds),
    }
