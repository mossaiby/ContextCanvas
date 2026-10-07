"""Baselines that share the LLM, prompts' output format and answer parser with ContextCanvas.

* closed_book   - question only, no evidence.
* full_context  - all candidate paragraphs as text (truncated to the context budget).
* bm25_k{K}     - the top-K paragraphs retrieved by BM25 against the question.
* oracle        - only the gold supporting paragraphs (an upper bound, not a competitor).

Every baseline returns the same record shape as the graph system so the statistics and report
code treat all systems identically.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_STOP = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "by", "with", "from", "and", "or", "is", "was",
    "are", "were", "be", "been", "who", "what", "which", "where", "when", "whose", "whom", "how", "why",
    "that", "this", "it", "its", "as", "did", "does", "do",
}


def tokenize(text: str) -> List[str]:
    return [t for t in (m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")) if t not in _STOP]


class BM25:
    """Okapi BM25 (Robertson & Zaragoza, 2009) over a small document set."""

    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [tokenize(d) for d in documents]
        self.k1, self.b = k1, b
        self.avgdl = (sum(len(d) for d in self.docs) / len(self.docs)) if self.docs else 0.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def scores(self, query: str) -> List[float]:
        q = tokenize(query)
        out = []
        for tf, d in zip(self.tf, self.docs):
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                f = tf[t]
                denom = f + self.k1 * (1 - self.b + self.b * len(d) / (self.avgdl or 1.0))
                s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / denom
            out.append(s)
        return out

    def top_k(self, query: str, k: int) -> List[int]:
        sc = self.scores(query)
        return sorted(range(len(sc)), key=lambda i: (-sc[i], i))[:k]


def format_passages(paragraphs: Sequence[Dict[str, Any]], max_chars: int) -> Tuple[str, List[int]]:
    """Concatenates paragraphs (title + text) up to a character budget; returns text and the
    idx values of the paragraphs actually included."""
    parts, used, total = [], [], 0
    for p in paragraphs:
        block = f"[{p.get('title', '')}] {p.get('paragraph_text', '')}".strip()
        if parts and total + len(block) + 2 > max_chars:
            break
        parts.append(block[:max_chars])
        used.append(int(p.get("idx", 0)))
        total += len(block) + 2
    return "\n\n".join(parts), used


def run_text_baseline(realizer: Any, case: Dict[str, Any], system: str, max_chars: int) -> Dict[str, Any]:
    """Answers one case with a text baseline; returns prediction, used support and evidence size."""
    question, paragraphs = case["question"], case["paragraphs"]
    if system == "closed_book":
        raw = realizer.answer_question(question, "", evidence_kind="none")
        return {"raw": raw, "support": [], "evidence_chars": 0}
    if system == "full_context":
        chosen = paragraphs
    elif system == "oracle":
        chosen = [p for p in paragraphs if p.get("is_supporting")]
    elif system.startswith("bm25_k"):
        k = int(system[len("bm25_k"):])
        bm25 = BM25([f"{p.get('title', '')} {p.get('paragraph_text', '')}" for p in paragraphs])
        chosen = [paragraphs[i] for i in bm25.top_k(question, k)]
    else:
        raise ValueError(f"Unknown baseline: {system}")
    text, used = format_passages(chosen, max_chars)
    raw = realizer.answer_question(question, text, evidence_kind="text")
    return {"raw": raw, "support": used, "evidence_chars": len(text)}
