"""Evaluation harness executing official and enhanced MuSiQue dataset evaluation metrics."""
from __future__ import annotations

import collections
import json
import re
import string
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List


def normalize_answer(s: str) -> str:
    """Enhanced SQuAD/MuSiQue string normalization with NFKD unicode handling."""
    if not s:
        return ""
    # 1. Unicode NFKD normalization & strip accents
    text = unicodedata.normalize("NFKD", s)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()

    # 2. Strip punctuation & symbols
    text = re.sub(r"[^\w\s]", " ", text)

    # 3. Remove leading/isolated articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    # 4. Collapse whitespace
    return " ".join(text.split())


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Computes token-level F1 between prediction and ground truth."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Computes normalized exact match."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_strict_em(prediction: str, ground_truth: str) -> float:
    """Computes strict, unnormalized exact match."""
    return float(prediction.strip() == ground_truth.strip())


def run_official_musique_evaluator(
    predictions_jsonl: Path,
    gold_jsonl: Path,
    evaluator_script_path: Path
) -> Dict[str, float]:
    """Invokes upstream official evaluate_v1.0.py as a subprocess and captures standard metrics."""
    if not evaluator_script_path.exists():
        raise FileNotFoundError(f"Official evaluator script not found at {evaluator_script_path}")

    cmd = [
        sys.executable,
        str(evaluator_script_path),
        str(predictions_jsonl),
        str(gold_jsonl)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        return compute_dataset_metrics(predictions_jsonl, gold_jsonl)


def compute_dataset_metrics(predictions_jsonl: Path, gold_jsonl: Path) -> Dict[str, Any]:
    """Computes strict, normalized, alias-aware EM and token F1 over dataset predictions."""
    preds = {}
    with open(predictions_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                preds[item["id"]] = item

    golds = {}
    with open(gold_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                golds[item["id"]] = item

    strict_em_total = 0.0
    norm_em_total = 0.0
    alias_em_total = 0.0
    f1_total = 0.0
    count = 0

    for qid, gold_item in golds.items():
        if qid not in preds:
            continue

        pred_ans = preds[qid].get("predicted_answer", "")
        primary_ans = gold_item.get("answer", "")
        aliases = [primary_ans] + gold_item.get("answer_aliases", [])

        # Strict EM on primary answer
        strict_em_total += compute_strict_em(pred_ans, primary_ans)

        # Normalized EM on primary answer
        norm_em_total += compute_exact_match(pred_ans, primary_ans)

        # Alias-aware EM & F1 across all gold references
        alias_em_total += max(compute_exact_match(pred_ans, a) for a in aliases)
        f1_total += max(compute_f1(pred_ans, a) for a in aliases)
        count += 1

    return {
        "strict_em": round((strict_em_total / count * 100), 2) if count else 0.0,
        "normalized_em": round((norm_em_total / count * 100), 2) if count else 0.0,
        "answer_em": round((alias_em_total / count * 100), 2) if count else 0.0,
        "alias_em": round((alias_em_total / count * 100), 2) if count else 0.0,
        "answer_f1": round((f1_total / count * 100), 2) if count else 0.0,
        "evaluated_cases": count,
    }