"""Evaluation runner for MuSiQue with optional relation-guided retrieval and failure-stage diagnostics."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import collections
import json
import re
import time
import urllib.request
from typing import Any, Dict, List, Optional, Set, Tuple

from engine.pipeline import ContextCanvasEngine, QUESTION_STOPWORDS
from evaluation.musique import (
    compute_dataset_metrics,
    compute_exact_match,
    compute_f1,
    normalize_answer,
    run_official_musique_evaluator,
)

RELATION_PATTERNS: Dict[str, Set[str]] = {
    "spouse": {"spouse", "wife", "husband", "married", "marry", "partner"},
    "child": {"child", "son", "daughter", "children", "offspring"},
    "parent": {"father", "mother", "parent", "parents"},
    "sibling": {"brother", "sister", "sibling", "siblings"},
    "border": {"border", "borders", "adjacent", "neighbor", "neighboring", "shares a border"},
    "headquarters": {"headquartered", "headquarters", "based", "hq", "located"},
    "location": {"located", "province", "district", "county", "state", "city", "country", "territory", "borough"},
    "birthplace": {"born", "birthplace", "birth place", "native of"},
    "education": {"educated", "school", "college", "university", "alumnus", "alumna", "studied"},
    "founded": {"founded", "established", "co-founded", "creator", "created", "started"},
    "league": {"league", "plays in", "competes in", "division", "club", "team"},
    "record_label": {"record label", "signed", "label", "records", "released by"},
    "work": {"author", "writer", "wrote", "book", "novel", "play", "composed", "performer", "album", "song"},
    "ownership": {"owner", "owns", "owned by", "property of", "subsidiary", "parent company"},
    "manufacturer": {"manufacturer", "manufactured", "maker", "made by", "aircraft"},
}

MEDIA_DISAMBIGUATION_MAP = {
    "performer": {"album", "song", "band", "musician", "singer"},
    "author": {"novel", "book", "play", "writer"},
    "director": {"film", "movie", "play"},
    "star": {"film", "movie", "series"},
}


def check_ollama_status(base_url: str = "http://localhost:11434") -> bool:
    host = base_url.replace("/v1", "").rstrip("/")
    try:
        req = urllib.request.Request(f"{host}/api/tags", headers={"User-Agent": "ContextCanvas"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def normalize_musique_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    qid = str(record.get("id", ""))
    question = record.get("question") or record.get("composed_question_text")
    answer = record.get("answer") or record.get("answer_text")
    if not qid or not question or answer is None:
        return None
    normalized_paragraphs = []
    for idx, p in enumerate(record.get("paragraphs") or record.get("contexts") or []):
        if not isinstance(p, dict):
            continue
        normalized_paragraphs.append({
            "idx": int(p.get("idx", idx)),
            "title": (p.get("title") or p.get("wikipedia_title") or "").strip(),
            "paragraph_text": (p.get("paragraph_text") or p.get("text") or "").strip(),
            "is_supporting": bool(p.get("is_supporting", False)),
        })
    aliases = [str(answer)]
    for alias in record.get("answer_aliases", []):
        if isinstance(alias, str) and alias not in aliases:
            aliases.append(alias)
    return {
        "id": qid,
        "question": str(question).strip(),
        "answer": str(answer).strip(),
        "answer_aliases": aliases,
        "paragraphs": normalized_paragraphs,
        "answerable": bool(record.get("answerable", True)),
    }


def load_musique_sample(jsonl_path: Path, max_samples: int = 100) -> List[Dict[str, Any]]:
    records = []
    if not jsonl_path.exists():
        print(f"Warning: {jsonl_path} not found. Generating synthetic validation sample.")
        return generate_mock_musique_sample()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            normalized = normalize_musique_record(json.loads(line))
            if normalized and normalized["answerable"]:
                records.append(normalized)
            if len(records) >= max_samples:
                break
    return records


def generate_mock_musique_sample() -> List[Dict[str, Any]]:
    return [{
        "id": "2hop__test_1",
        "question": "Which university employs the inventor of Project Prometheus?",
        "answer": "Munich Institute of Physics",
        "answer_aliases": ["Munich Institute of Physics", "MIP"],
        "paragraphs": [
            {"idx": 0, "title": "Dr. Elena Rostova", "paragraph_text": "Dr. Elena Rostova invented Project Prometheus in 2023.", "is_supporting": True},
            {"idx": 1, "title": "Munich Institute of Physics", "paragraph_text": "Munich Institute of Physics employs Dr. Elena Rostova.", "is_supporting": True},
            {"idx": 2, "title": "Unrelated Topic", "paragraph_text": "Cambridge University is located in England.", "is_supporting": False},
        ],
        "answerable": True,
    }]


def detect_question_relations(question: str) -> Set[str]:
    q_lower = question.lower()
    active = set()
    for rel_name, terms in RELATION_PATTERNS.items():
        if any(re.search(r"\b" + re.escape(t) + r"\b", q_lower) for t in terms):
            active.add(rel_name)
    return active


def select_hop1_candidates(question, paragraphs, max_candidates=2, min_score=0.20, debug=False):
    q_lower = question.lower()
    q_tokens = set(re.findall(r"\w+", q_lower)) - QUESTION_STOPWORDS
    q_entities = [m.lower() for m in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", question)]
    relation_tokens: Set[str] = set()
    for rel in detect_question_relations(question):
        relation_tokens.update(RELATION_PATTERNS[rel])
    scored, seen_titles = [], set()
    for p in paragraphs:
        title = p.get("title", "").strip()
        clean_title = re.sub(r"\s*\(.*?\)", "", title.lower()).strip()
        if not clean_title or clean_title in seen_titles:
            continue
        body_tokens = set(re.findall(r"\w+", p.get("paragraph_text", "").lower())) - QUESTION_STOPWORDS
        title_exact = 1.0 if clean_title in q_lower else 0.0
        title_entity = min(sum(1.0 for ent in q_entities if ent in clean_title), 2.0) / 2.0
        title_score = max(title_exact, title_entity)
        entity_overlap = min(len(q_tokens & body_tokens), 5) / 5.0
        relation_overlap = min(len(relation_tokens & body_tokens), 3) / 3.0 if relation_tokens else 0.0
        type_compat = 0.0
        for trigger, tags in MEDIA_DISAMBIGUATION_MAP.items():
            if trigger in q_lower and any(f"({tag}" in title.lower() for tag in tags):
                type_compat = 1.0
                break
        final = 0.30 * title_score + 0.35 * entity_overlap + 0.25 * relation_overlap + 0.10 * type_compat
        feats = {"title_score": round(title_score, 2), "entity_overlap": round(entity_overlap, 2),
                 "relation_overlap": round(relation_overlap, 2), "type_compat": type_compat, "final_score": round(final, 3)}
        if debug:
            print(f"      [Hop 1 Candidate] {title[:30]:<30} | Score: {final:.3f} | Accepted: {final >= min_score}")
        scored.append((p, final, feats))
        seen_titles.add(clean_title)
    scored.sort(key=lambda x: x[1], reverse=True)
    return [item for item in scored[:max_candidates] if item[1] >= min_score]


def select_hop2_candidates(bridge_entities, question, remaining_paragraphs, max_candidates=2, min_score=0.25, debug=False):
    q_lower = question.lower()
    q_tokens = set(re.findall(r"\w+", q_lower)) - QUESTION_STOPWORDS
    relation_tokens: Set[str] = set()
    for rel in detect_question_relations(question):
        relation_tokens.update(RELATION_PATTERNS[rel])
    clean_bridges = set()
    for e in bridge_entities:
        e_clean = re.sub(r"\s*\(.*?\)", "", e).strip().lower()
        if len(e_clean) <= 2 or e_clean in QUESTION_STOPWORDS or e_clean in q_lower:
            continue
        e_words = set(re.findall(r"\w+", e_clean)) - QUESTION_STOPWORDS
        if e_words and e_words.issubset(q_tokens):
            continue
        clean_bridges.add(e_clean)
    scored, seen_titles = [], set()
    for p in remaining_paragraphs:
        title = p.get("title", "").strip()
        clean_title = re.sub(r"\s*\(.*?\)", "", title.lower()).strip()
        if not clean_title or clean_title in seen_titles:
            continue
        body = p.get("paragraph_text", "").lower()
        bridge_title = 1.0 if any(b == clean_title or b in clean_title or (len(clean_title) > 3 and clean_title in b) for b in clean_bridges) else 0.0
        bridge_body = 1.0 if any(re.search(r"\b" + re.escape(b) + r"\b", body) for b in clean_bridges) else 0.0
        body_tokens = set(re.findall(r"\w+", body)) - QUESTION_STOPWORDS
        relation_overlap = min(len(relation_tokens & body_tokens), 3) / 3.0 if relation_tokens else 0.0
        final = 0.40 * bridge_title + 0.35 * bridge_body + 0.25 * relation_overlap
        feats = {"bridge_title_score": bridge_title, "bridge_body_score": bridge_body,
                 "relation_overlap": round(relation_overlap, 2), "final_score": round(final, 3)}
        if debug:
            print(f"      [Hop 2 Candidate] {title[:30]:<30} | Score: {final:.3f} | Accepted: {final >= min_score}")
        scored.append((p, final, feats))
        seen_titles.add(clean_title)
    scored.sort(key=lambda x: x[1], reverse=True)
    return [item for item in scored[:max_candidates] if item[1] >= min_score]


def _paragraph_source(qid: str, p: Dict[str, Any]) -> str:
    return f"musique:{qid}:{p.get('idx', 0)}"


def _source_paragraph_idx(source: str) -> Optional[int]:
    m = re.search(r":(\d+)$", source or "")
    return int(m.group(1)) if m else None


def _ingest_paragraph(engine: ContextCanvasEngine, qid: str, p: Dict[str, Any], debug: bool) -> int:
    title = p.get("title", "")
    body = p.get("paragraph_text", "")
    text = f"{title}. {body}".strip() if title else body.strip()
    if not text:
        return 0
    if debug:
        print(f"\n      [INGEST] {title or text[:40]}")
    t0 = time.perf_counter()
    ev_ids = engine.ingest_text(text, source=_paragraph_source(qid, p), title=title or None, debug=debug)
    print(f"    [{p.get('idx', 0)}] \"{(title or text)[:40]}\" -> {len(ev_ids)} event(s) in {time.perf_counter() - t0:.1f}s")
    return len(ev_ids)


def classify_case(
    prediction: str,
    aliases: List[str],
    paragraphs_ingested: int,
    events_extracted: int,
    query_meta: Dict[str, Any],
) -> str:
    """Attributes a case outcome to the pipeline stage that determined it."""
    if prediction and max(compute_exact_match(prediction, a) for a in aliases) == 1.0:
        return "success"
    if paragraphs_ingested == 0:
        return "retrieval_empty"
    if events_extracted == 0:
        return "extraction_zero_events"
    stage = query_meta.get("stage", "")
    if stage in {"graph_no_anchors", "graph_empty_blueprint"}:
        return stage
    gold_in_evidence = _gold_in_evidence(query_meta.get("evidence_text", ""), aliases)
    if stage == "qa_refusal":
        return "qa_refusal_answer_in_evidence" if gold_in_evidence else "qa_refusal_answer_not_in_evidence"
    if prediction and max(compute_f1(prediction, a) for a in aliases) > 0:
        return "answer_partial_match"
    return "answer_mismatch_answer_in_evidence" if gold_in_evidence else "answer_mismatch_answer_not_in_evidence"


def _gold_in_evidence(evidence_text: str, aliases: List[str]) -> bool:
    norm_evidence = f" {normalize_answer(evidence_text)} "
    return any(normalize_answer(a) and f" {normalize_answer(a)} " in norm_evidence for a in aliases)


def run_musique_evaluation(
    dataset_path: Path,
    output_dir: Path,
    evaluator_path: Optional[Path] = None,
    max_samples: int = 50,
    supporting_only: bool = False,
    top_k: int = 0,
    debug: bool = False,
) -> Dict[str, Any]:
    with open(ROOT_DIR / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    base_url = cfg.get("base_url", "http://localhost:11434/v1")
    print(f"Checking LLM endpoint ({base_url})...")
    if not check_ollama_status(base_url):
        print(f"Warning: Could not connect to Ollama at {base_url}. Ensure 'ollama serve' is running.")

    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_musique_sample(dataset_path, max_samples=max_samples)
    mode = "oracle_supporting" if supporting_only else (f"retrieval_top{top_k}" if top_k > 0 else "all_paragraphs")
    print(f"\nStarting evaluation on {len(samples)} MuSiQue case(s) [mode={mode}]...")

    engine = ContextCanvasEngine(db_path=str(output_dir / "musique_eval_kuzu"), config_path=str(ROOT_DIR / "config.json"))
    predictions, golds, case_diagnostics = [], [], []
    failure_stage_counts: Dict[str, int] = collections.defaultdict(int)

    for case_idx, case in enumerate(samples, start=1):
        engine.reset()
        qid, question = case["id"], case["question"]
        gold_answer, aliases = case["answer"], case["answer_aliases"]
        all_paragraphs = case.get("paragraphs", [])
        print(f"\n[{case_idx}/{len(samples)}] Case: {qid}\n  Question: \"{question}\"")

        ingested: List[Dict[str, Any]] = []
        total_events = 0
        hop_records: Dict[str, List[Dict[str, Any]]] = {"hop1": [], "hop2": []}

        if supporting_only:
            to_ingest = [p for p in all_paragraphs if p.get("is_supporting")]
            for p in to_ingest:
                total_events += _ingest_paragraph(engine, qid, p, debug)
                ingested.append(p)
        elif top_k > 0:
            hop1_max = max(1, top_k // 2)
            hop1_items = select_hop1_candidates(question, all_paragraphs, max_candidates=hop1_max, debug=debug)
            for p, score, feats in hop1_items:
                n = _ingest_paragraph(engine, qid, p, debug)
                total_events += n
                ingested.append(p)
                hop_records["hop1"].append({"title": p.get("title"), "score": score, "features": feats, "events": n})
            ingested_ids = {id(p) for p in ingested}
            remaining = [p for p in all_paragraphs if id(p) not in ingested_ids]
            hop2_items = select_hop2_candidates(set(engine.graph.entity_names()), question, remaining,
                                                max_candidates=top_k - hop1_max, debug=debug)
            for p, score, feats in hop2_items:
                n = _ingest_paragraph(engine, qid, p, debug)
                total_events += n
                ingested.append(p)
                hop_records["hop2"].append({"title": p.get("title"), "score": score, "features": feats, "events": n})
        else:
            for p in all_paragraphs:
                total_events += _ingest_paragraph(engine, qid, p, debug)
                ingested.append(p)

        print("  Querying Context Graph...", end="", flush=True)
        t0 = time.perf_counter()
        ans = engine.ask(question, debug=debug)
        print(f" Answered in {time.perf_counter() - t0:.1f}s")

        clean_pred = re.sub(r"^(?:ANSWER|STATUS)\s*:\s*", "", ans).strip()
        if clean_pred == "NOT_IN_EVIDENCE":
            clean_pred = ""
        print(f"  Predicted: \"{clean_pred or 'NOT_IN_EVIDENCE'}\" | Gold: \"{gold_answer}\"")

        query_meta = dict(getattr(engine, "last_query_status", {}))
        failure_stage = classify_case(clean_pred, aliases, len(ingested), total_events, query_meta)
        failure_stage_counts[failure_stage] += 1

        # Predicted support = paragraphs whose facts were shown to the answerer (never gold labels).
        support_idxs = sorted({
            i for i in (_source_paragraph_idx(s) for s in query_meta.get("evidence_sources", [])) if i is not None
        })
        predictions.append({
            "id": qid,
            "predicted_answer": clean_pred,
            "predicted_support_idxs": support_idxs,
            "predicted_answerable": bool(clean_pred),
        })
        golds.append({"id": qid, "answer": gold_answer, "answer_aliases": aliases})
        if not debug:
            query_meta.pop("evidence_text", None)
        case_diagnostics.append({
            "id": qid, "question": question, "gold": gold_answer, "aliases": aliases,
            "predicted": clean_pred, "failure_stage": failure_stage, "mode": mode,
            "paragraphs_ingested": len(ingested), "events_extracted": total_events,
            "hop_records": hop_records, "query_meta": query_meta,
        })

    engine.close()

    preds_file = output_dir / "musique_predictions.jsonl"
    golds_file = output_dir / "musique_gold.jsonl"
    with open(preds_file, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
    with open(golds_file, "w", encoding="utf-8") as f:
        for g in golds:
            f.write(json.dumps(g) + "\n")
    with open(output_dir / "musique_debug_cases.json", "w", encoding="utf-8") as f:
        json.dump(case_diagnostics, f, indent=2)

    if evaluator_path and evaluator_path.exists():
        metrics = run_official_musique_evaluator(preds_file, golds_file, evaluator_path)
    else:
        metrics = compute_dataset_metrics(preds_file, golds_file)
    metrics["mode"] = mode
    metrics["failure_stages"] = dict(failure_stage_counts)
    (output_dir / "musique_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n==================================================\n Final MuSiQue Evaluation Metrics:\n==================================================")
    print(json.dumps(metrics, indent=2))
    print("\nFailure Stage Breakdown:")
    for stage, count in sorted(failure_stage_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {stage:<40}: {count}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on MuSiQue-Ans")
    parser.add_argument("--data", type=Path, default=Path("external_data/musique_ans_dev.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("benchmark_musique_results"))
    parser.add_argument("--evaluator", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--supporting-only", action="store_true",
                        help="ORACLE setting: ingest only gold supporting paragraphs. Report separately.")
    parser.add_argument("--top-k", type=int, default=0,
                        help="If > 0 and not supporting-only, run relation-guided retrieval for top-k paragraphs.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    run_musique_evaluation(args.data, args.output, args.evaluator, max_samples=args.samples,
                           supporting_only=args.supporting_only, top_k=args.top_k, debug=args.debug)