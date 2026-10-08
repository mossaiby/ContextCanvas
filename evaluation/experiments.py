"""Paper experiments: every system on the same held-out MuSiQue sample.

Usage (from the repository root):

    python -m evaluation.experiments --data external_data/musique_ans_dev.jsonl \
        --out results/musique_main --n 500 --seed 13 --exclude-first 50 \
        --systems contextcanvas,contextcanvas_strict,closed_book,full_context,bm25_k5,oracle

    python -m evaluation.experiments ... --systems ablations    # all ablation variants
    python -m evaluation.report --results results/musique_main  # tables for the paper

Design:
* The first `--exclude-first` answerable questions of the file are the development set used
  while building the system; they are never sampled.
* The sample is drawn once with a fixed seed and stored in manifest.json; later invocations
  reuse it, so systems added later are evaluated on exactly the same questions.
* Each system writes predictions.jsonl incrementally and skips questions already present, so
  an interrupted run can be resumed.
* Extraction is cached on disk (see Realizer), so every ContextCanvas variant shares one set of
  extractions and differs only in the component being ablated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.musique import compute_exact_match, compute_f1  # noqa: E402

ALL_REPAIRS_OFF = {
    "date_migration": False, "placeholder_filter": False, "kinship_repair": False,
    "copula_normalization": False, "place_hierarchy": False, "possessive_links": False, "alias_mining": False,
}

# name -> ("graph", config overrides) or ("text", None)
SYSTEMS: Dict[str, Any] = {
    "contextcanvas": ("graph", {}),
    "contextcanvas_strict": ("graph", {"answer_policy": "strict"}),
    # Graph-guided retrieval: the graph selects documents, the reader reads them as text.
    "contextcanvas_retrieval": ("graph_retrieval", {"retrieval_k": 5, "retrieval_fill": True}),
    "contextcanvas_retrieval_nofill": ("graph_retrieval", {"retrieval_k": 5, "retrieval_fill": False}),
    "contextcanvas_retrieval_nomention": ("graph_retrieval", {"retrieval_k": 5, "retrieval_fill": True,
                                                              "ablations": {"mention_links": False}}),
    "cc_triples": ("graph", {"ablations": {"triples_only": True}}),
    "cc_no_role_labels": ("graph", {"ablations": {"role_labels": False}}),
    "cc_no_envelopes": ("graph", {"ablations": {"envelopes": False}}),
    "cc_no_repairs": ("graph", {"ablations": ALL_REPAIRS_OFF}),
    "cc_no_frame_guard": ("graph", {"ablations": {"frame_guard": False}}),
    "cc_no_kinship": ("graph", {"ablations": {"kinship_repair": False}}),
    "cc_no_place_hierarchy": ("graph", {"ablations": {"place_hierarchy": False}}),
    "cc_no_aliases": ("graph", {"ablations": {"alias_mining": False, "possessive_links": False}}),
    "cc_no_candidates": ("graph", {"ablations": {"candidates": False}}),
    "cc_no_relevance": ("graph", {"ablations": {"relevance_ranking": False}}),
    "cc_no_hub_pruning": ("graph", {"ablations": {"hub_pruning": False}}),
    "closed_book": ("text", None),
    "full_context": ("text", None),
    "bm25_k2": ("text", None),
    "bm25_k5": ("text", None),
    "oracle": ("text", None),
}
GROUPS = {
    "main": ["contextcanvas", "contextcanvas_strict", "contextcanvas_retrieval", "contextcanvas_retrieval_nofill",
             "contextcanvas_retrieval_nomention",
             "closed_book", "full_context", "bm25_k2", "bm25_k5", "oracle"],
    "ablations": [s for s in SYSTEMS if s.startswith("cc_")],
}


# ---------------------------------------------------------------------------------- data

def hop_count(record: Dict[str, Any]) -> int:
    decomposition = record.get("question_decomposition")
    if isinstance(decomposition, list) and decomposition:
        return len(decomposition)
    rid = str(record.get("id", ""))
    m = re.match(r"^(\d)hop", rid)
    if m:
        return int(m.group(1))
    words = {"double": 2, "triple": 3, "quadruple": 4}
    prefix = rid.split("__", 1)[0]
    if prefix in words:
        return words[prefix]
    return max(1, len(re.findall(r"\d+", rid.split("__", 1)[-1])))


def load_records(path: Path, include_unanswerable: bool) -> List[Dict[str, Any]]:
    from evaluation.eval_musique_runner import normalize_musique_record
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            rec = normalize_musique_record(raw)
            if rec is None:
                continue
            if not rec["answerable"] and not include_unanswerable:
                continue
            rec["hops"] = hop_count(raw)
            out.append(rec)
    return out


def draw_sample(records: List[Dict[str, Any]], n: int, seed: int, exclude_first: int, pool: str = "test") -> Dict[str, Any]:
    """pool="test": sample from everything except the development questions (paper runs).
    pool="dev": sample from the development questions only (pilot runs that must not touch,
    and so cannot leak information from, the held-out questions)."""
    answerable = [r for r in records if r["answerable"]]
    dev_ids = [r["id"] for r in answerable[:exclude_first]]
    dev = set(dev_ids)
    candidates = [r for r in records if (r["id"] in dev) == (pool == "dev")]
    rng = random.Random(seed)
    chosen = rng.sample(candidates, min(n, len(candidates)))
    return {"seed": seed, "exclude_first": exclude_first, "pool": pool, "dev_ids": dev_ids,
            "test_ids": [r["id"] for r in chosen]}


def config_fingerprint(config_path: Path) -> str:
    return hashlib.sha256(config_path.read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------------- scoring

def score(predicted: str, case: Dict[str, Any]) -> Dict[str, float]:
    refused = not predicted
    if not case["answerable"]:
        return {"em": float(refused), "f1": float(refused), "refused": refused}
    aliases = case["answer_aliases"]
    em = max(compute_exact_match(predicted, a) for a in aliases) if predicted else 0.0
    f1 = max(compute_f1(predicted, a) for a in aliases) if predicted else 0.0
    return {"em": em, "f1": f1, "refused": refused}


def clean_prediction(raw: str) -> str:
    pred = re.sub(r"^(?:ANSWER|STATUS)\s*:\s*", "", raw or "").strip()
    return "" if pred == "NOT_IN_EVIDENCE" else pred


def source_idx(source: str) -> Optional[int]:
    m = re.search(r":(\d+)$", source or "")
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------------- runners

def ingest_case(engine: Any, case: Dict[str, Any]) -> int:
    """Fresh graph from the case's paragraphs; returns the number of events stored."""
    engine.reset()
    engine.realizer.reset_usage()
    events = 0
    for p in case["paragraphs"]:
        title = p.get("title", "")
        text = f"{title}. {p.get('paragraph_text', '')}".strip() if title else p.get("paragraph_text", "")
        events += len(engine.ingest_text(text, source=f"musique:{case['id']}:{p['idx']}", title=title or None))
    return events


def run_graph_retrieval_case(engine: Any, case: Dict[str, Any], max_chars: int) -> Dict[str, Any]:
    """The graph ranks the case's paragraphs; the top k are given to the reader as text. With
    retrieval_fill, BM25 fills the remaining slots so the budget equals the BM25 top-k baseline."""
    from evaluation.baselines import BM25, format_passages
    from evaluation.eval_musique_runner import classify_case
    cfg = engine.realizer.config
    k, fill = int(cfg.get("retrieval_k", 5)), bool(cfg.get("retrieval_fill", True))
    events = ingest_case(engine, case)
    by_idx = {int(p["idx"]): p for p in case["paragraphs"]}
    ranked_sources = [(source_idx(s), s) for s in engine.rank_sources(case["question"])]
    ranked_sources = [(i, s) for i, s in ranked_sources if i in by_idx]
    meta = dict(engine.last_query_status)
    chosen = [i for i, _ in ranked_sources[:k]]
    from_graph = len(chosen)
    link_of = meta.get("source_links", {})
    link_counts = {kind: sum(1 for _, s in ranked_sources[:k] if link_of.get(s) == kind) for kind in ("event", "title", "mention")}
    if fill and len(chosen) < k:
        bm25 = BM25([f"{p.get('title', '')} {p.get('paragraph_text', '')}" for p in case["paragraphs"]])
        for pos in bm25.top_k(case["question"], len(case["paragraphs"])):
            idx = int(case["paragraphs"][pos]["idx"])
            if len(chosen) >= k:
                break
            if idx not in chosen:
                chosen.append(idx)
    text, used = format_passages([by_idx[i] for i in chosen], max_chars)
    raw = engine.realizer.answer_question(case["question"], text, evidence_kind="text") if used else "STATUS: NOT_IN_EVIDENCE"
    pred = clean_prediction(raw)
    stage = meta.get("stage") if not used else ("qa_success" if pred else "qa_refusal")
    meta.update({"stage": stage, "evidence_text": text})
    return {
        "raw": raw, "predicted": pred, "support": used, "evidence_chars": len(text),
        "failure_stage": classify_case(pred, case["answer_aliases"], len(case["paragraphs"]), events, meta)
        if case["answerable"] else None,
        "anchors": meta.get("anchors", []),
        "retrieval": {"from_graph": from_graph, "filled": len(chosen) - from_graph, "links": link_counts},
        "llm_output": (engine.realizer.last_answer_text or "")[-4000:] if used else "",
        "usage": engine.realizer.usage_summary(),
    }


def run_graph_case(engine: Any, case: Dict[str, Any]) -> Dict[str, Any]:
    from evaluation.eval_musique_runner import classify_case
    events = ingest_case(engine, case)
    raw = engine.ask(case["question"])
    meta = dict(engine.last_query_status)
    pred = clean_prediction(raw)
    support = sorted({i for i in (source_idx(s) for s in meta.get("evidence_sources", [])) if i is not None})
    return {
        "raw": raw, "predicted": pred, "support": support,
        "evidence_chars": meta.get("evidence_chars", 0),
        "failure_stage": classify_case(pred, case["answer_aliases"], len(case["paragraphs"]), events, meta)
        if case["answerable"] else None,
        "anchors": meta.get("anchors", []),
        "llm_output": (meta.get("llm_output") or "")[-4000:],
        "usage": engine.realizer.usage_summary(),
    }


def run_text_case(realizer: Any, case: Dict[str, Any], system: str, max_chars: int) -> Dict[str, Any]:
    from evaluation.baselines import run_text_baseline
    realizer.reset_usage()
    out = run_text_baseline(realizer, case, system, max_chars)
    return {
        "raw": out["raw"], "predicted": clean_prediction(out["raw"]), "support": out["support"],
        "evidence_chars": out["evidence_chars"], "failure_stage": None, "anchors": [],
        "llm_output": (realizer.last_answer_text or "")[-4000:], "usage": realizer.usage_summary(),
    }


def done_ids(path: Path) -> set:
    """Ids already answered. A final line cut off by an interrupted run is dropped from the
    file, so that question is simply run again."""
    if not path.exists():
        return set()
    good, ids = [], set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                ids.add(json.loads(line)["id"])
                good.append(line if line.endswith("\n") else line + "\n")
            except (json.JSONDecodeError, KeyError):
                print(f"  (dropping a truncated line in {path})")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(good)
    return ids


def run_system(system: str, cases: List[Dict[str, Any]], out_dir: Path, config_path: Path, cache_dir: Path) -> None:
    kind, overrides = SYSTEMS[system]
    sys_dir = out_dir / system
    sys_dir.mkdir(parents=True, exist_ok=True)
    pred_path = sys_dir / "predictions.jsonl"
    finished = done_ids(pred_path)
    todo = [c for c in cases if c["id"] not in finished]
    print(f"\n=== {system}: {len(finished)} done, {len(todo)} to run ===")
    if not todo:
        return

    base_overrides = {"extraction_cache_dir": str(cache_dir)}
    if kind in ("graph", "graph_retrieval"):
        from engine.pipeline import ContextCanvasEngine
        from engine.realizer import _deep_merge
        engine = ContextCanvasEngine(
            db_path=str(sys_dir / "kuzu"), config_path=str(config_path),
            config_overrides=_deep_merge(base_overrides, overrides),
        )
        if kind == "graph":
            runner = lambda case: run_graph_case(engine, case)
        else:
            # Same evidence budget as the text baselines.
            max_chars = int((engine.realizer.context_window_tokens - engine.realizer.output_reserve_tokens - 700) * 3.5)
            runner = lambda case: run_graph_retrieval_case(engine, case, max_chars)
        (sys_dir / "effective_config.json").write_text(json.dumps(
            {**engine.realizer.config, "ablations_effective": engine.ablations}, indent=2), encoding="utf-8")
    else:
        from engine.realizer import Realizer
        realizer = Realizer(config_path=str(config_path), overrides=base_overrides)
        # Same evidence budget as the graph system: context window minus answer reserve and prompt.
        max_chars = int((realizer.context_window_tokens - realizer.output_reserve_tokens - 700) * 3.5)
        runner = lambda case: run_text_case(realizer, case, system, max_chars)
        (sys_dir / "effective_config.json").write_text(json.dumps(
            {**realizer.config, "evidence_max_chars": max_chars}, indent=2), encoding="utf-8")

    with open(pred_path, "a", encoding="utf-8") as f:
        for i, case in enumerate(todo, 1):
            t0 = time.perf_counter()
            try:
                out = runner(case)
                error = None
            except Exception as exc:  # recorded, never silently dropped
                out = {"raw": "", "predicted": "", "support": [], "evidence_chars": 0, "failure_stage": "error",
                       "anchors": [], "llm_output": "", "usage": {}}
                error = repr(exc)
            scores = score(out["predicted"], case)
            record = {
                "id": case["id"], "system": system, "hops": case["hops"], "answerable": case["answerable"],
                "question": case["question"], "gold": case["answer"], "aliases": case["answer_aliases"],
                "predicted": out["predicted"], **scores,
                "support_pred": out["support"],
                "support_gold": [p["idx"] for p in case["paragraphs"] if p.get("is_supporting")],
                "failure_stage": out["failure_stage"], "anchors": out["anchors"],
                "evidence_chars": out["evidence_chars"], "usage": out["usage"],
                "retrieval": out.get("retrieval"),
                "wall_seconds": time.perf_counter() - t0, "error": error, "llm_output": out["llm_output"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  [{i}/{len(todo)}] {case['id']}  EM={scores['em']:.0f}  pred={out['predicted'][:40]!r}  gold={case['answer'][:40]!r}")
    if kind in ("graph", "graph_retrieval"):
        engine.close()


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=ROOT_DIR / "config.json")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--exclude-first", type=int, default=50,
                    help="First N answerable questions = development set; never sampled.")
    ap.add_argument("--include-unanswerable", action="store_true",
                    help="Sample from MuSiQue-Full style data (unanswerable cases count as correct when refused).")
    ap.add_argument("--pool", choices=["test", "dev"], default="test",
                    help="test: held-out questions (paper). dev: development questions only (pilot runs).")
    ap.add_argument("--systems", default="main", help="Comma-separated system names or group names (main, ablations).")
    ap.add_argument("--cache-dir", type=Path, default=ROOT_DIR / "results" / "extraction_cache",
                    help="Extraction cache shared by all runs; keyed by extraction model, prompt and text.")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    records = load_records(args.data, args.include_unanswerable)
    by_id = {r["id"]: r for r in records}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if config_fingerprint(args.config) != manifest["config_sha256_16"]:
            raise SystemExit("config.json changed since this experiment started: start a new --out directory "
                             "(results produced under different configurations must not be mixed).")
        if manifest.get("pool", "test") != args.pool:
            raise SystemExit(f"{args.out} was created with --pool {manifest.get('pool', 'test')}; use another --out.")
        print(f"Reusing sample from {manifest_path} ({len(manifest['test_ids'])} questions)")
    else:
        manifest = draw_sample(records, args.n, args.seed, args.exclude_first, pool=args.pool)
        cfg = json.loads(args.config.read_text(encoding="utf-8"))
        lock = ROOT_DIR / "data" / "propbank_lock.json"
        manifest.update({
            "extraction_model": cfg.get("extraction_model") or cfg.get("model_name"),
            "answer_model": cfg.get("answer_model") or cfg.get("model_name"),
            "propbank": json.loads(lock.read_text(encoding="utf-8")) if lock.exists() else None,
            "data": str(args.data), "include_unanswerable": args.include_unanswerable,
            "config_sha256_16": config_fingerprint(args.config),
            "config": json.loads(args.config.read_text(encoding="utf-8")),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    cases = [by_id[i] for i in manifest["test_ids"]]

    systems: List[str] = []
    for token in args.systems.split(","):
        token = token.strip()
        systems.extend(GROUPS.get(token, [token]))
    unknown = [s for s in systems if s not in SYSTEMS]
    if unknown:
        raise SystemExit(f"Unknown system(s): {unknown}. Known: {sorted(SYSTEMS)} or groups {sorted(GROUPS)}")

    cache_dir = args.cache_dir
    for system in dict.fromkeys(systems):
        run_system(system, cases, args.out, args.config, cache_dir)


if __name__ == "__main__":  # pragma: no cover  (script entry point; main() itself is tested)
    main()