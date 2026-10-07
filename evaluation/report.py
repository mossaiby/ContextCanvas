"""Builds every results table and number used by the paper from experiment outputs.

    python -m evaluation.report --results results/musique_main [--storage benchmark_summary.json] \
        --paper-dir paper/generated

Writes LaTeX tables (tab_*.tex), a macro file (results_macros.tex) and results.md. Nothing in the
paper's results section is typed by hand; rerunning this script after new runs updates the paper.
If a table's data are missing, a placeholder table is written so the paper still compiles.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.stats import group_by, mcnemar_exact_p, mean, paired_bootstrap_p, summarize  # noqa: E402

DISPLAY = {
    "contextcanvas": "ContextCanvas (best-supported)",
    "contextcanvas_strict": "ContextCanvas (strict)",
    "closed_book": "Closed-book LLM",
    "full_context": "Full-context LLM",
    "bm25_k2": "BM25 top-2 + LLM",
    "bm25_k5": "BM25 top-5 + LLM",
    "oracle": "Oracle paragraphs + LLM$^\\dagger$",
    "cc_triples": "triples only (no roles, no envelopes)",
    "cc_no_role_labels": "-- role meanings",
    "cc_no_envelopes": "-- context envelopes",
    "cc_no_repairs": "-- all semantic repairs",
    "cc_no_kinship": "-- kinship repair",
    "cc_no_place_hierarchy": "-- place hierarchies",
    "cc_no_aliases": "-- aliases and possessives",
    "cc_no_candidates": "-- path candidates",
    "cc_no_relevance": "-- relevance ranking",
    "cc_no_hub_pruning": "-- hub pruning",
}
MAIN_ORDER = ["closed_book", "bm25_k2", "bm25_k5", "full_context", "contextcanvas", "contextcanvas_strict", "oracle"]
REFERENCE = "contextcanvas"


def load(results_dir: Path) -> Dict[str, List[Dict]]:
    manifest = json.loads((results_dir / "manifest.json").read_text(encoding="utf-8"))
    order = {qid: i for i, qid in enumerate(manifest["test_ids"])}
    systems: Dict[str, List[Dict]] = {}
    for pred in sorted(results_dir.glob("*/predictions.jsonl")):
        with open(pred, "r", encoding="utf-8") as f:
            recs = [json.loads(l) for l in f if l.strip()]
        recs = sorted((r for r in recs if r["id"] in order), key=lambda r: order[r["id"]])
        if recs:
            systems[pred.parent.name] = recs
    return systems


def fmt(x: float, digits: int = 1) -> str:
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def fmt_p(p: Optional[float]) -> str:
    if p is None or math.isnan(p):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def holm(pvalues: Sequence[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values (step-down, monotone)."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def paired(a: List[Dict], b: List[Dict]):
    ids = [r["id"] for r in a if r["id"] in {x["id"] for x in b}]
    ia, ib = {r["id"]: r for r in a}, {r["id"]: r for r in b}
    return [ia[i] for i in ids], [ib[i] for i in ids]


def compare(sys_recs: List[Dict], ref_recs: List[Dict]) -> Dict[str, float]:
    a, b = paired(sys_recs, ref_recs)
    if not a:
        return {"n": 0, "d_em": float("nan"), "p_em": float("nan"), "d_f1": float("nan"), "p_f1": float("nan")}
    _, _, p_em = mcnemar_exact_p([r["em"] == 1 for r in a], [r["em"] == 1 for r in b])
    p_f1 = paired_bootstrap_p([r["f1"] for r in a], [r["f1"] for r in b])
    return {"n": len(a), "d_em": 100 * (mean([r["em"] for r in a]) - mean([r["em"] for r in b])), "p_em": p_em,
            "d_f1": 100 * (mean([r["f1"] for r in a]) - mean([r["f1"] for r in b])), "p_f1": p_f1}


_NOTES: Dict[str, str] = {}


def tex_escape(text: str) -> str:
    return (str(text).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%")
            .replace("&", "\\&").replace("#", "\\#"))


def model_macros(manifest: Dict) -> Dict[str, str]:
    pb = manifest.get("propbank") or {}
    return {
        "ResExtractionModel": "\\texttt{" + tex_escape(manifest.get("extraction_model") or "--") + "}",
        "ResAnswerModel": "\\texttt{" + tex_escape(manifest.get("answer_model") or "--") + "}",
        "ResPropBankRef": tex_escape(pb.get("ref") or "--"),
        "ResPropBankCommit": "\\texttt{" + tex_escape((pb.get("commit") or "--")[:10]) + "}",
    }


def build_models(configs: List[tuple], out_dir: Path) -> None:
    """One row per model configuration (label, results dir): the headline systems side by side.
    All configurations must have been run on the same questions."""
    rows, ref_ids = [], None
    for label, results_dir in configs:
        manifest = json.loads((results_dir / "manifest.json").read_text(encoding="utf-8"))
        if ref_ids is None:
            ref_ids = manifest["test_ids"]
        elif manifest["test_ids"] != ref_ids:
            raise SystemExit(f"{results_dir} was run on different questions; use the same --n/--seed/--exclude-first.")
        systems = load(results_dir)
        get = lambda name: summarize(systems[name]) if name in systems else None
        cc, full, bm, closed = get(REFERENCE), get("full_context"), get("bm25_k5"), get("closed_book")
        c = compare(systems[REFERENCE], systems["full_context"]) if cc and full else None
        rows.append([
            tex_escape(label),
            "\\texttt{" + tex_escape(manifest.get("extraction_model") or "--") + "}",
            "\\texttt{" + tex_escape(manifest.get("answer_model") or "--") + "}",
            fmt(cc["em"]) if cc else "--", fmt(cc["f1"]) if cc else "--",
            fmt(full["em"]) if full else "--", fmt(bm["em"]) if bm else "--", fmt(closed["em"]) if closed else "--",
            f"{c['d_em']:+.1f}" if c else "--", fmt_p(c["p_em"]) if c else "--",
        ])
    (out_dir / "tab_models.tex").write_text(tabular(
        "models", ["Configuration", "Extractor", "Reader", "CC EM", "CC F1", "Full EM", "BM25-5 EM", "Closed EM",
                   "$\\Delta$EM vs Full", "$p$"], rows, "lllrrrrrrr",
        note="CC: ContextCanvas (best-supported). $\\Delta$EM and $p$ (exact McNemar): ContextCanvas minus the "
             "full-context baseline with the same reader. Text baselines use only the reader model."), encoding="utf-8")


def tabular(name: str, header: List[str], rows: List[List[str]], align: str, note: str = "") -> str:
    """Returns the tabular body only; the note goes to tab_<name>_note.tex (written by build) so a
    paper may scale the table to the page width without shrinking its note."""
    lines = [r"\begin{tabular}{" + align + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows] or [r"\multicolumn{" + str(len(header)) + r"}{c}{(no results yet)} \\"]
    lines += [r"\bottomrule", r"\end{tabular}"]
    _NOTES[name] = ("% generated by evaluation/report.py - do not edit\n"
                    + (r"\par\smallskip{\footnotesize " + note + "}\n" if note else ""))
    return "% generated by evaluation/report.py - do not edit\n" + "\n".join(lines) + "\n"


def build(results_dir: Path, storage_path: Optional[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    systems = load(results_dir) if (results_dir / "manifest.json").exists() else {}
    manifest = json.loads((results_dir / "manifest.json").read_text()) if systems else {}
    summ = {s: summarize(r) for s, r in systems.items()}
    ref = systems.get(REFERENCE)
    macros: Dict[str, str] = {}
    md: List[str] = [f"# Results: {results_dir}\n"]

    # ---------------- main table
    rows, cost_rows = [], []
    for s in [x for x in MAIN_ORDER if x in summ]:
        m = summ[s]
        c = compare(systems[s], ref) if ref and s != REFERENCE else None
        rows.append([
            DISPLAY.get(s, s),
            f"{fmt(m['em'])} [{fmt(m['em_lo'])}, {fmt(m['em_hi'])}]",
            f"{fmt(m['f1'])} [{fmt(m['f1_lo'])}, {fmt(m['f1_hi'])}]",
            fmt(m["support_f1"]), fmt(m["refusal_rate"]),
            fmt_p(c["p_em"]) if c else "ref.",
        ])
        cost_rows.append([
            DISPLAY.get(s, s),
            fmt(mean([r["usage"].get("index_tokens", 0) for r in systems[s]]), 0),
            fmt(mean([r["usage"].get("query_tokens", 0) for r in systems[s]]), 0),
            fmt(mean([r["usage"].get("index_seconds", 0) for r in systems[s]])),
            fmt(mean([r["usage"].get("query_seconds", 0) for r in systems[s]])),
        ])
    n_test = len(manifest.get("test_ids", [])) if manifest else 0
    (out_dir / "tab_main.tex").write_text(tabular(
        "main", ["System", "EM [95\\% CI]", "F1 [95\\% CI]", "Sup.\\ F1", "Refusal \\%", "$p$ (EM)"],
        rows, "lrrrrr",
        note=f"$n={n_test}$ held-out questions. CIs: percentile bootstrap (10{{,}}000 resamples). "
             "$p$: exact McNemar test on per-question EM against ContextCanvas (best-supported). "
             "$^\\dagger$Upper bound: receives the gold supporting paragraphs."), encoding="utf-8")
    (out_dir / "tab_cost.tex").write_text(tabular(
        "cost", ["System", "Index tok./q", "Query tok./q", "Index s/q", "Query s/q"], cost_rows, "lrrrr",
        note="Mean language-model tokens and seconds per question. Index: extraction of the question's "
             "paragraphs (cached extractions are charged their original cost). Query: answering, including "
             "follow-up calls for truncated answers."), encoding="utf-8")
    macros.update(model_macros(manifest))
    macros["ResNTest"] = str(n_test) if n_test else "--"
    macros["ResCCFinalizeRate"] = "--"
    macros["ResCCEvidenceChars"] = "--"
    macros["ResNDev"] = str(len(manifest.get("dev_ids", []))) if manifest else "--"
    macros["ResSeed"] = str(manifest.get("seed", "--")) if manifest else "--"
    for s, key in [("contextcanvas", "CC"), ("contextcanvas_strict", "CCStrict"), ("full_context", "Full"),
                   ("bm25_k5", "BMfive"), ("bm25_k2", "BMtwo"), ("closed_book", "Closed"), ("oracle", "Oracle"),
                   ("cc_triples", "Triples")]:
        m = summ.get(s)
        macros[f"Res{key}EM"] = fmt(m["em"]) if m else "--"
        macros[f"Res{key}FOne"] = fmt(m["f1"]) if m else "--"
        macros[f"Res{key}Refusal"] = fmt(m["refusal_rate"]) if m else "--"
    if ref:
        fin = [r["usage"].get("finalize_calls", 0) > 0 for r in ref]
        macros["ResCCFinalizeRate"] = fmt(100 * mean([float(x) for x in fin]))
        macros["ResCCEvidenceChars"] = fmt(mean([r.get("evidence_chars", 0) for r in ref]), 0)
    md.append("## Main\n\n| system | EM | F1 | support F1 | refusal % |\n|---|---|---|---|---|")
    for s in [x for x in MAIN_ORDER if x in summ]:
        m = summ[s]
        md.append(f"| {s} | {fmt(m['em'])} [{fmt(m['em_lo'])}, {fmt(m['em_hi'])}] | {fmt(m['f1'])} | "
                  f"{fmt(m['support_f1'])} | {fmt(m['refusal_rate'])} |")

    # ---------------- per-hop table
    hop_systems = [x for x in MAIN_ORDER if x in systems]
    hops = sorted({r["hops"] for s in hop_systems for r in systems[s]})
    hop_rows = []
    for s in hop_systems:
        by_hop = group_by(systems[s], "hops")
        hop_rows.append([DISPLAY.get(s, s)] + [fmt(100 * mean([r["em"] for r in by_hop.get(h, [])])) if by_hop.get(h) else "--" for h in hops])
    if ref:
        counts = group_by(ref, "hops")
        hop_header = ["System"] + [f"{h}-hop ($n={len(counts.get(h, []))}$)" for h in hops]
    else:
        hop_header = ["System"] + [f"{h}-hop" for h in hops]
    (out_dir / "tab_hops.tex").write_text(tabular("hops", hop_header, hop_rows, "l" + "r" * len(hops),
                                                  note="Exact match (\\%) by number of reasoning hops."), encoding="utf-8")

    # ---------------- ablations
    abl = [s for s in DISPLAY if s.startswith("cc_") and s in systems]
    comps = {s: compare(systems[s], ref) for s in abl} if ref else {}
    if comps:
        adj_em = dict(zip(abl, holm([comps[s]["p_em"] for s in abl])))
        adj_f1 = dict(zip(abl, holm([comps[s]["p_f1"] for s in abl])))
    abl_rows = []
    if ref:
        abl_rows.append(["full system", fmt(summ[REFERENCE]["em"]), "", "", fmt(summ[REFERENCE]["f1"]), "", ""])
    for s in abl:
        c = comps[s]
        abl_rows.append([DISPLAY[s], fmt(summ[s]["em"]), f"{c['d_em']:+.1f}", fmt_p(adj_em[s]),
                         fmt(summ[s]["f1"]), f"{c['d_f1']:+.1f}", fmt_p(adj_f1[s])])
    (out_dir / "tab_ablation.tex").write_text(tabular(
        "ablation", ["Variant", "EM", "$\\Delta$EM", "$p_{\\text{Holm}}$", "F1", "$\\Delta$F1", "$p_{\\text{Holm}}$"],
        abl_rows, "lrrrrrr",
        note="Each row removes one component from the full system; all variants share the same cached "
             "extractions. EM: exact McNemar; F1: paired bootstrap; both Holm-corrected over all ablations."),
        encoding="utf-8")

    # ---------------- failure stages of the reference system
    fail_rows = []
    if ref:
        stages = group_by([r for r in ref if r.get("answerable", True)], "failure_stage")
        total = sum(len(v) for v in stages.values())
        for stage, recs in sorted(stages.items(), key=lambda kv: -len(kv[1])):
            fail_rows.append([str(stage).replace("_", "\\_"), str(len(recs)), fmt(100 * len(recs) / total)])
            macros["ResStage" + "".join(w.capitalize() for w in str(stage).split("_"))] = str(len(recs))
    (out_dir / "tab_failures.tex").write_text(tabular(
        "failures", ["Outcome", "Questions", "\\%"], fail_rows, "lrr",
        note="Outcome attribution for ContextCanvas. `answer in evidence' means a gold alias occurs in the "
             "evidence shown to the answerer."), encoding="utf-8")

    # ---------------- storage benchmark
    st_rows = []
    if storage_path and storage_path.exists():
        storage = json.loads(storage_path.read_text(encoding="utf-8")).get("storage", [])
        for m in storage:
            st_rows.append([str(m["events_count"]), fmt(m["events_per_second"]), str(m["disk_bytes"]),
                            fmt(m["bytes_per_event"]), fmt(m["median_query_latency_ms"], 2)])
    (out_dir / "tab_storage.tex").write_text(tabular(
        "storage", ["Events", "Inserts/s", "Disk bytes", "Bytes/event", "Median query ms"], st_rows, "rrrrr",
        note="Synthetic employment events; queries are served from the in-memory traversal mirror."), encoding="utf-8")

    if not (out_dir / "tab_models.tex").exists():
        (out_dir / "tab_models.tex").write_text(tabular(
            "models", ["Configuration", "Extractor", "Reader", "CC EM", "Full EM"], [], "lllrr"), encoding="utf-8")
    for name, note in _NOTES.items():
        (out_dir / f"tab_{name}_note.tex").write_text(note, encoding="utf-8")

    macro_lines = ["% generated by evaluation/report.py - do not edit"]
    for k, v in sorted(macros.items()):
        macro_lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    (out_dir / "results_macros.tex").write_text("\n".join(macro_lines) + "\n", encoding="utf-8")
    (out_dir / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote tables and macros to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--storage", type=Path, default=ROOT_DIR / "benchmark_summary.json")
    ap.add_argument("--paper-dir", type=Path, default=ROOT_DIR / "paper" / "generated")
    ap.add_argument("--models", default="",
                    help='Model comparison: "Label=results/dir,Label2=results/dir2" (main configuration first).')
    args = ap.parse_args()
    args.paper_dir.mkdir(parents=True, exist_ok=True)
    if args.models:
        configs = [(part.split("=", 1)[0].strip(), Path(part.split("=", 1)[1].strip()))
                   for part in args.models.split(",") if "=" in part]
        build_models(configs, args.paper_dir)
    build(args.results, args.storage, args.paper_dir)


if __name__ == "__main__":
    main()
