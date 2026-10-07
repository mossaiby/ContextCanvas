"""Intrinsic evaluation of extraction quality: humans and an LLM judge annotate extracted events.

See ANNOTATION_GUIDELINES.md for the procedure and the rules. Commands:

    # 1. sample extracted paragraphs (from the extraction cache) for annotation
    python -m evaluation.annotation export --cache results/extraction_cache --extractor qwen3.5:9b \
        --out annotation/sample.csv --n 200 --seed 7

    # 2. the paragraphs the two humans annotate (a subset of the sample)
    python -m evaluation.annotation subset annotation/sample.csv --n 35 --seed 11 --out annotation/human_subset.csv

    # 3. the LLM judge annotates the full sample:  python -m evaluation.llm_judge ...

    # 4. score any number of annotation files
    python -m evaluation.annotation score annotation/annotator_a.csv annotation/annotator_b.csv \
        annotation/annotator_llm.csv --labels "Annotator A,Annotator B,LLM judge" \
        --judge-meta annotation/annotator_llm.meta.json --out paper/generated/tab_annotation.tex

Annotators fill, per event row: event_correct (1/0), roles_correct (1/0, blank if the event is
wrong), context_correct (1/0, blank if no time/place or the event is wrong); and per paragraph,
on its first row, missed_facts (a count).

Scoring compares files on the paragraphs they share (a human subset against the judge's full
sample), reports each annotator's measures there, adds a full-sample column for any annotator
that covered more, and gives Cohen's kappa for every pair of annotators.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.stats import cohens_kappa, mean  # noqa: E402

FIELDS = ["paragraph_key", "paragraph_text", "event_index", "lemma", "sense_id", "roles", "role_meanings",
          "time", "place", "event_correct", "roles_correct", "context_correct", "missed_facts"]
JUDGMENTS = ("event_correct", "roles_correct", "context_correct", "missed_facts")


# ------------------------------------------------------------------------------- export / subset

def _role_meanings(sense_id: str, roles: Dict[str, str]) -> str:
    from semantics.propbank import GLOBAL_CATALOG  # imported lazily: loading the catalog is slow
    meanings = {r: GLOBAL_CATALOG.role_description(sense_id, r) for r in roles}
    return json.dumps({r: m for r, m in meanings.items() if m}, ensure_ascii=False) if any(meanings.values()) else ""


def export(cache_dir: Path, out: Path, n: int, seed: int, extractor: Optional[str] = None,
           texts_path: Optional[Path] = None) -> None:
    """Samples cached extractions. Paragraph texts come from texts.jsonl, which the realizer
    writes beside the cache. The cache is shared by all model configurations, so `extractor`
    restricts the sample to one extraction model."""
    texts: Dict[str, str] = {}
    texts_file = texts_path or (cache_dir / "texts.jsonl")
    if texts_file.exists():
        for line in texts_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if extractor is None or row.get("model") == extractor:
                    texts[row["key"]] = row["text"]
    models = {json.loads(l).get("model") for l in texts_file.read_text(encoding="utf-8").splitlines() if l.strip()} \
        if texts_file.exists() else set()
    if extractor is None and len(models) > 1:
        raise SystemExit(f"The cache holds extractions from several models {sorted(map(str, models))}; pass --extractor.")
    entries = sorted(p for p in cache_dir.glob("*/*.json") if p.stem in texts)
    rng = random.Random(seed)
    chosen = rng.sample(entries, min(n, len(entries)))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for path in chosen:
            payload = json.loads(path.read_text(encoding="utf-8")).get("payload", {})
            events = payload.get("events", []) or [{}]
            for i, ev in enumerate(events):
                roles = ev.get("roles", {}) or {}
                w.writerow({
                    "paragraph_key": path.stem, "paragraph_text": texts.get(path.stem, "") if i == 0 else "",
                    "event_index": i, "lemma": ev.get("lemma", ""), "sense_id": ev.get("sense_id", ""),
                    "roles": json.dumps(roles, ensure_ascii=False) if ev else "",
                    "role_meanings": _role_meanings(ev.get("sense_id", ""), roles) if ev else "",
                    "time": json.dumps(ev.get("time_context"), ensure_ascii=False) if ev.get("time_context") else "",
                    "place": json.dumps(ev.get("spatial_context"), ensure_ascii=False) if ev.get("spatial_context") else "",
                    **{k: "" for k in JUDGMENTS},
                })
    print(f"Wrote {len(chosen)} paragraphs to {out}")


def subset(sample: Path, n: int, seed: int, out: Path) -> None:
    """Whole paragraphs, in the sample's order."""
    rows = _read(sample)
    keys = list(dict.fromkeys(r["paragraph_key"] for r in rows))
    chosen = set(random.Random(seed).sample(keys, min(n, len(keys))))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            if r["paragraph_key"] in chosen:
                w.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"Wrote {len(chosen)} of {len(keys)} paragraphs to {out}")


# ------------------------------------------------------------------------------- scoring

def _read(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _keyed(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, str]]:
    return {(r["paragraph_key"], str(r["event_index"])): r for r in rows}


def _val(row: Dict[str, str], field: str) -> Optional[int]:
    v = (row.get(field) or "").strip()
    return int(v) if v != "" else None


def measures(rows: Sequence[Dict[str, str]]) -> Dict[str, float]:
    event_rows = [r for r in rows if r.get("lemma")]
    ev = [v for v in (_val(r, "event_correct") for r in event_rows) if v is not None]
    correct = [r for r in event_rows if _val(r, "event_correct") == 1]
    ro = [v for v in (_val(r, "roles_correct") for r in correct) if v is not None]
    cx = [v for v in (_val(r, "context_correct") for r in correct) if v is not None]
    missed = [v for v in (_val(r, "missed_facts") for r in rows if str(r["event_index"]) == "0") if v is not None]
    nan = float("nan")
    return {
        "n_paragraphs": len({r["paragraph_key"] for r in rows}), "n_events": len(ev),
        "event_precision": 100 * mean(ev) if ev else nan, "role_accuracy": 100 * mean(ro) if ro else nan,
        "context_accuracy": 100 * mean(cx) if cx else nan, "n_context": len(cx),
        "missed_per_paragraph": mean(missed) if missed else nan,
    }


def pair_kappa(a: Dict, b: Dict, keys: List, field: str) -> Tuple[float, int]:
    pairs = [(_val(a[k], field), _val(b[k], field)) for k in keys]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if not pairs:
        return float("nan"), 0
    return cohens_kappa([p[0] for p in pairs], [p[1] for p in pairs]), len(pairs)


def _f(x: float, d: int = 1) -> str:
    return "--" if x != x else f"{x:.{d}f}"


def _annotated_paragraphs(f: Dict) -> set:
    return {k[0] for k, r in f.items() if any((r.get(j) or "").strip() for j in JUDGMENTS)}


def score(paths: List[Path], labels: Optional[List[str]] = None, out: Optional[Path] = None,
          judge_meta: Optional[Path] = None) -> Dict:
    labels = labels or [p.stem for p in paths]
    if len(labels) != len(paths):
        raise SystemExit("--labels must name every file")
    files = [_keyed(_read(p)) for p in paths]

    # Paragraphs every annotator worked on (humans cover a subset; judge failures stay blank).
    shared_paragraphs = set.intersection(*(_annotated_paragraphs(f) for f in files))
    if not shared_paragraphs:
        raise SystemExit("The files have no annotated paragraphs in common.")
    shared_keys = sorted(k for k in set.intersection(*(set(f) for f in files)) if k[0] in shared_paragraphs)

    res: Dict = {"labels": labels, "shared": {}, "full": {}, "kappa": {}}
    for label, f in zip(labels, files):
        res["shared"][label] = measures([f[k] for k in shared_keys])
        own = _annotated_paragraphs(f)
        if len(own) > len(shared_paragraphs):
            res["full"][label] = measures([r for k, r in f.items() if k[0] in own])
    for (la, fa), (lb, fb) in itertools.combinations(zip(labels, files), 2):
        res["kappa"][f"{la} / {lb}"] = {
            field: pair_kappa(fa, fb, shared_keys, field) for field in ("event_correct", "roles_correct")}

    if judge_meta and judge_meta.exists():
        res["judge"] = json.loads(judge_meta.read_text(encoding="utf-8"))

    if out:
        _write_table(res, out)
    print(json.dumps(res, indent=2, default=str))
    return res


def _write_table(res: Dict, out: Path) -> None:
    from evaluation.report import tex_escape
    cols = [(l, res["shared"][l]) for l in res["labels"]] + [(f"{l} (full)", m) for l, m in res["full"].items()]
    rows = [
        ("Event precision (\\%)", "event_precision", 1), ("Role-direction accuracy (\\%)", "role_accuracy", 1),
        ("Time/place accuracy (\\%)", "context_accuracy", 1), ("Missed facts per paragraph", "missed_per_paragraph", 2),
        ("Paragraphs", "n_paragraphs", 0), ("Events judged", "n_events", 0),
    ]
    lines = ["% generated by evaluation/annotation.py - do not edit",
             r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
             "Measure & " + " & ".join(tex_escape(c) for c, _ in cols) + r" \\", r"\midrule"]
    for name, key, d in rows:
        lines.append(name + " & " + " & ".join(_f(float(m[key]), d) for _, m in cols) + r" \\")
    lines += [r"\midrule", r"Cohen's $\kappa$ (events / roles) & \multicolumn{" + str(len(cols)) + r"}{l}{} \\"]
    for pair, k in res["kappa"].items():
        lines.append(f"\\quad {tex_escape(pair)} & \\multicolumn{{{len(cols)}}}{{l}}"
                     f"{{{_f(k['event_correct'][0], 2)} / {_f(k['roles_correct'][0], 2)}}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    shared_n = next(iter(res["shared"].values()))["n_paragraphs"]
    judge = res.get("judge") or {}
    note = (f"Columns without `(full)': the {shared_n} paragraphs annotated by everyone; `(full)': all paragraphs "
            "that annotator judged. $\\kappa$ is computed on the shared paragraphs.")
    if judge:
        note += f" Judge model: \\texttt{{{tex_escape(judge.get('model', ''))}}}"
        if judge.get("failed"):
            note += f"; {len(judge['failed'])} paragraphs with invalid judge output are excluded"
        note += "."
    out.with_name(out.stem + "_note.tex").write_text(f"\\par\\smallskip{{\\footnotesize {note}}}\n", encoding="utf-8")

    full = next(iter(res["full"].values()), None)
    macros = {
        "AnnSharedParagraphs": str(shared_n),
        "AnnFullParagraphs": str(full["n_paragraphs"]) if full else "--",
        "AnnJudgeModel": ("\\texttt{" + tex_escape(judge.get("model", "--")) + "}") if judge else "--",
    }
    (out.parent / "annotation_macros.tex").write_text(
        "% generated by evaluation/annotation.py - do not edit\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items())), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--cache", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--n", type=int, default=200)
    e.add_argument("--seed", type=int, default=7)
    e.add_argument("--extractor", default=None, help="Only extractions made by this model (required if the cache holds several).")
    s = sub.add_parser("subset")
    s.add_argument("sample", type=Path)
    s.add_argument("--n", type=int, default=35)
    s.add_argument("--seed", type=int, default=11)
    s.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("score")
    c.add_argument("files", type=Path, nargs="+")
    c.add_argument("--labels", default="")
    c.add_argument("--judge-meta", type=Path, default=None)
    c.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.cmd == "export":
        export(args.cache, args.out, args.n, args.seed, args.extractor)
    elif args.cmd == "subset":
        subset(args.sample, args.n, args.seed, args.out)
    else:
        labels = [l.strip() for l in args.labels.split(",")] if args.labels else None
        score(args.files, labels, args.out, args.judge_meta)


if __name__ == "__main__":  # pragma: no cover  (script entry point; main() itself is tested)
    main()
