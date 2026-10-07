"""Tests for the hybrid annotation workflow: export/subset, LLM judge, N-annotator scoring."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from evaluation.annotation import FIELDS, score, subset
from evaluation.llm_judge import PLACEHOLDER_MODEL, group_paragraphs, load_rules, run, validate

import evaluation.llm_judge as _judge_mod
GUIDELINES = Path(_judge_mod.__file__).resolve().parent.parent / "ANNOTATION_GUIDELINES.md"


def _write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def _sample(n_paragraphs=4):
    rows = []
    for p in range(n_paragraphs):
        rows.append({"paragraph_key": f"p{p}", "paragraph_text": f"Walter{p} worked for DLR from 1988.", "event_index": 0,
                     "lemma": "employ", "sense_id": "employ.01", "roles": '{":ARG0": "DLR", ":ARG1": "Walter"}',
                     "time": '{"raw_expression": "1988"}'})
        rows.append({"paragraph_key": f"p{p}", "event_index": 1, "lemma": "bear", "sense_id": "bear.02",
                     "roles": '{":ARG1": "Walter"}'})
    return rows


class FakeClient:
    """Answers per paragraph; can reject parameters or return invalid output first."""
    def __init__(self, answers, reject=()):
        self.answers, self.reject, self.calls = answers, set(reject), []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        for param in self.reject:
            if param in kw:
                raise RuntimeError(f"Unsupported parameter: '{param}' is not supported with this model.")
        user = kw["messages"][1]["content"]
        key = next(k for k in self.answers if f"Walter{k[1:]} " in user)
        queue = self.answers[key]
        content = queue.pop(0) if len(queue) > 1 else queue[0]
        return NS(choices=[NS(message=NS(content=content))], usage=NS(prompt_tokens=10, completion_tokens=5))


GOOD = json.dumps({"events": [{"index": 0, "event_correct": 1, "roles_correct": 1, "context_correct": 1, "reason": "ok"},
                              {"index": 1, "event_correct": 0, "roles_correct": 1, "context_correct": None}],
                   "missed_facts": 1, "missed_examples": ["x"]})


def test_rules_come_verbatim_from_the_guidelines():
    rules = load_rules(GUIDELINES)
    assert rules.startswith("## What to fill in") and "missed_facts" in rules and "judge-rules" not in rules


def test_validation_blanks_fields_the_rules_leave_blank():
    rows = _sample(1)
    para = group_paragraphs(rows)[0]
    res = validate(json.loads(GOOD), para, rows)
    assert res["events"][0] == {"event_correct": 1, "roles_correct": 1, "context_correct": 1, "reason": "ok"}
    # Event 1 is wrong, so roles must be blank even though the judge filled them in.
    assert res["events"][1]["roles_correct"] is None and res["events"][1]["context_correct"] is None
    with pytest.raises(ValueError):
        validate({"events": [{"index": 0, "event_correct": 1, "roles_correct": 1, "context_correct": 1}],
                  "missed_facts": 0}, para, rows)  # event 1 missing
    with pytest.raises(ValueError):
        validate({"events": [{"index": 0, "event_correct": 2}, {"index": 1, "event_correct": 0}],
                  "missed_facts": 0}, para, rows)


def test_judge_writes_human_format_retries_and_records_failures(tmp_path):
    rows = _sample(3)
    _write(tmp_path / "sample.csv", rows)
    client = FakeClient({
        "p0": [GOOD],
        "p1": ["not json at all", GOOD],          # recovers on the retry
        "p2": ["still not json", "nope"],         # fails twice -> stays blank
    })
    cfg = {"model": "judge-x", "concurrency": 1}
    meta = run(tmp_path / "sample.csv", cfg, tmp_path / "llm.csv", GUIDELINES, client=client)
    out = list(csv.DictReader(open(tmp_path / "llm.csv")))
    assert [r["event_correct"] for r in out] == ["1", "0", "1", "0", "", ""]
    assert out[0]["missed_facts"] == "1" and out[1]["roles_correct"] == ""
    assert meta["failed"] == ["p2"] and meta["judged"] == 2 and meta["model"] == "judge-x"
    assert (tmp_path / "llm.log.jsonl").exists() and (tmp_path / "llm.meta.json").exists()

    # Resume: only the failed paragraph is asked again.
    client2 = FakeClient({"p0": [GOOD], "p1": [GOOD], "p2": [GOOD]})
    meta2 = run(tmp_path / "sample.csv", cfg, tmp_path / "llm.csv", GUIDELINES, client=client2)
    assert len(client2.calls) == 1 and meta2["failed"] == [] and meta2["judged"] == 3


def test_unsupported_parameters_are_dropped_and_recorded(tmp_path):
    _write(tmp_path / "sample.csv", _sample(2))
    client = FakeClient({"p0": [GOOD], "p1": [GOOD]}, reject=("temperature", "max_completion_tokens"))
    meta = run(tmp_path / "sample.csv", {"model": "m", "concurrency": 1}, tmp_path / "o.csv", GUIDELINES, client=client)
    assert meta["judged"] == 2 and meta["temperature"] is None
    assert "temperature" in meta["dropped_parameters"] and "max_completion_tokens->max_tokens" in meta["dropped_parameters"]
    assert "max_tokens" in client.calls[-1] and "temperature" not in client.calls[-1]


def test_placeholder_model_is_refused(tmp_path):
    _write(tmp_path / "sample.csv", _sample(1))
    try:
        run(tmp_path / "sample.csv", {"model": PLACEHOLDER_MODEL}, tmp_path / "o.csv", GUIDELINES, client=object())
    except SystemExit:
        return
    raise AssertionError("the placeholder model must not run")


def test_hybrid_scoring_humans_on_subset_judge_on_full_sample(tmp_path):
    rows = _sample(10)
    _write(tmp_path / "sample.csv", rows)
    subset(tmp_path / "sample.csv", 4, 11, tmp_path / "subset.csv")
    sub = list(csv.DictReader(open(tmp_path / "subset.csv")))
    assert len({r["paragraph_key"] for r in sub}) == 4

    def fill(rows, event1):
        out = []
        for r in rows:
            r = dict(r)
            if r["event_index"] == "0":
                r.update(event_correct="1", roles_correct="1", context_correct="1", missed_facts="0")
            else:
                r.update(event_correct=event1(r["paragraph_key"]))
                r["roles_correct"] = "1" if r["event_correct"] == "1" else ""
            out.append(r)
        return out
    _write(tmp_path / "a.csv", fill(sub, lambda k: "1"))
    _write(tmp_path / "b.csv", fill(sub, lambda k: "0" if k == sub[0]["paragraph_key"] else "1"))
    all_rows = list(csv.DictReader(open(tmp_path / "sample.csv")))
    _write(tmp_path / "j.csv", fill(all_rows, lambda k: "1"))
    (tmp_path / "j.meta.json").write_text(json.dumps({"model": "judge-x", "failed": []}))

    res = score([tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "j.csv"], ["Annotator A", "Annotator B", "LLM judge"],
                tmp_path / "gen" / "tab_annotation.tex", tmp_path / "j.meta.json")
    assert res["shared"]["Annotator A"]["n_paragraphs"] == 4
    assert res["full"]["LLM judge"]["n_paragraphs"] == 10 and "Annotator A" not in res["full"]
    assert res["shared"]["Annotator B"]["event_precision"] == pytest.approx(87.5)
    assert set(res["kappa"]) == {"Annotator A / Annotator B", "Annotator A / LLM judge", "Annotator B / LLM judge"}
    table = (tmp_path / "gen" / "tab_annotation.tex").read_text()
    assert "LLM judge (full)" in table and "judge-x" in (tmp_path / "gen" / "tab_annotation_note.tex").read_text()
    assert "\\newcommand{\\AnnFullParagraphs}{10}" in (tmp_path / "gen" / "annotation_macros.tex").read_text()


def test_export_separates_extractors_in_a_shared_cache(tmp_path):
    from evaluation.annotation import export
    cache = tmp_path / "cache"
    for key, model in (("aa1", "small:9b"), ("bb2", "gemma:12b")):
        (cache / key[:2]).mkdir(parents=True)
        (cache / key[:2] / f"{key}.json").write_text(json.dumps(
            {"payload": {"events": [{"lemma": "own", "sense_id": "own.01", "roles": {":ARG0": "A", ":ARG1": "B"}}]},
             "model": model}))
        with open(cache / "texts.jsonl", "a") as f:
            f.write(json.dumps({"key": key, "text": f"text {key}", "model": model}) + "\n")
    with pytest.raises(SystemExit):
        export(cache, tmp_path / "s.csv", 10, 1)          # ambiguous: two extractors
    export(cache, tmp_path / "s.csv", 10, 1, extractor="small:9b")
    rows = list(csv.DictReader(open(tmp_path / "s.csv")))
    assert {r["paragraph_key"] for r in rows} == {"aa1"} and rows[0]["role_meanings"]
