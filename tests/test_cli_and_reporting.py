"""Command-line entry points, the judge's edge cases, and report generation."""
import csv
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

import evaluation.annotation as annotation
import evaluation.llm_judge as llm_judge
import evaluation.report as report
from evaluation.annotation import FIELDS
from evaluation.llm_judge import Judge, group_paragraphs, load_rules, validate

ROOT = Path(llm_judge.__file__).resolve().parent.parent   # the repository, wherever the tests run from


@pytest.mark.parametrize("module", ["evaluation.experiments", "evaluation.report", "evaluation.annotation",
                                    "evaluation.llm_judge", "evaluation.eval_musique_runner"])
def test_scripts_find_the_repository_when_run_from_elsewhere(module, monkeypatch):
    """Running a script by path from another directory must still find the repository's packages."""
    path = importlib.import_module(module).__file__
    monkeypatch.setattr(sys, "path", [p for p in sys.path if Path(p or ".").resolve() != ROOT])
    spec = importlib.util.spec_from_file_location(f"_as_script_{module.replace('.', '_')}", path)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
    assert str(ROOT) in sys.path


# ------------------------------------------------------------------ annotation

def _write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def _rows(judged=True, failed=False):
    base = {"paragraph_key": "p0", "event_index": "0", "lemma": "own", "sense_id": "own.01", "roles": "{}"}
    return [{**base, "event_correct": "1" if judged else "", "roles_correct": "1" if judged else "",
             "missed_facts": "0" if judged else ""}]


def test_scoring_requires_consistent_inputs(tmp_path):
    _write(tmp_path / "a.csv", _rows())
    _write(tmp_path / "blank.csv", _rows(judged=False))
    for args in (([tmp_path / "a.csv"], ["only", "too many"]), ([tmp_path / "a.csv", tmp_path / "blank.csv"], None)):
        try:
            annotation.score(*args)
        except SystemExit:
            continue
        raise AssertionError("inconsistent inputs must be refused")


def test_kappa_without_overlapping_judgments():
    a = {("p", "0"): {"roles_correct": ""}}
    b = {("p", "0"): {"roles_correct": "1"}}
    k, n = annotation.pair_kappa(a, b, [("p", "0")], "roles_correct")
    assert n == 0 and k != k                                                  # NaN


def test_annotation_cli_and_failed_judge_note(tmp_path):
    cache = tmp_path / "cache" / "ab"
    cache.mkdir(parents=True)
    (cache / "abc.json").write_text(json.dumps({"payload": {"events": []}, "model": "m"}))
    (tmp_path / "cache" / "texts.jsonl").write_text(json.dumps({"key": "abc", "text": "Text.", "model": "m"}) + "\n")
    annotation.main(["export", "--cache", str(tmp_path / "cache"), "--out", str(tmp_path / "s.csv"), "--n", "5"])
    annotation.main(["subset", str(tmp_path / "s.csv"), "--n", "1", "--out", str(tmp_path / "sub.csv")])
    rows = list(csv.DictReader(open(tmp_path / "s.csv")))
    assert rows[0]["lemma"] == "" and rows[0]["paragraph_text"] == "Text."     # paragraph without events
    _write(tmp_path / "a.csv", _rows())
    _write(tmp_path / "b.csv", _rows())
    (tmp_path / "meta.json").write_text(json.dumps({"model": "judge-x", "failed": ["p9"]}))
    annotation.main(["score", str(tmp_path / "a.csv"), str(tmp_path / "b.csv"), "--labels", "A,B",
                     "--judge-meta", str(tmp_path / "meta.json"), "--out", str(tmp_path / "gen" / "tab.tex")])
    assert "1 paragraphs with invalid judge output" in (tmp_path / "gen" / "tab_note.tex").read_text()
    annotation.main(["score", str(tmp_path / "a.csv"), str(tmp_path / "b.csv")])     # labels default to file names


# ------------------------------------------------------------------ judge

def test_rules_markers_are_required(tmp_path):
    (tmp_path / "g.md").write_text("no markers")
    try:
        load_rules(tmp_path / "g.md")
    except SystemExit:
        return
    raise AssertionError("missing markers must stop the judge")


def _para(rows):
    return group_paragraphs(rows)[0]


EVENT_ROW = {"paragraph_key": "p", "event_index": "0", "lemma": "own", "sense_id": "own.01", "roles": "{}"}


@pytest.mark.parametrize("parsed", [
    {"events": "not a list", "missed_facts": 0},
    {"events": [{"index": 5, "event_correct": 1}], "missed_facts": 0},
    {"events": [{"index": 0, "event_correct": 0}, {"index": 0, "event_correct": 0}], "missed_facts": 0},
    {"events": [{"index": 0, "event_correct": 0}], "missed_facts": -1},
    {"events": [{"index": 0, "event_correct": 0}], "missed_facts": True},
])
def test_structurally_invalid_judgments_are_rejected(parsed):
    with pytest.raises(ValueError):
        validate(parsed, _para([EVENT_ROW]), [EVENT_ROW])


def test_lenient_binary_values_are_accepted():
    res = validate({"events": [{"index": 0, "event_correct": True, "roles_correct": " 1 "}], "missed_facts": 0},
                   _para([EVENT_ROW]), [EVENT_ROW])
    assert res["events"][0]["event_correct"] == 1 and res["events"][0]["roles_correct"] == 1


def test_messages_skip_rows_without_events():
    rows = [{**EVENT_ROW, "paragraph_text": "T"}, {**EVENT_ROW, "event_index": "1", "lemma": ""}]
    user = llm_judge.build_messages("rules", _para(rows), rows)[1]["content"]
    assert "[0] own" in user and "[1]" not in user


def test_judge_builds_an_openai_client_from_the_environment(monkeypatch):
    import openai
    created = {}

    class FakeOpenAI:
        def __init__(self, **kw):
            created.update(kw)
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.delenv("MY_KEY", raising=False)
    try:
        Judge({"model": "m", "api_key_env": "MY_KEY"})
    except SystemExit:
        pass
    else:
        raise AssertionError("a missing key must stop the judge")
    monkeypatch.setenv("MY_KEY", "secret")
    Judge({"model": "m", "api_key_env": "MY_KEY", "base_url": "https://example.org/v1"})
    assert created["api_key"] == "secret" and created["base_url"] == "https://example.org/v1"


class _Client:
    def __init__(self, error):
        self.error, self.calls = error, []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        if "response_format" in kw:
            raise RuntimeError(self.error)
        return NS(choices=[NS(message=NS(content=None))], usage=None)


def test_json_mode_is_dropped_once_and_other_errors_propagate():
    judge = Judge({"model": "m", "temperature": None}, client=_Client("response_format is not supported"))
    assert judge._call([]) == ("", {"prompt_tokens": 0, "completion_tokens": 0})
    assert judge.dropped == ["response_format"] and "response_format" not in judge.client.calls[-1]
    with pytest.raises(RuntimeError):
        Judge({"model": "m", "temperature": None}, client=_Client("server exploded"))._call([])


def test_judge_cli_tolerates_corrupt_log_lines(tmp_path, monkeypatch):
    _write(tmp_path / "s.csv", [{**EVENT_ROW, "paragraph_text": "Walter0 worked."}])
    (tmp_path / "o.log.jsonl").write_text("{corrupt\n")
    (tmp_path / "judge.json").write_text(json.dumps({"model": "m"}))
    answer = json.dumps({"events": [{"index": 0, "event_correct": 1, "roles_correct": 1}], "missed_facts": 0})

    class Client:
        chat = NS(completions=NS(create=lambda **kw: NS(choices=[NS(message=NS(content=answer))], usage=None)))
    real_init = Judge.__init__
    monkeypatch.setattr(Judge, "__init__", lambda self, cfg, client=None: real_init(self, cfg, Client()))
    llm_judge.main([str(tmp_path / "s.csv"), "--config", str(tmp_path / "judge.json"), "--out", str(tmp_path / "o.csv"),
                    "--guidelines", str(ROOT / "ANNOTATION_GUIDELINES.md")])
    assert json.loads((tmp_path / "o.meta.json").read_text())["judged"] == 1


# ------------------------------------------------------------------ report

def test_report_helpers():
    assert report.fmt_p(None) == "--" and report.fmt_p(0.0001) == "$<$0.001"
    assert report.compare([{"id": "a"}], [{"id": "b"}])["n"] == 0


def _results(path, ids, systems=("contextcanvas", "full_context")):
    path.mkdir()
    (path / "manifest.json").write_text(json.dumps({"test_ids": ids, "extraction_model": "e", "answer_model": "r"}))
    for s in systems:
        (path / s).mkdir()
        (path / s / "predictions.jsonl").write_text("".join(
            json.dumps({"id": q, "hops": 2, "answerable": True, "em": 1.0, "f1": 1.0, "refused": False, "usage": {}}) + "\n"
            for q in ids))


def test_report_cli_with_models_and_storage(tmp_path):
    _results(tmp_path / "main", ["q1", "q2"])
    _results(tmp_path / "other", ["q1", "q2"], systems=("contextcanvas",))
    storage = tmp_path / "bench.json"
    storage.write_text(json.dumps({"storage": [{"events_count": 100, "events_per_second": 50.0, "disk_bytes": 4096,
                                                "bytes_per_event": 40.96, "median_query_latency_ms": 1.5}]}))
    out = tmp_path / "gen"
    report.main(["--results", str(tmp_path / "main"), "--storage", str(storage), "--paper-dir", str(out),
                 "--models", f"Main={tmp_path / 'main'},Other={tmp_path / 'other'}"])
    assert "4096" in (out / "tab_storage.tex").read_text()
    models = (out / "tab_models.tex").read_text()
    assert "Main" in models and "Other" in models


def test_model_table_refuses_different_question_sets(tmp_path):
    _results(tmp_path / "a", ["q1"])
    _results(tmp_path / "b", ["q2"])
    try:
        report.build_models([("A", tmp_path / "a"), ("B", tmp_path / "b")], tmp_path)
    except SystemExit:
        return
    raise AssertionError("different question sets must be refused")


def test_main_table_reference_is_configurable(tmp_path):
    _results(tmp_path / "res", ["q1", "q2"], systems=("contextcanvas", "contextcanvas_retrieval", "full_context"))
    report.build(tmp_path / "res", None, tmp_path / "gen", reference="contextcanvas_retrieval")
    table = (tmp_path / "gen" / "tab_main.tex").read_text()
    retrieval_row = next(l for l in table.splitlines() if l.startswith("ContextCanvas retrieval + LLM"))
    assert retrieval_row.rstrip(" \\\\").endswith("ref.")
    assert "against ContextCanvas retrieval + LLM" in (tmp_path / "gen" / "tab_main_note.tex").read_text()
