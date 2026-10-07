"""The development runner (evaluation/eval_musique_runner.py), with a fake engine."""
import json
from types import SimpleNamespace as NS

import pytest

import evaluation.eval_musique_runner as runner


class FakeEngine:
    """Records ingestion; answers every question with a fixed response."""
    instances = []

    def __init__(self, db_path=None, config_path=None, answer="ANSWER: Munich Institute of Physics"):
        self.answer, self.ingested, self.closed = answer, [], False
        self.last_query_status = {}
        self.graph = NS(entity_names=lambda: ["Dr. Elena Rostova", "Project Prometheus"])
        FakeEngine.instances.append(self)

    def reset(self):
        self.last_query_status = {}

    def ingest_text(self, text, source, title=None, debug=False):
        self.ingested.append(source)
        return ["ev1", "ev2"]

    def ask(self, question, debug=False):
        idx = int(self.ingested[0].rsplit(":", 1)[1]) if self.ingested else 0
        self.last_query_status = {"stage": "qa_success", "evidence_text": "Munich Institute of Physics employs her.",
                                  "evidence_sources": [f"musique:x:{idx}", "bad"]}
        return self.answer

    def close(self):
        self.closed = True


@pytest.fixture
def setup(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"base_url": "http://localhost:11434/v1"}))
    monkeypatch.setattr(runner, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(runner, "ContextCanvasEngine", FakeEngine)
    monkeypatch.setattr(runner, "check_ollama_status", lambda url: False)
    monkeypatch.setattr(runner, "compute_dataset_metrics", lambda p, g: {"answer_em": 1.0, "source": "local"})
    monkeypatch.setattr(runner, "run_official_musique_evaluator", lambda p, g, e: {"answer_em": 1.0, "source": "official"})
    FakeEngine.instances = []
    return tmp_path


def test_ollama_status(monkeypatch):
    class Resp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(runner.urllib.request, "urlopen", lambda req, timeout: Resp())
    assert runner.check_ollama_status("http://h:1/v1") is True
    def down(req, timeout):
        raise OSError("refused")
    monkeypatch.setattr(runner.urllib.request, "urlopen", down)
    assert runner.check_ollama_status() is False


def test_record_normalization():
    assert runner.normalize_musique_record({"id": "", "question": "q", "answer": "a"}) is None
    rec = runner.normalize_musique_record({
        "id": 7, "composed_question_text": " Q? ", "answer_text": "A", "answer_aliases": ["A", "B", 3],
        "contexts": [{"wikipedia_title": "T", "text": " body "}, "junk"], "answerable": False})
    assert rec["answer_aliases"] == ["A", "B"] and rec["paragraphs"] == [
        {"idx": 0, "title": "T", "paragraph_text": "body", "is_supporting": False}]
    assert rec["question"] == "Q?" and rec["answerable"] is False


def test_sample_loading(tmp_path):
    assert runner.load_musique_sample(tmp_path / "missing.jsonl")[0]["id"] == "2hop__test_1"
    path = tmp_path / "d.jsonl"
    rows = [{"id": "u", "question": "q", "answer": "a", "answerable": False}] + \
           [{"id": f"q{i}", "question": "q", "answer": "a"} for i in range(3)]
    path.write_text("\n" + "\n".join(json.dumps(r) for r in rows))
    assert [r["id"] for r in runner.load_musique_sample(path, max_samples=2)] == ["q0", "q1"]


def test_relation_detection():
    assert runner.detect_question_relations("Who founded the company of her husband?") >= {"spouse", "founded"}
    assert runner.detect_question_relations("What colour is the sky?") == set()


def test_hop_candidate_selection(capsys):
    paragraphs = [
        {"title": "Thriller (album)", "paragraph_text": "An album by the performer Michael Jackson."},
        {"title": "Thriller (album)", "paragraph_text": "duplicate title"},
        {"title": "", "paragraph_text": "no title"},
        {"title": "Weather", "paragraph_text": "Rain."},
    ]
    hop1 = runner.select_hop1_candidates("Who is the performer of Thriller?", paragraphs, max_candidates=3, debug=True)
    assert [p["title"] for p, _, _ in hop1] == ["Thriller (album)"]
    assert "[Hop 1 Candidate]" in capsys.readouterr().out
    hop2 = runner.select_hop2_candidates(
        {"Michael Jackson", "Thriller", "ab", "Who", "Performer Thriller"},
        "Who is the performer of Thriller?",
        [{"title": "Michael Jackson", "paragraph_text": "Michael Jackson married Lisa."},
         {"title": "Michael Jackson", "paragraph_text": "dup"}, {"title": "", "paragraph_text": "x"},
         {"title": "Rain", "paragraph_text": "Weather."}], debug=True)
    assert [p["title"] for p, _, _ in hop2] == ["Michael Jackson"]
    assert "[Hop 2 Candidate]" in capsys.readouterr().out


def test_paragraph_ingestion(capsys):
    eng = FakeEngine()
    assert runner._ingest_paragraph(eng, "q", {"idx": 3, "title": "", "paragraph_text": " "}, debug=False) == 0
    assert runner._ingest_paragraph(eng, "q", {"idx": 3, "title": "T", "paragraph_text": "Body."}, debug=True) == 2
    assert "[INGEST] T" in capsys.readouterr().out and eng.ingested == ["musique:q:3"]


@pytest.mark.parametrize("pred, ingested, events, meta, stage", [
    ("Paris", 1, 1, {}, "success"),
    ("", 0, 0, {}, "retrieval_empty"),
    ("", 2, 0, {}, "extraction_zero_events"),
    ("", 2, 5, {"stage": "graph_no_anchors"}, "graph_no_anchors"),
    ("", 2, 5, {"stage": "qa_refusal", "evidence_text": "Paris is big"}, "qa_refusal_answer_in_evidence"),
    ("", 2, 5, {"stage": "qa_refusal", "evidence_text": "Lyon"}, "qa_refusal_answer_not_in_evidence"),
    ("Paris France", 2, 5, {"stage": "qa_success"}, "answer_partial_match"),
    ("Lyon", 2, 5, {"stage": "qa_success", "evidence_text": "Paris"}, "answer_mismatch_answer_in_evidence"),
    ("Lyon", 2, 5, {"stage": "qa_success", "evidence_text": "Rome"}, "answer_mismatch_answer_not_in_evidence"),
])
def test_failure_stage_attribution(pred, ingested, events, meta, stage):
    assert runner.classify_case(pred, ["Paris"], ingested, events, meta) == stage


RECORD = {"id": "q1", "question": "Which university employs the inventor of Project Prometheus?",
          "answer": "Munich Institute of Physics", "paragraphs": [
              {"idx": 0, "title": "Project Prometheus", "paragraph_text": "Dr. Elena Rostova invented Project Prometheus.", "is_supporting": True},
              {"idx": 1, "title": "Dr. Elena Rostova", "paragraph_text": "Munich Institute of Physics employs Dr. Elena Rostova.", "is_supporting": True},
              {"idx": 2, "title": "Weather", "paragraph_text": "Rain.", "is_supporting": False}]}


@pytest.mark.parametrize("kwargs, mode, ingested", [
    ({"supporting_only": True}, "oracle_supporting", 2),
    ({"top_k": 2, "debug": True}, "retrieval_top2", 2),
    ({}, "all_paragraphs", 3),
])
def test_full_runs_write_predictions_and_metrics(setup, kwargs, mode, ingested):
    out = setup / "out"
    data = setup / "data.jsonl"
    data.write_text(json.dumps(RECORD) + "\n")
    metrics = runner.run_musique_evaluation(data, out, **kwargs)
    assert metrics["mode"] == mode and metrics["source"] == "local" and metrics["failure_stages"] == {"success": 1}
    engine = FakeEngine.instances[-1]
    assert engine.closed and len(engine.ingested) == ingested
    pred = json.loads((out / "musique_predictions.jsonl").read_text())
    assert pred["predicted_answer"] == "Munich Institute of Physics" and pred["predicted_support_idxs"]
    case = json.loads((out / "musique_debug_cases.json").read_text())[0]
    assert ("evidence_text" in case["query_meta"]) == bool(kwargs.get("debug"))


def test_refusals_and_the_official_evaluator(setup, monkeypatch):
    monkeypatch.setattr(runner, "ContextCanvasEngine",
                        lambda **kw: FakeEngine(answer="STATUS: NOT_IN_EVIDENCE"))
    evaluator = setup / "evaluate.py"
    evaluator.write_text("# official")
    metrics = runner.main(["--data", str(setup / "missing.jsonl"), "--output", str(setup / "o"),
                           "--evaluator", str(evaluator), "--samples", "1"])
    assert metrics["source"] == "official"
    pred = json.loads((setup / "o" / "musique_predictions.jsonl").read_text())
    assert pred["predicted_answer"] == "" and pred["predicted_answerable"] is False
