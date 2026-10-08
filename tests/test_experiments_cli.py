"""The experiment driver (evaluation/experiments.py) end to end, with fake models."""
import json

import pytest

import engine.pipeline as pipeline_mod
import engine.realizer as realizer_mod
import evaluation.experiments as exp


class FakeRealizer:
    def __init__(self, config_path=None, overrides=None):
        self.config = {"model": "fake", **(overrides or {})}
        self.context_window_tokens, self.output_reserve_tokens = 8192, 2048
        self.last_answer_text = "FINAL ANSWER: Paris"

    def reset_usage(self):
        pass

    def usage_summary(self):
        return {"query_tokens": 10, "index_tokens": 0, "total_tokens": 10, "seconds": 0.1}

    def answer_question(self, question, evidence, candidate_entities=None, evidence_kind="graph"):
        return "ANSWER: Paris" if "Paris" in evidence else "STATUS: NOT_IN_EVIDENCE"


class FakeEngine:
    fail_on = None

    def __init__(self, db_path, config_path, config_overrides):
        self.realizer = FakeRealizer(overrides=config_overrides)
        self.ablations = {"triples_only": bool((config_overrides.get("ablations") or {}).get("triples_only"))}
        self.texts, self.last_query_status, self.closed = [], {}, False

    def reset(self):
        self.texts = []

    def ingest_text(self, text, source, title=None):
        self.texts.append((text, source))
        return ["ev"] if "Paris" in text else []

    def ask(self, question):
        if FakeEngine.fail_on and FakeEngine.fail_on in question:
            raise RuntimeError("engine crashed")
        hits = [s for t, s in self.texts if "Paris" in t]
        self.last_query_status = {"stage": "qa_success" if hits else "graph_no_anchors", "evidence_sources": hits,
                                  "evidence_chars": 42, "anchors": ["France"], "llm_output": "FINAL ANSWER: Paris"}
        return "ANSWER: Paris" if hits else "STATUS: NOT_IN_EVIDENCE"

    def rank_sources(self, question):
        hits = [s for t, s in self.texts if "Paris" in t]
        self.last_query_status = {"stage": "retrieval" if hits else "graph_no_anchors", "anchors": ["France"] if hits else []}
        return hits + ["musique:unknown:99"]          # a source that is not a paragraph is ignored

    def close(self):
        self.closed = True


def _record(i, answerable=True, with_answer=True, prefix="2hop"):
    return {"id": f"{prefix}__{i}_{i + 1}", "question": f"Q{i}: capital of France?", "answer": "Paris",
            "answerable": answerable, "paragraphs": [
                {"idx": 0, "title": "France", "paragraph_text": "The capital is Paris." if with_answer else "Wine.",
                 "is_supporting": True},
                {"idx": 1, "title": "", "paragraph_text": "Noise.", "is_supporting": False}]}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_mod, "ContextCanvasEngine", FakeEngine)
    monkeypatch.setattr(realizer_mod, "Realizer", FakeRealizer)
    monkeypatch.setattr(exp, "ROOT_DIR", tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "propbank_lock.json").write_text(json.dumps({"ref": "v3.4.0", "commit": "abc"}))
    data = tmp_path / "musique.jsonl"
    records = [_record(i, with_answer=i % 2 == 0) for i in range(12)] + [_record(99, answerable=False)]
    data.write_text("\n".join(json.dumps(r) for r in records) + "\n\n" + json.dumps({"id": "", "question": "x"}) + "\n")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"extraction_model": "small:9b", "answer_model": "big:27b"}))
    FakeEngine.fail_on = None
    return tmp_path, data, config


def _args(env, *extra):
    tmp, data, config = env
    return ["--data", str(data), "--out", str(tmp / "out"), "--config", str(config), "--n", "4", "--seed", "1",
            "--exclude-first", "4", "--cache-dir", str(tmp / "cache"), *extra]


def _predictions(tmp, system):
    return [json.loads(l) for l in (tmp / "out" / system / "predictions.jsonl").read_text().splitlines()]


def test_hop_counts_from_ids():
    assert exp.hop_count({"id": "triple__1_2_3"}) == 3
    assert exp.hop_count({"id": "4hop1__5_6_7_8"}) == 4
    assert exp.hop_count({"id": "custom__11_22"}) == 2
    assert exp.hop_count({"id": "plain"}) == 1


def test_scoring_and_parsing_helpers():
    unanswerable = {"answerable": False, "answer_aliases": ["x"]}
    assert exp.score("", unanswerable) == {"em": 1.0, "f1": 1.0, "refused": True}
    assert exp.score("Paris", unanswerable)["em"] == 0.0
    assert exp.score("", {"answerable": True, "answer_aliases": ["Paris"]})["f1"] == 0.0
    assert exp.clean_prediction("STATUS: NOT_IN_EVIDENCE") == "" and exp.clean_prediction("ANSWER: Lyon") == "Lyon"
    assert exp.source_idx("musique:q:7") == 7 and exp.source_idx("nothing") is None


def test_records_and_unanswerable_pool(env):
    _, data, _ = env
    assert len(exp.load_records(data, include_unanswerable=False)) == 12
    assert len(exp.load_records(data, include_unanswerable=True)) == 13


def test_main_runs_graph_and_text_systems_and_writes_the_manifest(env):
    tmp = env[0]
    exp.main(_args(env, "--systems", "contextcanvas,closed_book,bm25_k2,contextcanvas"))
    manifest = json.loads((tmp / "out" / "manifest.json").read_text())
    assert len(manifest["test_ids"]) == 4 and manifest["propbank"]["commit"] == "abc"
    assert (manifest["extraction_model"], manifest["answer_model"]) == ("small:9b", "big:27b")
    cc = _predictions(tmp, "contextcanvas")
    assert len(cc) == 4 and {r["failure_stage"] for r in cc} == {"success", "extraction_zero_events"}
    assert all(r["support_pred"] == [0] for r in cc if r["em"] == 1.0)
    assert {r["predicted"] for r in _predictions(tmp, "closed_book")} == {""}
    eff = json.loads((tmp / "out" / "bm25_k2" / "effective_config.json").read_text())
    assert eff["evidence_max_chars"] == int((8192 - 2048 - 700) * 3.5)


def test_resume_reuses_the_sample_and_skips_finished_questions(env, capsys):
    exp.main(_args(env, "--systems", "closed_book"))
    exp.main(_args(env, "--systems", "closed_book"))
    out = capsys.readouterr().out
    assert "Reusing sample" in out and "4 done, 0 to run" in out
    assert len(_predictions(env[0], "closed_book")) == 4


def test_errors_are_recorded_not_dropped(env):
    FakeEngine.fail_on = "Q"
    exp.main(_args(env, "--systems", "cc_triples"))
    preds = _predictions(env[0], "cc_triples")
    assert all(r["error"] and r["failure_stage"] == "error" for r in preds)


def test_unanswerable_questions_have_no_failure_stage(env):
    tmp, data, config = env
    exp.main(_args(env, "--include-unanswerable", "--n", "13", "--systems", "contextcanvas"))
    unans = [r for r in _predictions(tmp, "contextcanvas") if not r["answerable"]]
    assert unans and all(r["failure_stage"] is None for r in unans)


@pytest.mark.parametrize("change, message", [
    ("config", "config.json changed"),
    ("pool", "created with --pool"),
    ("system", "Unknown system"),
])
def test_main_refuses_inconsistent_runs(env, change, message):
    exp.main(_args(env, "--systems", "closed_book"))
    extra = ["--systems", "closed_book"]
    if change == "config":
        env[2].write_text(json.dumps({"extraction_model": "other", "answer_model": "big:27b"}))
    elif change == "pool":
        extra += ["--pool", "dev"]
    else:
        extra = ["--systems", "no_such_system"]
    try:
        exp.main(_args(env, *extra))
    except SystemExit as e:
        assert message in str(e)
    else:
        raise AssertionError("expected the run to be refused")


def test_groups_expand_to_their_systems():
    assert "cc_triples" in exp.GROUPS["ablations"] and "oracle" in exp.GROUPS["main"]


def test_graph_retrieval_reads_ranked_paragraphs_as_text(env):
    tmp = env[0]
    exp.main(_args(env, "--systems", "contextcanvas_retrieval,contextcanvas_retrieval_nofill"))
    filled = _predictions(tmp, "contextcanvas_retrieval")
    assert all(len(r["support_pred"]) == 2 for r in filled)                # k=5, but only 2 paragraphs exist
    assert {r["retrieval"]["from_graph"] for r in filled} == {0, 1}
    hits = [r for r in filled if r["retrieval"]["from_graph"] == 1]
    assert all(r["support_pred"][0] == 0 and r["em"] == 1.0 for r in hits)
    unfilled = _predictions(tmp, "contextcanvas_retrieval_nofill")
    misses = [r for r in unfilled if r["retrieval"]["from_graph"] == 0]
    assert misses and all(r["support_pred"] == [] and r["predicted"] == "" and r["llm_output"] == "" for r in misses)
    assert {r["failure_stage"] for r in misses} == {"extraction_zero_events"}
    assert {r["failure_stage"] for r in unfilled if r["retrieval"]["from_graph"]} == {"success"}


def test_filling_stops_at_k_paragraphs(env):
    engine = FakeEngine(db_path=None, config_path=None, config_overrides={"retrieval_k": 1, "retrieval_fill": True})
    case = exp.load_records(env[1], include_unanswerable=False)[0] | {"paragraphs": [
        {"idx": i, "title": f"T{i}", "paragraph_text": "France wine." if i else "Capital of France?", "is_supporting": False}
        for i in range(3)]}
    out = exp.run_graph_retrieval_case(engine, case, max_chars=10_000)
    assert out["retrieval"] == {"from_graph": 0, "filled": 1, "links": {"event": 0, "title": 0, "mention": 0}}
    assert len(out["support"]) == 1