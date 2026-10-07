"""Tests for the experiment, statistics, baseline and report machinery used by the paper."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evaluation.baselines import BM25, format_passages, run_text_baseline
from evaluation.stats import bootstrap_ci, cohens_kappa, mcnemar_exact_p, paired_bootstrap_p, set_f1
from evaluation.report import holm


def test_mcnemar_and_bootstrap_basics():
    b01, b10, p = mcnemar_exact_p([True] * 10 + [False] * 10, [True] * 10 + [True] * 10)
    assert (b01, b10) == (10, 0) and p < 0.01
    assert mcnemar_exact_p([True, False], [True, False])[2] == 1.0
    lo, hi = bootstrap_ci([1.0] * 50 + [0.0] * 50, seed=1)
    assert lo < 0.5 < hi
    assert bootstrap_ci([1.0, 0.0, 1.0], seed=3) == bootstrap_ci([1.0, 0.0, 1.0], seed=3)
    assert paired_bootstrap_p([1.0] * 30, [0.0] * 30) < 0.01
    assert paired_bootstrap_p([0.5] * 30, [0.5] * 30) == 1.0


def test_agreement_f1_and_holm():
    assert cohens_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert set_f1([1, 2], [2, 3]) == 0.5
    assert set_f1([], []) == 1.0
    adj = holm([0.01, 0.04, 0.03])
    assert adj[0] == pytest.approx(0.03) and adj[2] == pytest.approx(0.06) and adj[1] == pytest.approx(0.06)


def test_bm25_ranks_relevant_paragraph_first():
    docs = ["Kimbrough Memorial Stadium is in Canyon, Texas.", "Cork is a city in Ireland.", "Canyon is in Randall County."]
    assert BM25(docs).top_k("Which county is Kimbrough Memorial Stadium in?", 1) == [0]
    text, used = format_passages([{"idx": 3, "title": "A", "paragraph_text": "x" * 50},
                                  {"idx": 4, "title": "B", "paragraph_text": "y" * 50}], max_chars=70)
    assert used == [3]


def test_hop_count_and_dev_exclusion():
    from evaluation.experiments import draw_sample, hop_count
    assert hop_count({"id": "2hop__1_2"}) == 2
    assert hop_count({"id": "double__460946_294723"}) == 2
    assert hop_count({"id": "x", "question_decomposition": [1, 2, 3]}) == 3
    recs = [{"id": f"q{i}", "answerable": True} for i in range(100)]
    m = draw_sample(recs, n=30, seed=13, exclude_first=50)
    assert not set(m["test_ids"]) & set(m["dev_ids"]) and len(m["test_ids"]) == 30
    assert m == draw_sample(recs, n=30, seed=13, exclude_first=50)


def _fake_response(content, prompt_tokens=100, completion_tokens=20):
    choice = MagicMock()
    choice.message.content = content
    usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return MagicMock(choices=[choice], usage=usage)


def test_extraction_cache_charges_original_cost(tmp_path):
    from engine.realizer import Realizer
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"model_name": "m", "base_url": "http://x/v1", "api_key": "k"}))
    overrides = {"extraction_cache_dir": str(tmp_path / "cache")}
    payload = json.dumps({"events": [{"temp_id": "ev1", "lemma": "own", "sense_id": "own.01",
                                      "roles": {":ARG0": "A", ":ARG1": "B"}}]})
    r1 = Realizer(config_path=str(cfg), overrides=overrides)
    r1.client.chat.completions.create = MagicMock(return_value=_fake_response(payload))
    assert len(r1.extract_structured_context("A owns B.").events) == 1
    first = r1.usage_summary()
    r2 = Realizer(config_path=str(cfg), overrides=overrides)
    r2.client.chat.completions.create = MagicMock(side_effect=AssertionError("cache should have been used"))
    assert len(r2.extract_structured_context("A owns B.").events) == 1
    second = r2.usage_summary()
    assert second["extract_cache_hits"] == 1 and second["index_tokens"] == first["index_tokens"] == 120
    assert (tmp_path / "cache" / "texts.jsonl").exists()


def test_ablation_flags_change_rendering(tmp_path):
    from engine.pipeline import ContextCanvasEngine
    from semantics.schema import ExtractedEvent
    eng = ContextCanvasEngine(db_path=str(tmp_path / "db"), config_overrides={"ablations": {"triples_only": True}})
    eng.ingest_event_direct(ExtractedEvent(temp_id="e1", lemma="own", sense_id="own.01",
                                           roles={":ARG0": "Bombardier Inc.", ":ARG1": "Bombardier Aerospace"}))
    text = eng.graph.query_subgraph_blueprint("Bombardier Aerospace")
    assert "- (Bombardier Inc., own, Bombardier Aerospace)" in text and "ARG0" not in text and "ARG1" not in text
    eng.close()
    with pytest.raises(ValueError):
        ContextCanvasEngine(db_path=str(tmp_path / "db2"), config_overrides={"ablations": {"no_such_flag": False}})


def test_text_baselines_use_shared_answer_path():
    realizer = MagicMock()
    realizer.answer_question.return_value = "ANSWER: Randall County"
    case = {"question": "Which county is Kimbrough Memorial Stadium in?", "paragraphs": [
        {"idx": 0, "title": "Kimbrough Memorial Stadium", "paragraph_text": "It is in Canyon, Texas.", "is_supporting": True},
        {"idx": 1, "title": "Cork", "paragraph_text": "Cork is in Ireland.", "is_supporting": False}]}
    out = run_text_baseline(realizer, case, "bm25_k1", max_chars=10_000)
    assert out["support"] == [0] and realizer.answer_question.call_args.kwargs["evidence_kind"] == "text"
    assert run_text_baseline(realizer, case, "oracle", 10_000)["support"] == [0]
    run_text_baseline(realizer, case, "closed_book", 10_000)
    assert realizer.answer_question.call_args.kwargs["evidence_kind"] == "none"


def test_report_builds_tables_from_predictions(tmp_path):
    from evaluation.report import build
    res = tmp_path / "res"
    ids = [f"q{i}" for i in range(20)]
    (res).mkdir()
    (res / "manifest.json").write_text(json.dumps({"test_ids": ids, "dev_ids": ["d"], "seed": 13}))
    for system, correct in (("contextcanvas", 15), ("full_context", 10), ("cc_triples", 8)):
        d = res / system
        d.mkdir()
        with open(d / "predictions.jsonl", "w") as f:
            for i, q in enumerate(ids):
                ok = float(i < correct)
                f.write(json.dumps({"id": q, "hops": 2 + i % 2, "answerable": True, "em": ok, "f1": ok,
                                    "refused": False, "support_pred": [0], "support_gold": [0],
                                    "failure_stage": "success" if ok else "answer_mismatch_answer_in_evidence",
                                    "usage": {"query_tokens": 100, "index_tokens": 500, "seconds": 2.0}}) + "\n")
    out = tmp_path / "gen"
    build(res, None, out)
    main = (out / "tab_main.tex").read_text()
    assert "75.0" in main and "50.0" in main
    assert "triples only" in (out / "tab_ablation.tex").read_text()
    macros = (out / "results_macros.tex").read_text()
    assert "\\newcommand{\\ResCCEM}{75.0}" in macros and "\\newcommand{\\ResNTest}{20}" in macros


def test_report_without_results_writes_placeholders(tmp_path):
    from evaluation.report import build
    build(tmp_path / "missing", None, tmp_path / "gen")
    assert "no results yet" in (tmp_path / "gen" / "tab_main.tex").read_text()
    macros = (tmp_path / "gen" / "results_macros.tex").read_text()
    for name in ("ResCCEM", "ResNTest", "ResCCFinalizeRate", "ResCCEvidenceChars", "ResSeed", "ResNDev"):
        assert "\\newcommand{\\" + name + "}{--}" in macros, name
