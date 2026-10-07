"""Tests for answer-line parsing and evaluation failure-stage attribution."""
from engine.realizer import Realizer
from evaluation.eval_musique_runner import classify_case


def test_refusal_detection_matches_whole_answer_only():
    clean = Realizer._clean_answer
    assert clean("NOT_IN_EVIDENCE", "q") is None
    assert clean("There is no evidence linking these", "q") is None
    assert clean("Unknown Pleasures", "Which album?") == "Unknown Pleasures"
    assert clean("No Escape", "Which film?") == "No Escape"
    assert clean("The answer is Geneva.", "Where?") == "Geneva"


def test_failure_stage_attribution():
    aliases = ["Bombardier Inc."]
    assert classify_case("Bombardier Inc", aliases, 20, 100, {}) == "success"
    assert classify_case("", aliases, 0, 0, {}) == "retrieval_empty"
    assert classify_case("", aliases, 20, 0, {}) == "extraction_zero_events"
    assert classify_case("", aliases, 20, 50, {"stage": "graph_no_anchors"}) == "graph_no_anchors"
    meta = {"stage": "qa_refusal", "evidence_text": "own.01: ARG0=Bombardier Inc.; ARG1=Learjet"}
    assert classify_case("", aliases, 20, 50, meta) == "qa_refusal_answer_in_evidence"
    meta = {"stage": "qa_success", "evidence_text": "nothing relevant"}
    assert classify_case("Bombardier Aerospace", aliases, 20, 50, meta) == "answer_partial_match"
    assert classify_case("Canada", aliases, 20, 50, meta) == "answer_mismatch_answer_not_in_evidence"


def test_abbreviation_periods_survive_answer_cleaning():
    clean = Realizer._clean_answer
    assert clean("Bombardier Inc.", "Which company?") == "Bombardier Inc."
    assert clean("Geneva.", "Where?") == "Geneva"


def test_answer_prompt_states_one_refusal_criterion(tmp_path):
    import json
    from unittest.mock import MagicMock
    cfg = tmp_path / "config.json"
    for policy in ("best_supported", "strict"):
        cfg.write_text(json.dumps({"model_name": "m", "base_url": "http://x/v1", "api_key": "k", "answer_policy": policy}))
        r = Realizer(config_path=str(cfg))
        choice = MagicMock()
        choice.message.content = "FINAL ANSWER: NOT_IN_EVIDENCE"
        r.client.chat.completions.create = MagicMock(return_value=MagicMock(choices=[choice]))
        r.answer_question("q?", "facts")
        system_prompt = r.client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert system_prompt.count("NOT_IN_EVIDENCE") == 2      # the criterion (rule 4) + the output format (rule 5)
        assert "if the facts do not support an answer" not in system_prompt
        assert r.last_answer_text == "FINAL ANSWER: NOT_IN_EVIDENCE"


def test_truncated_reasoning_is_finalized_not_refused(tmp_path):
    import json
    from unittest.mock import MagicMock
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"model_name": "m", "base_url": "http://x/v1", "api_key": "k"}))
    r = Realizer(config_path=str(cfg))
    cut = MagicMock(); cut.message.content = "Ciudad Deportiva is in Nuevo Laredo, which is in the Municipality, which is in"
    final = MagicMock(); final.message.content = "FINAL ANSWER: Tamaulipas"
    r.client.chat.completions.create = MagicMock(side_effect=[MagicMock(choices=[cut]), MagicMock(choices=[final])])
    assert r.answer_question("Where?", "facts") == "ANSWER: Tamaulipas"
    assert r.client.chat.completions.create.call_count == 2