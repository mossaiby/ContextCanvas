"""LLM access layer (engine/realizer.py): JSON recovery, message parsing, extraction, answering."""
import json
from types import SimpleNamespace as NS

import pytest
from pydantic import ValidationError

import engine.realizer as realizer_mod
from engine.realizer import (
    Realizer, _clean_and_parse_json, _extract_json_content, _extract_message_text, _salvage_truncated_json,
)


def _msg(content="", **extra):
    fields = {k: v for k, v in extra.items() if k != "model_extra"}
    return NS(content=content, model_extra=extra.get("model_extra"), **fields)


def _resp(message):
    return NS(choices=[NS(message=message)], usage=NS(prompt_tokens=3, completion_tokens=2))


class ScriptedClient:
    """Returns (or raises) the scripted items in order; records every call."""
    def __init__(self, *script):
        self.script, self.calls = list(script), []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return _resp(item if not isinstance(item, str) else _msg(item))


@pytest.fixture
def make_realizer(tmp_path):
    def make(*script, **cfg):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"model_name": "m", "base_url": "http://x/v1", "api_key": "k", **cfg}))
        r = Realizer(config_path=str(path))
        r.client = ScriptedClient(*script)
        return r
    return make


# ------------------------------------------------------------------ JSON recovery

def test_json_cleanup_variants():
    assert _clean_and_parse_json("") is None
    assert _clean_and_parse_json('```json\n{"events": []}\n```') == {"events": []}
    assert _clean_and_parse_json('[{"lemma": "own"}]') == {"events": [{"lemma": "own"}]}
    # Regression: a bare array with several events used to be trimmed into invalid JSON and lost.
    assert _clean_and_parse_json('[{"lemma": "own"}, {"lemma": "found"}]') == {"events": [{"lemma": "own"}, {"lemma": "found"}]}
    assert _clean_and_parse_json('Here you go: {"events": []} Hope it helps') == {"events": []}
    assert _clean_and_parse_json('"just a string"') is None


def test_salvage_truncated_json():
    truncated = '{"events": [{"lemma": "own", "roles": {}}, {"lemma": "fou'
    assert _clean_and_parse_json(truncated) == {"events": [{"lemma": "own", "roles": {}}]}
    assert _salvage_truncated_json("no events key") is None
    assert _salvage_truncated_json('"events": [ no braces at all') is None
    assert _salvage_truncated_json('"events": [{"a": 1}') is None          # brace after the key
    assert _salvage_truncated_json('{"events": [{"a": } broken }') is None  # nothing recoverable


def test_message_fields_from_reasoning_and_model_extra():
    assert _extract_json_content(_msg("", reasoning='{"events": []}')) == '{"events": []}'
    assert _extract_json_content(_msg("", model_extra={"reasoning_content": "{}"})) == "{}"
    assert _extract_json_content(_msg("")) == ""
    # Reasoning comes first so the content's final answer line is the last one parsed.
    assert _extract_message_text(_msg("short", thinking="long reasoning")) == "long reasoning\nshort"
    assert _extract_message_text(_msg("FINAL ANSWER: X", thinking="r")) == "FINAL ANSWER: X"


# ------------------------------------------------------------------ configuration

def test_disable_thinking_sets_request_flags(make_realizer):
    r = make_realizer(disable_thinking=True)
    assert r.extra_body["think"] is False and r.extra_body["reasoning_effort"] == "none"


# ------------------------------------------------------------------ extraction

GOOD = '{"events": [{"lemma": "own", "sense_id": "own.01", "roles": {":ARG0": "A", ":ARG1": "B"}}]}'


def test_extraction_retries_after_an_error_and_ignores_a_corrupt_cache(make_realizer, tmp_path):
    r = make_realizer(RuntimeError("timeout"), GOOD, extraction_cache_dir=str(tmp_path / "cache"))
    path = r._cache_path(r._extraction_prompt(), "A owns B.")
    path.parent.mkdir(parents=True)
    path.write_text("{corrupt")
    assert len(r.extract_structured_context("A owns B.").events) == 1
    assert len(r.client.calls) == 2
    assert json.loads(path.read_text())["model"] == "m"                    # corrupt entry replaced


def test_extraction_gives_up_after_two_errors(make_realizer):
    r = make_realizer(RuntimeError("a"), RuntimeError("b"))
    assert r.extract_structured_context("x").events == []


def test_extraction_reprompts_on_unusable_json(make_realizer):
    r = make_realizer("not json", GOOD)
    assert len(r.extract_structured_context("A owns B.").events) == 1
    assert "Return ONLY valid JSON" in r.client.calls[1]["messages"][-1]["content"]


def test_extraction_survives_schema_validation_errors(make_realizer, monkeypatch):
    class Rejecting:
        @staticmethod
        def model_validate(data):
            raise ValidationError.from_exception_data("ExtractionPayload", [])
    r = make_realizer(GOOD, GOOD)
    monkeypatch.setattr(realizer_mod, "ExtractionPayload", Rejecting)
    assert isinstance(r.extract_structured_context("x"), Rejecting)        # the empty fallback
    assert len(r.client.calls) == 2                                         # both attempts were made


# ------------------------------------------------------------------ answering

@pytest.mark.parametrize("raw, question, expected", [
    ("<exact entity name>", "Who?", None),
    ("Stanley, North Dakota", "Where was he born?", "Stanley"),
    ("Stanley, North Dakota", "Which state is it in?", "Stanley, North Dakota"),
    ("Paris, france", "Where?", "Paris, france"),
])
def test_answer_cleaning(raw, question, expected):
    assert Realizer._clean_answer(raw, question) == expected


@pytest.mark.parametrize("kind, marker", [("graph", "Context Graph Facts"), ("text", "Passages:"), ("none", "Question:")])
def test_every_evidence_kind_uses_its_own_prompt(make_realizer, kind, marker):
    r = make_realizer("FINAL ANSWER: Paris")
    assert r.answer_question("Where?", "evidence", candidate_entities=["Paris"], evidence_kind=kind) == "ANSWER: Paris"
    messages = r.client.calls[0]["messages"]
    assert marker in messages[1]["content"]
    if kind == "none":
        assert "own knowledge" in messages[0]["content"] and "evidence" not in messages[1]["content"]
    if kind == "graph":
        assert "ENTITIES REACHABLE" in messages[1]["content"]


def test_strict_policy_states_the_strict_rule(make_realizer):
    r = make_realizer("FINAL ANSWER: NOT_IN_EVIDENCE", answer_policy="strict")
    assert r.answer_question("q", "facts") == "STATUS: NOT_IN_EVIDENCE"
    assert "every relation the question asks about" in r.client.calls[0]["messages"][0]["content"]


def test_answer_errors_and_empty_generations_are_refusals(make_realizer):
    r = make_realizer(RuntimeError("down"))
    assert r.answer_question("q", "facts") == "STATUS: NOT_IN_EVIDENCE"
    assert r.last_answer_text.startswith("<error:")
    assert make_realizer("").answer_question("q", "facts") == "STATUS: NOT_IN_EVIDENCE"


def test_truncated_answer_is_finalized(make_realizer):
    r = make_realizer("reasoning that never ends", "FINAL ANSWER: Lyon")
    assert r.answer_question("q", "facts") == "ANSWER: Lyon"
    assert "[finalize]" in r.last_answer_text and r.usage_summary()["finalize_calls"] == 1


def test_failed_finalize_is_a_refusal(make_realizer):
    r = make_realizer("reasoning that never ends", RuntimeError("down"))
    assert r.answer_question("q", "facts") == "STATUS: NOT_IN_EVIDENCE"
    assert "<finalize error" in r.last_answer_text
