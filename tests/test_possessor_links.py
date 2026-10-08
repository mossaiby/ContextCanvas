"""Owners named inside possessive arguments ("Orion Pictures' support") are linked to the event."""
from unittest.mock import MagicMock

import pytest

from engine.pipeline import ContextCanvasEngine
from semantics.schema import ExtractedEvent, ExtractionPayload


@pytest.fixture
def engine(tmp_path):
    eng = ContextCanvasEngine(db_path=str(tmp_path / "db"))
    yield eng
    eng.close()


@pytest.mark.parametrize("value, owner", [
    ("Orion Pictures' support", "Orion Pictures"),                 # UHF (film), debug trace
    ("Amblin Entertainment's headquarters", "Amblin Entertainment"),
    ("Algeria's status as a colony", "Algeria"),
    ("Louis Philippe\u2019s constitutional monarchy", "Louis Philippe"),
    ("Grant's First Stand", None),                                 # a title: continues in capitals
    ("Quebec's Abitibi-Témiscamingue region", None),               # capitalised head: a name
    ("the company's support", None),                               # no named owner
    ("Orion Pictures", None),
])
def test_possessive_arguments(value, owner):
    ev = ExtractedEvent(lemma="get", sense_id="get.01", roles={":ARG0": "Someone Else", ":ARG1": value})
    ContextCanvasEngine._link_possessor(ev)
    assert ev.roles.get(":ARGM-POSS") == owner
    assert ev.roles[":ARG1"] == value                              # the fact itself is unchanged


def test_owner_already_in_the_event_is_not_repeated():
    ev = ExtractedEvent(lemma="praise", sense_id="praise.01",
                        roles={":ARG0": "Orion Pictures", ":ARG1": "Orion Pictures' support"})
    ContextCanvasEngine._link_possessor(ev)
    assert ":ARGM-POSS" not in ev.roles


def test_only_the_first_possessive_is_linked():
    ev = ExtractedEvent(lemma="compare", sense_id="compare.01",
                        roles={":ARG0": "Acme Corp's budget", ":ARG1": "Zenith Ltd's budget"})
    ContextCanvasEngine._link_possessor(ev)
    assert ev.roles[":ARGM-POSS"] == "Acme Corp"
    ContextCanvasEngine._link_possessor(ev)                        # idempotent
    assert ev.roles[":ARGM-POSS"] == "Acme Corp"


def _ingest(engine, events, ablations=None):
    if ablations:
        engine.ablations.update(ablations)
    engine.realizer.extract_structured_context = MagicMock(
        return_value=ExtractionPayload.model_validate({"events": events}))
    engine.ingest_text("T. Body.", source="doc:0", title="T")


UHF = [   # the three facts question 2 of the debug trace needs
    {"temp_id": "a", "lemma": "write", "sense_id": "write.01", "roles": {":ARG0": "Yankovic and Levey", ":ARG1": "UHF"}},
    {"temp_id": "b", "lemma": "get", "sense_id": "get.01", "roles": {":ARG0": "Yankovic and Levey", ":ARG1": "Orion Pictures' support"}},
    {"temp_id": "c", "lemma": "found", "sense_id": "found.01", "roles": {":ARG0": "Mike Medavoy", ":ARG1": "Orion Pictures"}},
]


def test_the_owner_connects_the_question_entity_to_the_answer(engine):
    _ingest(engine, UHF)
    evidence = engine.graph.build_evidence(["UHF"])
    assert "Mike Medavoy" in evidence["entity_names"]
    assert "ARGM-POSS[possessor]=Orion Pictures" in evidence["text"]


def test_possessor_links_follow_the_ablation_switch(engine):
    _ingest(engine, UHF, ablations={"possessive_links": False})
    assert "Mike Medavoy" not in engine.graph.build_evidence(["UHF"])["entity_names"]
