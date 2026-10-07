"""Regressions from the first debug trace (MuSiQue development questions), one test per defect."""
from unittest.mock import MagicMock

import pytest

from engine.pipeline import ContextCanvasEngine
from semantics.schema import ExtractedEvent, ExtractionPayload
from semantics.text_utils import is_placeholder


@pytest.fixture
def engine(tmp_path):
    eng = ContextCanvasEngine(db_path=str(tmp_path / "db"))
    yield eng
    eng.close()


def _stored(engine, text, events, title):
    engine.realizer.extract_structured_context = MagicMock(
        return_value=ExtractionPayload.model_validate({"events": events}))
    engine.ingest_text(text, source="doc:0", title=title)
    out = []
    for ev in engine.graph.event_ids():
        roles = {r: engine.graph.mirror.nodes[e]["name"] for r, e in engine.graph._event_roles(ev)}
        out.append((engine.graph.mirror.nodes[ev]["sense_id"], roles))
    return out


@pytest.mark.parametrize("lemma, sense, expected", [
    ("bear.02", None, ("bear", "bear.02")),             # "Countess Charlotte Brabantina of Nassau"
    ("member.01", "member.01.01", ("member", "member.01")),  # "Juan Carlos Espinoza Mercado"
    ("marry.01", "marry_01.01", ("marry", "marry.01")),
    ("co-found", "found.01", ("co-found", "found.01")),  # normal events are untouched
])
def test_sense_written_into_the_lemma_field(lemma, sense, expected):
    data = {"lemma": lemma, "roles": {}}
    if sense:
        data["sense_id"] = sense
    ev = ExtractedEvent.model_validate(data)
    assert (ev.lemma, ev.sense_id) == expected


def test_birth_events_with_a_sense_in_the_lemma_get_the_birth_frame(engine):
    facts = _stored(engine, "Countess Charlotte Brabantina of Nassau. Body.", [{
        "lemma": "bear.02", "roles": {":ARG0": "William the Silent", ":ARG1": "Charlotte Brabantina"},
        "time_context": "17 September 1580"}], title="Countess Charlotte Brabantina of Nassau")
    # Previously stored under the unknown frame "bear_02.01", without role meanings.
    assert ("bear.02", {":ARG0": "William the Silent", ":ARG1": "Charlotte Brabantina"}) in facts


def test_nobody_is_their_own_mother(engine):
    facts = _stored(engine, "Miquette Giraudy. Body.", [{
        "lemma": "bear", "sense_id": "bear.02", "roles": {":ARG0": "Miquette Giraudy", ":ARG1": "Miquette Giraudy"},
        "time_context": "9 February 1953", "spatial_context": "Nice"}], title="Miquette Giraudy")
    births = [roles for sense, roles in facts if sense == "bear.02"]
    assert births == [{":ARG1": "Miquette Giraudy", ":location": "Nice"}]   # born in Nice, not her own mother


def test_no_organisation_founds_itself(engine):
    facts = _stored(engine, "DeSoto Records. Body.", [{
        "lemma": "found", "sense_id": "found.01", "roles": {":ARG0": "DeSoto Records", ":ARG1": "DeSoto Records"},
        "time_context": "1989"}], title="DeSoto Records")
    assert [roles for sense, roles in facts if sense == "found.01"] == [{":ARG1": "DeSoto Records"}]


def test_self_relations_beyond_the_agent_drop_the_later_role():
    ev = ExtractedEvent(lemma="locate", sense_id="locate.01", roles={":ARG1": "Tumaraa", ":ARG2": "Tumaraa"})
    from engine.pipeline import _drop_self_relations
    assert _drop_self_relations([ev])[0].roles == {":ARG1": "Tumaraa"}


@pytest.mark.parametrize("value, placeholder", [
    ("network executives (implied)", True),      # "Star Trek: Discovery"
    ("network producers (Implied)", True),
    ("Orion Pictures", False),
    ("Native Son (play)", False),                # a real disambiguating parenthesis
])
def test_participants_the_extractor_invented_are_placeholders(value, placeholder):
    assert is_placeholder(value) is placeholder
