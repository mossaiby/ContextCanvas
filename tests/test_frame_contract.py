"""The frame contract: system frames only for the verbs that express them (semantics/frames.py)."""
from unittest.mock import MagicMock

import pytest

from engine.pipeline import ContextCanvasEngine
from semantics.frames import CANONICAL_FRAMES, FRAME_VERBS, INTERNAL_FRAMES, frame_accepts, lemma_key
from semantics.schema import ExtractionPayload


def test_every_system_frame_lists_its_verbs():
    assert set(FRAME_VERBS) == set(CANONICAL_FRAMES) | set(INTERNAL_FRAMES)
    for sense, verbs in FRAME_VERBS.items():
        assert sense.split(".")[0] in verbs or sense == "parent.01" and "parent" in verbs


def test_lemma_keys():
    assert lemma_key(" Be_Located ") == "be located" and lemma_key("co-found") == "co-found"


@pytest.mark.parametrize("sense, lemma, accepted", [
    ("locate.01", "be located", True), ("locate.01", "be_located", True), ("found.01", "co-found", True),
    ("bear.02", "be born", True), ("distribute.01", "release", True),
    ("locate.01", "hold", False), ("locate.01", "be known as", False), ("locate.01", "be", False),
    ("member.01", "be home to", False), ("member.01", "become", False), ("employ.01", "be chairman", False),
    ("hold.01", "hold", True),                      # ordinary PropBank senses are not restricted
])
def test_frame_acceptance(sense, lemma, accepted):
    assert frame_accepts(sense, lemma) is accepted


@pytest.fixture
def engine(tmp_path):
    eng = ContextCanvasEngine(db_path=str(tmp_path / "db"))
    yield eng
    eng.close()


def _senses(engine, events, ablations=None):
    if ablations:
        engine.ablations.update(ablations)
    engine.realizer.extract_structured_context = MagicMock(
        return_value=ExtractionPayload.model_validate({"events": events}))
    engine.ingest_text("T. Body.", source="doc:0", title="T")
    return {engine.graph.mirror.nodes[e]["lemma"]: engine.graph.mirror.nodes[e]["sense_id"] for e in engine.graph.event_ids()}


TRACE_EVENTS = [   # from the Cyprus Popular Bank and Ciudad Deportiva paragraphs of the debug trace
    {"temp_id": "a", "lemma": "be behind", "sense_id": "locate.01", "roles": {":ARG1": "Cyprus Popular Bank", ":ARG2": "Bank of Cyprus"}},
    {"temp_id": "b", "lemma": "be home to", "sense_id": "member.01", "roles": {":ARG0": "Ciudad Deportiva", ":ARG1": "Toros de Nuevo Laredo"}},
    {"temp_id": "c", "lemma": "be located", "sense_id": "locate.01", "roles": {":ARG1": "Ciudad Deportiva", ":ARG2": "Nuevo Laredo"}},
]


def test_false_containment_and_membership_lose_the_frame_but_keep_the_fact(engine):
    senses = _senses(engine, TRACE_EVENTS)
    assert senses["be behind"] == "be_behind.01" and senses["be home to"] == "be_home_to.01"
    # A genuine location keeps its frame; "be located" is now also recognised by the location
    # repair, which used to miss it ("be_located" in its list vs. "be located" from the extractor).
    assert senses["locate"] == "locate.01" and "be located" not in senses
    assert len(engine.graph.event_ids()) == 3                         # no fact is lost


def test_the_check_can_be_switched_off_for_the_ablation(engine):
    senses = _senses(engine, TRACE_EVENTS, ablations={"frame_guard": False})
    assert senses["be behind"] == "locate.01" and senses["be home to"] == "member.01"


def test_copula_mislabelled_as_location_becomes_a_copula(engine):
    senses = _senses(engine, [{"temp_id": "a", "lemma": "be", "sense_id": "locate.01",
                               "roles": {":ARG1": "Cyprus Popular Bank", ":ARG2": "Second Largest Banking Group"}}])
    assert senses["be"] == "be.01"


def test_kinship_verbs_with_a_system_frame_are_still_repaired(engine):
    _senses(engine, [{"temp_id": "a", "lemma": "son", "sense_id": "parent.01",
                      "roles": {":ARG0": "Johan Heiberg", ":ARG1": "Peter Heiberg"}}])
    facts = {(engine.graph.mirror.nodes[e]["sense_id"],) for e in engine.graph.event_ids()}
    assert ("parent.01",) in facts


def test_prompt_states_the_contract():
    import json, tempfile, pathlib
    from engine.realizer import Realizer
    cfg = pathlib.Path(tempfile.mkdtemp()) / "c.json"
    cfg.write_text(json.dumps({"model_name": "m", "base_url": "http://x/v1", "api_key": "k"}))
    prompt = Realizer(str(cfg))._extraction_prompt()
    assert "only when the sentence states exactly that relation" in prompt
    assert "Name the entity itself, not a phrase around it" in prompt


# ------------------------------------------------------------------ verbs with an invalid sense

def test_listed_verbs_with_an_invented_sense_take_their_frame(engine):
    senses = _senses(engine, [{"temp_id": "a", "lemma": "co-found", "sense_id": "co-found.01",   # Kirtland Records
                               "roles": {":ARG0": "John Kirtland", ":ARG1": "Kirtland Records"}}])
    assert senses["co-found"] == "found.01"


def test_valid_senses_and_unlisted_verbs_are_left_alone(engine, monkeypatch):
    from semantics.propbank import GLOBAL_CATALOG, PropBankRoleset
    monkeypatch.setitem(GLOBAL_CATALOG.framesets, "establish.01",
                        PropBankRoleset(roleset_id="establish.01", name="establish", description="", roles={}))
    senses = _senses(engine, [
        {"temp_id": "a", "lemma": "establish", "sense_id": "establish.01", "roles": {":ARG0": "A Co", ":ARG1": "B Co"}},
        {"temp_id": "b", "lemma": "be behind", "sense_id": "be_behind.01", "roles": {":ARG1": "C Co", ":ARG2": "D Co"}},
    ])
    assert senses["establish"] == "establish.01" and senses["be behind"] == "be_behind.01"


def test_verb_to_frame_mapping_follows_the_ablation_switch(engine):
    senses = _senses(engine, [{"temp_id": "a", "lemma": "co-found", "sense_id": "co-found.01",
                               "roles": {":ARG0": "John Kirtland", ":ARG1": "Kirtland Records"}}],
                     ablations={"frame_guard": False})
    assert senses["co-found"] == "co-found.01"


def test_internal_frames_are_not_reachable_from_verbs():
    from semantics.frames import VERB_FRAMES
    assert "alias" not in VERB_FRAMES and "affiliate" not in VERB_FRAMES
    assert VERB_FRAMES["co-found"] == "found.01" and VERB_FRAMES["be located"] == "locate.01"