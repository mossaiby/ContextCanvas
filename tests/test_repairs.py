"""Tests for structural semantic repairs applied before events reach the graph."""
from unittest.mock import MagicMock

import pytest

from engine.pipeline import ContextCanvasEngine
from semantics.propbank import PropBankCatalog, PropBankRoleset
from semantics.schema import ExtractionPayload, SpatialContext


@pytest.fixture
def engine(tmp_path):
    eng = ContextCanvasEngine(db_path=str(tmp_path / "repair_db"))
    yield eng
    eng.close()


def _ingest(engine, title, events, idx=0):
    payload = ExtractionPayload.model_validate({"events": events})
    engine.realizer.extract_structured_context = MagicMock(return_value=payload)
    return engine.ingest_text(f"{title}. Body.", source=f"doc:{idx}", title=title)


def _bindings(engine):
    out = set()
    for ev in engine.graph.event_ids():
        sense = engine.graph.mirror.nodes[ev]["sense_id"]
        roles = {r: engine.graph.mirror.nodes[e]["name"] for r, e in engine.graph._event_roles(ev)}
        out.add((sense, tuple(sorted(roles.items()))))
    return out


def test_kinship_noun_phrases_become_directed_relations(engine):
    _ingest(engine, "Nannina", [
        {"temp_id": "ev1", "lemma": "be", "sense_id": "be.01",
         "roles": {":ARG0": "Nannina de' Medici", ":ARG1": "elder sister of Lorenzo de' Medici"}},
        {"temp_id": "ev2", "lemma": "be", "sense_id": "be.01",
         "roles": {":ARG0": "Fletcher Webster", ":ARG1": "son", ":ARG2": "Daniel Webster"}},
    ])
    b = _bindings(engine)
    assert ("sibling.01", ((":ARG0", "Nannina de' Medici"), (":ARG1", "Lorenzo de' Medici"))) in b
    assert ("parent.01", ((":ARG0", "Daniel Webster"), (":ARG1", "Fletcher Webster"))) in b


def test_child_lemma_direction_is_preserved(engine):
    _ingest(engine, "Milena", [{"temp_id": "ev1", "lemma": "be_child_of", "sense_id": "be_child_of.01",
                                "roles": {":ARG0": "Milena Reljin", ":ARG1": "Mita Reljin", ":ARG2": "Vukosava Reljin"}}])
    b = _bindings(engine)
    assert ("parent.01", ((":ARG0", "Mita Reljin"), (":ARG1", "Milena Reljin"))) in b
    assert ("parent.01", ((":ARG0", "Vukosava Reljin"), (":ARG1", "Milena Reljin"))) in b


def test_commas_inside_names_and_numbers_are_not_split(engine):
    _ingest(engine, "Philippe", [
        {"temp_id": "ev1", "lemma": "receive", "sense_id": "receive.01",
         "roles": {":ARG0": "Philippe, Duke of Orléans", ":ARG1": "dukedoms of Valois and Chartres"}},
        {"temp_id": "ev2", "lemma": "have", "sense_id": "have.01",
         "roles": {":ARG0": "Hughesville", ":ARG1": "2,197 residents"}},
    ])
    names = set(engine.graph.entity_names())
    assert "Philippe, Duke of Orléans" in names
    assert "2,197 residents" in names


def test_qualified_places_build_containment_chain(engine):
    _ingest(engine, "Tim DuBois", [{"temp_id": "ev1", "lemma": "born", "sense_id": "born.01",
                                    "roles": {":ARG0": "Tim DuBois", ":ARG2": "May 4, 1948"},
                                    "spatial_context": {"location_name": "Southwest City, Missouri"}}])
    _ingest(engine, "Southwest City", [{"temp_id": "ev1", "lemma": "locate", "sense_id": "locate.01",
                                        "roles": {":ARG1": "Southwest City", ":ARG2": "McDonald County, Missouri"}}], idx=1)
    b = _bindings(engine)
    assert ("bear.02", ((":ARG1", "Tim DuBois"), (":location", "Southwest City"))) in b
    assert ("locate.01", ((":ARG1", "Southwest City"), (":ARG2", "McDonald County"))) in b
    assert ("locate.01", ((":ARG1", "McDonald County"), (":ARG2", "Missouri"))) in b
    evidence = engine.graph.build_evidence(["Tim DuBois"])
    assert "McDonald County" in evidence["candidates"]


def test_dates_are_removed_from_every_core_role(engine):
    _ingest(engine, "Courthouse", [{"temp_id": "ev1", "lemma": "build", "sense_id": "build.01",
                                    "roles": {":ARG0": "1914", ":ARG1": "Mountrail County Courthouse"}}])
    assert "1914" not in set(engine.graph.entity_names())


def test_based_on_is_not_turned_into_location(engine):
    _ingest(engine, "Holes", [{"temp_id": "ev1", "lemma": "base", "sense_id": "base.01",
                               "roles": {":ARG0": "Holes (film)", ":ARG1": "Holes (novel)"}}])
    senses = {engine.graph.mirror.nodes[e]["sense_id"] for e in engine.graph.event_ids()}
    assert "locate.01" not in senses


def test_provenance_is_recorded_per_paragraph(engine):
    _ingest(engine, "CERN", [{"temp_id": "ev1", "lemma": "locate", "sense_id": "locate.01",
                              "roles": {":ARG1": "CERN", ":ARG2": "Geneva"}}], idx=7)
    bundle = engine.graph.build_evidence(["CERN"])
    assert bundle["sources"] == ["doc:7"]


def test_spatial_context_splits_qualified_location():
    sc = SpatialContext(location_name="Canyon, Texas")
    assert sc.location_name == "Canyon" and sc.parent_region == "Texas"
    sc2 = SpatialContext(location_name="Munich", parent_region="Bavaria")
    assert sc2.location_name == "Munich" and sc2.parent_region == "Bavaria"


def test_role_validation_remaps_instead_of_dropping(tmp_path):
    catalog = PropBankCatalog(cache_file=str(tmp_path / "c.json"), frames_dir=str(tmp_path / "none"))
    catalog.framesets["consist.01"] = PropBankRoleset(roleset_id="consist.01", name="consist", description="",
                                                      roles={":ARG1": "whole", ":ARG2": "parts"})
    catalog.framesets["own.01"] = PropBankRoleset(roleset_id="own.01", name="own", description="",
                                                  roles={":ARG0": "owner", ":ARG1": "possession"})
    assert catalog.validate_roles("consist.01", {":ARG0": "Band", ":ARG1": "Member"}) == {":ARG1": "Band", ":ARG2": "Member"}
    # Never guess the agent: an undeclared argument is not promoted into an empty :ARG0.
    assert catalog.validate_roles("own.01", {":ARG1": "Station", ":ARG2": "Network"}) == {":ARG1": "Station", ":ARG2": "Network"}
    assert catalog.validate_roles("own.01", {":ARG0": "A", ":ARG1": "B", ":ARG5": "C"}) == {":ARG0": "A", ":ARG1": "B"}
    # Undeclared arguments fill free non-agent slots.
    assert catalog.validate_roles("consist.01", {":ARG1": "Band", ":ARG3": "Member"}) == {":ARG1": "Band", ":ARG2": "Member"}


def test_canonical_frames_override_propbank_definitions(tmp_path):
    catalog = PropBankCatalog(cache_file=str(tmp_path / "c.json"), frames_dir=str(tmp_path / "none"))
    # Even if a PropBank release defines locate.01 without :ARG2, the system frame wins.
    assert set(catalog.framesets["locate.01"].roles) == {":ARG1", ":ARG2"}
    assert catalog.validate_roles("locate.01", {":ARG1": "Southwest City", ":ARG2": "Missouri"}) == {
        ":ARG1": "Southwest City", ":ARG2": "Missouri"}


def test_placeholders_do_not_become_hub_entities(engine):
    _ingest(engine, "Green", [
        {"temp_id": "ev1", "lemma": "make", "sense_id": "make.01", "roles": {":ARG0": "unknown", ":ARG1": "amulets"}},
        {"temp_id": "ev2", "lemma": "call", "sense_id": "call.01", "roles": {":ARG0": "unknown", ":ARG1": "sea"}},
    ])
    assert "unknown" not in {n.lower() for n in engine.graph.entity_names()}


def test_aliases_and_possessives_connect_entities(engine):
    payload = ExtractionPayload.model_validate({"events": [
        {"temp_id": "ev1", "lemma": "monitor", "sense_id": "monitor.01",
         "roles": {":ARG0": "ESA's Lander Control Center", ":ARG1": "Philae"},
         "spatial_context": {"location_name": "Cologne"}}]})
    engine.realizer.extract_structured_context = MagicMock(return_value=payload)
    engine.ingest_text("Philae. It was operated from ESA's Lander Control Center.", source="doc:0", title="Philae")
    payload2 = ExtractionPayload.model_validate({"events": [
        {"temp_id": "ev1", "lemma": "employ", "sense_id": "employ.01",
         "roles": {":ARG0": "European Space Agency", ":ARG1": "Ulrich Walter"}}]})
    engine.realizer.extract_structured_context = MagicMock(return_value=payload2)
    engine.ingest_text("Ulrich Walter. He trained at the European Space Agency (ESA).", source="doc:1", title="Ulrich Walter")
    bundle = engine.graph.build_evidence(["Ulrich Walter"], max_hops=3)
    assert "Cologne" in bundle["text"]
    assert "Cologne" in bundle["candidates"]


def test_candidates_cover_deeper_hops_and_relevant_events_first(engine):
    events = [{"temp_id": f"n{i}", "lemma": "locate", "sense_id": "locate.01",
               "roles": {":ARG1": f"Venue {i}", ":ARG2": "Sports City"}} for i in range(8)]
    events += [
        {"temp_id": "a", "lemma": "locate", "sense_id": "locate.01", "roles": {":ARG1": "Sports City", ":ARG2": "Town"}},
        {"temp_id": "b", "lemma": "locate", "sense_id": "locate.01", "roles": {":ARG1": "Town", ":ARG2": "Province"}},
        {"temp_id": "c", "lemma": "praise", "sense_id": "praise.01", "roles": {":ARG0": "Critic", ":ARG1": "Town"}},
    ]
    _ingest(engine, "Sports City", events)
    bundle = engine.graph.build_evidence(["Sports City"], max_candidates=6,
                                         question="In which province is Sports City located?")
    assert "Province" in bundle["candidates"]
    text = bundle["text"]
    assert text.index("Town; ARG2[location]=Province") < text.index("praise")


def test_answer_is_snapped_to_evidence_spelling(engine):
    from semantics.schema import ExtractedEvent
    engine.ingest_event_direct(ExtractedEvent(temp_id="e1", lemma="own", sense_id="own.01",
                                              roles={":ARG0": "Bombardier Inc.", ":ARG1": "Bombardier Aerospace"}))
    engine.realizer.answer_question = MagicMock(return_value="ANSWER: Bombardier Inc")
    assert engine.ask("Who owns Bombardier Aerospace?") == "ANSWER: Bombardier Inc."


def test_possessive_titles_are_not_ownership(engine):
    _ingest(engine, "Grant's First Stand", [
        {"temp_id": "ev1", "lemma": "record", "sense_id": "record.01",
         "roles": {":ARG0": "Grant Green", ":ARG1": "Grant's First Stand"}}])
    payload = ExtractionPayload.model_validate({"events": [
        {"temp_id": "ev1", "lemma": "own", "sense_id": "own.01",
         "roles": {":ARG0": "Green Party of the United States", ":ARG1": "National Women's Caucus"}}]})
    engine.realizer.extract_structured_context = MagicMock(return_value=payload)
    engine.ingest_text("Green Party. It has a National Women's Caucus.", source="doc:1", title="Green Party")
    senses = {engine.graph.mirror.nodes[e]["sense_id"] for e in engine.graph.event_ids()}
    assert "affiliate.01" not in senses
    assert "Grant" not in engine.graph.entity_names()
    assert "National Women" not in engine.graph.entity_names()


def test_possessive_of_standalone_entity_is_linked(engine):
    payload = ExtractionPayload.model_validate({"events": [
        {"temp_id": "ev1", "lemma": "visit", "sense_id": "visit.01",
         "roles": {":ARG0": "Ada Byron", ":ARG1": "Acme Corp's Research Lab"}}]})
    engine.realizer.extract_structured_context = MagicMock(return_value=payload)
    engine.ingest_text("Acme Corp. Founded in 1990, Acme Corp runs Acme Corp's Research Lab.", source="doc:0", title="Acme Corp")
    b = _bindings(engine)
    assert ("affiliate.01", ((":ARG0", "Acme Corp"), (":ARG1", "Acme Corp's Research Lab"))) in b


def test_plural_kin_phrases_and_self_relations_are_not_kinship(engine):
    _ingest(engine, "Frances Tupper", [
        {"temp_id": "ev1", "lemma": "bear", "sense_id": "bear.02",
         "roles": {":ARG0": "Frances Tupper", ":ARG1": "children of Frances Tupper and Charles Tupper"}},
        {"temp_id": "ev2", "lemma": "parent", "sense_id": "parent.01",
         "roles": {":ARG0": "Frances Tupper", ":ARG1": "Frances Tupper"}},
    ])
    b = _bindings(engine)
    assert not any(sense == "parent.01" for sense, _ in b)


def test_hub_anchor_is_dropped_when_a_specific_anchor_exists(engine):
    events = [{"temp_id": f"v{i}", "lemma": "locate", "sense_id": "locate.01",
               "roles": {":ARG1": f"Village {i}", ":ARG2": "Serbia"}} for i in range(30)]
    events.append({"temp_id": "t", "lemma": "overthrow", "sense_id": "overthrow.01",
                   "roles": {":ARG0": "Stefan Nemanja", ":ARG1": "Tihomir"}})
    _ingest(engine, "Serbia", events)
    engine.realizer.answer_question = MagicMock(return_value="ANSWER: x")
    engine.ask("Who is the child of the person who followed Tihomir of Serbia?")
    assert engine.last_query_status["matched_anchors"] == ["Tihomir", "Serbia"]
    assert engine.last_query_status["anchors"] == ["Tihomir"]
    assert "Village" not in engine.last_query_status["evidence_text"]