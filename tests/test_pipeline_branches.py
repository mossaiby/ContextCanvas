"""Repair rules, ingestion and question answering in engine/pipeline.py, branch by branch."""
from unittest.mock import MagicMock

import pytest

import engine.pipeline as pipeline_mod
from engine.pipeline import ContextCanvasEngine
from semantics.propbank import PropBankRoleset
from semantics.schema import ExtractedEvent, ExtractionPayload


@pytest.fixture
def engine(tmp_path):
    eng = ContextCanvasEngine(db_path=str(tmp_path / "db"))
    yield eng
    eng.close()


def _payload(events, discourse=()):
    return ExtractionPayload.model_validate({"events": events, "discourse": list(discourse)})


def _ingest(engine, text, events, title=None, discourse=(), debug=False):
    engine.realizer.extract_structured_context = MagicMock(return_value=_payload(events, discourse))
    return engine.ingest_text(text, source="doc:0", title=title, debug=debug)


def _facts(engine):
    out = set()
    for ev in engine.graph.event_ids():
        sense = engine.graph.mirror.nodes[ev]["sense_id"]
        roles = tuple(sorted((r, engine.graph.mirror.nodes[e]["name"]) for r, e in engine.graph._event_roles(ev)))
        out.add((sense, roles))
    return out


def _ev(lemma, roles, sense=None, **extra):
    return {"temp_id": "ev1", "lemma": lemma, "sense_id": sense or f"{lemma}.01", "roles": roles, **extra}


# ------------------------------------------------------------------ kinship

def test_spouse_noun_phrase_becomes_marriage(engine):
    _ingest(engine, "Jane. Body.", [_ev("be", {":ARG0": "Jane Smith", ":ARG1": "wife of John Smith"})], title="Jane")
    assert ("marry.01", ((":ARG0", "Jane Smith"), (":ARG1", "John Smith"))) in _facts(engine)


@pytest.mark.parametrize("roles, expected", [
    ({":ARG0": "Anna", ":ARG1": "Ben"}, {("Anna", "Ben")}),
    ({":ARG0": "Anna", ":ARG1": "Carl", ":ARG2": "Ben"}, {("Anna", "Ben"), ("Carl", "Ben")}),
])
def test_birth_giving_lemmas_become_parent_relations(engine, roles, expected):
    _ingest(engine, "T. Body.", [_ev("give_birth", roles)], title="T")
    parents = {(dict(r)[":ARG0"], dict(r)[":ARG1"]) for s, r in _facts(engine) if s == "parent.01"}
    assert parents == expected


def test_relational_noun_in_a_non_copula_keeps_the_original_event(engine):
    _ingest(engine, "T. Body.", [_ev("describe", {":ARG0": "Critic Joe", ":ARG1": "son of Peter Heiberg"})], title="T")
    senses = {s for s, _ in _facts(engine)}
    assert "parent.01" in senses and "describe.01" in senses


def test_events_without_core_roles_are_not_kinship(engine):
    assert engine._expand_kinship(ExtractedEvent(lemma="be", roles={":location": "Oslo"})) is None
    assert engine._subject_role(ExtractedEvent(lemma="be", roles={})) is None


# ------------------------------------------------------------------ copula, birth, locate

def test_subject_predicate_copula_is_normalized(engine):
    _ingest(engine, "T. Body.", [_ev("be", {":ARG0": "Paris", ":ARG1": "Capital City"})], title="T")
    assert ("be.01", ((":ARG1", "Paris"), (":ARG2", "Capital City"))) in _facts(engine)


def test_bear_01_is_not_a_birth(engine):
    ev = ExtractedEvent(lemma="bear", sense_id="bear.01", roles={":ARG0": "Atlas", ":ARG1": "the World"})
    assert engine._normalize_birth(ev) == [] and ev.sense_id == "bear.01"


def test_birth_place_fills_an_empty_envelope(engine):
    ev = ExtractedEvent(lemma="born", roles={":ARG0": "Ann Lee", ":ARG1": "Lyon"})
    engine._normalize_birth(ev)
    assert ev.roles == {":ARG1": "Ann Lee"} and ev.spatial_context.location_name == "Lyon"


def test_direct_ingestion_stores_derived_events_too(engine):
    first = engine.ingest_event_direct(ExtractedEvent(temp_id="loc", lemma="locate", sense_id="locate.01",
                                                      roles={":ARG1": "Kimbrough Stadium", ":ARG2": "Canyon, Texas"}))
    assert first == "loc" and len(engine.graph.event_ids()) == 2


def test_birth_place_conflicting_with_envelope_is_kept_as_locative(engine):
    ev = ExtractedEvent(lemma="born", roles={":ARG0": "Ann Lee", ":ARG1": "Lyon"}, spatial_context={"location_name": "Paris"})
    engine._normalize_birth(ev)
    assert ev.roles == {":ARG1": "Ann Lee", ":ARGM-LOC": "Lyon"} and ev.spatial_context.location_name == "Paris"


def test_birth_with_two_parents_creates_parent_events(engine):
    ev = ExtractedEvent(lemma="born", roles={":ARG0": "Ann Lee", ":ARG1": "Tom Lee", ":ARG2": "Sue Lee"})
    derived = engine._normalize_birth(ev)
    assert {(d.roles[":ARG0"], d.roles[":ARG1"]) for d in derived} == {("Tom Lee", "Ann Lee"), ("Sue Lee", "Ann Lee")}


@pytest.mark.parametrize("roles, expected", [
    ({":ARG0": "Learjet"}, {":ARG1": "Learjet"}),
    ({":ARG0": "Learjet", ":ARG1": "Wichita"}, {":ARG1": "Learjet", ":ARG2": "Wichita"}),
    ({":ARG1": "Learjet", ":location": "Wichita"}, {":ARG1": "Learjet", ":ARG2": "Wichita"}),
])
def test_locate_roles_are_normalized(roles, expected):
    ev = ExtractedEvent(lemma="headquarter", roles=dict(roles))
    ContextCanvasEngine._normalize_locate(ev)
    assert ev.roles == expected and ev.sense_id == "locate.01"


def test_place_chains_from_role_and_envelope_are_not_duplicated(engine):
    ev = ExtractedEvent(lemma="locate", sense_id="locate.01",
                        roles={":ARG1": "Kimbrough Stadium", ":ARG2": "Canyon, Texas"},
                        spatial_context={"location_name": "Canyon", "parent_region": "Texas"})
    derived = engine._expand_place_hierarchies(ev)
    assert [(d.roles[":ARG1"], d.roles[":ARG2"]) for d in derived] == [("Canyon", "Texas")]


def test_unknown_sense_is_disambiguated(engine, monkeypatch):
    monkeypatch.setattr(pipeline_mod.GLOBAL_CATALOG, "disambiguate_sense",
                        lambda lemma, ctx: PropBankRoleset(roleset_id="run.02", name="run", description="run", roles={}))
    ev = ExtractedEvent(lemma="run", sense_id="run.99", roles={":ARG0": "Ann"})
    engine._finalize_sense_and_roles(ev, "Ann ran the race")
    assert ev.sense_id == "run.02"


# ------------------------------------------------------------------ titles, anaphora, possessives

def test_title_is_derived_from_the_text_and_resolves_anaphora(engine):
    # "operate" is on no frame list, so the result does not depend on the installed PropBank frames.
    _ingest(engine, "Learjet (company). The company operates plants.", [_ev("operate", {":ARG0": "the company", ":ARG1": "Plants"})])
    assert ("operate.01", ((":ARG0", "Learjet"), (":ARG1", "Plants"))) in _facts(engine)
    assert ContextCanvasEngine._extract_document_title("no title here") is None


def test_possessor_entity_detection(engine):
    engine.ingest_event_direct(ExtractedEvent(lemma="own", roles={":ARG0": "Acme Holdings", ":ARG1": "Widget Co"}))
    assert engine._possessor_is_entity("Acme Holdings", "")                        # already in the graph
    assert not engine._possessor_is_entity("Grant", "Ulysses Grant visited Grant's Tomb.")  # part of a longer name
    assert engine._possessor_is_entity("Grant", "Grant visited Grant's Tomb.")


# ------------------------------------------------------------------ ingestion

def test_long_text_is_chunked_with_title_and_unique_ids(engine):
    engine.max_chunk_chars = 60
    text = "Learjet. " + "It builds business jets in Kansas. " * 4
    ids = _ingest(engine, text, [_ev("build", {":ARG0": "Learjet", ":ARG1": "Jets"})], title="Learjet")
    calls = engine.realizer.extract_structured_context.call_args_list
    assert len(calls) > 1 and all(c.args[0].startswith("Learjet") for c in calls)
    assert len(ids) == 1                                                             # duplicates collapse


def test_insert_failures_are_skipped_and_reported_in_debug(engine, monkeypatch, capsys):
    monkeypatch.setattr(engine.graph, "insert_event", MagicMock(side_effect=RuntimeError("disk full")))
    assert _ingest(engine, "T. The European Space Agency (ESA) hired him.", [_ev("own", {":ARG0": "A Co", ":ARG1": "B Co"})],
                   title="T", debug=True) == []
    assert "[INSERT FAILED]" in capsys.readouterr().out


def test_debug_lists_new_events_and_aliases(engine, capsys):
    _ingest(engine, "T. The European Space Agency (ESA) is in Paris.",
            [_ev("own", {":ARG0": "A Co", ":ARG1": "B Co"}, time_context="1990", spatial_context="Cologne")],
            title="T", debug=True)
    out = capsys.readouterr().out
    assert "[own / own.01]" in out and "time='1990'" in out and "[alias / alias.01]" in out


def test_discourse_links_follow_stored_ids_and_skip_bad_ones(engine, monkeypatch):
    events = [_ev("own", {":ARG0": "A Co", ":ARG1": "B Co"}), {**_ev("sell", {":ARG0": "A Co", ":ARG1": "B Co"}), "temp_id": "ev2"}]
    discourse = [{"source_id": "ev1", "target_id": "ev2", "relation": ":before"},
                 {"source_id": "ev1", "target_id": "nope", "relation": ":cause"},
                 {"source_id": "ev1", "target_id": "ev1", "relation": ":cause"}]
    _ingest(engine, "T. Body.", events, title="T", discourse=discourse)
    links = [(a, b) for a, b, d in engine.graph.mirror.edges(data=True) if d.get("edge_type") == "discourse"]
    assert len(links) == 1
    engine.reset()
    monkeypatch.setattr(engine.graph, "insert_discourse_relation", MagicMock(side_effect=RuntimeError("x")))
    assert len(_ingest(engine, "T. Body.", events, title="T", discourse=discourse[:1])) == 2


def test_alias_insert_failures_are_ignored(engine, monkeypatch):
    real_insert = engine.graph.insert_event

    def insert(ev):
        if ev.sense_id == "alias.01":
            raise RuntimeError("x")
        return real_insert(ev)
    monkeypatch.setattr(engine.graph, "insert_event", insert)
    ids = _ingest(engine, "T. The European Space Agency (ESA) is here.", [_ev("own", {":ARG0": "A Co", ":ARG1": "B Co"})], title="T")
    assert len(ids) == 1


# ------------------------------------------------------------------ anchoring and answering

def test_fallback_anchor_needs_all_words_of_a_span(engine):
    engine.ingest_event_direct(ExtractedEvent(temp_id="b1", lemma="bear", sense_id="bear.02", roles={":ARG1": "Frances Amelia Tupper"}))
    engine.ingest_event_direct(ExtractedEvent(temp_id="b2", lemma="bear", sense_id="bear.02", roles={":ARG1": "James Tupper"}))
    assert engine._find_candidate_entities("Where was Tupper Frances born?") == ["Frances Amelia Tupper"]


def test_snapping_leaves_refusals_alone():
    assert ContextCanvasEngine._snap_to_evidence("STATUS: NOT_IN_EVIDENCE", ["X"]) == "STATUS: NOT_IN_EVIDENCE"


def test_ask_debug_output_and_hub_pruning(engine, capsys):
    engine.graph.hub_degree = 1
    for i in range(3):
        engine.ingest_event_direct(ExtractedEvent(temp_id=f"s{i}", lemma="locate", sense_id="locate.01",
                                                  roles={":ARG1": f"Town {i}", ":ARG2": "Serbia"}))
    engine.ingest_event_direct(ExtractedEvent(temp_id="t", lemma="rule", sense_id="rule.01",
                                              roles={":ARG0": "Tihomir", ":ARG1": "Town 0"}))
    engine.realizer.answer_question = MagicMock(return_value="ANSWER: Town 0")
    engine.realizer.last_answer_text = "FINAL ANSWER: Town 0"
    assert engine.ask("Which town did Tihomir of Serbia rule?", debug=True) == "ANSWER: Town 0"
    out = capsys.readouterr().out
    assert "Anchors after hub pruning: ['Tihomir']" in out and "[DIAG] LLM Output" in out
    assert set(engine.last_query_status["matched_anchors"]) == {"Tihomir", "Serbia"}
    assert engine.last_query_status["anchors"] == ["Tihomir"]


def test_ask_with_an_anchor_but_no_evidence(engine, capsys):
    engine.graph._add_entity_node("ent_lonely", "Lonely Entity")
    assert engine.ask("Who?", target_entity="Lonely Entity", debug=True) == "STATUS: NOT_IN_EVIDENCE"
    assert engine.last_query_status["stage"] == "graph_empty_blueprint"
    assert "Evidence: EMPTY" in capsys.readouterr().out


def test_questions_naming_nothing_in_the_graph_are_refused(engine):
    engine.ingest_event_direct(ExtractedEvent(temp_id="b1", lemma="bear", sense_id="bear.02", roles={":ARG1": "Ann Lee"}))
    assert engine._find_candidate_entities("what is it?") == []
    assert engine.ask("what is it?") == "STATUS: NOT_IN_EVIDENCE"
    assert engine.last_query_status == {"stage": "graph_no_anchors", "anchors": []}