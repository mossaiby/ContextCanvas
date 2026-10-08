"""Graph-guided retrieval: the graph ranks documents, the reader reads them as text."""
import json
from unittest.mock import MagicMock

import pytest

from engine.pipeline import ContextCanvasEngine
from semantics.schema import ExtractedEvent, ExtractionPayload


@pytest.fixture
def engine(tmp_path):
    eng = ContextCanvasEngine(db_path=str(tmp_path / "db"))
    yield eng
    eng.close()


def _doc(engine, source, title, events):
    engine.realizer.extract_structured_context = MagicMock(
        return_value=ExtractionPayload.model_validate({"events": events}))
    engine.ingest_text(f"{title}. Body.", source=source, title=title)


def _ev(temp_id, lemma, roles):
    return {"temp_id": temp_id, "lemma": lemma, "sense_id": f"{lemma}.01", "roles": roles}


@pytest.mark.parametrize("title, names", [
    ("Pecos County, Texas", ["Pecos County, Texas", "Pecos County"]),
    ("Green (Steve Hillage album)", ["Green (Steve Hillage album)", "Green"]),
    ("Imperial", ["Imperial"]),
])
def test_title_names(title, names):
    assert ContextCanvasEngine._title_names(title) == names


def test_graph_source_distances(engine):
    _doc(engine, "d:0", "Imperial", [_ev("a", "locate", {":ARG1": "Imperial", ":ARG2": "Pecos County"})])
    _doc(engine, "d:1", "Fort Stockton", [_ev("b", "locate", {":ARG1": "Fort Stockton", ":ARG2": "Pecos County"})])
    sources, entities = engine.graph.source_distances(["Imperial"])
    assert sources == {"d:0": 0, "d:1": 1}
    assert entities[engine.graph.resolve_entity_id("Pecos County")] == 1
    assert engine.graph.source_distances(["Nobody"]) == ({}, {})


def test_the_bridge_entity_article_is_retrieved_through_its_title(engine):
    # Question 15 of the pilot: "Imperial is in Pecos County" was extracted, but the border
    # relation in the Pecos County article was not. Its title still makes it reachable.
    _doc(engine, "d:0", "Imperial, Texas", [_ev("a", "locate", {":ARG1": "Imperial", ":ARG2": "Pecos County"})])
    _doc(engine, "d:1", "Pecos County, Texas", [])                       # nothing extracted
    _doc(engine, "d:2", "Weather", [_ev("c", "rain", {":ARG0": "Clouds"})])
    ranked = engine.rank_sources("What county does the county where Imperial is located border?")
    assert ranked == ["d:0", "d:1"]
    status = engine.last_query_status
    assert status["stage"] == "retrieval" and status["anchors"] == ["Imperial"]
    assert status["source_distances"] == {"d:0": 0, "d:1": 1}


def test_questions_naming_nothing_retrieve_nothing(engine):
    _doc(engine, "d:0", "Imperial", [_ev("a", "locate", {":ARG1": "Imperial", ":ARG2": "Pecos County"})])
    assert engine.rank_sources("what is it?") == []
    assert engine.last_query_status["stage"] == "graph_no_anchors"


def test_hub_anchors_are_pruned_for_retrieval_too(engine):
    engine.graph.hub_degree = 1
    for i in range(3):
        _doc(engine, f"h:{i}", f"Town {i}", [_ev(f"t{i}", "locate", {":ARG1": f"Town {i}", ":ARG2": "Serbia"})])
    _doc(engine, "d:0", "Tihomir", [_ev("a", "rule", {":ARG0": "Tihomir", ":ARG1": "Town 0"})])
    engine.rank_sources("Which town did Tihomir of Serbia rule?")
    assert engine.last_query_status["anchors"] == ["Tihomir"]
    engine.ablations["hub_pruning"] = False
    engine.rank_sources("Which town did Tihomir of Serbia rule?")
    assert set(engine.last_query_status["anchors"]) == {"Tihomir", "Serbia"}


def test_reset_forgets_titles(engine):
    _doc(engine, "d:0", "Imperial", [_ev("a", "locate", {":ARG1": "Imperial", ":ARG2": "Pecos County"})])
    engine.reset()
    assert engine._source_titles == {}


# ------------------------------------------------------------------ mention links

def _plain(engine, source, title, body):
    """A document from which nothing was extracted."""
    engine.realizer.extract_structured_context = MagicMock(return_value=ExtractionPayload.model_validate({"events": []}))
    engine.ingest_text(f"{title}. {body}", source=source, title=title)


def test_a_document_mentioning_the_bridge_entity_is_reached(engine):
    # The pilot's Imperial question: the second gold paragraph is "Lancaster Crossing", which
    # mentions Pecos County but is not about it, and yielded no events.
    _doc(engine, "d:0", "Imperial, Texas", [_ev("a", "locate", {":ARG1": "Imperial", ":ARG2": "Pecos County"})])
    _plain(engine, "d:1", "Lancaster Crossing", "It lies on the line between Pecos County and Crockett County.")
    _plain(engine, "d:2", "Weather", "Rain falls.")
    assert engine.rank_sources("What county borders the county where Imperial is located?") == ["d:0", "d:1"]
    assert engine.last_query_status["source_links"] == {"d:0": "event", "d:1": "mention"}
    assert engine.last_query_status["source_distances"] == {"d:0": 0, "d:1": 1}


def test_links_at_equal_distance_rank_event_then_title_then_mention(engine):
    _doc(engine, "d:0", "Imperial", [_ev("a", "locate", {":ARG1": "Imperial", ":ARG2": "Pecos County"})])
    _plain(engine, "d:1", "Lancaster Crossing", "Near Pecos County.")                 # mention, distance 1
    _plain(engine, "d:2", "Pecos County, Texas", "A county.")                         # title, distance 1
    _doc(engine, "d:3", "Fort Stockton", [_ev("b", "locate", {":ARG1": "Fort Stockton", ":ARG2": "Pecos County"})])
    engine.rank_sources("Where is Imperial?")
    status = engine.last_query_status
    assert list(status["source_links"].items()) == [("d:0", "event"), ("d:3", "event"), ("d:2", "title"), ("d:1", "mention")]


def test_vague_names_and_hubs_create_no_mention_links(engine):
    engine.graph.hub_degree = 2
    _doc(engine, "d:0", "Imperial", [_ev("a", "record", {":ARG0": "Imperial", ":ARG1": "performances"}),
                                     _ev("b", "locate", {":ARG1": "Imperial", ":ARG2": "Texas"})])
    for i in range(3):    # Texas becomes a hub
        _doc(engine, f"h:{i}", f"Town {i}", [_ev(f"t{i}", "locate", {":ARG1": f"Town {i}", ":ARG2": "Texas"})])
    _plain(engine, "d:1", "Concert", "Their performances were loud.")                  # lower-case extractor phrase
    _plain(engine, "d:2", "Big State", "Texas is large.")                              # mentions only the hub
    ranked = engine.rank_sources("What did Imperial record?")
    assert "d:1" not in ranked and "d:2" not in ranked
    assert not ContextCanvasEngine._mentionable("Abc") and not ContextCanvasEngine._mentionable("film editor")
    assert ContextCanvasEngine._mentionable("Pecos County")


def test_mention_links_follow_the_ablation_switch(engine):
    _doc(engine, "d:0", "Imperial, Texas", [_ev("a", "locate", {":ARG1": "Imperial", ":ARG2": "Pecos County"})])
    _plain(engine, "d:1", "Lancaster Crossing", "It lies between Pecos County and Crockett County.")
    engine.ablations["mention_links"] = False
    assert engine.rank_sources("What county borders the county where Imperial is located?") == ["d:0"]