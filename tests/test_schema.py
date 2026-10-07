"""Coercion of messy extraction output into the schema (semantics/schema.py)."""
import pytest
from semantics.schema import (
    DiscourseRelation, ExtractedEvent, ExtractionPayload, SpatialContext, TemporalContext, _heuristic_role_for_key,
)


@pytest.mark.parametrize("raw, expected", [
    (None, ("", None, None)),
    ("  in 1990 ", ("in 1990", 1990, 1990)),
    (1987, ("1987", 1987, 1987)),
    (2001.0, ("2001.0", 2001, 2001)),
    ({"expression": "from 1950 to 1960"}, ("from 1950 to 1960", 1950, 1960)),
    ({"text": "spring", "start_year": "1999"}, ("spring", 1999, 1999)),
    ({"time": "x", "start_year": "early", "end_year": "late"}, ("x", None, None)),
    ({"raw_expression": None, "start_year": 1800, "end_year": 1810}, ("", 1800, 1810)),
    (["not", "a", "time"], ("", None, None)),
])
def test_temporal_context_coercion(raw, expected):
    tc = TemporalContext.model_validate(raw)
    assert (tc.raw_expression, tc.start_year, tc.end_year) == expected


@pytest.mark.parametrize("raw, expected", [
    (None, ("", None)),
    (" Paris ", ("Paris", None)),
    ({"location": "Lyon", "parent_region": "France"}, ("Lyon", "France")),
    ({"place": "Canyon, Texas", "parent_region": "United States"}, ("Canyon", "Texas, United States")),
    ({"name": None}, ("", None)),
    (42, ("", None)),
])
def test_spatial_context_coercion(raw, expected):
    sc = SpatialContext.model_validate(raw)
    assert (sc.location_name, sc.parent_region) == expected


def test_event_fills_ids_senses_and_lemmas_from_alternative_keys():
    ev = ExtractedEvent.model_validate({"id": "e7", "frame": "Employ-02", "roles": {}})
    assert (ev.temp_id, ev.sense_id, ev.lemma.lower()) == ("e7", "employ.02", "employ")
    assert ExtractedEvent.model_validate({"lemma": "own"}).sense_id == "own.01"
    bare = ExtractedEvent.model_validate({})
    assert (bare.temp_id, bare.sense_id, bare.lemma) == ("ev_1", "event.01", "event")
    # A frame without a sense number still yields a lemma.
    assert ExtractedEvent.model_validate({"sense_id": "", "frame": "found-1"}).lemma == "found"


@pytest.mark.parametrize("raw, expected", [
    (None, "event.01"), (7, "event.01"), ("Marry_3", "marry.03"), ("  co-write.1 ", "co-write.01"),
    ("!!!.02", "event.02"), ("give", "give.01"),
])
def test_sense_id_normalization(raw, expected):
    assert ExtractedEvent(sense_id=raw).sense_id == expected


def test_roles_from_lists_aliases_and_context_keys():
    ev = ExtractedEvent(roles=[
        {"role": "arg0", "value": "DLR"},
        {"argument": "ARGM-tmp", "entity": "1988"},
        {":owner": "ignored because ARG0 is set"},
        {"tag": "manner", "text": "quickly"},
        {"label": "", "val": ""},
        "not a dict",
    ])
    assert ev.roles == {":ARG0": "DLR", ":ARGM-TMP": "1988", ":manner": "quickly"}
    assert ExtractedEvent(roles="nonsense").roles == {}
    assert ExtractedEvent(roles={":": "x", " ": "y", ":ARG1": "  "}).roles == {}
    with pytest.raises(ValueError):  # pydantic's ValidationError is a ValueError
        ExtractedEvent(roles={":nonsense_role": "x"})
    # Context keys pass through; an alias never overwrites an explicit argument.
    assert ExtractedEvent(roles={":location": "Rome", ":place": "Paris"}).roles == {":location": "Rome"}
    assert ExtractedEvent(roles={"owner": "Bombardier Inc."}).roles == {":ARG0": "Bombardier Inc."}


@pytest.mark.parametrize("key, slot", [
    (":written_by", ":ARG0"), (":target_audience", ":ARG1"), (":hq_city", ":location"),
    (":year_of_x", ":time"), (":misc", ":ARG2"),
])
def test_heuristic_role_mapping(key, slot):
    assert _heuristic_role_for_key(key) == slot


def test_payload_shapes_and_role_collisions():
    assert len(ExtractionPayload.model_validate([{"lemma": "own"}]).events) == 1
    assert ExtractionPayload.model_validate({"lemma": "own", "roles": {}}).events[0].lemma == "own"
    payload = ExtractionPayload.model_validate({"events": [
        "junk",
        {"lemma": "found", "roles": {":ARG0": "A", "written_by": "B", "nested": {"x": 1}, "empty": ""}},
        {"lemma": "x", "roles": {f":ARG{i}": f"v{i}" for i in range(6)} | {"by_whom": "lost"}},
        {"lemma": "y", "roles": {":location": "L", "where_at": "dup"}},
    ], "discourse": None})
    assert payload.discourse == []
    assert payload.events[0].roles == {":ARG0": "A", ":ARG1": "B"}       # collision moves to next free slot
    assert "lost" not in payload.events[1].roles.values()                  # no free core slot
    assert payload.events[2].roles == {":location": "L"}                   # non-core collision dropped


@pytest.mark.parametrize("tc, kept", [
    (["bad"], None), ({"text": ""}, None), ({"time": "1990"}, "1990"), ({"start_year": 1990}, ""),
])
def test_payload_time_context_cleanup(tc, kept):
    ev = ExtractionPayload.model_validate({"events": [{"lemma": "x", "time_context": tc}]}).events[0]
    assert (ev.time_context.raw_expression if ev.time_context else None) == kept


@pytest.mark.parametrize("sc, kept", [
    (["bad"], None), ({"name": ""}, None), ({"place": " Rome "}, "Rome"), ("   ", None), ("Oslo", "Oslo"),
])
def test_payload_spatial_context_cleanup(sc, kept):
    ev = ExtractionPayload.model_validate({"events": [{"lemma": "x", "spatial_context": sc}]}).events[0]
    assert (ev.spatial_context.location_name if ev.spatial_context else None) == kept


@pytest.mark.parametrize("raw, relation", [
    ({"source": "e1", "target": "e2", "rel": "Before"}, ":before"),
    ({"src": "e1", "tgt": "e2", "type": "prior_to"}, ":before"),
    ({"from": "e1", "to": "e2", "relation": "happened after"}, ":after"),
    ({"source_id": "e1", "target_id": "e2", "relation": "enables"}, ":cause"),
    ({"source_id": "e1", "target_id": "e2", "relation": None}, ":cause"),
    ({"source_id": "e1", "target_id": "e2", "relation": ":contrast"}, ":contrast"),
])
def test_discourse_relation_normalization(raw, relation):
    d = DiscourseRelation.model_validate(raw)
    assert d.relation == relation and d.source_id == "e1" and d.target_id == "e2"
