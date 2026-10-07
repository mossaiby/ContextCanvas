"""Context graph (memory/context_graph.py): persistence, lifecycle, writes and evidence rendering."""
import os

import pytest

import memory.context_graph as cg
from memory.context_graph import ContextGraph, _role_sort_key
from semantics.schema import DiscourseRelation, ExtractedEvent


def _event(temp_id, lemma, roles, **ctx):
    return ExtractedEvent(temp_id=temp_id, lemma=lemma, sense_id=f"{lemma}.01", roles=roles, **ctx)


class FakeResult:
    def __init__(self, rows):
        self.rows = list(rows)

    def has_next(self):
        return bool(self.rows)

    def get_next(self):
        return self.rows.pop(0)


class FakeConn:
    """Answers the mirror-rebuild queries with fixed rows; can fail on chosen statements."""
    def __init__(self, rows=None, fail_on=(), fail_rollback=False):
        self.rows, self.fail_on, self.fail_rollback, self.log = rows or {}, fail_on, fail_rollback, []

    def execute(self, query, params=None):
        self.log.append(query)
        if query.startswith("ROLLBACK") and self.fail_rollback:
            raise RuntimeError("rollback failed")
        if any(f in query for f in self.fail_on):
            raise RuntimeError(f"failed: {query[:30]}")
        for key, rows in self.rows.items():
            if key in query:
                return FakeResult(rows)
        return FakeResult([])

    def close(self):
        raise RuntimeError("close failed")


class FakeKuzu:
    class Database:
        def __init__(self, path):
            pass

        def close(self):
            raise RuntimeError("close failed")

    class Connection:
        def __init__(self, db):
            pass

        def execute(self, query, params=None):
            return FakeResult([])


@pytest.fixture
def graph(tmp_path):
    g = ContextGraph(db_path=str(tmp_path / "db"))
    yield g
    g.close()


# ------------------------------------------------------------------ persistence

def test_mirror_is_rebuilt_from_database_rows(graph):
    graph.conn = FakeConn(rows={
        "RETURN e.id, e.lemma, e.sense_id": [("e1", "employ", "employ.01"), ("e2", "locate", "locate.01")],
        "RETURN e.id, r.role, n.id, n.name": [("e1", ":ARG0", "ent_dlr", "DLR"), ("e1", ":ARG1", "ent_walter", "Ulrich Walter"),
                                              ("e2", ":ARG1", "ent_dlr", "DLR"), ("e2", ":ARG2", "ent_cologne", "Cologne")],
        "t.start_year, t.end_year": [("e1", "time_e1", "1988", 1988, 1988), ("e2", "time_e2", "spring", -1, -1)],
        "p.location_name, p.parent_region": [("e1", "place_e1", "Cologne", "")],
        "c.confidence, c.is_speculative": [("e1", "ep_e1", "doc-7", 0.8, False), ("ghost", "ep_x", "doc-9", 1.0, False)],
        "r.relation": [("e1", "e2", ":before")],
    })
    graph._load_mirror_from_db()
    assert set(graph.event_ids()) == {"e1", "e2"}
    assert graph.resolve_entity_id("ulrich walter") == "ent_walter"
    assert graph.mirror.nodes["e1"]["source"] == "doc-7"
    assert graph.mirror.nodes["time_e2"]["context"]["start_year"] is None        # -1 means "no year"
    assert graph.mirror.nodes["place_e1"]["context"]["parent_region"] is None
    text = graph.build_evidence(["DLR"])["text"]
    assert "Ulrich Walter" in text and "Cologne" in text and "Certainty=80%" in text
    # The rebuilt signatures de-duplicate a re-ingested copy of e1.
    copy = _event("new", "employ", {":ARG0": "DLR", ":ARG1": "Ulrich Walter"}, time_context={"raw_expression": "1988"})
    assert graph.insert_event(copy) == "e1"


@pytest.mark.skipif(not hasattr(cg.kuzu, "__version__"), reason="needs the real kuzu package")
def test_persisted_graph_round_trip(tmp_path):
    path = str(tmp_path / "db")
    g = ContextGraph(db_path=path)
    g.insert_event(_event("e1", "employ", {":ARG0": "DLR", ":ARG1": "Ulrich Walter"},
                          time_context={"raw_expression": "1988 to 1990"}, spatial_context={"location_name": "Cologne"},
                          epistemic_context={"source": "doc-1", "confidence": 0.9}))
    g.insert_event(_event("e2", "locate", {":ARG1": "DLR", ":ARG2": "Cologne"}))
    g.insert_discourse_relation(DiscourseRelation(source_id="e1", target_id="e2", relation=":before"))
    before = g.build_evidence(["DLR"])["text"]
    g.close()
    reopened = ContextGraph(db_path=path)
    try:
        assert set(reopened.event_ids()) == {"e1", "e2"}
        assert reopened.build_evidence(["DLR"])["text"] == before
    finally:
        reopened.close()


def test_failing_queries_degrade_to_empty_results(graph):
    graph.conn = FakeConn(fail_on=("MATCH", "CREATE NODE", "CREATE REL"))
    graph._init_schema()                                   # existing tables are not an error
    assert graph._rows("MATCH (e:Event) RETURN e.id") == []


# ------------------------------------------------------------------ lifecycle

def test_close_tolerates_failing_close_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(cg, "kuzu", FakeKuzu)
    g = ContextGraph(db_path=str(tmp_path / "db"))
    g.conn = FakeConn()
    g.close()
    assert g.conn is None and g.kuzu_db is None


def test_reset_removes_database_files_and_keeps_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(cg, "kuzu", FakeKuzu)
    path = tmp_path / "db"
    g = ContextGraph(db_path=str(path), hub_degree=7, triples_mode=True)
    path.write_text("single-file database")
    (tmp_path / "db.wal").write_text("wal")
    (tmp_path / "db.lock").mkdir()                        # not a file: left alone
    g.reset()
    assert not path.exists() and not (tmp_path / "db.wal").exists() and (tmp_path / "db.lock").exists()
    assert g.hub_degree == 7 and g.triples_mode is True
    path.mkdir()
    (path / "data").write_text("x")
    g.reset()
    assert not path.exists()                               # directory databases are removed too


def test_reset_survives_files_it_cannot_delete(tmp_path, monkeypatch):
    monkeypatch.setattr(cg, "kuzu", FakeKuzu)
    path = tmp_path / "db"
    g = ContextGraph(db_path=str(path))
    path.write_text("x")
    (tmp_path / "db.wal").write_text("wal")

    def refuse(p):
        raise OSError("busy")
    monkeypatch.setattr(cg.os, "remove", refuse)
    g.reset()
    assert path.exists() and (tmp_path / "db.wal").exists()


# ------------------------------------------------------------------ entity ids and writes

def test_entity_lookup_and_allocation(graph):
    assert graph.resolve_entity_id("") is None
    graph.insert_event(_event("e1", "own", {":ARG0": "Bombardier Inc.", ":ARG1": "Learjet"}))
    existing = graph.get_canonical_entity_id("Bombardier")
    assert existing == graph.resolve_entity_id("Bombardier Inc.")
    assert graph.get_canonical_entity_id("Cessna").startswith("ent_")
    assert graph._allocate_entity_id("learjet", reserved=set()) != graph.resolve_entity_id("Learjet")
    assert graph._allocate_entity_id("x", reserved={"ent_x", "ent_x_2"}) == "ent_x_3"


def test_duplicates_and_conflicting_ids(graph):
    first = graph.insert_event(_event("e1", "own", {":ARG0": "A Corp", ":ARG1": "B Corp"}))
    assert graph.insert_event(_event("e9", "own", {":ARG0": "A Corp", ":ARG1": "B Corp"})) == first
    with pytest.raises(ValueError):
        graph.insert_event(_event("e1", "own", {":ARG0": "C Corp", ":ARG1": "D Corp"}))


@pytest.mark.parametrize("fail_rollback", [False, True])
def test_failed_event_write_rolls_back_and_leaves_mirror_untouched(graph, fail_rollback):
    graph.conn = FakeConn(fail_on=("CREATE (e:Event",), fail_rollback=fail_rollback)
    with pytest.raises(RuntimeError):
        graph.insert_event(_event("e1", "own", {":ARG0": "A Corp", ":ARG1": "B Corp"}))
    assert "ROLLBACK;" in graph.conn.log and graph.event_ids() == []


@pytest.mark.parametrize("fail_rollback", [False, True])
def test_discourse_relation_validation_and_rollback(graph, fail_rollback):
    graph.insert_event(_event("e1", "own", {":ARG0": "A Corp", ":ARG1": "B Corp"}))
    graph.insert_event(_event("e2", "sell", {":ARG0": "A Corp", ":ARG1": "B Corp"}))
    with pytest.raises(ValueError):
        graph.insert_discourse_relation(DiscourseRelation(source_id="e1", target_id="missing", relation=":cause"))
    graph.insert_discourse_relation(DiscourseRelation(source_id="e1", target_id="e2", relation=":before"))
    assert "Discourse: --:before--> [e2]" in graph.build_evidence(["A Corp"])["text"]
    graph.conn = FakeConn(fail_on=("DiscourseLink",), fail_rollback=fail_rollback)
    with pytest.raises(RuntimeError):
        graph.insert_discourse_relation(DiscourseRelation(source_id="e2", target_id="e1", relation=":cause"))
    discourse = [(a, b) for a, b, d in graph.mirror.edges(data=True) if d.get("edge_type") == "discourse"]
    assert discourse == [("e1", "e2")]                      # the failed write left no edge


# ------------------------------------------------------------------ evidence

def test_role_order():
    assert sorted([":time", ":ARGM-LOC", ":ARG1", ":ARG0"], key=_role_sort_key) == [":ARG0", ":ARG1", ":ARGM-LOC", ":time"]


def test_hubs_are_reached_but_not_expanded(graph):
    graph.hub_degree = 2
    graph.insert_event(_event("e0", "visit", {":ARG0": "Anna", ":ARG1": "Paris"}))
    for i in range(3):
        graph.insert_event(_event(f"h{i}", "visit", {":ARG0": f"Tourist {i}", ":ARG1": "Paris"}))
    names = graph.build_evidence(["Anna"])["entity_names"]
    assert "Paris" in names and not any(n.startswith("Tourist") for n in names)


def test_rendering_variants(graph):
    graph.insert_event(_event("e1", "exist", {":ARG1": "Atlantis"}, time_context={"raw_expression": "long ago"}))
    text = graph.build_evidence(["Atlantis"])["text"]
    assert "Temporal Scope: long ago" in text and "[" not in text.split("Temporal Scope:")[1]
    graph.render_envelopes = False
    assert "Temporal Scope" not in graph.build_evidence(["Atlantis"])["text"]
    graph.triples_mode = True
    assert "- (Atlantis, exist, -)" in graph.build_evidence(["Atlantis"])["text"]


def test_anchor_without_events_and_character_budget(graph):
    assert graph.build_evidence(["Nobody"])["text"] == ""
    graph._add_entity_node("ent_lonely", "Lonely Entity")
    assert graph.build_evidence(["Lonely Entity"])["anchors"] == ["Lonely Entity"]
    for i in range(5):
        graph.insert_event(_event(f"e{i}", "own", {":ARG0": "Holding Co", ":ARG1": f"Subsidiary Number {i}"}))
    bundle = graph.build_evidence(["Holding Co"], max_chars=150)
    assert 1 <= len(bundle["event_ids"]) < 5


def test_path_candidates_and_blueprint(graph):
    graph.insert_event(_event("e1", "own", {":ARG0": "Bombardier Inc.", ":ARG1": "Learjet"}))
    graph.insert_event(_event("e2", "locate", {":ARG1": "Learjet", ":ARG2": "Wichita"}))
    paths = graph.find_candidate_answers_from_paths("Bombardier Inc.")
    assert [name for name, _ in paths] == ["Learjet", "Wichita"]
    assert graph.find_candidate_answers_from_paths("Nobody") == []
    assert graph.query_subgraph_blueprint("Nobody") == "No matching context graph entries found."
