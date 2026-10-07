"""Comprehensive test suite verifying schemas, context graph, evaluator, and storage scaling."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from engine.pipeline import ContextCanvasEngine
from engine.realizer import Realizer
from evaluation.benchmark_storage import benchmark_storage_scaling, measure_directory_bytes
from evaluation.musique import (
    compute_dataset_metrics,
    compute_exact_match,
    compute_f1,
    normalize_answer,
    run_official_musique_evaluator,
)
from memory.context_graph import ContextGraph
from semantics.propbank import GLOBAL_CATALOG, PropBankCatalog
from semantics.schema import (
    DiscourseRelation,
    EpistemicContext,
    ExtractedEvent,
    ExtractionPayload,
    SpatialContext,
    TemporalContext,
)


def test_propbank_candidate_disambiguation():
    candidates = GLOBAL_CATALOG.get_candidate_senses("strike")
    assert len(candidates) >= 2

    sense_hit = GLOBAL_CATALOG.disambiguate_sense("strike", "The baseball player struck the ball with the bat.")
    assert sense_hit is not None
    assert sense_hit.roleset_id == "strike.01"

    sense_labor = GLOBAL_CATALOG.disambiguate_sense("strike", "Factory workers organized a strike against the employer.")
    assert sense_labor is not None
    assert sense_labor.roleset_id == "strike.02"


def test_propbank_validate_roles_filtering():
    roles = {":ARG0": "Company", ":ARG1": "Employee", ":ARG5": "InvalidExtraRole"}
    valid = GLOBAL_CATALOG.validate_roles("employ.01", roles)
    assert ":ARG0" in valid
    assert ":ARG1" in valid
    assert ":ARG5" not in valid


def test_schema_temporal_interval_parsing():
    tc_single = TemporalContext(raw_expression="Recorded in 2022")
    assert tc_single.start_year == 2022
    assert tc_single.end_year == 2022

    tc_interval = TemporalContext(raw_expression="Active between 2018 and 2024")
    assert tc_interval.start_year == 2018
    assert tc_interval.end_year == 2024


def test_schema_invalid_role_rejection():
    with pytest.raises(ValueError, match="Invalid PropBank role identifier"):
        ExtractedEvent(
            temp_id="e1",
            lemma="invent",
            sense_id="invent.01",
            roles={":completely_invalid_key_xyz": "Dr. Rostova"}
        )


def test_schema_role_alias_normalization():
    ev = ExtractedEvent(
        temp_id="e1",
        lemma="invent",
        sense_id="invent.01",
        roles={":creator": "Dr. Elena Rostova", ":creation": "Project Prometheus"}
    )
    assert ev.roles[":ARG0"] == "Dr. Elena Rostova"
    assert ev.roles[":ARG1"] == "Project Prometheus"


def test_context_graph_transactional_insert(tmp_path):
    db_path = str(tmp_path / "test_context_db")
    cg = ContextGraph(db_path=db_path)

    event = ExtractedEvent(
        temp_id="ev_1",
        lemma="employ",
        sense_id="employ.01",
        roles={":ARG0": "Munich Institute of Physics", ":ARG1": "Dr. Elena Rostova"},
        time_context=TemporalContext(raw_expression="from 2021 to 2025", start_year=2021, end_year=2025),
        spatial_context=SpatialContext(location_name="Munich", parent_region="Bavaria"),
        epistemic_context=EpistemicContext(source="Official Register", confidence=0.95, is_speculative=False)
    )

    ev_id = cg.insert_event(event)
    assert ev_id == "ev_1"

    blueprint = cg.query_subgraph_blueprint("Dr. Elena Rostova")
    assert "Munich Institute of Physics" in blueprint
    assert "Temporal Scope: from 2021 to 2025 [2021-2025]" in blueprint
    assert "Spatial Scope: Munich (Bavaria)" in blueprint
    assert "Epistemic Envelope: Source='Official Register', Certainty=95% [VERIFIED]" in blueprint

    cg.close()


def test_context_graph_discourse_link(tmp_path):
    db_path = str(tmp_path / "test_disc_db")
    cg = ContextGraph(db_path=db_path)

    ev1 = ExtractedEvent(
        temp_id="ev_fund",
        lemma="fund",
        sense_id="fund.01",
        roles={":ARG0": "ERC", ":ARG1": "Project Prometheus"}
    )
    ev2 = ExtractedEvent(
        temp_id="ev_invent",
        lemma="invent",
        sense_id="invent.01",
        roles={":ARG0": "Dr. Rostova", ":ARG1": "Project Prometheus"}
    )

    cg.insert_event(ev1)
    cg.insert_event(ev2)
    cg.insert_discourse_relation(DiscourseRelation(source_id="ev_fund", target_id="ev_invent", relation=":cause"))

    blueprint = cg.query_subgraph_blueprint("Project Prometheus")
    assert "Discourse: --:cause--> [ev_invent]" in blueprint
    cg.close()


def test_evaluator_musique_metrics_calculation(tmp_path):
    preds_file = tmp_path / "preds.jsonl"
    gold_file = tmp_path / "gold.jsonl"

    preds_data = [
        {"id": "q1", "predicted_answer": "University of Cambridge"},
        {"id": "q2", "predicted_answer": "Elena Rostova"}
    ]
    gold_data = [
        {"id": "q1", "answer": "Cambridge University", "answer_aliases": ["University of Cambridge"]},
        {"id": "q2", "answer": "Dr. Elena Rostova", "answer_aliases": ["Elena Rostova"]}
    ]

    with open(preds_file, "w", encoding="utf-8") as f:
        for p in preds_data:
            f.write(json.dumps(p) + "\n")

    with open(gold_file, "w", encoding="utf-8") as f:
        for g in gold_data:
            f.write(json.dumps(g) + "\n")

    metrics = compute_dataset_metrics(preds_file, gold_file)
    assert metrics["answer_em"] == 100.0
    assert metrics["answer_f1"] == 100.0
    assert metrics["evaluated_cases"] == 2


def test_evaluator_string_normalization():
    assert normalize_answer("The Munich Institute of Physics.") == "munich institute of physics"
    assert compute_exact_match("A Project Alpha", "Project Alpha") == 1.0
    assert compute_f1("Elena Rostova", "Dr. Elena Rostova") > 0.6


def test_storage_benchmark_monotonic_disk_bytes(tmp_path):
    metrics = benchmark_storage_scaling(sizes=[10, 50], base_dir=tmp_path / "bench_test")
    assert len(metrics) == 2
    assert metrics[0]["disk_bytes"] > 0
    assert metrics[1]["disk_bytes"] >= metrics[0]["disk_bytes"]
    assert metrics[0]["events_per_second"] > 0.0


def test_pipeline_end_to_end(tmp_path):
    db_path = str(tmp_path / "test_pipe_db")
    engine = ContextCanvasEngine(db_path=db_path)

    ev = ExtractedEvent(
        temp_id="ev_test",
        lemma="locate",
        sense_id="locate.01",
        roles={":ARG1": "CERN Laboratory", ":ARG2": "Geneva"}
    )
    engine.ingest_event_direct(ev)

    engine.realizer.answer_question = MagicMock(return_value="ANSWER: Geneva")
    ans = engine.ask("Where is CERN Laboratory located?")
    assert ans == "ANSWER: Geneva"

    ans_none = engine.ask("Where is Unrelated Facility located?")
    assert ans_none == "STATUS: NOT_IN_EVIDENCE"

    engine.close()