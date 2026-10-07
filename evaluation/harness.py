"""Automated diagnostic evaluation harness for Context Graph reasoning axes."""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock

from engine.pipeline import ContextCanvasEngine
from semantics.schema import (
    DiscourseRelation,
    EpistemicContext,
    ExtractedEvent,
    SpatialContext,
    TemporalContext,
)


class DiagnosticBenchmarkHarness:
    """Executes controlled test scenarios across semantic axes."""

    def __init__(self, db_path: str = "benchmark_eval_kuzu"):
        self.engine = ContextCanvasEngine(db_path=db_path)

    def close(self) -> None:
        self.engine.close()

    def run_all(self) -> Dict[str, Any]:
        """Runs all reasoning axes and aggregates results."""
        results = {
            "temporal_updating": self.eval_temporal_updating(),
            "multi_hop_transitivity": self.eval_multi_hop(),
            "epistemic_calibration": self.eval_epistemic_calibration(),
            "negative_constraints": self.eval_negative_constraints(),
        }
        total_tests = sum(len(r["cases"]) for r in results.values())
        total_passed = sum(r["passed"] for r in results.values())
        results["summary"] = {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "accuracy_percent": round((total_passed / total_tests) * 100, 2) if total_tests else 0.0,
        }
        return results

    def eval_temporal_updating(self) -> Dict[str, Any]:
        """Tests whether temporal scopes distinguish historical vs. current states."""
        self.engine.reset()

        # Ingest past job
        self.engine.ingest_event_direct(ExtractedEvent(
            temp_id="ev_job_1",
            lemma="employ",
            sense_id="employ.01",
            roles={":ARG0": "Munich Institute of Physics", ":ARG1": "Dr. Elena Rostova"},
            time_context=TemporalContext(raw_expression="from 2018 to 2021", start_year=2018, end_year=2021),
            spatial_context=SpatialContext(location_name="Munich", parent_region="Germany")
        ))

        # Ingest subsequent job
        self.engine.ingest_event_direct(ExtractedEvent(
            temp_id="ev_job_2",
            lemma="employ",
            sense_id="employ.01",
            roles={":ARG0": "Oxford University", ":ARG1": "Dr. Elena Rostova"},
            time_context=TemporalContext(raw_expression="from 2022 to 2026", start_year=2022, end_year=2026),
            spatial_context=SpatialContext(location_name="Oxford", parent_region="United Kingdom")
        ))
        self.engine.graph.insert_discourse_relation(DiscourseRelation(
            source_id="ev_job_1", target_id="ev_job_2", relation=":before"
        ))

        blueprint = self.engine.graph.query_subgraph_blueprint("Dr. Elena Rostova")
        passed = (
            "Temporal Scope: from 2018 to 2021 [2018-2021]" in blueprint
            and "Temporal Scope: from 2022 to 2026 [2022-2026]" in blueprint
            and "Discourse: --:before--> [ev_job_2]" in blueprint
        )

        return {
            "axis": "temporal_updating",
            "passed": 1 if passed else 0,
            "cases": [{"query": "Elena Rostova career timeline", "expected_in_blueprint": [":before", "Oxford", "Munich"], "passed": passed}]
        }

    def eval_multi_hop(self) -> Dict[str, Any]:
        """Tests 2-hop traversal: Entity A -> Event 1 -> Entity B -> Event 2 -> Entity C."""
        self.engine.reset()

        self.engine.ingest_event_direct(ExtractedEvent(
            temp_id="ev_hop1",
            lemma="employ",
            sense_id="employ.01",
            roles={":ARG0": "Max Planck Institute", ":ARG1": "Dr. Klaus Vogel"}
        ))
        self.engine.ingest_event_direct(ExtractedEvent(
            temp_id="ev_hop2",
            lemma="invent",
            sense_id="invent.01",
            roles={":ARG0": "Dr. Klaus Vogel", ":ARG1": "Detector Helios"}
        ))

        blueprint = self.engine.graph.query_subgraph_blueprint("Detector Helios", max_hops=2)
        passed = "Max Planck Institute" in blueprint and "Dr. Klaus Vogel" in blueprint

        return {
            "axis": "multi_hop_transitivity",
            "passed": 1 if passed else 0,
            "cases": [{"query": "Detector Helios -> Employing Institution", "passed": passed}]
        }

    def eval_epistemic_calibration(self) -> Dict[str, Any]:
        """Tests that speculative claims are explicitly labeled in the context envelope."""
        self.engine.reset()

        self.engine.ingest_event_direct(ExtractedEvent(
            temp_id="ev_rumor",
            lemma="invent",
            sense_id="invent.01",
            roles={":ARG0": "Lab Omega", ":ARG1": "Room-Temperature Superconductor"},
            epistemic_context=EpistemicContext(source="Anonymous Forum Leak", confidence=0.35, is_speculative=True)
        ))

        blueprint = self.engine.graph.query_subgraph_blueprint("Room-Temperature Superconductor")
        passed = "[SPECULATIVE]" in blueprint and "Certainty=35%" in blueprint

        return {
            "axis": "epistemic_calibration",
            "passed": 1 if passed else 0,
            "cases": [{"query": "Superconductor certainty check", "passed": passed}]
        }

    def eval_negative_constraints(self) -> Dict[str, Any]:
        """Tests that queries on absent facts return STATUS: NOT_IN_EVIDENCE."""
        self.engine.reset()

        self.engine.ingest_event_direct(ExtractedEvent(
            temp_id="ev_fact",
            lemma="locate",
            sense_id="locate.01",
            roles={":ARG1": "CERN", ":ARG2": "Geneva"}
        ))

        # Query completely unlinked entity
        ans = self.engine.ask("What is the valuation of Startup X?", target_entity="Startup X")
        passed = ans == "STATUS: NOT_IN_EVIDENCE"

        return {
            "axis": "negative_constraints",
            "passed": 1 if passed else 0,
            "cases": [{"query": "Startup X valuation", "expected": "STATUS: NOT_IN_EVIDENCE", "passed": passed}]
        }