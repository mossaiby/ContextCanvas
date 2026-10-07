"""Tests for Realizer, Extraction Pipeline, and End-to-End Orchestrator."""
import json
from unittest.mock import MagicMock, patch
import pytest

from engine.pipeline import ContextCanvasEngine
from engine.realizer import Realizer
from semantics.schema import ExtractionPayload, ExtractedEvent


class TestRealizer:
    @pytest.fixture
    def realizer(self):
        return Realizer(config_path="config.json")

    def test_extract_structured_context_success(self, realizer):
        payload_data = {
            "events": [
                {
                    "temp_id": "ev_1",
                    "lemma": "invent",
                    "sense_id": "invent.01",
                    "roles": {":ARG0": "Alice", ":ARG1": "Widget"},
                    "time_context": {"raw_expression": "in 2024"},
                    "spatial_context": {"location_name": "Berlin"},
                    "epistemic_context": {"source": "Test", "confidence": 1.0, "is_speculative": False}
                }
            ],
            "discourse": []
        }
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps(payload_data)
        realizer.client.chat.completions.create = MagicMock(return_value=MagicMock(choices=[mock_choice]))

        result = realizer.extract_structured_context("Alice invented Widget in 2024 in Berlin.")
        assert len(result.events) == 1
        assert result.events[0].lemma == "invent"
        assert result.events[0].sense_id == "invent.01"
        assert result.events[0].roles[":ARG0"] == "Alice"

    def test_extract_structured_context_retry_on_invalid_json(self, realizer):
        bad_choice = MagicMock()
        bad_choice.message.content = "Not JSON output"

        good_choice = MagicMock()
        good_choice.message.content = json.dumps({
            "events": [{
                "temp_id": "ev_1",
                "lemma": "fund",
                "sense_id": "fund.01",
                "roles": {":ARG0": "Agency", ":ARG1": "Project"},
                "epistemic_context": {"source": "Doc", "confidence": 1.0, "is_speculative": False}
            }],
            "discourse": []
        })

        realizer.client.chat.completions.create = MagicMock(side_effect=[
            MagicMock(choices=[bad_choice]),
            MagicMock(choices=[good_choice])
        ])

        result = realizer.extract_structured_context("Agency funded Project.")
        assert len(result.events) == 1
        assert result.events[0].lemma == "fund"
        assert realizer.client.chat.completions.create.call_count == 2

    def test_answer_question_routing(self, realizer):
        mock_choice = MagicMock()
        mock_choice.message.content = "ANSWER: Cambridge University"
        realizer.client.chat.completions.create = MagicMock(return_value=MagicMock(choices=[mock_choice]))

        ans = realizer.answer_question("Where is the lab located?", "Blueprint evidence")
        assert ans == "ANSWER: Cambridge University"


class TestContextCanvasEngine:
    @pytest.fixture
    def engine(self, tmp_path):
        db_path = str(tmp_path / "engine_test_db")
        eng = ContextCanvasEngine(db_path=db_path)
        yield eng
        eng.close()

    def test_ingest_text_with_sense_disambiguation(self, engine):
        payload_data = {
            "events": [
                {
                    "temp_id": "ev_strike",
                    "lemma": "strike",
                    "sense_id": "strike.01",
                    "roles": {":ARG0": "Workers", ":ARG1": "Automobile Plant"},
                    "epistemic_context": {"source": "News", "confidence": 0.9, "is_speculative": False}
                }
            ],
            "discourse": []
        }
        engine.realizer.extract_structured_context = MagicMock(
            return_value=ExtractionPayload.model_validate(payload_data)
        )

        event_ids = engine.ingest_text("Workers strike against the automobile plant.")
        assert len(event_ids) == 1
        assert event_ids[0] == "ev_strike"

        blueprint = engine.graph.query_subgraph_blueprint("Workers")
        assert "Workers" in blueprint

    def test_ask_entity_detection_and_fallback(self, engine):
        ev = ExtractedEvent(
            temp_id="ev_loc",
            lemma="locate",
            sense_id="locate.01",
            roles={":ARG1": "Fermi National Accelerator Laboratory", ":ARG2": "Illinois"}
        )
        engine.ingest_event_direct(ev)

        engine.realizer.answer_question = MagicMock(return_value="ANSWER: Illinois")
        response = engine.ask("Where is Fermi National Accelerator Laboratory located?")
        assert response == "ANSWER: Illinois"

        # Absent entity
        empty_response = engine.ask("Where is Unregistered Lab?")
        assert empty_response == "STATUS: NOT_IN_EVIDENCE"