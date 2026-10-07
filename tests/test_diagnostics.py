"""Tests for Diagnostic Harness and PropBank Catalog remote loading."""
from unittest.mock import MagicMock, patch
import pytest

from evaluation.harness import DiagnosticBenchmarkHarness
from semantics.propbank import PropBankCatalog, PropBankRoleset


def test_diagnostic_harness_execution(tmp_path):
    harness = DiagnosticBenchmarkHarness(db_path=str(tmp_path / "harness_db"))
    results = harness.run_all()

    assert "summary" in results
    assert results["summary"]["total_tests"] == 4
    assert results["summary"]["total_passed"] >= 3
    assert results["temporal_updating"]["passed"] == 1
    assert results["multi_hop_transitivity"]["passed"] == 1
    assert results["epistemic_calibration"]["passed"] == 1
    assert results["negative_constraints"]["passed"] == 1

    harness.close()


def test_propbank_catalog_custom_cache_path(tmp_path):
    cache_file = tmp_path / "custom_catalog.json"
    catalog = PropBankCatalog(cache_file=str(cache_file))

    # Add custom definition and verify cache write/read
    catalog.framesets["test.01"] = PropBankRoleset(
        roleset_id="test.01",
        name="run tests",
        description="execute test verification",
        roles={":ARG0": "tester", ":ARG1": "target code"}
    )
    catalog.lemma_to_senses["test"] = ["test.01"]

    candidates = catalog.get_candidate_senses("test")
    assert len(candidates) == 1
    assert candidates[0].roleset_id == "test.01"


def test_propbank_disambiguate_unknown_lemma():
    catalog = PropBankCatalog(cache_file="nonexistent.json")
    assert catalog.disambiguate_sense("completely_unknown_lemma_xyz", "Context here") is None