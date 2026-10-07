"""Storage benchmark helpers (evaluation/benchmark_storage.py) and MuSiQue metrics (evaluation/musique.py)."""
import json
import sys

import pytest

import evaluation.benchmark_storage as bench
from evaluation.musique import compute_dataset_metrics, run_official_musique_evaluator


# ------------------------------------------------------------------ storage benchmark

def _db_layout(tmp_path):
    """A single-file database with a sidecar, a directory database, and a database whose name
    extends the first one's (the prefix trap)."""
    (tmp_path / "kuzu_size_100").write_bytes(b"x" * 10)
    (tmp_path / "kuzu_size_100.wal").write_bytes(b"x" * 5)
    (tmp_path / "kuzu_size_1000").write_bytes(b"x" * 1000)
    (tmp_path / "kuzu_dir").mkdir()
    (tmp_path / "kuzu_dir" / "nested").mkdir()
    (tmp_path / "kuzu_dir" / "nested" / "data").write_bytes(b"x" * 7)
    (tmp_path / "kuzu_dir.lock").mkdir()


def test_measuring_counts_only_the_database_and_its_sidecars(tmp_path):
    _db_layout(tmp_path)
    assert bench.measure_directory_bytes(tmp_path / "kuzu_size_100") == 15        # not the 1000-event database
    assert bench.measure_directory_bytes(tmp_path / "kuzu_dir") == 7
    assert bench.measure_directory_bytes(tmp_path / "not_created_yet") == 0
    assert bench.measure_directory_bytes(tmp_path / "missing_parent" / "db") == 0


def test_removal_never_touches_a_different_database(tmp_path):
    _db_layout(tmp_path)
    bench._remove_path_safely(tmp_path / "kuzu_size_100")
    assert not (tmp_path / "kuzu_size_100").exists() and not (tmp_path / "kuzu_size_100.wal").exists()
    assert (tmp_path / "kuzu_size_1000").exists()                                  # regression: prefix match
    bench._remove_path_safely(tmp_path / "kuzu_dir")
    assert not (tmp_path / "kuzu_dir").exists() and not (tmp_path / "kuzu_dir.lock").exists()
    bench._remove_path_safely(tmp_path / "missing_parent" / "db")                  # nothing to do


def test_benchmark_reports_every_size_and_cleans_up(tmp_path):
    base = tmp_path / "bench"
    results = bench.benchmark_storage_scaling(sizes=[3, 5], base_dir=base)
    assert [r["events_count"] for r in results] == [3, 5]
    assert all(r["events_per_second"] > 0 and r["median_query_latency_ms"] >= 0 for r in results)
    assert not base.exists()


def test_benchmark_cli(monkeypatch, capsys):
    monkeypatch.setattr(bench, "benchmark_storage_scaling", lambda: [{"events_count": 1}])
    bench.main()
    assert json.loads(capsys.readouterr().out) == [{"events_count": 1}]


# ------------------------------------------------------------------ MuSiQue metrics

def _jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
    return path


@pytest.fixture
def files(tmp_path):
    preds = _jsonl(tmp_path / "preds.jsonl", [{"id": "q1", "predicted_answer": "the Paris"}])
    golds = _jsonl(tmp_path / "golds.jsonl", [{"id": "q1", "answer": "Paris", "answer_aliases": ["City of Light"]},
                                              {"id": "q2", "answer": "Lyon"}])
    return tmp_path, preds, golds


def test_unanswered_gold_questions_are_not_counted(files):
    _, preds, golds = files
    metrics = compute_dataset_metrics(preds, golds)
    assert metrics["evaluated_cases"] == 1
    assert metrics["strict_em"] == 0.0 and metrics["normalized_em"] == 100.0 and metrics["answer_f1"] == 100.0


def _script(path, body):
    path.write_text(body)
    return path


def test_official_evaluator_output_is_used(files):
    tmp, preds, golds = files
    script = _script(tmp / "official.py", "import json, sys\nprint(json.dumps({'answer_f1': 42.0, 'args': sys.argv[1:]}))\n")
    metrics = run_official_musique_evaluator(preds, golds, script)
    assert metrics["answer_f1"] == 42.0 and metrics["args"] == [str(preds), str(golds)]


def test_unparseable_evaluator_output_falls_back_to_local_metrics(files):
    tmp, preds, golds = files
    script = _script(tmp / "chatty.py", "print('Evaluation finished!')\n")
    assert run_official_musique_evaluator(preds, golds, script)["evaluated_cases"] == 1


def test_missing_or_failing_evaluator_is_an_error(files):
    tmp, preds, golds = files
    with pytest.raises(FileNotFoundError):
        run_official_musique_evaluator(preds, golds, tmp / "absent.py")
    import subprocess
    with pytest.raises(subprocess.CalledProcessError):
        run_official_musique_evaluator(preds, golds, _script(tmp / "crash.py", "import sys\nsys.exit(3)\n"))
