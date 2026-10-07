"""Storage footprint and query-latency benchmark measuring actual on-disk byte growth."""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

from memory.context_graph import ContextGraph
from semantics.schema import (
    EpistemicContext,
    ExtractedEvent,
    SpatialContext,
    TemporalContext,
)


def _database_files(path: Path) -> List[Path]:
    """The database itself plus its sidecar files ("<name>.wal", "<name>.lock", ...).

    Matching "<name>.*" rather than the prefix "<name>*" matters: the prefix pattern for
    kuzu_size_100 also matches kuzu_size_1000, a different benchmark database.
    """
    if not path.parent.exists():
        return []
    return [p for p in path.parent.iterdir() if p.name == path.name or p.name.startswith(path.name + ".")]


def _remove_path_safely(target: Path) -> None:
    """Removes a database (file or directory) together with its sidecar files."""
    for path in _database_files(target):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def measure_directory_bytes(dir_path: Path) -> int:
    """Calculates total physical disk footprint across database directory or file(s)."""
    if dir_path.is_dir():
        return sum(f.stat().st_size for f in dir_path.glob("**/*") if f.is_file())
    # Single-file database (or one not yet created): the file plus its sidecar files.
    return sum(f.stat().st_size for f in _database_files(dir_path) if f.is_file())


def benchmark_storage_scaling(
    sizes: List[int] = [100, 500, 1000],
    base_dir: Path = Path("benchmark_storage_data")
) -> List[Dict[str, Any]]:
    """Benchmarks event insert throughput, query latency, and physical byte allocation."""
    results = []
    base_dir.mkdir(parents=True, exist_ok=True)

    for n in sizes:
        db_path = base_dir / f"kuzu_size_{n}"
        _remove_path_safely(db_path)

        cg = ContextGraph(db_path=str(db_path))

        t0 = time.perf_counter()
        for i in range(n):
            ev = ExtractedEvent(
                temp_id=f"ev_{i}",
                lemma="employ",
                sense_id="employ.01",
                roles={":ARG0": f"Institution_{i % 20}", ":ARG1": f"Researcher_{i}"},
                time_context=TemporalContext(
                    raw_expression=f"in {2000 + (i % 25)}",
                    start_year=2000 + (i % 25),
                    end_year=2000 + (i % 25)
                ),
                spatial_context=SpatialContext(location_name=f"City_{i % 10}", parent_region="Region"),
                epistemic_context=EpistemicContext(
                    source="Synthetic Benchmark",
                    confidence=0.95,
                    is_speculative=False
                )
            )
            cg.insert_event(ev)
        insert_duration = time.perf_counter() - t0
        events_per_sec = n / insert_duration if insert_duration > 0 else 0.0

        # Query latency benchmark
        q_times = []
        for i in range(min(50, n)):
            t_q = time.perf_counter()
            _ = cg.query_subgraph_blueprint(f"Researcher_{i}")
            q_times.append(time.perf_counter() - t_q)

        median_latency = sorted(q_times)[len(q_times) // 2] * 1000 if q_times else 0.0

        # Close database handle to force WAL flush and release file locks
        cg.close()

        # Measure verified disk allocation
        disk_bytes = measure_directory_bytes(db_path)
        bytes_per_event = disk_bytes / n if n > 0 else 0.0

        results.append({
            "events_count": n,
            "events_per_second": round(events_per_sec, 2),
            "disk_bytes": disk_bytes,
            "bytes_per_event": round(bytes_per_event, 2),
            "median_query_latency_ms": round(median_latency, 3)
        })

        _remove_path_safely(db_path)

    if base_dir.exists():
        shutil.rmtree(base_dir, ignore_errors=True)

    return results


def main() -> None:
    print(json.dumps(benchmark_storage_scaling(), indent=2))


if __name__ == "__main__":  # pragma: no cover  (script entry point; main() itself is tested)
    main()