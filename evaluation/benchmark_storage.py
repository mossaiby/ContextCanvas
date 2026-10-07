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


def _remove_path_safely(target: Path) -> None:
    """Safely removes a file, directory, or wildcard-associated database files."""
    if target.is_dir():
        shutil.rmtree(target, ignore_errors=True)
    elif target.is_file():
        target.unlink(missing_ok=True)

    if target.parent.exists():
        for sibling in target.parent.glob(f"{target.name}*"):
            if sibling.is_dir():
                shutil.rmtree(sibling, ignore_errors=True)
            elif sibling.is_file():
                sibling.unlink(missing_ok=True)


def measure_directory_bytes(dir_path: Path) -> int:
    """Calculates total physical disk footprint across database directory or file(s)."""
    if not dir_path.exists():
        if dir_path.parent.exists():
            return sum(f.stat().st_size for f in dir_path.parent.glob(f"{dir_path.name}*") if f.is_file())
        return 0

    if dir_path.is_file():
        total = 0
        for f in dir_path.parent.glob(f"{dir_path.name}*"):
            if f.is_file():
                total += f.stat().st_size
        return total

    return sum(f.stat().st_size for f in dir_path.glob("**/*") if f.is_file())


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


if __name__ == "__main__":
    metrics = benchmark_storage_scaling()
    print(json.dumps(metrics, indent=2))