#!/usr/bin/env python3
"""Unified test and evaluation runner for Context Canvas."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from evaluation.benchmark_storage import benchmark_storage_scaling
from evaluation.harness import DiagnosticBenchmarkHarness


def run_unit_tests() -> bool:
    print("=" * 70)
    print(" 1. Running Unit & Integration Test Suite (pytest)")
    print("=" * 70)
    return subprocess.run([sys.executable, "-m", "pytest", "tests/"]).returncode == 0


def run_reasoning_diagnostics() -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print(" 2. Running Diagnostic Reasoning Benchmark Harness")
    print("=" * 70)
    harness = DiagnosticBenchmarkHarness(db_path="eval_diagnostic_kuzu")
    try:
        results = harness.run_all()
    finally:
        harness.close()
    summary = results["summary"]
    print(f"-> Diagnostic Tests Passed: {summary['total_passed']}/{summary['total_tests']} ({summary['accuracy_percent']}%)")
    for axis, data in results.items():
        if axis != "summary":
            print(f"   [{'PASSED' if data.get('passed', 0) > 0 else 'FAILED'}] {axis}")
    return results


def run_storage_benchmarks() -> List[Dict[str, Any]]:
    print("\n" + "=" * 70)
    print(" 3. Running Storage Footprint & Scaling Benchmark")
    print("=" * 70)
    metrics = benchmark_storage_scaling(sizes=[100, 500, 1000], base_dir=Path("eval_storage_data"))
    for m in metrics:
        print(f"-> N = {m['events_count']:<4} | {m['events_per_second']:>6.1f} events/sec | "
              f"{m['disk_bytes']:>8} bytes ({m['bytes_per_event']} B/event) | "
              f"Latency: {m['median_query_latency_ms']:.2f} ms")
    return metrics


def main() -> None:
    if not run_unit_tests():
        print("\nUnit tests failed. Halting evaluation.")
        sys.exit(1)
    diag_results = run_reasoning_diagnostics()
    storage_results = run_storage_benchmarks()
    out_file = Path("benchmark_summary.json")
    out_file.write_text(json.dumps({"diagnostics": diag_results, "storage": storage_results}, indent=2), encoding="utf-8")
    print(f"\nSaved full benchmark summary to {out_file}")


if __name__ == "__main__":
    main()