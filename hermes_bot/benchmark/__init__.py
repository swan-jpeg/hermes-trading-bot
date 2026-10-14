"""Reproducible local benchmark for MVP retrieval, graph and pipeline.

This package is standalone and opt-in: it is not collected by the normal unit
test suite.  Run it as a module or import ``run_benchmark`` programmatically.
"""

from hermes_bot.benchmark.generator import SyntheticDataset, generate_dataset
from hermes_bot.benchmark.runner import BenchmarkConfig, BenchmarkResult, run_benchmark

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "SyntheticDataset",
    "generate_dataset",
    "run_benchmark",
]
