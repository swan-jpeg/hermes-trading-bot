"""CLI entry point: ``python -m hermes_bot.benchmark``.

All output is written to stdout; no files are created in the repository.
"""
from __future__ import annotations

import argparse
import sys

from hermes_bot.benchmark.runner import BenchmarkConfig, run_benchmark


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reproducible local benchmark for Hermes MVP components."
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--max-total-seconds",
        type=float,
        default=60.0,
    )
    parser.add_argument(
        "--max-scenario-seconds",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--max-rss-mb",
        type=float,
        default=2048.0,
    )
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human-readable summary.",
    )
    args = parser.parse_args(argv)

    config = BenchmarkConfig(
        sizes=tuple(args.sizes) if args.sizes else (10, 100, 1000),
        seed=args.seed,
        max_total_seconds=args.max_total_seconds,
        max_scenario_seconds=args.max_scenario_seconds,
        max_rss_bytes=int(args.max_rss_mb * 1024 * 1024),
        include_pipeline=not args.skip_pipeline,
    )
    result = run_benchmark(config)
    if args.json:
        print(result.to_json())
    else:
        print(result.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
