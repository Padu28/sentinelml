"""Command-line entry point.

    python -m sentinelml.cli generate-sample      # write data/sample/*.csv
    python -m sentinelml.cli run-experiments      # full policy comparison (≈1–2 min)
    python -m sentinelml.cli run-experiments --quick
    python -m sentinelml.cli pipeline             # sample + experiments (used by Docker & CI)
"""
from __future__ import annotations

import argparse

from sentinelml.common.config import PROJECT_ROOT, get_settings
from sentinelml.common.logging import get_logger
from sentinelml.ingestion.loaders import load_stream

log = get_logger("sentinelml.cli")


def generate_sample() -> None:
    out = PROJECT_ROOT / "data" / "sample"
    out.mkdir(parents=True, exist_ok=True)
    df = load_stream("sudden", seed=7)
    cols = ["timestamp", *get_settings().features, "label"]
    df[cols].head(600).to_csv(out / "stream_sample.csv", index=False)
    df[df["quality_fault"]][cols].head(120).to_csv(out / "quality_fault_sample.csv", index=False)
    log.info("Sample data written to %s", out)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="sentinelml")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate-sample")
    e = sub.add_parser("run-experiments")
    e.add_argument("--quick", action="store_true", help="2 seeds instead of 5")
    pl = sub.add_parser("pipeline")
    pl.add_argument("--quick", action="store_true")
    a = p.parse_args(argv)

    from sentinelml.experiments.runner import run_experiments

    if a.cmd == "generate-sample":
        generate_sample()
    elif a.cmd == "run-experiments":
        run_experiments(quick=a.quick)
    elif a.cmd == "pipeline":
        generate_sample()
        run_experiments(quick=a.quick)
        log.info("Pipeline complete → database: %s", get_settings().db_path)


if __name__ == "__main__":
    main()
