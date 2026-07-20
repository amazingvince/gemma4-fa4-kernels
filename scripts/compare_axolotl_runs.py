#!/usr/bin/env python3
"""Compare an Axolotl baseline report with the project 12B FA4 report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gemma4_fa4.axolotl_harness import (
    HARNESS_SCHEMA_VERSION,
    HarnessComparisonError,
    compare_reports,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
        result = compare_reports(baseline, candidate)
    except (OSError, json.JSONDecodeError, HarnessComparisonError, ValueError) as exc:
        result = {
            "schema_version": HARNESS_SCHEMA_VERSION,
            "passed": False,
            "error": str(exc),
            "baseline_report": str(args.baseline),
            "candidate_report": str(args.candidate),
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(f"Axolotl comparison failed: {exc}", file=sys.stderr)
        return 1
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
