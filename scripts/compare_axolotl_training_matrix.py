#!/usr/bin/env python3
"""Apply the prospective three-seed gate to matched SDPA and FA4 reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gemma4_fa4.axolotl_training_harness import (
    PROSPECTIVE_MATRIX_SEEDS,
    TRAINING_MATRIX_SCHEMA_VERSION,
    compare_full_training_matrix,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        reports = {
            seed: (
                json.loads(
                    (args.root / f"seed-{seed}" / "sdpa" / "report.json").read_text(
                        encoding="utf-8"
                    )
                ),
                json.loads(
                    (args.root / f"seed-{seed}" / "native" / "report.json").read_text(
                        encoding="utf-8"
                    )
                ),
            )
            for seed in PROSPECTIVE_MATRIX_SEEDS
        }
        result = compare_full_training_matrix(reports)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result = {
            "schema_version": TRAINING_MATRIX_SCHEMA_VERSION,
            "passed": False,
            "error": str(exc),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
