#!/usr/bin/env python3
"""Compare matched pure-SDPA control and candidate full-training reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gemma4_fa4.axolotl_training_harness import (
    TRAINING_SCHEMA_VERSION,
    compare_full_training_reports,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("control", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare_full_training_reports(
            json.loads(args.control.read_text(encoding="utf-8")),
            json.loads(args.candidate.read_text(encoding="utf-8")),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result = {
            "schema_version": TRAINING_SCHEMA_VERSION,
            "passed": False,
            "error": str(exc),
        }
        status = 1
    else:
        status = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
