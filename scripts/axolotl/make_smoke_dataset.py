#!/usr/bin/env python3
"""Generate the deterministic, deliberately overlength EXP-0036 chat dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_OUTPUT = Path("agent_space/axolotl-exp0036/gemma4-12b-smoke.jsonl")


def _row(index: int) -> dict[str, object]:
    sentence = (
        f"Record {index:02d} checks prepared attention on a fixed text-only sequence with "
        "causal ordering deterministic labels distinct keys and values and no optimizer update. "
    )
    prompt = (sentence * 150).strip()
    answer = (
        "The invariant is preserved when every token uses the same locked mask scale dtype and "
        "head mapping across all compared attention backends."
    )
    return {
        "id": f"exp0036-{index:02d}",
        "conversations": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }


def write_dataset(path: Path, *, records: int = 12) -> dict[str, object]:
    if records < 8:
        raise ValueError("the eight-step harness requires at least eight records")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(_row(index), sort_keys=True, separators=(",", ":")) + "\n"
        for index in range(records)
    ).encode("utf-8")
    path.write_bytes(payload)
    return {
        "path": str(path),
        "records": records,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--records", type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(write_dataset(args.output, records=args.records), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

