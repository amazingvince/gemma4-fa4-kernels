#!/usr/bin/env python3
"""Generate the deterministic, sequence-bounded EXP-0036 chat dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_OUTPUT = Path("agent_space/axolotl-exp0036/gemma4-12b-smoke.jsonl")
PROMPT_REPETITIONS = 35
PROMPT_PADDING_TOKENS = 9


def _row(
    index: int,
    *,
    prompt_repetitions: int = PROMPT_REPETITIONS,
    prompt_padding_tokens: int = PROMPT_PADDING_TOKENS,
) -> dict[str, object]:
    sentence = (
        f"Record {index:02d} checks prepared attention on a fixed text-only sequence with "
        "causal ordering deterministic labels distinct keys and values and no optimizer update. "
    )
    prompt = (sentence * prompt_repetitions).strip() + (" padding" * prompt_padding_tokens)
    answer = (
        "The invariant is preserved when every token uses the same locked mask scale dtype and "
        "head mapping across all compared attention backends."
    )
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }


def write_dataset(
    path: Path,
    *,
    records: int = 12,
    prompt_repetitions: int = PROMPT_REPETITIONS,
    prompt_padding_tokens: int = PROMPT_PADDING_TOKENS,
) -> dict[str, object]:
    if records < 8:
        raise ValueError("the eight-step harness requires at least eight records")
    if prompt_repetitions < 1:
        raise ValueError("prompt_repetitions must be positive")
    if prompt_padding_tokens < 0:
        raise ValueError("prompt_padding_tokens must be nonnegative")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(
            _row(
                index,
                prompt_repetitions=prompt_repetitions,
                prompt_padding_tokens=prompt_padding_tokens,
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
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
    parser.add_argument("--prompt-repetitions", type=int, default=PROMPT_REPETITIONS)
    parser.add_argument("--prompt-padding-tokens", type=int, default=PROMPT_PADDING_TOKENS)
    args = parser.parse_args()
    print(
        json.dumps(
            write_dataset(
                args.output,
                records=args.records,
                prompt_repetitions=args.prompt_repetitions,
                prompt_padding_tokens=args.prompt_padding_tokens,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
