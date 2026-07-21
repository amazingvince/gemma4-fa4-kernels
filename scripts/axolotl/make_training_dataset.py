#!/usr/bin/env python3
"""Materialize a deterministic real-data slice for EXP-0047."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import urllib.request
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

DATASET_ID = "tatsu-lab/alpaca"
DATASET_REVISION = "dce01c9b08f87459cf36a430d809084718273017"
DATASET_LICENSE = "CC-BY-NC-4.0"
SOURCE_PATH = "data/train-00000-of-00001-a09b74b3ef9c3b56.parquet"
SOURCE_URL = (
    f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/{SOURCE_PATH}"
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_records(
    rows: list[dict[str, Any]],
    *,
    records: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Select and convert source rows without executing dataset-side code."""

    candidates: list[tuple[int, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        instruction = str(row.get("instruction", "")).strip()
        answer = str(row.get("output", "")).strip()
        context = str(row.get("input", "")).strip()
        if not instruction or not answer:
            continue
        prompt = instruction if not context else f"{instruction}\n\nInput:\n{context}"
        candidates.append(
            (
                index,
                {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": answer},
                    ]
                },
            )
        )
    random.Random(seed).shuffle(candidates)
    if len(candidates) < records:
        raise ValueError(f"source has only {len(candidates)} usable rows, need {records}")
    selected = candidates[:records]
    return [item for _index, item in selected], [index for index, _item in selected]


def write_dataset(output: Path, *, records: int = 256, seed: int = 4721) -> dict[str, Any]:
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "gemma4-fa4-exp0047"})
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - pinned HTTPS
        source_bytes = response.read()
    source = pq.read_table(io.BytesIO(source_bytes)).to_pylist()
    converted, source_indices = build_records(source, records=records, seed=seed)
    payload = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in converted
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    provenance = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "license": DATASET_LICENSE,
        "source_path": SOURCE_PATH,
        "source_url": SOURCE_URL,
        "source_sha256": _sha256(source_bytes),
        "sha256": _sha256(payload),
        "records": records,
        "selection_seed": seed,
        "selected_source_indices_sha256": _sha256(
            json.dumps(source_indices, separators=(",", ":")).encode("utf-8")
        ),
    }
    provenance_path = output.with_suffix(output.suffix + ".provenance.json")
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=int, default=256)
    parser.add_argument("--seed", type=int, default=4721)
    args = parser.parse_args()
    result = write_dataset(args.output, records=args.records, seed=args.seed)
    print(json.dumps({"path": str(args.output), **result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
