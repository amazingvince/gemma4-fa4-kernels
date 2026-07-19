import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_ladders_are_well_formed():
    for path in sorted((ROOT / "configs/benchmarks").glob("*.json")):
        data = json.loads(path.read_text())
        assert data["schema_version"] == 1
        names = set()
        for item in data["entries"]:
            assert item["name"] not in names
            names.add(item["name"])
            assert item["layer"] in {"sliding_attention", "full_attention"}
            assert item["mode"] in {"fwd", "bwd", "fwd_bwd"}
            assert item["batch"] > 0 and item["seqlen"] > 0
            qh = item.get("q_heads", 32)
            kvh = item.get("kv_heads", 16 if item["layer"] == "sliding_attention" else 4)
            assert qh > 0 and kvh > 0 and qh % kvh == 0
