import builtins
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "record_result", ROOT / "scripts/record_result.py"
)
assert SPEC and SPEC.loader
RECORD_RESULT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECORD_RESULT)


def valid_record():
    return {
        "exp_id": "EXP-0001",
        "kernel": "upstream-fa4",
        "arch": "sm_90",
        "decision": "baseline",
        "hypothesis": "The pinned baseline compiles on the policy stack.",
        "timestamp": "2026-07-19T00:00:00+0000",
        "git_sha": "0" * 40,
        "environment": {},
        "results": [],
    }


def test_record_schema_requires_hypothesis():
    record = valid_record()
    assert RECORD_RESULT.validate_record(record) == []
    del record["hypothesis"]
    assert any(
        "hypothesis" in problem for problem in RECORD_RESULT.validate_record(record)
    )


def test_fallback_validator_rejects_every_used_constraint(monkeypatch):
    original_import = builtins.__import__

    def without_jsonschema(name, *args, **kwargs):
        if name == "jsonschema":
            raise ImportError
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_jsonschema)
    record = valid_record()
    record.update(exp_id="EXP-99", arch="sm_80", environment=[], results={})
    problems = RECORD_RESULT.validate_record(record)
    assert any("does not match" in problem for problem in problems)
    assert any("not in" in problem for problem in problems)
    assert any("environment must be an object" in problem for problem in problems)
    assert any("results must be an array" in problem for problem in problems)


def test_fallback_validator_rejects_non_object_roots(monkeypatch):
    original_import = builtins.__import__

    def without_jsonschema(name, *args, **kwargs):
        if name == "jsonschema":
            raise ImportError
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_jsonschema)
    for record in (None, False, 7, 0.25, "not a record", []):
        assert RECORD_RESULT.validate_record(record) == ["record must be an object"]


def test_fallback_validator_rejects_non_object_results_items(monkeypatch):
    original_import = builtins.__import__

    def without_jsonschema(name, *args, **kwargs):
        if name == "jsonschema":
            raise ImportError
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_jsonschema)
    record = valid_record()
    record["results"] = [{}, "not an object", 7]
    problems = RECORD_RESULT.validate_record(record)
    assert "results[1] must be an object" in problems
    assert "results[2] must be an object" in problems
