#!/usr/bin/env python3
"""Static validator for the writing-cute-dsl-kernels skill package."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "SKILL.md",
    "README.md",
    "manifest.json",
    "references/01-execution-model.md",
    "references/02-layouts-tensors-partitioning.md",
    "references/03-kernel-family-patterns.md",
    "references/04-architecture-playbooks.md",
    "references/05-memory-pipelines-synchronization.md",
    "references/06-jit-integration-versioning.md",
    "references/07-correctness-debugging-profiling.md",
    "references/08-source-map.md",
    "references/09-agent-assisted-development.md",
    "references/10-production-exemplars-and-learning-map.md",
    "MERGE_NOTES.md",
    "templates/kernel-design-brief.md",
    "templates/review-checklist.md",
    "templates/benchmark-report.md",
    "tests/evaluation-prompts.md",
    "tests/static-checks.md",
]

REQUIRED_SKILL_TERMS = [
    "Version Gate",
    "Architecture Router",
    "Required Design Workflow",
    "Deliverable Contract",
    "SM80",
    "SM90",
    "tcgen05",
    "predication",
    "acquire",
    "release",
    "Exemplar-First Gate",
    "Agentic Development Gate",
    "SM103",
    "SM120",
    "Cluster Launch Control",
    "Low-Level",
]

REQUIRED_SOURCE_DOMAINS = [
    "https://docs.nvidia.com/cutlass/",
    "https://github.com/NVIDIA/cutlass",
    "https://github.com/Dao-AILab/flash-attention",
    "https://github.com/ScalingIntelligence/KernelBench",
]


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        raise ValueError("SKILL.md must begin with YAML frontmatter")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise ValueError("SKILL.md frontmatter is not closed")
    raw = text[4:end]
    body = text[end + 5 :]
    values: dict[str, str] = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        if ":" not in line:
            raise ValueError(f"unsupported frontmatter line: {line!r}")
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return values, body


def markdown_files() -> Iterable[Path]:
    yield from ROOT.rglob("*.md")


def relative_links(text: str) -> Iterable[str]:
    # Excludes image links and absolute/protocol/anchor links.
    for match in re.finditer(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", text):
        target = match.group(1).split("#", 1)[0].strip()
        if not target or "://" in target or target.startswith("#"):
            continue
        yield target


def main() -> int:
    errors: list[str] = []

    for rel in REQUIRED_FILES:
        if not (ROOT / rel).is_file():
            fail(f"missing required file: {rel}", errors)

    skill_path = ROOT / "SKILL.md"
    if skill_path.is_file():
        text = skill_path.read_text(encoding="utf-8")
        try:
            fm, body = parse_frontmatter(text)
        except ValueError as exc:
            fail(str(exc), errors)
            fm, body = {}, ""

        if set(fm) != {"name", "description"}:
            fail(
                "SKILL.md frontmatter must contain exactly name and description",
                errors,
            )
        name = fm.get("name", "")
        if name != ROOT.name:
            fail(f"frontmatter name {name!r} must equal directory {ROOT.name!r}", errors)
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
            fail("skill name must use lowercase letters, numbers, and hyphens", errors)
        if not fm.get("description", "").startswith("Use when"):
            fail("description must begin with 'Use when'", errors)
        if sum(len(v) for v in fm.values()) > 1024:
            fail("frontmatter values exceed 1024 characters", errors)
        if len(fm.get("description", "")) > 500:
            fail("description should stay at or below 500 characters", errors)
        for term in REQUIRED_SKILL_TERMS:
            if term.lower() not in body.lower():
                fail(f"SKILL.md missing required term/section: {term}", errors)

    source_map = ROOT / "references/08-source-map.md"
    if source_map.is_file():
        source_text = source_map.read_text(encoding="utf-8")
        for domain in REQUIRED_SOURCE_DOMAINS:
            if domain not in source_text:
                fail(f"source map missing required source domain: {domain}", errors)

    reference_requirements = {
        "references/04-architecture-playbooks.md": [
            "SM103", "SM120/SM121", "blackwell_geforce", "not the SM100 tcgen05/TMEM model",
            "PipelineClcFetchAsync", "16-byte response", "cross-proxy fence"
        ],
        "references/06-jit-integration-versioning.md": [
            "bypasses that implicit cache", "compatible only with the TVM FFI backend", "two-pass",
            "cute.compile_to"
        ],
        "references/09-agent-assisted-development.md": [
            "One-Change Experimental Loop", "Human-Review Boundaries", "fast_p"
        ],
        "references/10-production-exemplars-and-learning-map.md": [
            "FlashAttention-4", "QuACK", "KernelBench", "Evidence Hierarchy"
        ],
        "tests/evaluation-prompts.md": [
            "SM120 Architecture Mismatch", "Blank-Page Agent Pressure", "Fake Tensor and Custom Cache ABI",
            "Blackwell CLC Scheduler Race", "Inline-PTX Escape-Hatch Review"
        ],
    }
    for rel, terms in reference_requirements.items():
        path = ROOT / rel
        if not path.is_file():
            continue
        ref_text = path.read_text(encoding="utf-8").lower()
        for term in terms:
            if term.lower() not in ref_text:
                fail(f"{rel} missing merged requirement: {term}", errors)

    # Unfinished markers are allowed only in fillable templates.
    unfinished = re.compile(r"\b(?:TBD|TODO|FIXME|PLACEHOLDER)\b", re.IGNORECASE)
    for path in markdown_files():
        if "templates" in path.parts:
            continue
        matches = list(unfinished.finditer(path.read_text(encoding="utf-8")))
        if matches:
            fail(f"unfinished marker in {path.relative_to(ROOT)}", errors)

    for path in markdown_files():
        text = path.read_text(encoding="utf-8")
        for target in relative_links(text):
            resolved = (path.parent / target).resolve()
            try:
                resolved.relative_to(ROOT.resolve())
            except ValueError:
                fail(f"relative link escapes package in {path.relative_to(ROOT)}: {target}", errors)
                continue
            if not resolved.exists():
                fail(f"broken relative link in {path.relative_to(ROOT)}: {target}", errors)

    manifest_path = ROOT / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            fail(f"manifest.json invalid: {exc}", errors)
        else:
            if manifest.get("name") != ROOT.name:
                fail("manifest name does not match directory", errors)
            if manifest.get("entrypoint") != "SKILL.md":
                fail("manifest entrypoint must be SKILL.md", errors)

    if errors:
        print("STATIC VALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    md_count = sum(1 for _ in markdown_files())
    total_words = sum(
        len(path.read_text(encoding="utf-8").split()) for path in markdown_files()
    )
    print("STATIC VALIDATION PASSED")
    print(f"Package: {ROOT.name}")
    print(f"Markdown files: {md_count}")
    print(f"Approximate words: {total_words}")
    print("Note: GPU compilation and agent scenario evaluations are not static checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
