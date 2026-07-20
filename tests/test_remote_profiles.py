from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_profile(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip("'\"")
    return values


def test_remote_profile_placeholders_are_safe_and_arch_specific():
    expected = {"h100": ("sm_90", "sm_90a"), "b300": ("sm_103", "sm_103a")}
    for name, (fa_arch, cute_arch) in expected.items():
        values = parse_profile(ROOT / f"remote/{name}.env.example")
        assert values["REMOTE_HOST"] == "CHANGE_ME"
        assert values["REMOTE_USER"] == "CHANGE_ME"
        assert values["REMOTE_ROOT"].startswith(("~/", "/"))
        assert " " not in values["REMOTE_ROOT"]
        assert values["EXPECTED_ARCH"] == fa_arch
        assert values["FLASH_ATTENTION_ARCH"] == fa_arch
        assert values["CUTE_DSL_ARCH"] == cute_arch


def test_remote_sync_keeps_tracked_agent_space_provenance_files():
    script = (ROOT / "scripts/remote/sync.sh").read_text()
    for include in (
        "--include 'agent_space/README.md'",
        "--include 'agent_space/h100-check-exp0006.json'",
        "--include 'agent_space/h100-check-exp0012.json'",
        "--include 'agent_space/h100-check-exp0013.json'",
        "--include 'agent_space/h100-check-exp0014.json'",
        "--include 'agent_space/h100-check-precommit.json'",
    ):
        assert include in script
        assert script.index(include) < script.index("--exclude 'agent_space/*'")


def test_bundle_json_scan_ignores_upstream_and_virtualenvs():
    script = (ROOT / "scripts/verify_bundle.sh").read_text()
    assert "part == '.upstream' or part.startswith('.venv')" in script


def test_direct_environment_setup_finishes_with_a_strict_target_check():
    script = (ROOT / "scripts/setup_env.sh").read_text()
    assert 'CHECK_ARGS=(--profile "$PROFILE" --expect-arch "$EXPECTED_ARCH" --strict)' in script
    assert "CHECK_ARGS+=(--require-transformers)" in script
    assert 'check_env.py" "${CHECK_ARGS[@]}"' in script
