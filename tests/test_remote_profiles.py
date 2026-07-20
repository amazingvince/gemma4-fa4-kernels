from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dev_dependencies_cover_cpu_probe_runtime() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text().splitlines()
    dev = next(line for line in pyproject if line.startswith("dev = "))
    assert '"numpy>=' in dev


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
        assert values["REMOTE_GPU_LOCK_FILE"].startswith("/tmp/")
        assert values["REMOTE_GPU_LOCK_WAIT_SECONDS"].isdigit()
        assert values["REMOTE_GPU_REQUIRE_IDLE"] in {"0", "1"}


def test_gpu_run_uses_an_exclusive_lease_and_idle_preflight():
    script = (ROOT / "scripts/remote/gpu-run.sh").read_text()
    assert "flock --exclusive --conflict-exit-code 75" in script
    assert "--query-compute-apps=pid,process_name,used_gpu_memory" in script
    assert "REMOTE_GPU_REQUIRE_IDLE" in script
    assert "--no-venv" in script
    assert 'remote_exec "$LEASE_COMMAND"' in script


def test_remote_sync_keeps_tracked_agent_space_provenance_files():
    script = (ROOT / "scripts/remote/sync.sh").read_text()
    tracked_query = 'git -C "$REPO_ROOT" ls-files -z -- agent_space'
    assert tracked_query in script
    assert 'AGENT_SPACE_FILES+=("--include=$tracked_path")' in script
    assert 'AGENT_SPACE_DIRS["$tracked_dir/"]=1' in script
    assert 'AGENT_SPACE_INCLUDES=("--include=agent_space/")' in script
    assert '"${AGENT_SPACE_INCLUDES[@]}"' in script
    assert script.index('"${AGENT_SPACE_INCLUDES[@]}"') < script.index("--exclude 'agent_space/*'")


def test_remote_sync_handles_windows_linked_worktrees_without_silent_data_loss():
    script = (ROOT / "scripts/remote/sync.sh").read_text()
    assert "gitdir=$(sed -n 's/^gitdir: //p' \"$REPO_ROOT/.git\")" in script
    assert 'gitdir=$(wslpath -u "$gitdir")' in script
    assert 'git --git-dir="$gitdir" --work-tree="$REPO_ROOT"' in script
    assert 'echo "Unable to enumerate tracked agent_space files' in script
    assert 'done <"$tracked_list"' in script


def test_cache_object_inspection_never_rewrites_the_compiler_cache():
    script = (ROOT / "scripts/inspect_cute_cache_objects.sh").read_text()
    assert 'cp -- "$object" "$object_copy"' in script
    assert 'objcopy --dump-section .lrodata="$device_image" "$object_copy"' in script
    assert 'digest_after=$(sha256sum "$object"' in script
    assert "cache object changed during inspection" in script


def test_bundle_json_scan_ignores_upstream_and_virtualenvs():
    script = (ROOT / "scripts/verify_bundle.sh").read_text()
    assert "part == '.upstream' or part.startswith('.venv')" in script


def test_direct_environment_setup_finishes_with_a_strict_target_check():
    script = (ROOT / "scripts/setup_env.sh").read_text()
    assert 'CHECK_ARGS=(--profile "$PROFILE" --expect-arch "$EXPECTED_ARCH" --strict)' in script
    assert "CHECK_ARGS+=(--require-transformers)" in script
    assert 'check_env.py" "${CHECK_ARGS[@]}"' in script
