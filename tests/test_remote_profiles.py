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
