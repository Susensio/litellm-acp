import json
import os
from pathlib import Path

import pytest

from litellm_acp import ACPAdapterNotFoundError, register_adapter
from litellm_acp.adapters import (
    DEFAULT_ADAPTERS,
    load_adapters,
    resolve_adapter,
    resolve_and_check_adapter,
)

NO_CONFIG = Path("/nonexistent/adapters.toml")


def test_default_registry_known_adapters():
    assert resolve_adapter("claude").binary == "claude-agent-acp"
    assert resolve_adapter("codex").binary == "codex-acp"


def test_unknown_name_requires_explicit_registration():
    with pytest.raises(ACPAdapterNotFoundError) as excinfo:
        resolve_adapter("some-custom-binary")

    message = str(excinfo.value)
    assert "acp/some-custom-binary" in message
    assert "register_adapter" in message
    assert "adapters.toml" in message


def test_register_adapter_overrides_default():
    register_adapter("claude", "my-claude-fork", args=["--flag"])

    assert resolve_adapter("claude").argv() == ["my-claude-fork", "--flag"]


def test_missing_default_binary_shows_its_install_command(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")

    with pytest.raises(ACPAdapterNotFoundError) as excinfo:
        resolve_and_check_adapter("claude")

    assert "npm install -g @agentclientprotocol/claude-agent-acp" in str(excinfo.value)


def test_missing_custom_binary_shows_generic_hint():
    register_adapter("definitely-missing", "totally-not-a-real-binary-xyz")

    with pytest.raises(ACPAdapterNotFoundError) as excinfo:
        resolve_and_check_adapter("definitely-missing")

    message = str(excinfo.value)
    assert "totally-not-a-real-binary-xyz" in message
    assert "acp/definitely-missing" in message
    assert "register_adapter" in message


def test_binary_present_on_path_resolves(tmp_path, monkeypatch):
    fake_bin = tmp_path / "fake-adapter-bin"
    fake_bin.write_text("#!/bin/sh\necho hi\n")
    fake_bin.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    register_adapter("fake-present", "fake-adapter-bin")
    spec = resolve_and_check_adapter("fake-present")
    assert spec.binary == "fake-adapter-bin"


def test_readme_lists_each_default_install_command():
    readme = (Path(__file__).parents[1] / "README.md").read_text()

    for spec in DEFAULT_ADAPTERS.values():
        assert spec.install_hint in readme


def test_no_config_means_defaults():
    assert load_adapters({}, NO_CONFIG) == dict(DEFAULT_ADAPTERS)


def test_config_file_adds_adapters(tmp_path):
    config = tmp_path / "adapters.toml"
    config.write_text("[adapters.mine]\nbinary = 'my-binary'\nargs = ['--config']\n")

    adapters = load_adapters({"LITELLM_ACP_CONFIG": str(config)}, NO_CONFIG)

    assert adapters["mine"].argv() == ["my-binary", "--config"]
    assert adapters["claude"] == DEFAULT_ADAPTERS["claude"]


def test_default_config_path_is_read_without_the_env_variable(tmp_path):
    config = tmp_path / "adapters.toml"
    config.write_text("[adapters]\nmine = 'my-binary'\n")

    assert load_adapters({}, config)["mine"].binary == "my-binary"


def test_env_variable_overrides_config_file_overrides_defaults(tmp_path):
    config = tmp_path / "adapters.toml"
    config.write_text("[adapters]\nclaude = 'from-config'\ncodex = 'from-config'\n")
    environ = {
        "LITELLM_ACP_CONFIG": str(config),
        "LITELLM_ACP_ADAPTERS": json.dumps({"codex": {"binary": "from-env", "args": ["--x"]}}),
    }

    adapters = load_adapters(environ, NO_CONFIG)

    assert adapters["claude"].binary == "from-config"
    assert adapters["codex"].argv() == ["from-env", "--x"]


def test_overriding_a_default_drops_its_install_command():
    # The npm command installs the official adapter, not whatever replaced it.
    adapters = load_adapters({"LITELLM_ACP_ADAPTERS": '{"claude": "my-fork"}'}, NO_CONFIG)

    assert adapters["claude"].install_hint is None


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        '["a", "list"]',
        '{"broken": {"args": ["no binary"]}}',
        '{"broken": 42}',
    ],
)
def test_malformed_env_variable_is_skipped_not_fatal(raw, caplog):
    adapters = load_adapters({"LITELLM_ACP_ADAPTERS": raw}, NO_CONFIG)

    assert "broken" not in adapters
    assert adapters["claude"] == DEFAULT_ADAPTERS["claude"]
    assert caplog.records


def test_unparseable_config_file_is_skipped(tmp_path, caplog):
    config = tmp_path / "adapters.toml"
    config.write_text("this is = = not toml")

    assert load_adapters({"LITELLM_ACP_CONFIG": str(config)}, NO_CONFIG) == dict(DEFAULT_ADAPTERS)
    assert caplog.records
