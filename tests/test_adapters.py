import os

import pytest

from litellm_acp import ACPAdapterNotFoundError, register_adapter
from litellm_acp.adapters import resolve_adapter, resolve_and_check_adapter


def test_default_registry_known_adapters():
    assert resolve_adapter("claude").binary == "claude-agent-acp"
    assert resolve_adapter("codex").binary == "codex-acp"
    gemini = resolve_adapter("gemini")
    assert gemini.binary == "gemini"
    assert gemini.args == ["--acp"]


def test_unknown_name_falls_back_to_literal_binary():
    spec = resolve_adapter("some-custom-binary")
    assert spec.argv() == ["some-custom-binary"]


def test_register_adapter_overrides_default():
    register_adapter("claude", "my-claude-fork", args=["--flag"])
    try:
        spec = resolve_adapter("claude")
        assert spec.argv() == ["my-claude-fork", "--flag"]
    finally:
        register_adapter("claude", "claude-agent-acp")


def test_missing_binary_raises_helpful_error():
    register_adapter("definitely-missing", "totally-not-a-real-binary-xyz")
    with pytest.raises(ACPAdapterNotFoundError) as excinfo:
        resolve_and_check_adapter("definitely-missing")
    message = str(excinfo.value)
    assert "totally-not-a-real-binary-xyz" in message
    assert "acp/definitely-missing" in message


def test_binary_present_on_path_resolves(tmp_path, monkeypatch):
    fake_bin = tmp_path / "fake-adapter-bin"
    fake_bin.write_text("#!/bin/sh\necho hi\n")
    fake_bin.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    register_adapter("fake-present", "fake-adapter-bin")
    spec = resolve_and_check_adapter("fake-present")
    assert spec.binary == "fake-adapter-bin"


def test_env_var_override_registers_adapter(monkeypatch):
    monkeypatch.setenv("LITELLM_ACP_ADAPTERS", '{"envtest": {"binary": "env-test-binary", "args": ["--x"]}}')
    from litellm_acp.adapters import _load_env_override

    _load_env_override()
    spec = resolve_adapter("envtest")
    assert spec.argv() == ["env-test-binary", "--x"]
