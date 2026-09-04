"""Registry mapping short model names (``acp/<name>``) to local ACP adapter binaries.

Ships with defaults for the well-known adapters. Extend it in three ways,
from lowest to highest precedence:

1. Built-in defaults (below).
2. An optional TOML config file (``$LITELLM_ACP_CONFIG`` or
   ``~/.config/litellm-acp/adapters.toml``), loaded once at import time.
3. An optional ``LITELLM_ACP_ADAPTERS`` environment variable (JSON), also
   loaded at import time, applied after the config file.
4. Calling :func:`register_adapter` at runtime, which always wins.
"""

from __future__ import annotations

import json
import os
import shutil
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ACPAdapterNotFoundError

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "litellm-acp" / "adapters.toml"


@dataclass(frozen=True)
class AdapterSpec:
    """How to spawn a local ACP adapter as a subprocess."""

    binary: str
    args: list[str] = field(default_factory=list)

    def argv(self) -> list[str]:
        return [self.binary, *self.args]


ADAPTER_REGISTRY: dict[str, AdapterSpec] = {
    "claude": AdapterSpec(binary="claude-agent-acp"),
    "codex": AdapterSpec(binary="codex-acp"),
}


def register_adapter(name: str, binary: str, args: list[str] | None = None) -> None:
    """Register (or override) an adapter at runtime.

    Example:
        >>> import litellm_acp
        >>> litellm_acp.register_adapter("mistral", "mistral-acp-agent")
        >>> litellm.completion(model="acp/mistral", messages=[...])
    """
    ADAPTER_REGISTRY[name] = AdapterSpec(binary=binary, args=list(args or []))


def _apply_overrides(raw: dict) -> None:
    adapters = raw.get("adapters", raw)
    for name, value in adapters.items():
        if isinstance(value, str):
            register_adapter(name, value)
        elif isinstance(value, dict):
            register_adapter(name, value["binary"], value.get("args"))


def _load_config_file() -> None:
    config_path = Path(os.environ.get("LITELLM_ACP_CONFIG", DEFAULT_CONFIG_PATH))
    if not config_path.is_file():
        return
    try:
        with config_path.open("rb") as fh:
            raw = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return
    _apply_overrides(raw)


def _load_env_override() -> None:
    raw_json = os.environ.get("LITELLM_ACP_ADAPTERS")
    if not raw_json:
        return
    try:
        raw = json.loads(raw_json)
    except json.JSONDecodeError:
        return
    _apply_overrides(raw)


def _load_overrides() -> None:
    _load_config_file()
    _load_env_override()


_load_overrides()


def resolve_adapter(name: str) -> AdapterSpec:
    """Look up the configured adapter for a bare model name like ``claude``."""
    spec = ADAPTER_REGISTRY.get(name)
    if spec is None:
        raise ACPAdapterNotFoundError(name)
    return spec


def resolve_and_check_adapter(name: str) -> AdapterSpec:
    """Like :func:`resolve_adapter`, but raises if the binary isn't on PATH."""
    spec = resolve_adapter(name)
    if shutil.which(spec.binary) is None:
        raise ACPAdapterNotFoundError(name, spec.binary)
    return spec
