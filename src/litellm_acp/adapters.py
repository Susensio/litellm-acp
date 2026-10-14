"""Registry mapping short model names (``acp/<name>``) to local ACP adapter binaries.

Sources, from lowest to highest precedence:

1. Built-in defaults (below).
2. An optional TOML config file (``$LITELLM_ACP_CONFIG`` or
   ``~/.config/litellm-acp/adapters.toml``).
3. An optional ``LITELLM_ACP_ADAPTERS`` environment variable (JSON).
4. Calling :func:`register_adapter` at runtime, which always wins.

:func:`load_adapters` resolves 1-3 from an environment it's given; the
registry is built from the real one once, at import time.
"""

import json
import logging
import os
import shutil
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ACPAdapterNotFoundError

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "litellm-acp" / "adapters.toml"


@dataclass(frozen=True)
class AdapterSpec:
    """How to spawn a local ACP adapter as a subprocess."""

    binary: str
    args: list[str] = field(default_factory=list)
    # Shown when the binary isn't on PATH; None falls back to a generic hint.
    install_hint: str | None = None

    def argv(self) -> list[str]:
        return [self.binary, *self.args]


DEFAULT_ADAPTERS: Mapping[str, AdapterSpec] = {
    "claude": AdapterSpec(
        binary="claude-agent-acp",
        install_hint="npm install -g @agentclientprotocol/claude-agent-acp",
    ),
    "codex": AdapterSpec(
        binary="codex-acp",
        install_hint="npm install -g @agentclientprotocol/codex-acp",
    ),
}


def load_adapters(
    environ: Mapping[str, str], default_config_path: Path = DEFAULT_CONFIG_PATH
) -> dict[str, AdapterSpec]:
    """The built-in defaults, overridden by the config file, then by the env variable.

    Unreadable sources and malformed entries are logged and skipped, never fatal:
    this runs at import time, where an exception would break `import litellm_acp`.
    """
    adapters = dict(DEFAULT_ADAPTERS)
    config_path = Path(environ.get("LITELLM_ACP_CONFIG", default_config_path))
    if config_path.is_file():
        try:
            with config_path.open("rb") as fh:
                adapters.update(_parse_overrides(tomllib.load(fh), str(config_path)))
        except (OSError, tomllib.TOMLDecodeError):
            logger.warning("failed to read adapter config %s", config_path, exc_info=True)
    if raw_json := environ.get("LITELLM_ACP_ADAPTERS"):
        try:
            adapters.update(_parse_overrides(json.loads(raw_json), "LITELLM_ACP_ADAPTERS"))
        except json.JSONDecodeError:
            logger.warning("failed to parse LITELLM_ACP_ADAPTERS as JSON", exc_info=True)
    return adapters


def _parse_overrides(raw: object, source: str) -> dict[str, AdapterSpec]:
    adapters = raw.get("adapters", raw) if isinstance(raw, dict) else None
    if not isinstance(adapters, dict):
        logger.warning("ignoring %s: expected a table of adapters", source)
        return {}
    parsed: dict[str, AdapterSpec] = {}
    for name, value in adapters.items():
        if isinstance(value, str):
            parsed[name] = AdapterSpec(binary=value)
        elif isinstance(value, dict) and isinstance(value.get("binary"), str):
            parsed[name] = AdapterSpec(binary=value["binary"], args=list(value.get("args") or []))
        else:
            logger.warning("ignoring adapter %r in %s: it needs a 'binary'", name, source)
    return parsed


ADAPTER_REGISTRY: dict[str, AdapterSpec] = load_adapters(os.environ)


def register_adapter(name: str, binary: str, args: list[str] | None = None) -> None:
    """Register (or override) an adapter at runtime.

    Example:
        >>> import litellm_acp
        >>> litellm_acp.register_adapter("mistral", "mistral-acp-agent")
        >>> litellm.completion(model="acp/mistral", messages=[...])
    """
    ADAPTER_REGISTRY[name] = AdapterSpec(binary=binary, args=list(args or []))


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
        raise ACPAdapterNotFoundError(name, spec.binary, spec.install_hint)
    return spec
