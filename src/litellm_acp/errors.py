"""Errors raised by litellm-acp, with actionable install instructions."""

from __future__ import annotations

_INSTALL_HINTS: dict[str, str] = {
    "claude": "npm install -g @agentclientprotocol/claude-agent-acp",
    "codex": "npm install -g @agentclientprotocol/codex-acp",
    "gemini": "npm install -g @google/gemini-cli   # ships the `gemini` binary; litellm-acp runs `gemini --acp`",
}

_GENERIC_HINT = (
    "install the adapter for '{name}' and make sure the `{binary}` binary is on your PATH, "
    "or point litellm-acp at a different binary with:\n"
    "    import litellm_acp\n"
    "    litellm_acp.register_adapter('{name}', '<binary-name>')"
)


class ACPAdapterNotFoundError(Exception):
    """Raised when the local CLI binary for an ``acp/<name>`` model isn't on PATH."""

    def __init__(self, name: str, binary: str) -> None:
        hint = _INSTALL_HINTS.get(name) or _GENERIC_HINT.format(name=name, binary=binary)
        message = (
            f"litellm-acp: could not find the '{binary}' executable on your PATH "
            f"for model 'acp/{name}'.\n"
            f"Install it with:\n    {hint}\n"
            f"Then verify with:\n    which {binary}"
        )
        super().__init__(message)
        self.name = name
        self.binary = binary
