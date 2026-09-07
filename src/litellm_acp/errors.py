"""Errors raised by litellm-acp, with actionable install instructions."""

_INSTALL_HINTS: dict[str, str] = {
    "claude": "npm install -g @agentclientprotocol/claude-agent-acp",
    "codex": "npm install -g @agentclientprotocol/codex-acp",
}

_GENERIC_HINT = (
    "install the adapter for '{name}' and make sure the `{binary}` binary is on your PATH, "
    "or point litellm-acp at a different binary with:\n"
    "    import litellm_acp\n"
    "    litellm_acp.register_adapter('{name}', '<binary-name>')"
)

_REGISTRATION_HINT = (
    "register it with:\n"
    "    import litellm_acp\n"
    "    litellm_acp.register_adapter('{name}', '<binary-name>')\n"
    "or configure it with LITELLM_ACP_ADAPTERS or "
    "~/.config/litellm-acp/adapters.toml"
)


class ACPAdapterNotFoundError(Exception):
    """Raised when an ``acp/<name>`` model is unregistered or its binary is unavailable."""

    def __init__(self, name: str, binary: str | None = None) -> None:
        if binary is None:
            message = (
                f"litellm-acp: no ACP adapter is registered for model 'acp/{name}'.\n"
                + _REGISTRATION_HINT.format(name=name)
            )
            super().__init__(message)
            self.name = name
            self.binary = None
            return

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
