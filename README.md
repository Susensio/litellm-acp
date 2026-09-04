# litellm-acp

**Bring Your Own Subscription (BYOS)** for [LiteLLM](https://github.com/BerriAI/litellm).

Route `litellm.completion()` / `litellm.acompletion()` calls to local CLI agents you already
pay for with a flat-rate subscription (Claude Pro, ChatGPT Plus, ...) via the
[Agent Client Protocol](https://agentclientprotocol.com) (ACP), instead of a pay-per-token API.

```python
import litellm
import litellm_acp  # registers the "acp" custom provider on import

response = litellm.completion(
    model="acp/claude",
    messages=[{"role": "user", "content": "hello"}],
)
print(response.choices[0].message.content)
```

`litellm_acp` spawns the corresponding ACP adapter binary as a subprocess, speaks the ACP
JSON-RPC protocol over its stdio, and translates the reply back into a standard LiteLLM
`ModelResponse`.

## Install

```bash
pip install litellm-acp
```

You also need the ACP adapter binaries for whichever models you use, installed globally and
on your `PATH` (litellm-acp calls them directly - no `npx`):

| `acp/<name>` | install |
| --- | --- |
| `acp/claude` | `npm install -g @agentclientprotocol/claude-agent-acp` |
| `acp/codex` | `npm install -g @agentclientprotocol/codex-acp` |
| `acp/agy` | not a built-in default - see [docs/agy.md](docs/agy.md) |

Each adapter authenticates using that CLI's own login flow (e.g. `claude` / `codex` login),
not an API key - that's the whole point.

If a binary isn't on your `PATH`, `litellm.completion()` raises a `CustomLLMError` with the
exact install command instead of a confusing subprocess traceback.

## Adding your own adapter

Only registered adapters can be used.
Unknown `acp/<name>` models fail before any subprocess is started.
Add a custom adapter once by one of these methods, from highest to lowest precedence:

**1. Runtime registration** (always wins):

```python
import litellm_acp

litellm_acp.register_adapter("mistral", "mistral-acp-agent", args=["--flag"])
litellm.completion(model="acp/mistral", messages=[...])
```

**2. `LITELLM_ACP_ADAPTERS` environment variable** (JSON):

```bash
export LITELLM_ACP_ADAPTERS='{"mistral": {"binary": "mistral-acp-agent", "args": ["--flag"]}}'
```

**3. A TOML config file**, at `~/.config/litellm-acp/adapters.toml` by default (override the
path with `LITELLM_ACP_CONFIG`):

```toml
[adapters.mistral]
binary = "mistral-acp-agent"
args = ["--flag"]
```

The config file is the recommended way to make custom adapters available across local projects.

## How it works

- Each `completion()` call is **stateless**: litellm-acp spawns a fresh adapter subprocess,
  opens one ACP session, sends the prompt, collects the reply, and tears the subprocess down.
- Since ACP sessions don't accept resent OpenAI-style message history, the full conversation
  is flattened into a single labeled transcript (`[System]`, `[User]`, `[Assistant]`, ...) and
  sent as one ACP prompt. A lone trailing user message is sent verbatim.
- `acompletion()` is the native implementation (the ACP SDK is asyncio-based end to end); the
  sync `completion()` drives the same coroutine, safely detecting whether an event loop is
  already running instead of assuming `asyncio.run()` is safe to call.
- litellm-acp advertises no filesystem/terminal capabilities during ACP's `initialize()`, so
  well-behaved adapters won't ask for local file or shell access - it's a headless bridge, not
  an IDE.

## Caveats

- No streaming yet (`stream=True` isn't implemented) - `astreaming`/`streaming` raise the
  default `CustomLLMError`.
- No true multi-turn *server-side* session reuse - see "How it works" above.
- Token usage isn't reported (local CLI agents don't return LiteLLM-shaped usage stats), so
  cost tracking / budgets that depend on `usage` won't see real numbers.
- LiteLLM resolves `custom_provider_map` entries by their exact `acp/` prefix, but bare model
  names you register (`claude`, `codex`, ...) can collide with names LiteLLM already
  recognizes for *other* providers if you ever call them without the `acp/` prefix - always use
  the full `acp/<name>` form.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The test suite spawns a bundled fake ACP agent (`tests/fixtures/fake_acp_agent.py`) that
speaks the real ACP wire protocol, so it doesn't require any real subscription CLI to be
installed.

## License

MIT
