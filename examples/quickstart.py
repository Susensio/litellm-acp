"""Minimal litellm-acp usage example.

Prerequisite: install at least one ACP adapter globally, e.g.

    npm install -g @agentclientprotocol/claude-agent-acp

and be logged in / subscribed via that CLI's own auth flow so it can run
without an API key.
"""

import litellm

import litellm_acp  # noqa: F401 - registers the "acp" custom provider

response = litellm.completion(
    model="acp/claude",
    messages=[{"role": "user", "content": "In one sentence, what is the ACP protocol?"}],
)

print(response.choices[0].message.content)
