"""End-to-end tests: drive the real ACP wire protocol against a fake local agent.

No real subscription CLI (claude-agent-acp, codex-acp, gemini) is required -
these spawn tests/fixtures/fake_acp_agent.py, which speaks ACP for real over
stdio via the official `acp` SDK's agent-side helpers.
"""

from __future__ import annotations

import litellm

import litellm_acp  # noqa: F401 - import side effect registers the "acp" provider
from litellm_acp import register_adapter


def test_sync_completion_round_trips_through_fake_agent(fake_agent_argv):
    binary, *args = fake_agent_argv
    register_adapter("fake-sync", binary, args=args)

    response = litellm.completion(
        model="acp/fake-sync",
        messages=[{"role": "user", "content": "hello from litellm-acp"}],
    )

    assert response.choices[0].message.content == "echo: hello from litellm-acp"
    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].finish_reason == "stop"


async def test_async_completion_round_trips_through_fake_agent(fake_agent_argv):
    binary, *args = fake_agent_argv
    register_adapter("fake-async", binary, args=args)

    response = await litellm.acompletion(
        model="acp/fake-async",
        messages=[{"role": "user", "content": "async hello"}],
    )

    assert response.choices[0].message.content == "echo: async hello"


def test_multi_turn_history_is_flattened_before_reaching_agent(fake_agent_argv):
    binary, *args = fake_agent_argv
    register_adapter("fake-multiturn", binary, args=args)

    response = litellm.completion(
        model="acp/fake-multiturn",
        messages=[
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "how are you"},
        ],
    )

    content = response.choices[0].message.content
    assert content.startswith("echo: ")
    assert "[User]\nhi" in content
    assert "[Assistant]\nhello" in content
